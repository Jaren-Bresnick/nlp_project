import os
import json
import torch
import logging
import re
from kagglehub import dataset_download
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, DatasetDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure GPU utilization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths
output_dir = "./results"
dataset_path = "./dataset_split"
evaluation_results_dir = "./evaluation_results"
os.makedirs(evaluation_results_dir, exist_ok=True)

# Step 1: Download Dataset from Kaggle
logger.info("Downloading dataset from Kaggle...")
dataset_name = "Cornell-University/arxiv"
path = dataset_download(dataset_name)
input_file = os.path.join(path, "arxiv-metadata-oai-snapshot.json")
logger.info(f"Dataset downloaded to: {input_file}")

# Step 2: Extract Relevant Articles (2020 and Beyond)
logger.info("Filtering relevant articles from 2020 and beyond...")
articles = []
with open(input_file, 'r') as file:
    for line in file:
        try:
            # Parse each line as a JSON object
            article = json.loads(line)
            
            # Get the article's updated date
            updated_date = article.get("update_date", "1900-01-01")  # Default to an old date if not present
            year = int(updated_date.split("-")[0])  # Extract the year from the date
            
            # Filter based on categories of interest and year
            if (
                any(cat in article.get('categories', '') for cat in ["cs.AI", "cs.LG", "cs.CV", "cs.RO", "cs.IR", "cs.ET", "cs.NE", "cs.MA", "cs.CL", "cs.SI"])
                and year >= 2020
            ):
                title = re.sub(r'\s+', ' ', article.get('title', 'No Title').strip())
                abstract = re.sub(r'\s+', ' ', article.get('abstract', 'No Abstract').strip())
                articles.append(f"{title}\n{abstract}\n\n")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            continue

logger.info(f"Total number of relevant articles from 2020 and beyond: {len(articles)}")

# Create Dataset
logger.info("Creating dataset...")
dataset = Dataset.from_dict({"text": articles})

# Step 3: Split Dataset into Train, Validation, and Test
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)  # Split train (80%) and temp (20%)
temp_dataset = split_dataset["test"].train_test_split(test_size=0.5, seed=42)  # Split temp into val (10%) and test (10%)

split_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": temp_dataset["train"],
    "test": temp_dataset["test"],
})

# Save Dataset
split_dataset.save_to_disk(dataset_path)
logger.info(f"Dataset saved to {dataset_path}.")

# Step 4: Tokenize Dataset
logger.info("Tokenizing dataset...")
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 5: Fine-Tune SciBERT for MLM
logger.info("Fine-tuning SciBERT...")
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,  # Implementing gradient clipping
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Step 6: Evaluate on Test Set
logger.info("Evaluating the fine-tuned model on the test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
logger.info(f"Test set evaluation results: {test_results}")

# Save fine-tuned model results
with open(os.path.join(evaluation_results_dir, "test_results_fine_tuned.json"), "w") as f:
    json.dump(test_results, f)

logger.info("Training and evaluation of fine-tuned model complete.")

