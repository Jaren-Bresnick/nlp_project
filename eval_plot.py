import os
import json
import torch
import logging
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

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

# Load the test dataset
logger.info("Loading test dataset...")
split_dataset = load_from_disk(dataset_path)
test_dataset = split_dataset["test"]

# Define the compute_metrics function
def compute_metrics(model, tokenizer, dataset):
    model.eval()
    total_loss = 0
    total_predictions = 0
    correct_predictions = 0
    total_samples = 0

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    def tokenize_and_collate(examples):
        texts = [example['text'] for example in examples]
        tokenized_inputs = tokenizer(texts, truncation=True, max_length=512)
        # Convert to list of dictionaries
        examples = [{k: v[i] for k, v in tokenized_inputs.items()} for i in range(len(texts))]
        # Use data collator to handle masking and padding
        batch = data_collator(examples)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=tokenize_and_collate)

    for batch in dataloader:
        batch_size = batch['input_ids'].size(0)
        total_samples += batch_size

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

            # Get predictions for masked positions
            masked_positions = batch['labels'] != -100
            predictions = logits.argmax(dim=-1)
            masked_labels = batch['labels'][masked_positions]
            masked_predictions = predictions[masked_positions]

            correct_predictions += (masked_predictions == masked_labels).sum().item()
            total_predictions += masked_labels.numel()
            total_loss += outputs.loss.item() * batch_size  # Multiply by batch size

    accuracy = correct_predictions / total_predictions
    average_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(average_loss)).item()
    return {"accuracy": accuracy, "perplexity": perplexity}

# Initialize the results dictionary
all_results = {}

# Evaluate the fine-tuned SciBERT
logger.info("Evaluating fine-tuned SciBERT...")
fine_tuned_model = AutoModelForMaskedLM.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(output_dir)

fine_tuned_metrics = compute_metrics(fine_tuned_model, fine_tuned_tokenizer, test_dataset)
all_results["Fine-Tuned SciBERT"] = fine_tuned_metrics

# Evaluate baseline models
logger.info("Evaluating baseline models...")
baseline_models = {
    "Baseline SciBERT": "allenai/scibert_scivocab_uncased",
    "Baseline BERT": "bert-base-uncased",
}

for base_model_name, base_model_path in baseline_models.items():
    logger.info(f"Evaluating {base_model_name}...")
    baseline_model = AutoModelForMaskedLM.from_pretrained(base_model_path).to(device)
    baseline_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    metrics = compute_metrics(baseline_model, baseline_tokenizer, test_dataset)
    all_results[base_model_name] = metrics

# Save all results to one file
with open(os.path.join(evaluation_results_dir, "all_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)

# Optionally, print the results to the console
for model_name, metrics in all_results.items():
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.4f}")
    print()

# Combine Results for Plotting
logger.info("Plotting results...")
results = {model_name: metrics["perplexity"] for model_name, metrics in all_results.items()}

# Step 8: Plot Comparisons
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), results.values(), color="skyblue")
plt.title("Model Perplexity Comparison")
plt.ylabel("Perplexity (Lower is Better)")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(evaluation_results_dir, "model_comparison.png"))
plt.close()

logger.info("All evaluations complete. Results and plots saved.")
