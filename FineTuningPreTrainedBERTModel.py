from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load and preprocess the dataset
def load_and_prepare_data():
    # Assuming CSV file
    dataset = load_dataset("csv", data_files={"train": "Data.csv", "test": "Data.csv"})
    labels = dataset["train"].unique("classification")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return dataset, label2id, id2label

# Tokenize data
def tokenize_data(examples, tokenizer, label2id):
    tokens = tokenizer(examples['mutation_description'], truncation=True, padding=True, max_length=128)
    tokens['labels'] = [label2id[label] for label in examples['classification']]
    return tokens

# Main function
def fine_tune_model():
    dataset, label2id, id2label = load_and_prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Apply tokenization
    tokenized_datasets = dataset.map(lambda x: tokenize_data(x, tokenizer, label2id), batched=True)

    # Load pre-trained model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Keep evaluation at 'epoch'
        save_strategy="epoch",       # Match saving strategy with evaluation
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,  # Requires matching save/eval strategy
    )

    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model fine-tuned and saved!")

# Execute fine-tuning
if __name__ == "__main__":
    fine_tune_model()
