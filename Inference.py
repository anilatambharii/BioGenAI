from transformers import BertTokenizer, BertForSequenceClassification
import torch  # Import PyTorch

# Define the path to your fine-tuned model
model_path = r"C:\Users\anils\projects\BioGenAI\fine_tuned_model"

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Define a function to classify mutation descriptions
def classify_mutation(description):
    # Tokenize the input description
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Pass the inputs through the model
    outputs = model(**inputs)
    # Get the predicted label index
    predictions = torch.argmax(outputs.logits, dim=1)
    # Map the index to the label
    label = model.config.id2label[predictions.item()]
    return label

# Example mutation description
mutation = "BRCA1 gene variant c.5266dupC linked to breast cancer"
result = classify_mutation(mutation)
print(f"Classification: {result}")
