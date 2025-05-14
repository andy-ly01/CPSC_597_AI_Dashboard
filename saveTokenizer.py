#tokenizer file was missing from ./fine_tuned_model directory
#script saves the tokenizer of a fine-tuned model to a specified directory.
#will implement into trainmodel later
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained("./fine_tuned_model")
print("Tokenizer saved to ./fine_tuned_model")
