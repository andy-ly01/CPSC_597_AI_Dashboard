import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from cleanData import load_and_clean
from streamlitCallback import StreamlitCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def main(progress_placeholder=None, progress_bar_placeholder=None, status_placeholder=None, num_epochs=3, batch_size=4):
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    #convert string labels negative, neutral, positive to numeric IDs
    labelToid = {"negative": 0, "neutral": 1, "positive": 2}
    idToLabel = {v: k for k, v in labelToid.items()} 

    train_df["labels"] = train_df["sentiment_label"].map(labelToid)
    test_df["labels"] = test_df["sentiment_label"].map(labelToid)
    
    #splits and creates hugging face data set
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    #initialize a tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=idToLabel,
        label2id=labelToid
    )

    #tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["review_text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )

    #tokenization map to the dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    #remove columns we don't need for training(possibly remove reviewTime? good to keep to know data age)
    remove_cols = ["review_text", "star_rating", "sentiment_label", "asin", "reviewTime"]
    for col in remove_cols:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([col])
        if col in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns([col])

    #set dataset format to PyTorch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    #training Arguments
    training_args = TrainingArguments(
        output_dir="./results",        # save checkpoints
        num_train_epochs=num_epochs,           # tune this as needed
        per_device_train_batch_size=batch_size, #smaller batch size allows for higher it/s (counterintuitive?)
        per_device_eval_batch_size=batch_size,
        logging_steps=100,
        save_steps=500,               #save checkpoint every 500 steps
        learning_rate=2e-5,
        weight_decay=0.01
    )

    # simple metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    # trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[StreamlitCallback(progress_placeholder, progress_bar_placeholder, status_placeholder)]
    )

    trainer.train()

    #evaluate on the test set
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    #save your fine-tuned model
    trainer.save_model("./fine_tuned_model")  # creates a folder with weights, config, etc

if __name__ == "__main__":
    print("Is CUDA available?", torch.cuda.is_available()) #checks if GPU is available
    main()