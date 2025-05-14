# predictionPipeline.py

import pandas as pd
import torch
from datasets import Dataset
from transformers import pipeline, AutoTokenizer
import transformers
from trustScore import trustScore
import time

def run_prediction_pipeline_with_progress(
    input_csv_path="data/processed/test_clean.csv",
    output_csv_path="data/processed/reviews_with_scores.csv",
    progress_bar=None,
    status_placeholder=None
):
    # load model + tokenizer
    model_path = "./fine_tuned_model"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    transformers.logging.set_verbosity_error()
    device_id = 0 if torch.cuda.is_available() else -1

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=tokenizer,
        truncation=True,
        padding=True,
        max_length=512,
        device=device_id
    )

    # load cleaned data
    df = pd.read_csv(input_csv_path)

    # convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    num_rows = len(dataset)
    batch_size = 32
    num_batches = (num_rows + batch_size - 1) // batch_size

    all_sentiments = []
    all_confidences = []
    all_trust_scores = []

    # track time
    start_time = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_rows)

        batch = dataset.select(range(start_idx, end_idx))
        results = sentiment_pipeline(batch["review_text"])

        for text, res in zip(batch["review_text"], results):
            label = res["label"].lower()
            confidence = res["score"]
            trust = trustScore(text, label, conf=confidence)

            all_sentiments.append(label)
            all_confidences.append(confidence)
            all_trust_scores.append(trust)

        # update Streamlit progress after each batch
        if progress_bar and status_placeholder:
            progress = (batch_idx + 1) / num_batches
            progress_bar.progress(min(progress, 1.0))

            elapsed = time.time() - start_time
            speed = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            remaining_batches = num_batches - (batch_idx + 1)
            eta_seconds = remaining_batches / speed if speed > 0 else 0
            eta_min = eta_seconds / 60

            status_placeholder.text(
                f"Prediction progress: {int(progress * 100)}% complete "
                f"({batch_idx + 1}/{num_batches} batches) | ETA: {eta_min:.1f} min"
            )

    # attach new predictions
    df["sentiment"] = all_sentiments
    df["confidence"] = all_confidences
    df["trust_score"] = all_trust_scores

    # save
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved analyzed data to {output_csv_path}")

if __name__ == "__main__":
    run_prediction_pipeline_with_progress()
