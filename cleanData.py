import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean(filepath):
    #reads CSV input
    df = pd.read_csv(filepath)

    #keeping needed columns
    needed_cols = ["reviewText", "overall", "asin", "reviewTime"]
    df = df[needed_cols]

    df.rename(columns={
        "reviewText": "review_text",
        "overall": "star_rating"
    }, inplace=True)

    #removes missing or duplicate reviews
    df.dropna(subset=["review_text"], inplace=True)
    df.drop_duplicates(subset=["review_text"], inplace=True)

    #converts the rating (1–5) into sentiment labels
    def rating_to_label(rating):
        if rating <= 2:
            return "negative"
        elif rating == 3:
            return "neutral"
        else:
            return "positive"

    df["sentiment_label"] = df["star_rating"].apply(rating_to_label)

    #text normalization
    df["review_text"] = df["review_text"].str.lower().str.strip()

    #split dataframe train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_and_clean("data/raw/raw_reviews.csv")

    #saves cleaned splits
    train_df.to_csv("data/processed/train_clean.csv", index=False)
    test_df.to_csv("data/processed/test_clean.csv", index=False)
