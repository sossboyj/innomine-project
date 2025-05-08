import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from textblob import TextBlob
import os

# Download stopwords
nltk.download("stopwords")

# Initialize tokenizer and stopword set
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words("english"))

# Keywords commonly indicating a "problem signal"
PROBLEM_KEYWORDS = [
    "stuck", "error", "can't", "issue", "bug", "frustrated", "problem",
    "help", "how do i", "why does", "not working", "fail", "failed", "fix", "broken"
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alpha
    tokens = tokenizer.tokenize(text)
    cleaned = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(cleaned)

def label_post(text):
    """
    Improved labeling logic:
    Only returns True if the post contains a problem keyword
    AND has slightly negative sentiment.
    """
    if not isinstance(text, str):
        return False
    text = text.lower()
    keyword_match = any(keyword in text for keyword in PROBLEM_KEYWORDS)
    sentiment_score = TextBlob(text).sentiment.polarity

    # Require both keyword and mild negativity to reduce false positives
    return keyword_match and sentiment_score < 0.1

def main():
    input_path = "data/raw_data.json"
    output_path = "data/cleaned_labeled_posts.csv"

    if not os.path.exists(input_path):
        print("raw_data.json not found. Please scrape posts first.")
        return

    df = pd.read_json(input_path)

    # Combine title and body
    df["combined_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)

    # Clean text
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Label data
    df["problem_signal"] = df["combined_text"].apply(label_post)

    # Check class balance
    label_counts = df["problem_signal"].value_counts()
    print("\n Label Distribution:")
    print(label_counts)

    if len(label_counts) < 2:
        print("Only one class detected. Please adjust labeling criteria.")
        return

    # Save labeled data
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Cleaned and labeled posts saved to data/cleaned_labeled_posts.csv")

if __name__ == "__main__":
    main()
