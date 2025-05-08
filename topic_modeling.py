# topic_modeling.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import os

def load_data():
    """Load cleaned and labeled posts from CSV."""
    df = pd.read_csv("data/cleaned_labeled_posts.csv")
    return df

def extract_topics(docs, n_topics=10, n_top_words=10):
    """Run NMF to extract topics and top words."""
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tfidf = tfidf_vectorizer.fit_transform(docs)

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(", ".join(top_words))

    doc_topics = W.argmax(axis=1)
    return topics, doc_topics, tfidf_vectorizer, nmf

def save_outputs(df, topics, doc_topics):
    """Add topic columns and save to CSV for frontend use."""
    df["topic_id"] = doc_topics
    df["topic"] = df["topic_id"].apply(lambda x: f"Topic {x}")  # <- Required for Streamlit filters
    df["topic_keywords"] = df["topic_id"].apply(lambda x: topics[x])
    
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/ranked_problems.csv", index=False)
    print("Topics added and saved to output/ranked_problems.csv")

def main():
    df = load_data()
    print("Loaded", len(df), "posts")

    topics, doc_topics, _, _ = extract_topics(df["clean_text"])

    print("\n📚 Topics Discovered:")
    for i, topic in enumerate(topics):
        print(f"🔹 Topic {i}: {topic}")

    save_outputs(df, topics, doc_topics)

if __name__ == "__main__":
    main()
