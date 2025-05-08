# visualizations/problem_clustering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import os

def run_clustering(n_clusters=5):
    file_path = "data/cleaned_labeled_posts.csv"

    if not os.path.exists(file_path):
        print(" Data file not found.")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    if "clean_text" not in df.columns or df["clean_text"].isnull().all():
        print("'clean_text' column missing or empty.")
        return pd.DataFrame()

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["clean_text"])

    # UMAP for 2D projection
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X.toarray())
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    # KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    return df[["x", "y", "cluster", "title", "clean_text"]]
