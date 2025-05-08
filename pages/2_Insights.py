# pages/2_Insights.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
from collections import Counter

st.set_page_config(page_title="Insights & Trends", layout="wide")

st.title("Developer Topic Insights")
st.markdown("Explore what developers are struggling with based on topic frequency.")

try:
    df = pd.read_csv("output/ranked_problems.csv")

    topic_freq = df["topic_keywords"].value_counts().reset_index()
    topic_freq.columns = ["Topic", "Count"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=topic_freq, x="Count", y="Topic", palette="viridis")
    plt.title("Topic Frequency")
    plt.xlabel("Number of Posts")
    plt.ylabel("Topics")
    st.pyplot(plt)

except FileNotFoundError:
    st.error("ranked_problems.csv not found. Please run `topic_modeling.py` to generate it.")

st.markdown("---")
st.header("Topic Trends Over Time")

if "created_utc" in df.columns:
    # Convert to datetime
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit='s', errors='coerce')
    df = df.dropna(subset=["created_utc"])

    # Group by month and topic
    df["month"] = df["created_utc"].dt.to_period("M").astype(str)
    trend_data = df.groupby(["month", "topic_keywords"]).size().reset_index(name="count")

    # Show dropdown to filter by topic
    selected_topic = st.selectbox("Select a topic to see its trend:", trend_data["topic_keywords"].unique())
    filtered_trend = trend_data[trend_data["topic_keywords"] == selected_topic]

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=filtered_trend, x="month", y="count", marker="o")
    plt.xticks(rotation=45)
    plt.title(f" Trend for Topic: {selected_topic}")
    plt.xlabel("Month")
    plt.ylabel("Number of Mentions")
    st.pyplot(plt)

else:
    st.warning("`created_utc` column not found. Make sure your raw data includes timestamps.")

st.markdown("---")
st.header("Keyword Co-Occurrence Network")

# Tokenize clean_text
token_lists = df["clean_text"].dropna().apply(lambda x: x.split())

# Generate pairs of words from each post
pairs = []
for tokens in token_lists:
    pairs.extend(combinations(set(tokens), 2))

# Count co-occurring pairs
pair_counts = Counter(pairs)
top_pairs = dict(pair_counts.most_common(30))  # top 30 pairs only

# Build graph
G = nx.Graph()
for (word1, word2), weight in top_pairs.items():
    G.add_edge(word1, word2, weight=weight)

# Plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
edges = G.edges(data=True)
weights = [e[2]['weight'] for e in edges]

nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray",
        width=[w * 0.1 for w in weights], font_size=10, node_size=1500)
st.pyplot(plt)