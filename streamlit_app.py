# streamlit_app.py

import streamlit as st
import pandas as pd
from config.topic_labels import TOPIC_LABELS

# first Streamlit command
st.set_page_config(page_title="Innomine – Developer Pain Points", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("output/ranked_problems.csv")
        assert "topic_id" in df.columns and "title" in df.columns
        return df
    except Exception as e:
        st.error(f" Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

st.title(" Innomine: Reddit-Based Developer Problem Scanner")
st.markdown("Analyze real-world issues developers face – discover topics, trends, and opportunities for new tools or fixes.")

if df.empty:
    st.warning(" No data available. Please ensure `output/ranked_problems.csv` exists.")
else:
    # Show Top 5 Trending Topics
    st.markdown("### Trending Topics")
    top_topics = df["topic_id"].value_counts().head(5).index.tolist()
    selected_trending = st.selectbox("Quick jump to a trending topic:", [f"{t} – {TOPIC_LABELS.get(int(t), 'Unknown')}" for t in top_topics])
    trending_topic_id = int(selected_trending.split("–")[0].strip()) if selected_trending else None

    # Sidebar filters
    with st.sidebar:
        st.header("Filter")
        selected_topics = st.multiselect("Select Topic(s):", sorted(df["topic_id"].astype(str).unique()))

    # Apply filters (from sidebar or trending)
    if selected_topics:
        filtered_df = df[df["topic_id"].astype(str).isin(selected_topics)]
    elif trending_topic_id is not None:
        filtered_df = df[df["topic_id"] == trending_topic_id]
    else:
        filtered_df = df

    # Display results
    st.subheader(f"🧾 Showing {len(filtered_df)} posts")
    for _, row in filtered_df.iterrows():
        with st.expander(f"Topic {row['topic_id']} – {row['topic_keywords']}"):
            st.markdown(f"**Title**: {row['title']}")
            st.markdown(f"**Post**: {row['text']}")
            st.markdown(f"[View on Reddit]({row['url']})")
