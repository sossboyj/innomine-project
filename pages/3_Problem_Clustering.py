import streamlit as st
from visualizations.problem_clustering import run_clustering
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Problem Clustering", layout="wide")

st.title(" Problem Clustering")
st.markdown("We use **UMAP + KMeans** to group similar developer issues into semantic clusters.")

# Load data
df = run_clustering()

# Plot scatter chart
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["title"],
    title="🔍 Clustered Developer Problems (UMAP + KMeans)"
)

# Use container-wide plotly chart with unique key
clicked = st.plotly_chart(fig, use_container_width=True, key="umap_plot", click_event=True)

# Show clicked point info
st.subheader(" Selected Problem Details")

# Access click data through `st.session_state` if click support is wired
if "plotly_click" in st.session_state and st.session_state.plotly_click:
    event_data = st.session_state.plotly_click
    point_idx = event_data["points"][0]["pointIndex"]
    clicked_row = df.iloc[point_idx]

    st.write(f"**Cluster:** {clicked_row['cluster']}")
    st.write(f"**Title:** {clicked_row['title']}")
    st.write(f"**Text:** {clicked_row['clean_text']}")
else:
    st.info("Click on a point in the chart to see details here.")
