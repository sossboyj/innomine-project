# pages/1_Model_Metrics.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

# Set Streamlit config
st.set_page_config(page_title="Model Metrics", layout="wide")

# Title
st.title(" Classifier Performance Comparison")

# Model files and labels
model_info = {
    "Logistic Regression": {
        "model_path": "ml_models/problem_classifier.pkl",
        "vectorizer_path": "ml_models/tfidf_vectorizer.pkl"
    },
    "Random Forest": {
        "model_path": "ml_models/random_forest_classifier.pkl",
        "vectorizer_path": "ml_models/rf_tfidf_vectorizer.pkl"
    }
}

# Load labeled data
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_labeled_posts.csv")

df = load_data()
X_text = df["clean_text"]
y_true = df["problem_signal"]

# Visualization layout
for model_name, paths in model_info.items():
    st.subheader(f"🔍 {model_name}")

    try:
        model = joblib.load(paths["model_path"])
        vectorizer = joblib.load(paths["vectorizer_path"])

        X_vec = vectorizer.transform(X_text)
        y_pred = model.predict(X_vec)

        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.markdown("#### Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap="Blues"))

        # Confusion Matrix
        st.markdown("####  Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=model.classes_, yticklabels=model.classes_)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Precision-Recall Curve
        if hasattr(model, "predict_proba"):
            st.markdown("####  Precision-Recall Curve")
            y_scores = model.predict_proba(X_vec)[:, 1]
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = average_precision_score(y_true, y_scores)
            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall, precision, label=f"Avg Precision = {avg_precision:.2f}")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall Curve")
            ax_pr.legend()
            st.pyplot(fig_pr)

    except FileNotFoundError as e:
        st.warning(f" {model_name} model or vectorizer not found.")
    except Exception as e:
        st.error(f" Error loading {model_name}: {e}")
