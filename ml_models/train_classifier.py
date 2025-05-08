import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# load cleaned and labeled data
df_path = "data/cleaned_labeled_posts.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(" cleaned_labeled_posts.csv not found in /data. Run text_cleaner.py first.")

df = pd.read_csv(df_path)

# prepare features and labels
X = df["clean_text"].fillna("")
y = df["problem_signal"]

# vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
X_vec = vectorizer.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# train model (Logistic Regression)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# cross-validation for global score
cv_score = cross_val_score(model, X_vec, y, cv=5, scoring='f1').mean()
print(f"\n Cross-Validated F1 Score: {cv_score:.2f}")

# save model and vectorizer
output_dir = "ml_models"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, f"{output_dir}/problem_classifier.pkl")
joblib.dump(vectorizer, f"{output_dir}/tfidf_vectorizer.pkl")
print("\n Model and vectorizer saved to /ml_models/")
