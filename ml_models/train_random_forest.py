import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# load data
df_path = "data/cleaned_labeled_posts.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(" cleaned_labeled_posts.csv not found. Please run text_cleaner.py first.")

df = pd.read_csv(df_path)

# preprocess
X = df["clean_text"].fillna("")
y = df["problem_signal"]

# vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)

print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# cross-Validated F1 Score
cv_f1 = cross_val_score(model, X_vec, y, cv=5, scoring="f1_macro").mean()
print(f"\n Cross-Validated F1 Score: {cv_f1:.2f}")

# save Model & Vectorizer
os.makedirs("ml_models", exist_ok=True)
joblib.dump(model, "ml_models/random_forest_classifier.pkl")
joblib.dump(vectorizer, "ml_models/rf_tfidf_vectorizer.pkl")
print(" Random Forest model and vectorizer saved to /ml_models/")
