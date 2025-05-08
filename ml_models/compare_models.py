import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/cleaned_labeled_posts.csv")
X = df["clean_text"].fillna("")
y = df["problem_signal"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Run evaluation
for name, model in models.items():
    print(f"\n Testing: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(" Classification Report:\n", classification_report(y_test, y_pred))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    cv_score = cross_val_score(model, X_vec, y, cv=5, scoring="f1").mean()
    print(f" Cross-Validated F1 Score: {cv_score:.2f}")
