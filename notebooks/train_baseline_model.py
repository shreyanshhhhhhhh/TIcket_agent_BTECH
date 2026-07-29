import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import json
import os

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

for df in [train_df, val_df, test_df]:
    df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df["text"])
X_val = vectorizer.transform(val_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train, y_val, y_test = train_df["category"], val_df["category"], test_df["category"]

clf = MultinomialNB()
clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
test_preds = clf.predict(X_test)

print("=== BASELINE: TF-IDF + Naive Bayes ===")
print(f"Val Accuracy: {accuracy_score(y_val, val_preds):.4f} | F1: {f1_score(y_val, val_preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f} | F1: {f1_score(y_test, test_preds, average='macro'):.4f}")
print(classification_report(y_test, test_preds))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/baseline_nb_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

results = {
    "model": "TF-IDF + Naive Bayes",
    "val_accuracy": accuracy_score(y_val, val_preds),
    "val_f1_macro": f1_score(y_val, val_preds, average="macro"),
    "test_accuracy": accuracy_score(y_test, test_preds),
    "test_f1_macro": f1_score(y_test, test_preds, average="macro"),
}
with open("models/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model and results.")