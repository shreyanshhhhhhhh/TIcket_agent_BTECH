import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import json
import os

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

for df in [train_df, val_df, test_df]:
    df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_train = embedder.encode(train_df["text"].tolist(), show_progress_bar=True)
X_val = embedder.encode(val_df["text"].tolist(), show_progress_bar=True)
X_test = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)

y_train, y_val, y_test = train_df["category"], val_df["category"], test_df["category"]

clf = SVC(kernel="linear", probability=True, class_weight="balanced")
clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
test_preds = clf.predict(X_test)

print("=== COMPARISON MODEL: Embeddings + SVM ===")
print(f"Val Accuracy: {accuracy_score(y_val, val_preds):.4f} | F1: {f1_score(y_val, val_preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f} | F1: {f1_score(y_test, test_preds, average='macro'):.4f}")
print(classification_report(y_test, test_preds))

joblib.dump(clf, "models/svm_classifier.joblib")

results = {
    "model": "Embeddings (MiniLM) + SVM",
    "val_accuracy": float(accuracy_score(y_val, val_preds)),
    "val_f1_macro": float(f1_score(y_val, val_preds, average="macro")),
    "test_accuracy": float(accuracy_score(y_test, test_preds)),
    "test_f1_macro": float(f1_score(y_test, test_preds, average="macro")),
}
with open("models/svm_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model and results.")