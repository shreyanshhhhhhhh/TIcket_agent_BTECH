import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import json
import os

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

for df in [train_df, val_df, test_df]:
    df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
X_train = embedder.encode(train_df["text"].tolist(), show_progress_bar=True)
X_val = embedder.encode(val_df["text"].tolist(), show_progress_bar=True)
X_test = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)

y_train, y_val, y_test = train_df["category"], val_df["category"], test_df["category"]

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

val_preds = clf.predict(X_val)
test_preds = clf.predict(X_test)
test_probs = clf.predict_proba(X_test)

print("\n=== MAIN MODEL: Embeddings + Logistic Regression ===")
print(f"Val Accuracy: {accuracy_score(y_val, val_preds):.4f} | F1: {f1_score(y_val, val_preds, average='macro'):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.4f} | F1: {f1_score(y_test, test_preds, average='macro'):.4f}")
print(classification_report(y_test, test_preds))

cm = confusion_matrix(y_test, test_preds, labels=clf.classes_)
print("Confusion Matrix (test set):")
print(cm)
print("Labels order:", clf.classes_)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/logreg_classifier.joblib")

# Save embeddings + test predictions for later use (calibration, error analysis)
np.save("models/X_test_embeddings.npy", X_test)
test_df["predicted"] = test_preds
test_df["confidence"] = test_probs.max(axis=1)
test_df.to_csv("data/processed/test_with_predictions.csv", index=False)

results = {
    "model": "Embeddings (MiniLM) + Logistic Regression",
    "val_accuracy": float(accuracy_score(y_val, val_preds)),
    "val_f1_macro": float(f1_score(y_val, val_preds, average="macro")),
    "test_accuracy": float(accuracy_score(y_test, test_preds)),
    "test_f1_macro": float(f1_score(y_test, test_preds, average="macro")),
}
with open("models/logreg_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model, embeddings, and predictions.")