import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

with open("data/raw/eval_holdout.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

eval_df = pd.DataFrame(eval_data)
eval_df["text"] = eval_df["title"].astype(str) + ". " + eval_df["description"].astype(str)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_eval = embedder.encode(eval_df["text"].tolist(), show_progress_bar=True)

clf = joblib.load("models/logreg_classifier.joblib")
preds = clf.predict(X_eval)

print("=== HELD-OUT EVAL SET RESULTS (true generalization test) ===")
print(f"Accuracy: {accuracy_score(eval_df['category'], preds):.4f}")
print(f"F1 (macro): {f1_score(eval_df['category'], preds, average='macro'):.4f}")
print(classification_report(eval_df["category"], preds))