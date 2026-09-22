"""
Evaluate SetFit on the held-out eval set (never used in training).
Run from project root: python notebooks/evaluate_setfit_holdout.py
"""
import json
import os
import sys

import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_PATH = os.path.join(BASE_DIR, "data", "raw", "eval_holdout.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "setfit_classifier")


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"SetFit model not found at {MODEL_PATH}")
        return 1

    with open(HOLDOUT_PATH, encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_df = pd.DataFrame(eval_data)
    eval_df["text"] = eval_df["title"].astype(str) + ". " + eval_df["description"].astype(str)

    model = SetFitModel.from_pretrained(MODEL_PATH)
    preds = model.predict(eval_df["text"].tolist())

    print("=== SETFIT HELD-OUT EVAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(eval_df['category'], preds):.4f}")
    print(f"F1 (macro): {f1_score(eval_df['category'], preds, average='macro'):.4f}")
    print(classification_report(eval_df["category"], preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
