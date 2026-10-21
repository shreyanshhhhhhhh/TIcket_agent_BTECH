"""
Train the second-stage classifier: BAAI/bge-base-en-v1.5 + logistic regression.

Stage 1 stays the existing MiniLM + logistic regression model. This script
fits the BGE head on the training split only, then picks the "confusing"
rule on the validation split: when stage 1's top two probabilities are
within 0.15, the BGE prediction replaces it.

The held-out test split is scored once, after the rule is chosen.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
STAGE1_EMBEDDER_NAME = "all-MiniLM-L6-v2"


def ticket_text(df: pd.DataFrame) -> pd.Series:
    return df["title"].astype(str) + ". " + df["description"].astype(str)


def predict_detail(clf, embeddings: np.ndarray):
    probs = clf.predict_proba(embeddings)
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    rows = np.arange(len(probs))
    confidence = probs[rows, top1]
    runner_up = probs[rows, top2]
    labels = clf.classes_[top1]
    return labels, confidence, confidence - runner_up


def apply_two_stage(stage1_labels, stage1_conf, stage1_margin, bge_labels, margin_max, conf_max):
    confusing = (stage1_margin < margin_max) | (stage1_conf < conf_max)
    final = np.where(confusing, bge_labels, stage1_labels)
    return final, confusing


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    y_train = train_df["category"].to_numpy()
    y_val = val_df["category"].to_numpy()
    y_test = test_df["category"].to_numpy()

    print("Encoding stage-1 MiniLM vectors...")
    mini = SentenceTransformer(STAGE1_EMBEDDER_NAME)
    x_train_mini = mini.encode(ticket_text(train_df).tolist(), show_progress_bar=True)
    x_val_mini = mini.encode(ticket_text(val_df).tolist(), show_progress_bar=True)
    x_test_mini = mini.encode(ticket_text(test_df).tolist(), show_progress_bar=True)

    stage1 = joblib.load(os.path.join(MODELS_DIR, "logreg_classifier.joblib"))
    s1_val_labels, s1_val_conf, s1_val_margin = predict_detail(stage1, x_val_mini)
    s1_test_labels, s1_test_conf, s1_test_margin = predict_detail(stage1, x_test_mini)

    print(f"Encoding {BGE_MODEL_NAME}...")
    bge = SentenceTransformer(BGE_MODEL_NAME)
    x_train_bge = bge.encode(ticket_text(train_df).tolist(), show_progress_bar=True)
    x_val_bge = bge.encode(ticket_text(val_df).tolist(), show_progress_bar=True)
    x_test_bge = bge.encode(ticket_text(test_df).tolist(), show_progress_bar=True)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train_bge, y_train)

    bge_val_labels, _, _ = predict_detail(clf, x_val_bge)
    bge_test_labels, _, _ = predict_detail(clf, x_test_bge)

    stage1_val = score(y_val, s1_val_labels)
    stage1_test = score(y_test, s1_test_labels)
    bge_val = score(y_val, bge_val_labels)
    bge_test = score(y_test, bge_test_labels)
    print("Stage-1 val", stage1_val, "test", stage1_test)
    print("BGE-only val", bge_val, "test", bge_test)

    # "Confusing" means the top two stage-1 probabilities are close.
    # Margins above 0.15 are a clear winner, so BGE does not overrule them.
    # The confidence gate stays off: a low score with a wide gap is still a
    # single leading class. Pick the margin with the best validation F1,
    # then the one that calls BGE on fewer tickets.
    candidates = []
    for margin_max in (0.05, 0.08, 0.10, 0.12, 0.15):
        conf_max = 0.0
        preds, confusing = apply_two_stage(
            s1_val_labels, s1_val_conf, s1_val_margin, bge_val_labels, margin_max, conf_max
        )
        metrics = score(y_val, preds)
        candidates.append(
            {
                "margin_max": margin_max,
                "conf_max": conf_max,
                "val_accuracy": metrics["accuracy"],
                "val_f1_macro": metrics["f1_macro"],
                "bge_rate": float(confusing.mean()),
            }
        )

    candidates.sort(
        key=lambda row: (row["val_f1_macro"], row["val_accuracy"], -row["bge_rate"]),
        reverse=True,
    )
    best = candidates[0]
    print("Chosen gate", best)

    test_preds, test_confusing = apply_two_stage(
        s1_test_labels,
        s1_test_conf,
        s1_test_margin,
        bge_test_labels,
        best["margin_max"],
        best["conf_max"],
    )
    two_stage_test = score(y_test, test_preds)
    two_stage_test["bge_rate"] = float(test_confusing.mean())
    print("Two-stage test", two_stage_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "bge_logreg_classifier.joblib")
    joblib.dump(clf, model_path)

    results = {
        "model": "Two-stage: MiniLM logistic regression, then BGE logistic regression when confusing",
        "bge_encoder": BGE_MODEL_NAME,
        "stage1_encoder": STAGE1_EMBEDDER_NAME,
        "text_format": "title + '. ' + description",
        "query_prefix": None,
        "confusion_margin_max": best["margin_max"],
        "confusion_confidence_max": best["conf_max"],
        "stage1_val": stage1_val,
        "stage1_test": stage1_test,
        "bge_only_val": bge_val,
        "bge_only_test": bge_test,
        "two_stage_val": {
            "accuracy": best["val_accuracy"],
            "f1_macro": best["val_f1_macro"],
            "bge_rate": best["bge_rate"],
        },
        "two_stage_test": two_stage_test,
        "val_gate_search_top5": candidates[:5],
        "n_features": int(clf.n_features_in_),
        "classes": [str(c) for c in clf.classes_],
    }
    results_path = os.path.join(MODELS_DIR, "bge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {model_path} and {results_path}")


if __name__ == "__main__":
    main()
