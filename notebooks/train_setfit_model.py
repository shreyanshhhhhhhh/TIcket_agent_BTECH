"""
Train a SetFit classifier on train.csv and evaluate on val/test splits.
Run from project root: python notebooks/train_setfit_model.py
"""
import json
import os
import sys

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "setfit_classifier")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "setfit_results.json")
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def load_splits():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "val.csv"))
    test_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test.csv"))

    for df in [train_df, val_df, test_df]:
        df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)

    labels = sorted(train_df["category"].unique().tolist())
    return train_df, val_df, test_df, labels


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "category"]].rename(columns={"category": "label"}))


def main():
    os.chdir(BASE_DIR)
    train_df, val_df, test_df, labels = load_splits()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Labels ({len(labels)}): {labels}")

    train_dataset = to_dataset(train_df)
    val_dataset = to_dataset(val_df)
    test_dataset = to_dataset(test_df)

    print(f"\nLoading SetFit backbone: {BACKBONE}")
    model = SetFitModel.from_pretrained(BACKBONE, labels=labels)

    # num_iterations controls contrastive pair sampling (SetFit's key step)
    num_iterations = 20 if len(train_df) < 1500 else 30

    args = TrainingArguments(
        batch_size=16,
        num_epochs=1,
        num_iterations=num_iterations,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        column_mapping={"text": "text", "label": "label"},
    )

    print("\n=== TRAINING SETFIT ===")
    trainer.train()

    print("\n=== VALIDATION ===")
    val_preds = model.predict(val_df["text"].tolist())
    val_acc = accuracy_score(val_df["category"], val_preds)
    val_f1 = f1_score(val_df["category"], val_preds, average="macro")
    print(f"Val Accuracy: {val_acc:.4f} | F1 (macro): {val_f1:.4f}")

    print("\n=== TEST ===")
    test_preds = model.predict(test_df["text"].tolist())
    test_probs = model.predict_proba(test_df["text"].tolist())
    test_acc = accuracy_score(test_df["category"], test_preds)
    test_f1 = f1_score(test_df["category"], test_preds, average="macro")

    print(f"Test Accuracy: {test_acc:.4f} | F1 (macro): {test_f1:.4f}")
    print(classification_report(test_df["category"], test_preds))

    cm = confusion_matrix(test_df["category"], test_preds, labels=labels)
    print("Confusion Matrix (test set):")
    print(cm)
    print("Labels order:", labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)

    test_df = test_df.copy()
    test_df["predicted"] = test_preds
    test_df["confidence"] = [float(probs.max()) for probs in test_probs]
    test_out = os.path.join(BASE_DIR, "data", "processed", "test_with_predictions_setfit.csv")
    test_df.to_csv(test_out, index=False)

    results = {
        "model": f"SetFit ({BACKBONE})",
        "train_size": len(train_df),
        "num_iterations": num_iterations,
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}")
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved test predictions to {test_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
