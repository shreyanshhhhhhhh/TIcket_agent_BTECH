import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = {
    "Baseline (TF-IDF + NB)": "models/baseline_results.json",
    "Main (Embeddings + LogReg)": "models/logreg_results.json",
    "SVM (Embeddings + SVM)": "models/svm_results.json",
    "SetFit (Fine-tuned MiniLM)": "models/setfit_results.json",
}

rows = []
for name, path in files.items():
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.isfile(full_path):
        print(f"Skipping missing results: {path}")
        continue
    with open(full_path, encoding="utf-8") as f:
        r = json.load(f)
    rows.append({
        "Model": name,
        "Val Accuracy": round(r["val_accuracy"], 4),
        "Val F1 (macro)": round(r["val_f1_macro"], 4),
        "Test Accuracy": round(r["test_accuracy"], 4),
        "Test F1 (macro)": round(r["test_f1_macro"], 4),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
out = os.path.join(BASE_DIR, "docs", "model_comparison_table.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
