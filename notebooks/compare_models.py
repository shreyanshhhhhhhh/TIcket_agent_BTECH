import json
import pandas as pd

files = {
    "Baseline (TF-IDF + NB)": "models/baseline_results.json",
    "Main (Embeddings + LogReg)": "models/logreg_results.json",
    "SVM (Embeddings + SVM)": "models/svm_results.json",
}

rows = []
for name, path in files.items():
    with open(path) as f:
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
df.to_csv("docs/model_comparison_table.csv", index=False)
print("\nSaved to docs/model_comparison_table.csv")