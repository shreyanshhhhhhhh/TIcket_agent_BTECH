import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import json

with open("data/raw/all_tickets_full.json", "r", encoding="utf-8") as f:
    import json as j
    data = j.load(f)

df = pd.DataFrame(data)
df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Generating embeddings for full dataset...")
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["category"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies, f1_scores = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    accuracies.append(acc)
    f1_scores.append(f1)
    print(f"Fold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}")

print(f"\nMean Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
print(f"Mean F1 (macro): {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")

results = {
    "fold_accuracies": accuracies,
    "fold_f1_scores": f1_scores,
    "mean_accuracy": float(np.mean(accuracies)),
    "std_accuracy": float(np.std(accuracies)),
    "mean_f1": float(np.mean(f1_scores)),
    "std_f1": float(np.std(f1_scores)),
}
with open("docs/cross_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)