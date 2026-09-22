"""Quick classification evaluation summary for test/val splits."""
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

embedder = SentenceTransformer("all-MiniLM-L6-v2")
clf = joblib.load("models/logreg_classifier.joblib")

for split in ["test", "val"]:
    df = pd.read_csv(f"data/processed/{split}.csv")
    df["text"] = df["title"].astype(str) + ". " + df["description"].astype(str)
    X = embedder.encode(df["text"].tolist(), show_progress_bar=False)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    conf = probs.max(axis=1)

    print(f"=== {split.upper()} SET ({len(df)} tickets) ===")
    print(f"Accuracy: {accuracy_score(df['category'], preds):.4f}")
    print(f"F1 (macro): {f1_score(df['category'], preds, average='macro'):.4f}")
    print(f"Mean confidence: {conf.mean():.4f}")
    print(classification_report(df["category"], preds))

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(df["category"], preds, labels=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("Labels:", labels)
    for i, row in enumerate(cm):
        print(f"  {labels[i]}: {row.tolist()}")

    mis = (df["category"].values != preds).sum()
    print(f"Misclassified: {mis}/{len(df)} ({100 * mis / len(df):.1f}%)\n")
