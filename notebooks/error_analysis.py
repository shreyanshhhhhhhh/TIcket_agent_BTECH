import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/processed/test_with_predictions.csv")

labels = sorted(df["category"].unique())
cm = confusion_matrix(df["category"], df["predicted"], labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("docs/confusion_matrix.png")
print("Saved confusion matrix plot to docs/confusion_matrix.png")

# Show misclassified examples
misclassified = df[df["category"] != df["predicted"]]
print(f"\nTotal misclassified: {len(misclassified)} out of {len(df)}")
print("\nSample misclassified tickets:")
for _, row in misclassified.head(10).iterrows():
    print(f"\nActual: {row['category']} | Predicted: {row['predicted']} | Confidence: {row['confidence']:.2f}")
    print(f"Text: {row['text'][:150]}")

misclassified.to_csv("docs/misclassified_examples.csv", index=False)