import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/test_with_predictions.csv")
df["correct"] = (df["category"] == df["predicted"]).astype(int)

bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
df["conf_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)

calibration = df.groupby("conf_bin", observed=True).agg(
    avg_confidence=("confidence", "mean"),
    actual_accuracy=("correct", "mean"),
    count=("correct", "count")
).reset_index()

print("=== CALIBRATION TABLE ===")
print(calibration.to_string(index=False))

plt.figure(figsize=(7, 7))
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.plot(calibration["avg_confidence"], calibration["actual_accuracy"], "o-", label="Model calibration")
plt.xlabel("Predicted Confidence")
plt.ylabel("Actual Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.grid(True)
plt.savefig("docs/reliability_diagram.png")
print("\nSaved reliability diagram to docs/reliability_diagram.png")

calibration.to_csv("docs/calibration_table.csv", index=False)