import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

with open("data/raw/all_tickets_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[["title", "description", "category", "priority", "resolution"]]

train, temp = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["category"], random_state=42)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")