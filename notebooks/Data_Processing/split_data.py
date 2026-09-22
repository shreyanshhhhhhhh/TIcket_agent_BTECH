import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Split ticket pool into train/val/test CSVs")
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "data", "raw", "all_tickets_full.json"),
        help="Input JSON pool (use all_tickets_full_v2.json after scaling)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(BASE_DIR, "data", "processed"),
        help="Output directory for CSV splits",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df[["title", "description", "category", "priority", "resolution"]]
    df = df.drop_duplicates(subset=["title"], keep="first")

    train, temp = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["category"], random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    print(f"Source: {args.input}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Category counts (train):")
    print(train["category"].value_counts().to_string())


if __name__ == "__main__":
    main()