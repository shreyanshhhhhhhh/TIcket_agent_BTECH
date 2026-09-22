"""
Validate merged v2 ticket pool before train/val/test split.
Run: python notebooks/Data_Processing/validate_v2_pool.py
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOL_PATH = os.path.join(BASE_DIR, "data", "raw", "all_tickets_full_v2.json")
HOLDOUT_PATH = os.path.join(BASE_DIR, "data", "raw", "eval_holdout_v2.json")

VALID_CATEGORIES = {
    "Infrastructure", "Application", "Security", "Database",
    "Storage", "Network", "Access Management",
}
VALID_PRIORITIES = {"Low", "Medium", "High", "Critical"}


def validate_pool(path: str, label: str) -> bool:
    if not os.path.isfile(path):
        print(f"[SKIP] {label}: file not found ({path})")
        return True

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    ok = True
    print(f"\n=== {label} ({len(df)} tickets) ===")

    missing = 0
    for col in ["title", "description", "category", "priority", "resolution"]:
        if col not in df.columns:
            print(f"FAIL missing column: {col}")
            ok = False
            continue
        missing += df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
    print(f"Missing/empty fields: {missing}")

    bad_cats = set(df["category"].unique()) - VALID_CATEGORIES
    if bad_cats:
        print(f"FAIL invalid categories: {bad_cats}")
        ok = False

    bad_pri = set(df["priority"].unique()) - VALID_PRIORITIES
    if bad_pri:
        print(f"FAIL invalid priorities: {bad_pri}")
        ok = False

    dup_titles = len(df) - df["title"].astype(str).str.strip().str.lower().nunique()
    print(f"Duplicate titles: {dup_titles}")

    print("Category balance:")
    counts = df["category"].value_counts()
    print(counts.to_string())
    if counts.min() < counts.max() * 0.7:
        print("WARN: categories are imbalanced (>30% spread)")

    return ok


def main():
    pool_ok = validate_pool(POOL_PATH, "Training pool v2")
    holdout_ok = validate_pool(HOLDOUT_PATH, "Holdout v2")

    if os.path.isfile(POOL_PATH) and os.path.isfile(HOLDOUT_PATH):
        pool = pd.read_json(POOL_PATH)
        holdout = pd.read_json(HOLDOUT_PATH)
        overlap = len(
            set(pool["title"].astype(str).str.strip().str.lower())
            & set(holdout["title"].astype(str).str.strip().str.lower())
        )
        print(f"\nTitle overlap pool/holdout: {overlap}")
        if overlap:
            print("FAIL: holdout titles leak into training pool")
            pool_ok = False

    if pool_ok and holdout_ok:
        print("\nValidation PASSED")
        return 0
    print("\nValidation FAILED — fix issues before splitting")
    return 1


if __name__ == "__main__":
    sys.exit(main())
