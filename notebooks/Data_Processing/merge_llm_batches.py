"""
Merge pipe-delimited LLM batch files into one JSON pool for training.

Usage (from project root):
  python notebooks/Data_Processing/merge_llm_batches.py
  python notebooks/Data_Processing/merge_llm_batches.py --input-dir data/raw/llm_batches --output data/raw/all_tickets_full_v2.json

Expected batch format (first line may be header):
  title|description|category|priority|resolution|type
"""
import argparse
import glob
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQUIRED = {"title", "description", "category", "priority", "resolution"}


def load_batch(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "type" not in df.columns:
        df["type"] = "base"
    return df


def main():
    parser = argparse.ArgumentParser(description="Merge LLM pipe-delimited ticket batches")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(BASE_DIR, "data", "raw", "llm_batches"),
        help="Folder containing llm_batch_*.txt or *.csv files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "data", "raw", "all_tickets_full_v2.json"),
        help="Merged JSON output path",
    )
    parser.add_argument(
        "--include-existing",
        default=os.path.join(BASE_DIR, "data", "raw", "all_tickets_full.json"),
        help="Optional existing pool JSON to merge in (set empty to skip)",
    )
    args = parser.parse_args()

    frames = []
    patterns = [
        os.path.join(args.input_dir, "*.txt"),
        os.path.join(args.input_dir, "*.csv"),
    ]
    batch_files = sorted({p for pat in patterns for p in glob.glob(pat)})

    for path in batch_files:
        try:
            df = load_batch(path)
            frames.append(df)
            print(f"Loaded {len(df):4d} from {os.path.basename(path)}")
        except Exception as exc:
            print(f"SKIP {path}: {exc}")

    if args.include_existing and os.path.isfile(args.include_existing):
        with open(args.include_existing, encoding="utf-8") as f:
            existing = pd.DataFrame(json.load(f))
        frames.append(existing)
        print(f"Loaded {len(existing):4d} from existing pool")

    if not frames:
        print("No data loaded. Add batch files to input-dir first.")
        return 1

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["title", "description", "category"])
    merged["title"] = merged["title"].astype(str).str.strip()
    merged["description"] = merged["description"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["title"], keep="first")

    records = merged.to_dict(orient="records")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("-" * 50)
    print(f"Total merged tickets: {len(records)}")
    print("Category counts:")
    print(merged["category"].value_counts().to_string())
    print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
