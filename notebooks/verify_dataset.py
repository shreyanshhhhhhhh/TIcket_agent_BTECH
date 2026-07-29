import pandas as pd
import glob
import os

RAW_DIR = "data/raw/reference"
CLEAN_DIR = "data/raw/reference/english_only"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ---------------------------------------------------------
# STEP 1: Verify and classify all reference files
# ---------------------------------------------------------

reference_files = glob.glob(f"{RAW_DIR}/*.csv")
summary = []

print("="*80)
print("STEP 1: CLASSIFYING ALL REFERENCE FILES")
print("="*80)

for f in reference_files:
    fname = os.path.basename(f)
    print(f"\nProcessing: {fname}")
    try:
        df = pd.read_csv(f)

        # Find language column
        lang_col = None
        for col in df.columns:
            if 'lang' in col.lower():
                lang_col = col
                break

        if lang_col:
            english_mask = df[lang_col].astype(str).str.lower().isin(['en', 'eng', 'english'])
            df_english = df[english_mask].copy()
            usable = len(df_english) > 0
        else:
            # No language column — check manually using a simple heuristic:
            # if text columns contain mostly ASCII/English-like characters, assume English
            text_col = None
            for col in df.columns:
                if df[col].dtype == object:
                    text_col = col
                    break

            if text_col:
                sample_text = " ".join(df[text_col].astype(str).head(50).tolist())
                # crude check: ratio of ascii characters
                ascii_ratio = sum(1 for c in sample_text if ord(c) < 128) / max(len(sample_text), 1)
                usable = ascii_ratio > 0.95
                df_english = df.copy() if usable else df.iloc[0:0]
            else:
                usable = False
                df_english = df.iloc[0:0]

            print(f"  No language column — heuristic check: {'looks English' if usable else 'not confirmed English'}")

        out_path = f"{CLEAN_DIR}/english_{fname}"
        df_english.to_csv(out_path, index=False)

        summary.append({
            "file": fname,
            "total_rows": len(df),
            "english_rows": len(df_english),
            "columns": df.columns.tolist(),
            "usable": len(df_english) > 100  # arbitrary usefulness threshold
        })

        print(f"  Total: {len(df)} -> English: {len(df_english)}")

    except Exception as e:
        print(f"  Error: {e}")
        summary.append({"file": fname, "error": str(e)})

print("\n" + "="*80)
print("CLASSIFICATION SUMMARY")
print("="*80)
for s in summary:
    if "error" in s:
        print(f"\n{s['file']}: ERROR - {s['error']}")
        continue
    status = "USABLE" if s["usable"] else "SKIP (too few English rows)"
    print(f"\n{s['file']} [{status}]")
    print(f"  {s['total_rows']} total -> {s['english_rows']} English")
    print(f"  Columns: {s['columns']}")

# ---------------------------------------------------------
# STEP 2: Inspect queue/category and priority values
# ---------------------------------------------------------

print("\n" + "="*80)
print("STEP 2: CATEGORY (queue) AND PRIORITY VALUES IN USABLE FILES")
print("="*80)

usable_files = [s["file"] for s in summary if s.get("usable")]

for fname in usable_files:
    path = f"{CLEAN_DIR}/english_{fname}"
    df = pd.read_csv(path)
    print(f"\n--- {fname} ---")

    if "queue" in df.columns:
        print("Queue (category) value counts:")
        print(df["queue"].value_counts())

    if "priority" in df.columns:
        print("\nPriority value counts:")
        print(df["priority"].value_counts())

    if "Topic_group" in df.columns:
        print("Topic_group value counts:")
        print(df["Topic_group"].value_counts())

    if "Document" in df.columns:
        print("\nSample documents:")
        for doc in df["Document"].head(5).tolist():
            print(f"  - {str(doc)[:150]}")

# ---------------------------------------------------------
# STEP 3: Combine usable files into ONE clean reference set
# with columns mapped to your project schema
# ---------------------------------------------------------

print("\n" + "="*80)
print("STEP 3: BUILDING UNIFIED REFERENCE DATASET")
print("="*80)

combined_rows = []

for fname in usable_files:
    path = f"{CLEAN_DIR}/english_{fname}"
    df = pd.read_csv(path)

    # Map known schema (subject/body/answer/queue/priority) -> your schema
    if set(["subject", "body", "queue", "priority"]).issubset(df.columns):
        mapped = pd.DataFrame({
            "title": df["subject"],
            "description": df["body"],
            "category_raw": df["queue"],
            "priority_raw": df["priority"],
            "resolution": df["answer"] if "answer" in df.columns else "",
            "source_file": fname
        })
        combined_rows.append(mapped)
        print(f"Mapped {len(mapped)} rows from {fname}")
    else:
        print(f"Skipped mapping for {fname} (schema doesn't match subject/body/queue/priority)")

if combined_rows:
    reference_combined = pd.concat(combined_rows, ignore_index=True)
    reference_combined = reference_combined.dropna(subset=["title", "description"])
    reference_combined = reference_combined.drop_duplicates(subset=["title", "description"])

    out_path = "data/raw/reference/reference_combined_clean.csv"
    reference_combined.to_csv(out_path, index=False)

    print(f"\nFinal combined reference dataset: {len(reference_combined)} rows")
    print(f"Saved to: {out_path}")
    print(f"\nUnique category_raw values (compare to your 7 target categories):")
    print(reference_combined["category_raw"].value_counts())
    print(f"\nUnique priority_raw values:")
    print(reference_combined["priority_raw"].value_counts())
else:
    print("\nNo files matched the expected schema for combining.")

print("\n" + "="*80)
print("DONE")
print("="*80)