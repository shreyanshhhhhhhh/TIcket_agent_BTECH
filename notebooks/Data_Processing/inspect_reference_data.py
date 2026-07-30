import pandas as pd
import glob
import os

reference_files = glob.glob("data/raw/reference/*.csv")
os.makedirs("data/raw/reference/english_only", exist_ok=True)

summary = []

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
            # Common English codes: 'en', 'eng', 'english'
            english_values = df[lang_col].astype(str).str.lower().isin(['en', 'eng', 'english'])
            df_english = df[english_values]
        else:
            # No language column found — assume it might already be English, keep as-is but flag it
            df_english = df
            print(f"  No language column found — keeping all {len(df)} rows, verify manually")

        out_path = f"data/raw/reference/english_only/english_{fname}"
        df_english.to_csv(out_path, index=False)

        summary.append({
            "file": fname,
            "total_rows": len(df),
            "english_rows": len(df_english),
            "columns": df.columns.tolist()
        })

        print(f"  Total rows: {len(df)} -> English rows: {len(df_english)}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for s in summary:
    print(f"\n{s['file']}")
    print(f"  {s['total_rows']} total -> {s['english_rows']} English")
    print(f"  Columns: {s['columns']}")