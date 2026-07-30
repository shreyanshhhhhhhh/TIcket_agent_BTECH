import json
from collections import Counter

with open("data/raw/all_tickets_full.json", "r", encoding="utf-8") as f:
    full = json.load(f)

with open("data/raw/eval_holdout.json", "r", encoding="utf-8") as f:
    eval_set = json.load(f)

print(f"Training pool: {len(full)}")
print(f"Eval holdout: {len(eval_set)}")

print("\nType breakdown (training pool):")
print(Counter(t.get("type", "unknown") for t in full))

print("\nCategory breakdown (training pool):")
print(Counter(t["category"] for t in full))

# Check for missing fields
required = ["title", "description", "priority", "resolution", "category"]
issues = 0
for i, t in enumerate(full):
    for f_name in required:
        if f_name not in t or not str(t.get(f_name, "")).strip():
            issues += 1
print(f"\nMissing/empty field issues: {issues}")

# Check duplicates
descs = [t["description"] for t in full]
print(f"Duplicate descriptions: {len(descs) - len(set(descs))}")