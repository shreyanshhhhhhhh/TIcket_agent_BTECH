import json

with open("data/raw/all_tickets_base.json", "r", encoding="utf-8") as f:
    tickets = json.load(f)

print(f"Total tickets: {len(tickets)}")

# Check category balance
from collections import Counter
cat_counts = Counter(t["category"] for t in tickets)
print("\nCategory breakdown:")
for cat, count in cat_counts.items():
    print(f"  {cat}: {count}")

# Check for missing/empty fields
required_fields = ["title", "description", "priority", "resolution", "category"]
issues = 0
for i, t in enumerate(tickets):
    for field in required_fields:
        if field not in t or not str(t.get(field, "")).strip():
            print(f"  Ticket {i} missing/empty '{field}'")
            issues += 1

print(f"\nTotal field issues found: {issues}")

# Check for duplicates
descriptions = [t["description"] for t in tickets]
duplicates = len(descriptions) - len(set(descriptions))
print(f"Duplicate descriptions: {duplicates}")