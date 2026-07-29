from google import genai
import os
import json
import time
import random
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"

CATEGORIES = [
    "Infrastructure", "Application", "Security",
    "Database", "Storage", "Network", "Access Management"
]

CATEGORY_PAIRS = [
    ("Network", "Database"),
    ("Security", "Access Management"),
    ("Infrastructure", "Storage"),
    ("Application", "Network"),
]

def call_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`").replace("json", "", 1).strip()
            return json.loads(text)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return []

def call_gemini_text(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return None

def generate_paraphrase(ticket, style):
    prompt = f"""
Rewrite this IT ticket description in a {style} tone, keeping the same meaning and technical facts:
"{ticket['description']}"
Return ONLY the rewritten description text, nothing else, no quotes.
"""
    new_desc = call_gemini_text(prompt)
    if new_desc:
        new_ticket = ticket.copy()
        new_ticket["description"] = new_desc
        new_ticket["type"] = "paraphrase"
        return new_ticket
    return None

def generate_ambiguous(cat1, cat2, n=10):
    prompt = f"""
Generate {n} IT support tickets that are genuinely ambiguous between '{cat1}' and '{cat2}' categories
(the root cause could plausibly be either).
Return ONLY a valid JSON array. Each object must have: title, description, most_likely_category (either '{cat1}' or '{cat2}'), priority, resolution.
"""
    tickets = call_gemini(prompt)
    for t in tickets:
        t["category"] = t.get("most_likely_category", cat1)
        t["type"] = "ambiguous"
    return tickets

def generate_priority_mismatch(category, n=8):
    prompt = f"""
Generate {n} IT support tickets for category '{category}' where the WORDING TONE does not match the priority level
(e.g., calm/polite wording but priority is Critical, OR panicked/urgent wording but priority is Low).
Return ONLY a valid JSON array with: title, description, priority, resolution.
"""
    tickets = call_gemini(prompt)
    for t in tickets:
        t["category"] = category
        t["type"] = "priority_mismatch"
    return tickets

def generate_eval_holdout(category, n=17):
    prompt = f"""
Generate {n} realistic IT support tickets for category '{category}' for a FINAL EVALUATION set.
Use different phrasing style and structure than typical training examples - be creative with wording.
Return ONLY a valid JSON array with: title, description, priority, resolution.
"""
    tickets = call_gemini(prompt)
    for t in tickets:
        t["category"] = category
        t["type"] = "eval_holdout"
    return tickets

def main():
    # Load base tickets
    with open("data/raw/all_tickets_base.json", "r", encoding="utf-8") as f:
        base_tickets = json.load(f)

    augmented = []

    # 1. Paraphrased variants (~150)
    print("=== Generating paraphrased variants ===")
    styles = ["casual Slack-message", "formal complaint", "frustrated customer"]
    sample = random.sample(base_tickets, min(150, len(base_tickets)))
    for i, ticket in enumerate(sample):
        style = styles[i % len(styles)]
        new_t = generate_paraphrase(ticket, style)
        if new_t:
            augmented.append(new_t)
        if (i + 1) % 10 == 0:
            print(f"  Paraphrased {i+1}/{len(sample)}")
        time.sleep(6)

    with open("data/raw/paraphrased.json", "w", encoding="utf-8") as f:
        json.dump([t for t in augmented if t["type"] == "paraphrase"], f, indent=2)
    print(f"Paraphrased total: {len([t for t in augmented if t['type']=='paraphrase'])}")

    # 2. Ambiguous cases (~80)
    print("\n=== Generating ambiguous cases ===")
    ambiguous_all = []
    for cat1, cat2 in CATEGORY_PAIRS:
        tickets = generate_ambiguous(cat1, cat2, n=10)
        ambiguous_all.extend(tickets)
        print(f"  {cat1}/{cat2}: {len(tickets)} tickets")
        time.sleep(6)
    with open("data/raw/ambiguous.json", "w", encoding="utf-8") as f:
        json.dump(ambiguous_all, f, indent=2)
    augmented.extend(ambiguous_all)
    print(f"Ambiguous total: {len(ambiguous_all)}")

    # 3. Priority-mismatch cases (~50, spread across 6 categories)
    print("\n=== Generating priority-mismatch cases ===")
    mismatch_all = []
    for cat in CATEGORIES[:6]:
        tickets = generate_priority_mismatch(cat, n=8)
        mismatch_all.extend(tickets)
        print(f"  {cat}: {len(tickets)} tickets")
        time.sleep(6)
    with open("data/raw/priority_mismatch.json", "w", encoding="utf-8") as f:
        json.dump(mismatch_all, f, indent=2)
    augmented.extend(mismatch_all)
    print(f"Priority-mismatch total: {len(mismatch_all)}")

    # Save combined training pool (base + augmentation)
    full_pool = base_tickets + augmented
    with open("data/raw/all_tickets_full.json", "w", encoding="utf-8") as f:
        json.dump(full_pool, f, indent=2)
    print(f"\nFull training pool total: {len(full_pool)}")

    # 4. Held-out eval set (~120, kept separate)
    print("\n=== Generating held-out eval set ===")
    eval_tickets = []
    for cat in CATEGORIES:
        tickets = generate_eval_holdout(cat, n=17)
        eval_tickets.extend(tickets)
        print(f"  {cat}: {len(tickets)} tickets")
        time.sleep(6)
    with open("data/raw/eval_holdout.json", "w", encoding="utf-8") as f:
        json.dump(eval_tickets, f, indent=2)
    print(f"Eval holdout total: {len(eval_tickets)}")

    print(f"\n=== ALL DONE. Training pool: {len(full_pool)} | Eval holdout: {len(eval_tickets)} ===")

if __name__ == "__main__":
    main()