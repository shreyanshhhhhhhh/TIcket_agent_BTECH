from google import genai
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CATEGORIES = [
    "Infrastructure", "Application", "Security",
    "Database", "Storage", "Network", "Access Management"
]

def call_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`").replace("json", "", 1).strip()
            return json.loads(text)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return []

def generate_base(category, n=12):
    prompt = f"""
Generate {n} realistic IT support tickets for the category '{category}'.
Return ONLY a valid JSON array, no markdown, no extra text.
Each object must have: title, description (2-5 sentences with realistic technical detail: error codes, OS versions, device names), priority (Low/Medium/High/Critical), resolution.
Vary tone (formal, casual, frustrated, calm) and length (short and long) across tickets.
"""
    tickets = call_gemini(prompt)
    for t in tickets:
        t["category"] = category
        t["type"] = "base"
    return tickets

def main():
    os.makedirs("data/raw", exist_ok=True)
    all_tickets = []

    print("=== FULL RUN: Generating base tickets (all 7 categories) ===")
    for cat in CATEGORIES:
        print(f"\nCategory: {cat}")
        cat_tickets = []
        for batch in range(9):   # 9 batches x 12 = ~108 tickets per category
            batch_tickets = generate_base(cat, n=12)
            cat_tickets.extend(batch_tickets)
            print(f"  Batch {batch+1}: {len(batch_tickets)} tickets")
            time.sleep(6)   # respects 15 RPM limit

        # Save per-category file immediately (safety net in case of crash)
        with open(f"data/raw/{cat.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(cat_tickets, f, indent=2)

        all_tickets.extend(cat_tickets)
        print(f"  Category total: {len(cat_tickets)} tickets")

    with open("data/raw/all_tickets_base.json", "w", encoding="utf-8") as f:
        json.dump(all_tickets, f, indent=2)

    print(f"\n=== DONE. Total base tickets generated: {len(all_tickets)} ===")

if __name__ == "__main__":
    main()