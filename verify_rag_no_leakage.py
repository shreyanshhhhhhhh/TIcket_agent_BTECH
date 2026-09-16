"""
Verify RAG knowledge base was built from train split only.
Run after rebuilding Chroma: python verify_rag_no_leakage.py
"""
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "api"))

from rag_retrieval import retrieve_similar_tickets, _VECTORSTORE  # noqa: E402


def main():
    test = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test.csv"))

    indexed_count = _VECTORSTORE._collection.count()

    exact_memorization = 0
    high_sim_on_test = 0

    for _, row in test.iterrows():
        query = f"{row['title']}. {row['description']}"
        top = retrieve_similar_tickets(query, k=1)[0]
        matched_title = str(top["title"]).strip().lower()
        sim = top["similarity_score"]
        if matched_title == str(row["title"]).strip().lower() and sim > 0.95:
            exact_memorization += 1
        if sim > 0.95:
            high_sim_on_test += 1

    sample_title = "URGENT: Database corruption error"
    sample_desc = (
        "System just threw a 'Database page corruption detected' error on the CRM instance! "
        "We cannot access any records. Please help, the sales team is panicking!"
    )
    sample_top = retrieve_similar_tickets(f"{sample_title}. {sample_desc}", k=1)[0]

    print("=" * 70)
    print("RAG LEAKAGE VERIFICATION")
    print("=" * 70)
    print(f"Indexed documents:               {indexed_count} (expected ~695)")
    print(f"Test tickets checked:            {len(test)}")
    print(f"Exact memorization hits (>0.95): {exact_memorization}")
    print(f"High similarity on test (>0.95): {high_sim_on_test}")
    print("-" * 70)
    print("Sample ticket (test_single_ticket.py):")
    print(f"  Matched title:  {sample_top['title']}")
    print(f"  Similarity:     {sample_top['similarity_score']}")
    print(f"  Exact match:    {sample_top['title'].strip().lower() == sample_title.lower()}")
    print("=" * 70)

    primary_fixed = (
        680 <= indexed_count <= 700
        and sample_top["title"].strip().lower() != sample_title.lower()
        and sample_top["similarity_score"] < 0.95
    )
    print(f"Primary leakage fix:  {'PASS' if primary_fixed else 'FAIL'}")
    if exact_memorization:
        print(
            f"Note: {exact_memorization} test tickets matched train entries with "
            "identical titles (secondary data-split issue)."
        )
    return 0 if primary_fixed else 1


if __name__ == "__main__":
    raise SystemExit(main())
