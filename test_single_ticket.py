import argparse
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "api"))

from agent import process_ticket, available_classifiers

sample_ticket = {
    "title": "URGENT: Database corruption error",
    "description": (
        "System just threw a 'Database page corruption detected' error on the CRM instance! "
        "We cannot access any records. Please help, the sales team is panicking!"
    ),
    "ground_truth_category": "Database",
    "expected_department": "Database Administration (DBA) Team",
}


def run_one(model_id: str, label: str):
    full_text = f"{sample_ticket['title']}. {sample_ticket['description']}"
    print("=" * 70)
    print(f"MODEL: {label} ({model_id})")
    print("=" * 70)
    result = process_ticket(full_text, classifier_model=model_id)
    print(f"Predicted Category: {result['category']}")
    print(f"Confidence Score:   {result['classifier_confidence']:.4f}")
    print(f"Routed Department:  {result['department']}")
    print(f"Best RAG Similarity:{result['best_similarity']:.4f}")
    print(f"Agent Decision:     {result['decision']}")
    print(f"Reason:             {result['reason']}")
    cat_ok = result["category"] == sample_ticket["ground_truth_category"]
    dept_ok = result["department"] == sample_ticket["expected_department"]
    print(f"Category Match:     {'CORRECT [PASS]' if cat_ok else 'INCORRECT [FAIL]'}")
    print(f"Department Match:   {'CORRECT [PASS]' if dept_ok else 'INCORRECT [FAIL]'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test ticket through AI pipeline")
    parser.add_argument(
        "--model",
        choices=[m["id"] for m in available_classifiers()],
        default="all",
        help="Classifier to use (default: run all available models)",
    )
    args = parser.parse_args()

    models = available_classifiers()
    if args.model == "all":
        for m in models:
            run_one(m["id"], m["name"])
    else:
        match = next(m for m in models if m["id"] == args.model)
        run_one(match["id"], match["name"])


if __name__ == "__main__":
    main()
