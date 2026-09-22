"""
Phase 5: Agentic Decision Layer using LangGraph.

Pipeline: classify -> route -> retrieve similar tickets -> decide (auto-resolve or escalate)
"""

import joblib
import numpy as np
from typing import TypedDict, List, Dict, Optional, Literal
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from routing import route_ticket
from rag_retrieval import retrieve_similar_tickets
import os

# Get the project root directory regardless of where this script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ClassifierModelId = Literal["logreg", "setfit"]

# ---------------------------------------------------------
# Load models once (at module load time, not per-request)
# ---------------------------------------------------------
print("Loading classifiers and embedder...")
CLASSIFIER = joblib.load(os.path.join(BASE_DIR, "models", "logreg_classifier.joblib"))
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

SETFIT_MODEL = None
SETFIT_PATH = os.path.join(BASE_DIR, "models", "setfit_classifier")
if os.path.isdir(SETFIT_PATH):
    try:
        from setfit import SetFitModel
        SETFIT_MODEL = SetFitModel.from_pretrained(SETFIT_PATH)
        print("SetFit classifier loaded.")
    except Exception as exc:
        print(f"SetFit not available: {exc}")
else:
    print("SetFit model directory not found — only logreg available.")


def available_classifiers() -> List[Dict]:
    models = [
        {
            "id": "logreg",
            "name": "Logistic Regression",
            "description": "MiniLM embeddings + linear classifier (baseline, ~70% test accuracy)",
        }
    ]
    if SETFIT_MODEL is not None:
        models.append(
            {
                "id": "setfit",
                "name": "SetFit",
                "description": "Contrastive fine-tuned MiniLM (~71% test accuracy)",
            }
        )
    return models

# Thresholds (tunable — these decide escalation behavior)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.50   # was 0.55
RETRIEVAL_SIMILARITY_THRESHOLD = 0.40    # was 0.45, slightly relaxed# below this, treat retrieval as weak evidence


# ---------------------------------------------------------
# State definition — what flows through the graph
# ---------------------------------------------------------
class TicketState(TypedDict):
    ticket_text: str
    classifier_model: str
    category: Optional[str]
    classifier_confidence: Optional[float]
    department: Optional[str]
    retrieved: Optional[List[Dict]]
    best_similarity: Optional[float]
    decision: Optional[str]          # "auto_resolve" | "escalate"
    suggested_resolution: Optional[str]
    reason: Optional[str]


# ---------------------------------------------------------
# Node 1: Classification
# ---------------------------------------------------------
def _classify_with_logreg(text: str) -> tuple[str, float]:
    embedding = EMBEDDER.encode([text])
    probs = CLASSIFIER.predict_proba(embedding)[0]
    pred_idx = int(np.argmax(probs))
    return CLASSIFIER.classes_[pred_idx], float(probs[pred_idx])


def _classify_with_setfit(text: str) -> tuple[str, float]:
    if SETFIT_MODEL is None:
        raise ValueError("SetFit model is not loaded")
    probs = SETFIT_MODEL.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    labels = SETFIT_MODEL.labels
    return labels[pred_idx], float(probs[pred_idx])


def classify_node(state: TicketState) -> TicketState:
    text = state["ticket_text"]
    model_id = state.get("classifier_model", "logreg")

    if model_id == "setfit":
        category, confidence = _classify_with_setfit(text)
    else:
        category, confidence = _classify_with_logreg(text)

    state["category"] = category
    state["classifier_confidence"] = round(confidence, 4)
    return state


# ---------------------------------------------------------
# Node 2: Routing
# ---------------------------------------------------------
def route_node(state: TicketState) -> TicketState:
    state["department"] = route_ticket(state["category"])
    return state


# ---------------------------------------------------------
# Node 3: RAG Retrieval
# ---------------------------------------------------------
def retrieve_node(state: TicketState) -> TicketState:
    results = retrieve_similar_tickets(state["ticket_text"], k=3)
    state["retrieved"] = results
    state["best_similarity"] = results[0]["similarity_score"] if results else 0.0
    return state


# ---------------------------------------------------------
# Node 4: Decision (the actual "agentic" logic)
# ---------------------------------------------------------
def decide_node(state: TicketState) -> TicketState:
    conf = state["classifier_confidence"]
    sim = state["best_similarity"]

    if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD and sim >= RETRIEVAL_SIMILARITY_THRESHOLD:
        state["decision"] = "auto_resolve"
        state["suggested_resolution"] = state["retrieved"][0]["resolution"]
        state["reason"] = (
            f"High classifier confidence ({conf:.2f}) and strong retrieval match "
            f"(similarity {sim:.2f}) — auto-suggesting resolution."
        )
    else:
        state["decision"] = "escalate"
        state["suggested_resolution"] = None
        state["reason"] = (
            f"Low confidence (classifier: {conf:.2f}, retrieval similarity: {sim:.2f}) — "
            f"escalating to {state['department']} for human review."
        )
    return state


# ---------------------------------------------------------
# Build the LangGraph state machine
# ---------------------------------------------------------
def build_agent_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify", classify_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_edge("decide", END)

    return graph.compile()


# ---------------------------------------------------------
# Public function to process a new ticket end-to-end
# ---------------------------------------------------------
# Build the graph ONCE at module load time, not per-request
_COMPILED_AGENT = build_agent_graph()

def process_ticket(ticket_text: str, classifier_model: str = "logreg") -> Dict:
    if classifier_model not in {m["id"] for m in available_classifiers()}:
        raise ValueError(f"Unknown classifier_model: {classifier_model}")

    initial_state: TicketState = {
        "ticket_text": ticket_text,
        "classifier_model": classifier_model,
        "category": None,
        "classifier_confidence": None,
        "department": None,
        "retrieved": None,
        "best_similarity": None,
        "decision": None,
        "suggested_resolution": None,
        "reason": None,
    }
    final_state = _COMPILED_AGENT.invoke(initial_state)
    final_state["classifier_model"] = classifier_model
    return final_state

if __name__ == "__main__":
    test_tickets = [
        "My VPN keeps disconnecting every few minutes on Windows 11, error 807",
        "The printer on the 3rd floor is jammed and won't print anything",
        "I think someone accessed my account without permission, seeing logins from an unknown IP",
    ]

    for t in test_tickets:
        print("=" * 80)
        print(f"TICKET: {t}")
        result = process_ticket(t)
        print(f"Category: {result['category']} (confidence: {result['classifier_confidence']})")
        print(f"Department: {result['department']}")
        print(f"Best similarity: {result['best_similarity']}")
        print(f"Decision: {result['decision'].upper()}")
        print(f"Reason: {result['reason']}")
        if result["suggested_resolution"]:
            print(f"Suggested Resolution: {result['suggested_resolution']}")
        print()