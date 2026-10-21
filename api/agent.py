"""
Phase 5: Agentic Decision Layer using LangGraph.

Pipeline: classify -> route -> retrieve similar tickets -> decide (auto-resolve or escalate)

Classification is two-stage. Stage 1 is the original MiniLM + logistic
regression model. When that prediction is confusing — the top two class
probabilities are within the validation-chosen margin — stage 2 re-encodes
the ticket with BAAI/bge-base-en-v1.5 and a logistic regression trained on
those vectors. Routing and the rest of the pipeline use whichever label won.
"""

import json
import joblib
import numpy as np
from typing import TypedDict, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from routing import route_ticket
from rag_retrieval import retrieve_similar_tickets
import os

# Get the project root directory regardless of where this script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ---------------------------------------------------------
# Load models once (at module load time, not per-request)
# ---------------------------------------------------------
print("Loading stage-1 classifier, BGE second stage, and embedders...")
CLASSIFIER = joblib.load(os.path.join(BASE_DIR, "models", "logreg_classifier.joblib"))
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
BGE_CLASSIFIER = joblib.load(os.path.join(BASE_DIR, "models", "bge_logreg_classifier.joblib"))
BGE_EMBEDDER = SentenceTransformer(BGE_MODEL_NAME)

with open(os.path.join(BASE_DIR, "models", "bge_results.json"), encoding="utf-8") as _bge_results_file:
    _BGE_RESULTS = json.load(_bge_results_file)

# Stage 1 is confusing when the gap between the top two probabilities is
# below this value. Selected on the validation split by
# notebooks/train_bge_second_stage.py. The confidence cutoff is stored
# alongside it and stays at 0, so only a close top-two race triggers BGE.
CONFUSION_MARGIN_MAX = float(_BGE_RESULTS["confusion_margin_max"])
CONFUSION_CONFIDENCE_MAX = float(_BGE_RESULTS["confusion_confidence_max"])

# Thresholds (tunable — these decide escalation behavior)
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.50   # was 0.55
RETRIEVAL_SIMILARITY_THRESHOLD = 0.40    # was 0.45, slightly relaxed# below this, treat retrieval as weak evidence


def _embed_dim(model) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


if BGE_CLASSIFIER.n_features_in_ != _embed_dim(BGE_EMBEDDER):
    raise RuntimeError(
        f"BGE classifier expects {BGE_CLASSIFIER.n_features_in_} features, "
        f"but {BGE_MODEL_NAME} outputs {_embed_dim(BGE_EMBEDDER)}."
    )


def _predict(classifier, embedder, text: str) -> Dict:
    """Return the top class, its probability, and the gap to second place."""
    embedding = embedder.encode([text])
    probs = classifier.predict_proba(embedding)[0]
    order = np.argsort(probs)
    top1 = int(order[-1])
    top2 = int(order[-2])
    confidence = float(probs[top1])
    return {
        "category": str(classifier.classes_[top1]),
        "confidence": confidence,
        "runner_up": str(classifier.classes_[top2]),
        "margin": confidence - float(probs[top2]),
    }


def _is_confusing(prediction: Dict) -> bool:
    return (
        prediction["margin"] < CONFUSION_MARGIN_MAX
        or prediction["confidence"] < CONFUSION_CONFIDENCE_MAX
    )


# ---------------------------------------------------------
# State definition — what flows through the graph
# ---------------------------------------------------------
class TicketState(TypedDict):
    ticket_text: str
    category: Optional[str]
    classifier_confidence: Optional[float]
    classifier_stage: Optional[str]  # "minilm" | "bge"
    stage1_category: Optional[str]
    stage1_confidence: Optional[float]
    stage1_margin: Optional[float]
    department: Optional[str]
    retrieved: Optional[List[Dict]]
    best_similarity: Optional[float]
    decision: Optional[str]          # "auto_resolve" | "escalate"
    suggested_resolution: Optional[str]
    reason: Optional[str]


# ---------------------------------------------------------
# Node 1: Classification
# ---------------------------------------------------------
def classify_node(state: TicketState) -> TicketState:
    text = state["ticket_text"]
    first = _predict(CLASSIFIER, EMBEDDER, text)
    state["stage1_category"] = first["category"]
    state["stage1_confidence"] = round(first["confidence"], 4)
    state["stage1_margin"] = round(first["margin"], 4)

    if _is_confusing(first):
        second = _predict(BGE_CLASSIFIER, BGE_EMBEDDER, text)
        state["category"] = second["category"]
        state["classifier_confidence"] = round(second["confidence"], 4)
        state["classifier_stage"] = "bge"
    else:
        state["category"] = first["category"]
        state["classifier_confidence"] = round(first["confidence"], 4)
        state["classifier_stage"] = "minilm"
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
def _stage_note(state: TicketState) -> str:
    if state.get("classifier_stage") != "bge":
        return ""
    return (
        f"Stage-1 was inconclusive ({state['stage1_category']} at "
        f"{state['stage1_confidence']:.2f}, margin {state['stage1_margin']:.2f}). "
        f"BGE second layer classified it as {state['category']} "
        f"({state['classifier_confidence']:.2f}). "
    )


def decide_node(state: TicketState) -> TicketState:
    conf = state["classifier_confidence"]
    sim = state["best_similarity"]
    note = _stage_note(state)

    if conf >= CLASSIFIER_CONFIDENCE_THRESHOLD and sim >= RETRIEVAL_SIMILARITY_THRESHOLD:
        state["decision"] = "auto_resolve"
        state["suggested_resolution"] = state["retrieved"][0]["resolution"]
        state["reason"] = (
            f"{note}High classifier confidence ({conf:.2f}) and strong retrieval match "
            f"(similarity {sim:.2f}) — auto-suggesting resolution."
        )
    else:
        state["decision"] = "escalate"
        state["suggested_resolution"] = None
        state["reason"] = (
            f"{note}Low confidence (classifier: {conf:.2f}, retrieval similarity: {sim:.2f}) — "
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

def process_ticket(ticket_text: str) -> Dict:
    initial_state: TicketState = {
        "ticket_text": ticket_text,
        "category": None,
        "classifier_confidence": None,
        "classifier_stage": None,
        "stage1_category": None,
        "stage1_confidence": None,
        "stage1_margin": None,
        "department": None,
        "retrieved": None,
        "best_similarity": None,
        "decision": None,
        "suggested_resolution": None,
        "reason": None,
    }
    final_state = _COMPILED_AGENT.invoke(initial_state)
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
        print(
            f"Category: {result['category']} (confidence: {result['classifier_confidence']}, "
            f"stage: {result['classifier_stage']}, stage-1: {result['stage1_category']} "
            f"{result['stage1_confidence']})"
        )
        print(f"Department: {result['department']}")
        print(f"Best similarity: {result['best_similarity']}")
        print(f"Decision: {result['decision'].upper()}")
        print(f"Reason: {result['reason']}")
        if result["suggested_resolution"]:
            print(f"Suggested Resolution: {result['suggested_resolution']}")
        print()