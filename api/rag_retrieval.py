from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "models", "chroma_db")
COLLECTION_NAME = "ticket_resolutions"

# Load once at import time (main thread) to avoid ChromaDB thread-safety issues
# when FastAPI handles requests in worker threads.
print("Loading Chroma knowledge base...")
_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_VECTORSTORE = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_EMBEDDINGS,
    persist_directory=CHROMA_PATH,
)


def load_knowledge_base():
    """Returns the cached Chroma vector store."""
    return _VECTORSTORE


def retrieve_similar_tickets(query_text: str, k: int = 5):
    """
    Given a new ticket's text, retrieve the top-k most similar past tickets
    along with their similarity scores and metadata (category, priority, resolution).
    """
    vectorstore = _VECTORSTORE

    # similarity_search_with_score returns (Document, distance_score) pairs
    # lower distance = more similar (Chroma uses L2/cosine distance by default)
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    retrieved = []
    for doc, score in results:
        similarity = 1 - score  # convert distance to a rough similarity indicator
        retrieved.append({
            "title": doc.metadata.get("title"),
            "category": doc.metadata.get("category"),
            "priority": doc.metadata.get("priority"),
            "resolution": doc.metadata.get("resolution"),
            "similarity_score": round(similarity, 4),
            "matched_text": doc.page_content,
        })

    return retrieved


if __name__ == "__main__":
    # Quick test with a sample new ticket
    test_query = "My VPN keeps disconnecting every few minutes on Windows 11, error 807"
    print(f"Query: {test_query}\n")

    results = retrieve_similar_tickets(test_query, k=3)

    for i, r in enumerate(results, 1):
        print(f"--- Match {i} (similarity: {r['similarity_score']}) ---")
        print(f"Category: {r['category']} | Priority: {r['priority']}")
        print(f"Matched text: {r['matched_text'][:150]}")
        print(f"Resolution: {r['resolution']}\n")