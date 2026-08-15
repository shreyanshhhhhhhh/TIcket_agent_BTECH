from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "models", "chroma_db")
COLLECTION_NAME = "ticket_resolutions"

_VECTORSTORE = None

def load_knowledge_base():
    """Loads and caches the existing persisted Chroma vector store."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _VECTORSTORE = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
    return _VECTORSTORE

def retrieve_similar_tickets(query_text: str, k: int = 5):
    """
    Given a new ticket's text, retrieve the top-k most similar past tickets
    along with their similarity scores and metadata (category, priority, resolution).
    """
    vectorstore = load_knowledge_base()

    # similarity_search_with_score returns (Document, distance_score) pairs
    # Chroma returns L2 / Cosine distance. We convert to normalized similarity (0.0 to 1.0).
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    retrieved = []
    for doc, score in results:
        # Convert Chroma L2/cosine distance to a 0.0-1.0 similarity score scale
        # Typical distance ranges from 0.0 (identical) to ~1.4 (dissimilar)
        similarity = max(0.0, 1.0 - (score / 1.5))
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