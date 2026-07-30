import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = "models/chroma_db"
COLLECTION_NAME = "ticket_resolutions"

def build_knowledge_base():
    """
    Loads all tickets, embeds their descriptions, and stores them
    as LangChain Documents in a persistent Chroma vector store.
    """
    print("Loading tickets...")
    with open("data/raw/all_tickets_full.json", "r", encoding="utf-8") as f:
        tickets = json.load(f)
    print(f"Loaded {len(tickets)} tickets.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Converting tickets into LangChain Documents...")
    documents = []
    for i, t in enumerate(tickets):
        text = f"{t['title']}. {t['description']}"
        doc = Document(
            page_content=text,
            metadata={
                "title": t["title"],
                "category": t["category"],
                "priority": t["priority"],
                "resolution": t["resolution"],
                "type": t.get("type", "base"),
            },
            id=f"ticket_{i}",
        )
        documents.append(doc)

    print("Building Chroma vector store (this embeds everything)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )

    print(f"\nKnowledge base built successfully with {len(documents)} tickets.")
    return vectorstore


if __name__ == "__main__":
    build_knowledge_base()