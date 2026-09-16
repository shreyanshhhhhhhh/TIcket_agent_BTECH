import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "models", "chroma_db")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train.csv")
COLLECTION_NAME = "ticket_resolutions"

FORBIDDEN_SOURCES = ("test.csv", "val.csv", "eval_holdout.json", "all_tickets_full.json")


def build_knowledge_base(source_path: str = TRAIN_CSV):
    """
    Build Chroma index from the train split only.
    Test, val, and holdout data must never be indexed to prevent data leakage.
    """
    if os.path.basename(source_path) != "train.csv":
        raise ValueError(f"Refusing to index non-train data: {source_path}")
    if any(name in source_path for name in FORBIDDEN_SOURCES if name != "train.csv"):
        raise ValueError(f"Refusing to index forbidden source: {source_path}")

    print("Loading train split only (no test/val leakage)...")
    df = pd.read_csv(source_path)
    print(f"Loaded {len(df)} train tickets.")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Converting tickets into LangChain Documents...")
    documents = []
    for i, row in df.iterrows():
        text = f"{row['title']}. {row['description']}"
        doc = Document(
            page_content=text,
            metadata={
                "title": row["title"],
                "category": row["category"],
                "priority": row["priority"],
                "resolution": row["resolution"],
                "source": "train",
            },
            id=f"train_{i}",
        )
        documents.append(doc)

    print("Building Chroma vector store (this embeds everything)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )

    print(f"\nKnowledge base built successfully with {len(documents)} train-only tickets.")
    return vectorstore


if __name__ == "__main__":
    build_knowledge_base()
