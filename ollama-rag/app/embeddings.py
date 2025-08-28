from typing import List
import ollama
from langchain.schema import Document

EMBED_MODEL = "nomic-embed-text"  # Ollama's embedding model


def embed_text(text: str) -> List[float]:
    """Generate embedding vector for a single text chunk using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def embed_documents(docs: List[Document]) -> List[dict]:
    """
    Create embeddings for document chunks.
    Returns list of dicts: {"text": chunk, "embedding": vector}
    """
    embedded_chunks = []
    for doc in docs:
        text = doc.page_content
        vector = embed_text(text)
        embedded_chunks.append({
            "text": text,
            "embedding": vector,
            "metadata": doc.metadata
        })
    return embedded_chunks


if __name__ == "__main__":
    from loader import load_documents, chunk_documents

    raw_docs = load_documents()
    chunks = chunk_documents(raw_docs)
    print(f"Embedding {len(chunks)} chunks...")

    embedded = embed_documents(chunks[:3])  # test on first 3 chunks
    print("Sample embedding:", embedded[0]["embedding"][:5])  # show first 5 numbers
