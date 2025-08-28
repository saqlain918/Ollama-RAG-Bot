import chromadb
from chromadb.api.types import Documents, Embeddings

# Initialize Chroma client (persisted locally)
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Create or get a collection
collection = chroma_client.get_or_create_collection(
    name="cv_chunks",
    embedding_function=None  # We'll pass embeddings manually
)

def add_to_chroma(chunks_with_embeddings):
    """
    Insert chunks + embeddings into ChromaDB.
    chunks_with_embeddings = [{"text":..., "embedding":..., "metadata":...}]
    """
    ids = [str(i) for i in range(len(chunks_with_embeddings))]
    documents: Documents = [c["text"] for c in chunks_with_embeddings]
    embeddings: Embeddings = [c["embedding"] for c in chunks_with_embeddings]
    metadatas = [c["metadata"] for c in chunks_with_embeddings]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    # Removed print here


def query_chroma(query_embedding, top_k=3):
    """
    Search ChromaDB for similar chunks.
    Returns top_k results.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


if __name__ == "__main__":
    from loader import load_documents, chunk_documents
    from embeddings import embed_documents, embed_text

    # Load + chunk documents
    raw_docs = load_documents()
    chunks = chunk_documents(raw_docs)

    # Embed chunks
    embedded_chunks = embed_documents(chunks[:5])  # testing on 5 chunks

    # Add to DB
    add_to_chroma(embedded_chunks)

    # Test query
    query = "What programming languages are in the CV?"
    query_vec = embed_text(query)
    results = query_chroma(query_vec)
    # Removed print here too
