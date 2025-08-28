import streamlit as st
from loader import load_documents, chunk_documents
from embeddings import embed_documents
from vector_store import add_to_chroma
from rag import answer_query

# --- Streamlit UI ---
st.set_page_config(page_title="Local CV RAG", page_icon="📄")

st.title("RAG with Ollama + ChromaDB")

# Sidebar - data loading
st.sidebar.header("Data Preparation")
if st.sidebar.button("Load & Index CVs"):
    raw_docs = load_documents()
    if not raw_docs:
        st.warning("No CV files found in `data/` folder.")
    else:
        chunks = chunk_documents(raw_docs)
        embedded_chunks = embed_documents(chunks)
        add_to_chroma(embedded_chunks)
        st.success(f"Indexed {len(embedded_chunks)} chunks from {len(raw_docs)} document(s).")

# Main chat interface
st.header("Ask questions ")
query = st.text_input("Enter your question:", placeholder="e.g. What programming languages do I know?")

if query:
    with st.spinner("Thinking..."):
        result = answer_query(query, top_k=4)

    st.subheader("✅ Answer")
    st.write(result["answer"])


