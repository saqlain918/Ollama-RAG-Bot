from typing import List, Dict
import textwrap
import ollama

from embeddings import embed_text
from vector_store import query_chroma

# Choose your local Ollama chat model (swap: "llama3", "mistral", etc.)
OLLAMA_LLM = "llama2"


def _format_context(results: Dict) -> str:
    """
    Build a readable context string from Chroma query results.
    Expects `results` to contain 'documents' and 'metadatas'.
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    formatted_chunks: List[str] = []
    for i, chunk in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        source = meta.get("source") or meta.get("file_path") or "unknown_source"
        page = meta.get("page") or meta.get("page_number") or "?"
        formatted = f"[Chunk {i+1} | {source} | page {page}]\n{chunk}".strip()
        formatted_chunks.append(formatted)

    return "\n\n---\n\n".join(formatted_chunks)


def _build_prompt(user_query: str, context: str) -> str:
    """
    Create a strict RAG prompt that instructs the model to use only the provided context.
    """
    system_rules = textwrap.dedent(f"""
    You are an assistant answering questions strictly using the CONTEXT provided below,
    which is extracted from a CV/resume.

    RULES:
    - Only use the information in the CONTEXT.
    - If the answer is not explicitly present, reply exactly with: "Not found in CV".
    - Do NOT use outside knowledge.
    - Do NOT guess or assume details.
    - Prefer bullet points for lists (skills, tools, roles).
    - When possible, cite the chunk number and source.

    CONTEXT:
    {context}
    """).strip()

    user_block = f"Question: {user_query}\n\nAnswer:"
    return system_rules + "\n\n" + user_block


def answer_query(query: str, top_k: int = 4, model: str = OLLAMA_LLM) -> Dict:
    """
    Full RAG step:
      1) Embed the query with Ollama embeddings
      2) Retrieve top_k chunks from Chroma
      3) Build a grounded prompt
      4) Ask Ollama LLM for the final answer

    Returns a dict: {"answer": str, "sources": List[Dict]}
    """
    # 1) Query embedding
    q_vec = embed_text(query)

    # 2) Retrieve similar chunks
    results = query_chroma(q_vec, top_k=top_k)

    # 3) Build prompt with retrieved context
    context = _format_context(results)
    prompt = _build_prompt(query, context)

    # 4) Generate answer from Ollama LLM (force deterministic output)
    completion = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0}  # 🔒 prevents creativity
    )
    answer_text = completion.get("response", "").strip()

    # Pack minimal source info (ids + metadatas)
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    sources = []
    for i in range(len(ids)):
        sources.append({
            "id": ids[i],
            "source": (metas[i].get("source") or metas[i].get("file_path") or "unknown_source") if i < len(metas) else "unknown_source",
            "page": metas[i].get("page") if i < len(metas) else None,
            "snippet": docs[i][:240] + ("..." if len(docs[i]) > 240 else "")
        })

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # Quick manual test (ensure you've already populated Chroma via vector_store.py)
    test_q = "Summarize my work experience and key skills."
    result = answer_query(test_q, top_k=4)
    print("\n=== ANSWER ===\n", result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(s)
