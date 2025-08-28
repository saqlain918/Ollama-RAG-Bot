## OllamaRAG-LocalAI

A local Retrieval-Augmented Generation (RAG) system powered by Ollama LLMs. This project allows you to ask questions over your documents, and the model answers strictly based on the provided context using local AI, embeddings, and a vector database (ChromaDB).

## What it does

- Processes your documents locally for question answering
- Uses Retrieval-Augmented Generation to provide accurate, context-based answers
- Stores document embeddings in ChromaDB for fast semantic search
- Runs completely offline with no data sent to external servers
- Provides interactive interface via Streamlit
- Ensures answers are grounded in your actual data

## Features

- Ollama LLM Integration: Local language model for question answering
- RAG Implementation: Ensures answers are grounded in your data
- ChromaDB Storage: Fast semantic search with vector embeddings
- Fully Local Setup: No data sent to external servers
- Interactive UI: User-friendly Streamlit interface
- Document Processing: Automatic chunking and embedding generation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OllamaRAG-LocalAI.git
   cd OllamaRAG-LocalAI
   ```

2. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama locally:
   Follow instructions at https://ollama.com to set up LLM models on your machine

## How to use

### Step 1: Load and chunk documents
- Place your PDFs or text files in the `data/` folder
- Run the ingestion script:
  ```
  python vector_store.py
  ```
  This will:
  - Load documents
  - Split them into chunks
  - Create embeddings
  - Store embeddings in ChromaDB

### Step 2: Run the Streamlit App
```
streamlit run app.py
```
- Open the displayed URL (usually http://localhost:8501)
- Ask questions in the input box
- System retrieves relevant chunks and generates grounded answers

### Step 3: Direct querying via RAG script
```
python rag.py
```
Test queries in terminal:
```python
test_q = "Summarize my work experience and key skills."
result = answer_query(test_q, top_k=4)
print(result["answer"])
```

## Project Structure

```
local_rag_ollama/
├── data/                 # Store your CV PDFs or text files
├── app/
│   ├── main.py           # Entry point (Streamlit UI + RAG flow)
│   ├── loader.py         # Document loading + chunking
│   ├── embeddings.py     # Generate embeddings with Ollama
│   ├── vector_store.py   # ChromaDB storage & retrieval
│   └── rag.py            # Orchestrates query → retrieve → answer
├── requirements.txt      # Dependencies
└── README.md
```

## Key Concepts

**Chunking:** Documents split into smaller pieces for effective retrieval

**ChromaDB:** Stores embeddings and metadata for semantic search

**Ollama LLM:** Generates answers based on retrieved context only

**Top-K Retrieval:** Default is 4 chunks per query (configurable in rag.py)

## Recommended Workflow

1. Add new documents to `data/` folder
2. Re-run embedding ingestion (`vector_store.py`) to update ChromaDB
3. Ask questions through Streamlit or terminal (`rag.py`)
4. Review and refine results as needed

## Dependencies

- langchain
- chromadb
- streamlit
- numpy
- pandas

## Requirements

- Python 3.10+
- Ollama installed locally
- Sufficient storage for document embeddings
- RAM for model execution

## Use Cases

- Personal document Q&A systems
- Local knowledge base queries
- Private document analysis
- Offline research assistance
- Secure document processing
- Educational material exploration

## Notes

- Fully local setup ensures data privacy
- No internet required after initial setup
- Answers are strictly based on provided documents
- Configurable retrieval parameters
- Supports multiple document formats
- Scalable to large document collections

## Optional Enhancements

- Add pagination for answers in Streamlit
- Support multiple document sources
- Experiment with different Ollama models (llama2, mistral, qwen2.5)
- Implement document versioning
- Add advanced filtering options

## Built with

- Python
- Ollama
- ChromaDB
- Streamlit
- LangChain
