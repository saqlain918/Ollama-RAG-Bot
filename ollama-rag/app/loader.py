import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

DATA_DIR = "data"

def load_documents():
    """Load all documents (PDF/TXT) from the data folder."""
    docs = []
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        else:
            print(f"Skipping unsupported file: {file_name}")
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks for embedding."""
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    raw_docs = load_documents()
    chunks = chunk_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} docs, split into {len(chunks)} chunks.")
