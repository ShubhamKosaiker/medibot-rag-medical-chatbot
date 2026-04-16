"""
build_vectorstore.py
────────────────────
One-time script to ingest PDF files and build the FAISS vector store.
Run this BEFORE launching the Streamlit app:

    python build_vectorstore.py

Pipeline:
    1. Load all PDFs from the data/ directory
    2. Split into overlapping text chunks (preserves context across boundaries)
    3. Generate embeddings with MiniLM-L6-v2 (lightweight, runs on CPU)
    4. Save the FAISS index to vectorstore/db_faiss/
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ── Chunking config ────────────────────────────────────────────────────────────
# Medical text is dense — larger chunks keep more clinical context per retrieval.
# Overlap of 100 ensures no sentence is split across chunk boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_pdf_files(data_path: str):
    """Load every PDF in data_path using LangChain's DirectoryLoader."""
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No PDF files found in '{data_path}'. "
            "Place your source PDFs there and re-run."
        )
    print(f"[1/4] Loaded {len(documents)} document page(s) from {data_path}")
    return documents


def create_chunks(documents):
    """
    Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries to split on paragraphs → sentences →
    words in that priority order, so chunk boundaries are semantically cleaner.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"[2/4] Created {len(chunks)} text chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_embedding_model():
    """
    Load the sentence-transformer embedding model.
    MiniLM-L6-v2 is fast and CPU-friendly while still producing high-quality
    semantic embeddings (384-dimensional vectors).
    """
    print("[3/4] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2 ...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_and_save_vectorstore(chunks, embedding_model):
    """
    Build a FAISS index from the text chunks and save it to disk.
    FAISS (Facebook AI Similarity Search) enables millisecond-level
    nearest-neighbour lookup across thousands of embeddings.
    """
    print(f"[4/4] Building FAISS index and saving to '{DB_FAISS_PATH}' ...")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"\nDone! Vector store saved to '{DB_FAISS_PATH}'.")
    print("You can now launch the app:  streamlit run MEDICAL_CHATBOT.py")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    build_and_save_vectorstore(chunks, embedding_model)
