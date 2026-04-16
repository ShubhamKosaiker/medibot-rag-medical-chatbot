"""
build_vectorstore.py
────────────────────
One-time script to ingest knowledge sources and build the FAISS vector store.
Run this BEFORE launching the Streamlit app:

    python build_vectorstore.py

Supports THREE source types — all are combined into one FAISS index:
  1. PDFs in the data/ folder         (e.g. Gale Encyclopedia of Medicine)
  2. Wikipedia medical articles        (queried by topic list below)
  3. Web pages / URLs                  (add URLs to WEB_URLS list below)

Pipeline:
    → Load documents from all sources
    → Split into overlapping text chunks
    → Embed with MiniLM-L6-v2
    → Save FAISS index to vectorstore/db_faiss/
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH     = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# ── Chunking config ───────────────────────────────────────────────────────────
# Medical text is dense — larger chunks keep more clinical context per retrieval.
# Overlap of 100 ensures sentences are never silently cut at chunk boundaries.
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── Wikipedia topics ──────────────────────────────────────────────────────────
# Add or remove medical topics here. Each entry pulls the Wikipedia article
# for that topic and adds it to the knowledge base.
# doc_content_chars_max limits how much of each article is loaded (keeps chunks manageable).
WIKIPEDIA_TOPICS = [
    "Diabetes mellitus",
    "Hypertension",
    "Asthma",
    "Cancer",
    "Pneumonia",
    "Tuberculosis",
    "Malaria",
    "HIV/AIDS",
    "Heart failure",
    "Stroke",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Depression (mood)",
    "Anxiety disorder",
    "Kidney failure",
    "Liver cirrhosis",
    "Anemia",
    "Arthritis",
    "Osteoporosis",
    "Hypothyroidism",
]

# ── Web URLs ──────────────────────────────────────────────────────────────────
# Add any publicly accessible medical reference pages here.
# Leave empty to skip web loading.
WEB_URLS = [
    # Example: "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdfs(data_path: str):
    """Load all PDFs from data/. Returns empty list (not error) if folder is empty."""
    if not os.path.exists(data_path) or not any(
        f.endswith(".pdf") for f in os.listdir(data_path)
    ):
        print(f"  [PDFs] No PDFs found in '{data_path}' — skipping.")
        return []
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"  [PDFs] Loaded {len(docs)} page(s) from {data_path}")
    return docs


def load_wikipedia(topics: list[str]):
    """
    Load Wikipedia articles for each topic in the list.
    WikipediaLoader fetches the article summary + body for the given query.
    doc_content_chars_max=4000 keeps each article to ~4000 characters to avoid
    very long documents that would produce too many chunks from one source.
    """
    if not topics:
        return []
    all_docs = []
    for topic in topics:
        try:
            docs = WikipediaLoader(
                query=topic,
                load_max_docs=1,           # one article per topic
                doc_content_chars_max=4000,
            ).load()
            all_docs.extend(docs)
            print(f"  [Wikipedia] Loaded: {topic}")
        except Exception as e:
            print(f"  [Wikipedia] Failed '{topic}': {e}")
    print(f"  [Wikipedia] Total: {len(all_docs)} article(s)")
    return all_docs


def load_web_pages(urls: list[str]):
    """Load content from web URLs using LangChain's WebBaseLoader."""
    if not urls:
        return []
    try:
        docs = WebBaseLoader(urls).load()
        print(f"  [Web] Loaded {len(docs)} page(s)")
        return docs
    except Exception as e:
        print(f"  [Web] Failed: {e}")
        return []


def create_chunks(documents):
    """
    Split documents into overlapping chunks.
    RecursiveCharacterTextSplitter tries paragraph → sentence → word splits
    in that order, so boundaries are semantically cleaner than a hard cut.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  [Chunking] {len(documents)} document(s) -> {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_embedding_model():
    """
    MiniLM-L6-v2: 384-dim sentence embeddings, fast on CPU, no GPU needed.
    The same model must be used at both index-time and query-time.
    """
    print("  [Embeddings] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_and_save(chunks, embedding_model):
    """Build FAISS index and save to disk."""
    print(f"  [FAISS] Building index over {len(chunks)} chunks ...")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"  [FAISS] Saved to '{DB_FAISS_PATH}'")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n== MediBot Vector Store Builder ==================================\n")

    print("Step 1/4  Loading knowledge sources...")
    pdf_docs  = load_pdfs(DATA_PATH)
    wiki_docs = load_wikipedia(WIKIPEDIA_TOPICS)
    web_docs  = load_web_pages(WEB_URLS)

    all_docs = pdf_docs + wiki_docs + web_docs
    if not all_docs:
        print("\nNo documents loaded from any source. Add PDFs to data/ or "
              "check your WIKIPEDIA_TOPICS / WEB_URLS lists.")
        exit(1)

    print(f"\n  Total documents loaded: {len(all_docs)}")

    print("\nStep 2/4  Chunking documents...")
    chunks = create_chunks(all_docs)

    print("\nStep 3/4  Loading embedding model...")
    embedding_model = get_embedding_model()

    print("\nStep 4/4  Building and saving FAISS index...")
    build_and_save(chunks, embedding_model)

    print("\n==================================================================")
    print("Done! Launch the app with:  streamlit run MEDICAL_CHATBOT.py")
    print(f"Knowledge base: {len(pdf_docs)} PDF pages  |  "
          f"{len(wiki_docs)} Wikipedia articles  |  {len(web_docs)} web pages")
