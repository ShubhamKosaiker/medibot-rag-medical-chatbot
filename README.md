# MediBot — AI Medical Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot grounded on the *Gale Encyclopedia of Medicine*. Ask about conditions, symptoms, treatments, and medications — every answer is cited back to the source document.

---

## Architecture

```
User Query
    │
    ▼
[Question Condensation LLM]   ← rewrites follow-up questions using chat history
    │
    ▼
[FAISS Vector Store]          ← top-k semantic search over embedded PDF chunks
    │
    ▼
[Answer LLM + System Prompt]  ← LLaMA 3.1 8B, grounded strictly on retrieved context
    │
    ▼
[Streamlit UI]                ← streaming response + collapsible source citations
```

**Tech Stack**

| Component | Tool |
|-----------|------|
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (local, CPU) |
| RAG Framework | LangChain `ConversationalRetrievalChain` |
| UI | Streamlit |
| Source Data | Gale Encyclopedia of Medicine (PDF) |

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd medical_bots
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Add your PDF(s)

Place your source PDF(s) inside the `data/` folder.

### 4. Build the vector store

This only needs to run once (or whenever you add new PDFs):

```bash
python build_vectorstore.py
```

This will:
- Load all PDFs from `data/`
- Split them into 800-character chunks (100-char overlap)
- Embed them with MiniLM-L6-v2
- Save the FAISS index to `vectorstore/db_faiss/`

### 5. Launch the app

```bash
streamlit run MEDICAL_CHATBOT.py
```

---

## Project Structure

```
medical_bots/
├── MEDICAL_CHATBOT.py          # Main Streamlit app
├── build_vectorstore.py        # One-time PDF ingestion & FAISS index builder
├── connect_memory_with_llm.py  # CLI interface for quick testing
├── requirements.txt
├── .gitignore
├── data/                       # Place source PDFs here (git-ignored)
└── vectorstore/                # FAISS index stored here (git-ignored)
```

---

## Key Design Decisions

- **Two LLMs in the chain**: A non-streaming `condense_llm` rewrites follow-up questions using chat history, while a streaming `answer_llm` generates the final grounded answer. This avoids streaming noise from the condensation step appearing in the UI.
- **Windowed memory** (`k=10`): Keeps only the last 10 exchanges so long sessions don't overflow the LLM's context window.
- **Strict system prompt**: The answer LLM is instructed to stay within retrieved context and never fabricate medical information.
- **Source citations**: Every answer shows the source PDF filenames in a collapsible expander.

---

## Disclaimer

MediBot is for **informational purposes only** and does not constitute professional medical advice. Always consult a qualified healthcare provider.
