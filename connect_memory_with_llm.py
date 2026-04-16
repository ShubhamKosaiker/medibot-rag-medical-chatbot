"""
connect_memory_with_llm.py
──────────────────────────
CLI interface for MediBot — useful for quick testing without launching Streamlit.

Usage:
    python connect_memory_with_llm.py

Requires:
    - GROQ_API_KEY in .env
    - Vector store built (run build_vectorstore.py first)
"""

import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"
RETRIEVAL_K = 3   # number of chunks to retrieve per query

# ── System prompt ──────────────────────────────────────────────────────────────
# Instructs the LLM to stay strictly within the retrieved context
# and never fabricate medical information.
CUSTOM_PROMPT_TEMPLATE = """
You are MediBot, a medical information assistant trained on the Gale Encyclopedia of Medicine.

Rules:
- Use ONLY the context below. Never fabricate information.
- If the answer is not in the context, say: "I don't have enough information to answer that."
- Be concise and use accurate medical terminology.

Context:
{context}

Question:
{question}

Answer:"""


def load_llm():
    """Load the Groq LLM (LLaMA 3.1 8B). Temperature=0.3 for factual but fluent answers."""
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )


def load_vectorstore():
    """Load the pre-built FAISS vector store from disk."""
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{DB_FAISS_PATH}'. "
            "Run 'python build_vectorstore.py' first."
        )
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # allow_dangerous_deserialization is safe — we built this index ourselves
    return FAISS.load_local(DB_FAISS_PATH, embedding_model,
                            allow_dangerous_deserialization=True)


def build_qa_chain(vectorstore):
    """
    Build a RetrievalQA chain.
    RetrievalQA (vs ConversationalRetrievalChain) is simpler — no memory,
    good for stateless single-turn CLI queries.
    """
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",                                      # stuff = concatenate all chunks into one prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading vector store and LLM…")
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)
    print("Ready. Type your medical question (Ctrl+C to quit).\n")

    while True:
        try:
            user_query = input("You: ").strip()
            if not user_query:
                continue
            response = qa_chain.invoke({"query": user_query})
            print(f"\nMediBot: {response['result']}")
            # Print source file names for transparency
            sources = {
                os.path.basename(doc.metadata.get("source", "unknown"))
                for doc in response["source_documents"]
            }
            print(f"Sources: {', '.join(sources)}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break
