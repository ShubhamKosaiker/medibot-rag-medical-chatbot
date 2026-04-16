"""
MEDICAL_CHATBOT.py
──────────────────
MediBot — RAG-powered medical information assistant.
Trained on the Gale Encyclopedia of Medicine.

Run:
    streamlit run MEDICAL_CHATBOT.py

Requirements:
    - GROQ_API_KEY must be set in your .env file
    - Vector store must already be built (run build_vectorstore.py first)
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory  # windowed — avoids unbounded context growth
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler          # for streaming
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

load_dotenv(find_dotenv())

# ── Constants ─────────────────────────────────────────────────────────────────
DB_FAISS_PATH = "vectorstore/db_faiss"

# Simple greeting detection — these bypass the RAG pipeline entirely
GREETINGS = {"hi", "hello", "hey", "hii", "howdy", "good morning", "good evening", "sup"}

# Number of past exchanges to keep in memory.
# 10 is a good balance: enough context for follow-up questions without
# exceeding the LLM's context window on long sessions.
MEMORY_WINDOW = 10

# Number of chunks retrieved from FAISS per query.
# 3 gives focused, relevant context; increase to 5 if answers feel incomplete.
RETRIEVAL_K = 3


# ── System prompt ─────────────────────────────────────────────────────────────
# This prompt is injected into the answer LLM call (not the condense-question call).
# It establishes MediBot's persona, strict grounding rules, and safety guardrails.
ANSWER_PROMPT = PromptTemplate.from_template("""
You are MediBot, a medical information assistant trained exclusively on the \
Gale Encyclopedia of Medicine.

Rules you must follow:
- Answer ONLY from the context provided below. Never fabricate information.
- If the answer is not in the context, respond:
  "I don't have enough information in my knowledge base to answer that. \
Please consult a qualified healthcare professional."
- Use accurate medical terminology but explain jargon when it appears.
- Never provide personal diagnoses or prescribe treatments — educational info only.
- Be concise and structured. Use bullet points for lists of symptoms/steps.

Context:
{context}

Question: {question}

Answer:""")


# ── Streaming callback ────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    """
    Pipes LLM tokens into a Streamlit placeholder as they arrive.
    This gives a real-time typing effect instead of waiting for the full response.

    How it works:
      - on_llm_new_token() is called by LangChain each time the LLM emits a token.
      - We accumulate tokens in self.text and re-render the placeholder each time.
      - The trailing "▌" acts as a blinking cursor during generation.
      - on_llm_end() does a final clean render without the cursor.
    """
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")   # blinking cursor effect

    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)          # final render, no cursor


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediBot · AI Medical Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.stApp { background: #F0F4F8; }

/* ── GLOBAL TEXT FIX ──────────────────────────────────────────────────────────
   Streamlit may default to white text in dark-mode environments.
   These rules force dark text across all main-area elements so text stays
   visible regardless of the user's OS colour scheme.                         */
.stApp p, .stApp span, .stApp label,
.stApp li, .stApp a, .stApp div,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown em,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #1E293B !important;
}

/* ── MAIN BLOCK ── */
.block-container {
    padding: 1.8rem 2.5rem 1rem !important;
    max-width: 900px !important;
}

/* Title */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: #0D3D3D !important;
    font-size: 2rem !important;
    letter-spacing: -0.4px !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Divider */
hr { border-color: #0D9488 !important; margin: 0.6rem 0 1rem !important; }

/* Info box */
[data-testid="stInfo"] {
    background: #F0FDFA !important;
    border: 1px solid #99F6E4 !important;
    border-left: 4px solid #0D9488 !important;
    border-radius: 12px !important;
    font-size: 0.83rem !important;
    color: #134E4A !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stInfo"] p,
[data-testid="stInfo"] span,
[data-testid="stInfo"] div { color: #134E4A !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.65rem !important;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04) !important;
    color: #1E293B !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #EFF9F8 !important;
    border-color: #B2E8E3 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 13px !important;
    border: 2px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    font-size: 0.88rem !important;
    font-family: 'Sora', sans-serif !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #0D9488 !important; }

/* Caption */
.stCaption p { color: #94A3B8 !important; font-size: 0.71rem !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0D3D3D !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.18) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 0.7rem 0 !important;
}

/* Sidebar metric */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    font-size: 0.65rem !important;
    color: rgba(255,255,255,0.42) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] div {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #2DD4BF !important;
}

/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.78rem !important;
}

/* Sidebar clear button */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(239,68,68,0.1) !important;
    color: #FCA5A5 !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    border-radius: 9px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 0.48rem !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(239,68,68,0.2) !important;
    color: #FEE2E2 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    """Return True if the input is a simple greeting — skip RAG for these."""
    return text.strip().lower() in GREETINGS


@st.cache_resource
def get_vectorstore():
    """
    Load the FAISS vector store from disk. Decorated with @st.cache_resource so
    it is loaded only ONCE per Streamlit server session (not on every page rerun).

    Raises FileNotFoundError with a helpful message if the index hasn't been built yet.
    """
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{DB_FAISS_PATH}'. "
            "Run  'python build_vectorstore.py'  first to build it."
        )
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # allow_dangerous_deserialization=True is safe here because WE built this
    # index ourselves from our own PDFs — it is not loaded from an external source.
    return FAISS.load_local(DB_FAISS_PATH, embedding_model,
                            allow_dangerous_deserialization=True)


def get_qa_chain(vectorstore, stream_handler=None):
    """
    Build and return a ConversationalRetrievalChain.

    Two separate LLMs are used intentionally:
      - condense_llm: No streaming. Quickly rewrites the user's question using
        chat history so follow-up questions ("What about its side effects?")
        become self-contained ("What are the side effects of metformin?").
      - answer_llm: Streaming enabled. Generates the final grounded answer from
        the retrieved FAISS chunks. The StreamHandler pushes tokens to the UI
        in real time so the user sees a typing effect.

    Memory is stored in session state so it persists across Streamlit reruns
    but resets when the user clears the conversation.
    """
    if "memory" not in st.session_state:
        # ConversationBufferWindowMemory keeps only the last k exchanges.
        # This prevents the context from growing unboundedly on long sessions
        # which would eventually exceed the LLM's token limit and cause errors.
        st.session_state.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    # Non-streaming LLM — used only for the internal question-condensation step.
    # Temperature=0 makes it deterministic for this rewriting task.
    condense_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

    # Streaming LLM — used for the final answer generation shown to the user.
    answer_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.environ["GROQ_API_KEY"],
        streaming=bool(stream_handler),
        callbacks=[stream_handler] if stream_handler else [],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=answer_llm,
        condense_question_llm=condense_llm,          # separate non-streaming LLM for condensation
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},  # inject our system prompt
    )


def get_unique_sources(source_documents) -> list[str]:
    """
    Extract and deduplicate source file names from retrieved documents.
    Each chunk carries metadata with the original PDF filename.
    """
    seen, sources = set(), []
    for doc in source_documents:
        name = os.path.basename(doc.metadata.get("source", "Unknown source"))
        if name not in seen:
            seen.add(name)
            sources.append(name)
    return sources


# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction, so persistent state
# must live in st.session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now().strftime("%b %d")
if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding:1.2rem 0.2rem 1rem;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:0.9rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
            <div style="width:40px;height:40px;background:linear-gradient(135deg,#0D9488,#2DD4BF);
                border-radius:11px;display:flex;align-items:center;justify-content:center;
                font-size:20px;box-shadow:0 4px 14px rgba(13,148,136,0.45);">⚕️</div>
            <span style="font-family:'Playfair Display',serif;font-size:1.42rem;
                color:#fff;letter-spacing:-0.3px;">MediBot</span>
        </div>
        <div style="font-size:0.65rem;color:#99F6E4;opacity:0.65;letter-spacing:0.06em;padding-left:2px;">
            AI Medical Assistant · RAG-Powered
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    st.caption("Session")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Date", st.session_state.session_start)
    with col2:
        st.metric("Questions", st.session_state.question_count)

    st.divider()

    # Recent questions (last 5, most recent first)
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        st.caption("Recent Questions")
        for msg in reversed(user_msgs[-5:]):
            preview = msg["content"][:38] + "…" if len(msg["content"]) > 38 else msg["content"]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.07);
                border-left:2px solid #2DD4BF;border-radius:7px;
                padding:0.42rem 0.68rem;margin-bottom:0.3rem;
                font-size:0.71rem;color:rgba(255,255,255,0.6);
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                💬 {preview}
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    # Tech stack info
    with st.expander("⚙️ Tech Stack"):
        st.markdown("""
        <div style="font-size:0.76rem;line-height:2.1;color:rgba(255,255,255,0.75)!important;">
        📚 <b style="color:#2DD4BF;">Source</b> — Gale Encyclopedia of Medicine<br>
        🤖 <b style="color:#2DD4BF;">LLM</b> — LLaMA 3.1 · 8B (Groq)<br>
        🔍 <b style="color:#2DD4BF;">Embeddings</b> — MiniLM-L6-v2<br>
        🗄️ <b style="color:#2DD4BF;">Vector DB</b> — FAISS (local)<br>
        🔗 <b style="color:#2DD4BF;">Framework</b> — LangChain<br>
        🖥️ <b style="color:#2DD4BF;">UI</b> — Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Clear conversation — resets messages, question counter, and LangChain memory
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        if "memory" in st.session_state:
            del st.session_state.memory   # also wipe LangChain's ConversationBufferWindowMemory
        st.rerun()


# ── MAIN AREA ────────────────────────────────────────────────────────────────

# Header row
col_icon, col_text = st.columns([0.07, 0.93])
with col_icon:
    st.markdown("""
    <div style="width:48px;height:48px;background:linear-gradient(135deg,#0E6060,#0D9488);
        border-radius:13px;display:flex;align-items:center;justify-content:center;
        font-size:23px;box-shadow:0 5px 16px rgba(13,148,136,0.3);margin-top:6px;">🩺</div>
    """, unsafe_allow_html=True)
with col_text:
    st.title("MediBot")
    st.caption("Powered by Gale Encyclopedia of Medicine · RAG + LLaMA 3.1 + FAISS")

st.divider()

# Welcome message shown only when chat is empty
if len(st.session_state.messages) == 0:
    st.info(
        "👋 **Welcome to MediBot!**  \n"
        "I'm trained on the **Gale Encyclopedia of Medicine**. Ask me about conditions, "
        "symptoms, treatments, or medications — every answer is grounded in the source "
        "document with citations.  \n\n"
        "🩺 Conditions · 💊 Treatments · 🔬 Symptoms · 🧬 Diagnoses · 💉 Medications"
    )

# Render the full chat history on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render source expanders for assistant messages that have sources stored
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# ── Chat input & response ─────────────────────────────────────────────────────
prompt = st.chat_input("Ask a medical question… e.g. What are symptoms of diabetes?")

if prompt:
    # Display and store the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if is_greeting(prompt):
        # Short-circuit: no need to hit the vector store for a greeting
        reply = ("Hello! 👋 I'm **MediBot**. Ask me anything about medical conditions, "
                 "symptoms, or treatments!")
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})

    else:
        try:
            vectorstore = get_vectorstore()

            with st.chat_message("assistant"):
                # Create an empty placeholder — StreamHandler will fill it token by token
                placeholder = st.empty()
                stream_handler = StreamHandler(placeholder)

                # Build the chain with the streaming handler attached to the answer LLM
                qa_chain = get_qa_chain(vectorstore, stream_handler)

                # Invoke the chain — streaming tokens appear in real time via the callback
                with st.spinner("Searching knowledge base…"):
                    response = qa_chain.invoke({"question": prompt})

                answer = response["answer"]
                sources = get_unique_sources(response["source_documents"])

                # Show sources in a collapsible expander (cleaner than inline text)
                if sources:
                    with st.expander("📄 View Sources"):
                        for src in sources:
                            st.markdown(f"- `{src}`")

            st.session_state.question_count += 1
            # Store sources alongside the message for re-rendering on page rerun
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except FileNotFoundError as e:
            # Friendly error if build_vectorstore.py hasn't been run yet
            st.error(f"⚠️ Vector store missing: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚕️ MediBot is for informational purposes only and does not constitute "
    "professional medical advice. Always consult a qualified healthcare provider."
)
