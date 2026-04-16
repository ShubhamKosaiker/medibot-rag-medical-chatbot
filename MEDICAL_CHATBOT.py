"""
MEDICAL_CHATBOT.py
──────────────────
MediBot — RAG-powered medical information assistant.
Trained on the Gale Encyclopedia of Medicine + Wikipedia medical articles.

Run:
    streamlit run MEDICAL_CHATBOT.py

Requirements:
    - GROQ_API_KEY in .env
    - Vector store built first (python build_vectorstore.py)
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

load_dotenv(find_dotenv())

# ── Constants ─────────────────────────────────────────────────────────────────
DB_FAISS_PATH  = "vectorstore/db_faiss"
GREETINGS      = {"hi", "hello", "hey", "hii", "howdy", "good morning", "good evening", "sup"}
MEMORY_WINDOW  = 10   # max past exchanges kept in context
RETRIEVAL_K    = 3    # chunks retrieved per query

# ── Prompts ───────────────────────────────────────────────────────────────────

# Used by the condense LLM to rewrite follow-up questions using chat history.
# E.g. "What are its symptoms?" → "What are the symptoms of diabetes mellitus?"
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the question to be "
     "fully self-contained (no pronouns referring to prior messages). "
     "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Used by the answer LLM. Context is the retrieved FAISS chunks injected at runtime.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are MediBot, a medical information assistant trained on the "
     "Gale Encyclopedia of Medicine and Wikipedia medical articles.\n\n"
     "Rules:\n"
     "- Answer ONLY from the Context below. Never fabricate information.\n"
     "- If the answer is not in the Context, say: \"I don't have enough information "
     "in my knowledge base to answer that. Please consult a qualified healthcare professional.\"\n"
     "- Use accurate medical terminology but explain jargon when it appears.\n"
     "- Never provide personal diagnoses or prescribe treatments.\n"
     "- Be concise and structured; use bullet points for lists.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


# ── Streaming callback ────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    """
    Streams LLM tokens into a Streamlit placeholder in real time.

    LangChain calls on_llm_new_token() for each emitted token.
    We accumulate tokens in self.text and re-render the placeholder,
    creating a typing effect. The trailing cursor '▌' is removed on finish.
    """
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)


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
   Force dark text everywhere in the main area so the app looks correct
   regardless of the user's OS light/dark mode setting.                       */
.stApp p, .stApp span, .stApp label, .stApp li, .stApp a,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stAlert"] p,
[data-testid="stAlert"] div { color: #1E293B !important; }

/* ── MAIN BLOCK ── */
.block-container { padding: 1.8rem 2.5rem 1rem !important; max-width: 900px !important; }

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

/* ── Chat input bottom bar ──────────────────────────────────────────────────
   The sticky footer can inherit dark-mode colours. Force it light so typed
   text is always visible.                                                   */
[data-testid="stBottom"],
[data-testid="stBottom"] > div { background: #F0F4F8 !important; }

[data-testid="stChatInput"] {
    border-radius: 13px !important;
    border: 2px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    color: #1E293B !important;
    background: #FFFFFF !important;
    caret-color: #0D9488 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #94A3B8 !important; }
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
[data-testid="stSidebar"] div { color: rgba(255,255,255,0.8) !important; }
[data-testid="stSidebar"] .stCaption p {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; margin: 0.7rem 0 !important; }

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
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(239,68,68,0.2) !important;
    color: #FEE2E2 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETINGS


@st.cache_resource
def get_vectorstore():
    """
    Load FAISS index once per server session.
    @st.cache_resource persists the loaded object across all Streamlit reruns,
    avoiding a 5-10s reload on every user interaction.
    """
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{DB_FAISS_PATH}'. "
            "Run  python build_vectorstore.py  first."
        )
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # safe: we built this index ourselves from our own PDFs
    return FAISS.load_local(DB_FAISS_PATH, embedding_model,
                            allow_dangerous_deserialization=True)


def make_llm(streaming: bool = False, callbacks: list = None):
    """Create a ChatGroq LLM instance."""
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3 if streaming else 0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        streaming=streaming,
        callbacks=callbacks or [],
    )


def get_windowed_history() -> list:
    """
    Return the last MEMORY_WINDOW pairs of messages as LangChain message objects.
    Keeps the LLM context bounded — prevents token limit errors on long sessions.
    """
    history = st.session_state.get("lc_history", [])
    # Each pair = HumanMessage + AIMessage, so slice last MEMORY_WINDOW*2 items
    return history[-(MEMORY_WINDOW * 2):]


def get_unique_sources(docs) -> list:
    """Deduplicate source filenames from retrieved FAISS documents."""
    seen, sources = set(), []
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "Unknown"))
        if name not in seen:
            seen.add(name)
            sources.append(name)
    return sources


def answer_question(question: str, vectorstore, stream_handler=None) -> tuple[str, list, list]:
    """
    Full RAG pipeline using LCEL (LangChain Expression Language):

    1. Condense — rewrite follow-up questions to be self-contained using history.
       Uses a non-streaming LLM so the intermediate question never appears in the UI.
    2. Retrieve — FAISS nearest-neighbour search over embedded chunks.
    3. Answer — streaming LLM reads retrieved context + history, generates answer.

    Returns: (answer_text, source_docs, updated_lc_history)
    """
    chat_history = get_windowed_history()
    retriever    = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # ── Step 1: Condense question ────────────────────────────────────────────
    if chat_history:
        # Only invoke condense step when there's actual history to consider
        condense_chain = CONDENSE_PROMPT | make_llm(streaming=False) | StrOutputParser()
        standalone_q   = condense_chain.invoke({
            "chat_history": chat_history,
            "question":     question,
        })
    else:
        standalone_q = question   # no history → use question as-is

    # ── Step 2: Retrieve relevant chunks ────────────────────────────────────
    docs    = retriever.invoke(standalone_q)
    context = "\n\n".join(doc.page_content for doc in docs)

    # ── Step 3: Generate streaming answer ────────────────────────────────────
    callbacks  = [stream_handler] if stream_handler else []
    answer_llm = make_llm(streaming=bool(stream_handler), callbacks=callbacks)
    answer_chain = ANSWER_PROMPT | answer_llm | StrOutputParser()
    answer = answer_chain.invoke({
        "chat_history": chat_history,
        "question":     question,
        "context":      context,
    })

    # Update LangChain message history for next turn
    updated_history = (st.session_state.get("lc_history", [])
                       + [HumanMessage(content=question),
                          AIMessage(content=answer)])

    return answer, docs, updated_history


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "lc_history"     not in st.session_state: st.session_state.lc_history     = []
if "session_start"  not in st.session_state: st.session_state.session_start  = datetime.now().strftime("%b %d")
if "question_count" not in st.session_state: st.session_state.question_count = 0


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
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

    st.caption("Session")
    col1, col2 = st.columns(2)
    with col1: st.metric("Date",      st.session_state.session_start)
    with col2: st.metric("Questions", st.session_state.question_count)
    st.divider()

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

    with st.expander("⚙️ Tech Stack"):
        st.markdown("""
        <div style="font-size:0.76rem;line-height:2.1;color:rgba(255,255,255,0.75)!important;">
        📚 <b style="color:#2DD4BF;">Source</b> — Gale Encyclopedia + Wikipedia<br>
        🤖 <b style="color:#2DD4BF;">LLM</b> — LLaMA 3.1 · 8B (Groq)<br>
        🔍 <b style="color:#2DD4BF;">Embeddings</b> — MiniLM-L6-v2<br>
        🗄️ <b style="color:#2DD4BF;">Vector DB</b> — FAISS (local)<br>
        🔗 <b style="color:#2DD4BF;">Framework</b> — LangChain LCEL<br>
        🖥️ <b style="color:#2DD4BF;">UI</b> — Streamlit
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages       = []
        st.session_state.lc_history     = []
        st.session_state.question_count = 0
        st.rerun()


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
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

if len(st.session_state.messages) == 0:
    st.info(
        "👋 **Welcome to MediBot!**  \n"
        "I'm trained on the **Gale Encyclopedia of Medicine** and **Wikipedia medical articles**. "
        "Ask me about conditions, symptoms, treatments, or medications — every answer is "
        "grounded in the source documents with citations.  \n\n"
        "🩺 Conditions · 💊 Treatments · 🔬 Symptoms · 🧬 Diagnoses · 💉 Medications"
    )

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a medical question… e.g. What are symptoms of diabetes?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if is_greeting(prompt):
        reply = ("Hello! 👋 I'm **MediBot**. Ask me anything about medical conditions, "
                 "symptoms, or treatments!")
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})

    else:
        try:
            vectorstore = get_vectorstore()

            with st.chat_message("assistant"):
                placeholder    = st.empty()
                stream_handler = StreamHandler(placeholder)

                with st.spinner("Searching knowledge base…"):
                    answer, docs, updated_history = answer_question(
                        prompt, vectorstore, stream_handler
                    )

                sources = get_unique_sources(docs)
                if sources:
                    with st.expander("📄 View Sources"):
                        for src in sources:
                            st.markdown(f"- `{src}`")

            # Persist history and stats
            st.session_state.lc_history     = updated_history
            st.session_state.question_count += 1
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources,
            })

        except FileNotFoundError as e:
            st.error(f"⚠️ Vector store missing: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚕️ MediBot is for informational purposes only and does not constitute "
    "professional medical advice. Always consult a qualified healthcare provider."
)
