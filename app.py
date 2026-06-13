import streamlit as st
from langchain_core.messages import HumanMessage
import time
import config 
from graph import studymate_graph, get_mermaid_diagram

# Page Config
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
/* ===== Global ===== */
.stApp {
    background: linear-gradient(180deg, #f4f7fb 0%, #eaf1f8 100%);
}
body, .stMarkdown, .stCaption, p {
    color: #1f2a44;
}

/* ===== Hero Header ===== */
.hero {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
    border-radius: 18px;
    padding: 32px 36px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(59,130,246,0.25);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.3em;
    margin: 0;
    font-weight: 800;
    letter-spacing: 0.5px;
}
.hero p {
    color: #eaf2ff;
    margin: 8px 0 0;
    font-size: 1em;
}
.hero .stack-line {
    margin-top: 10px;
    font-size: 0.85em;
    color: #dbeafe;
}

/* ===== Badges (Intent) ===== */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 700;
    margin: 2px 0;
    letter-spacing: 0.3px;
}
.b-research  { background:#e0edff; color:#2563eb; border:1px solid #93c5fd; }
.b-analysis  { background:#fde8e8; color:#dc2626; border:1px solid #fca5a5; }
.b-writing   { background:#e7f8ed; color:#16a34a; border:1px solid #86efac; }
.b-flashcard { background:#fef6e0; color:#d97706; border:1px solid #fcd34d; }
.b-qa        { background:#f1e9fb; color:#7c3aed; border:1px solid #c4b5fd; }

.trace-tag {
    font-size: 11px;
    color: #6b7280;
    margin-left: 6px;
}

/* ===== Quick Start Section ===== */
.qs-title {
    font-size: 0.95em;
    font-weight: 700;
    color: #334155;
    margin: 4px 0 10px 0;
}

div[data-testid="column"] .stButton button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #dbeafe;
    background: #ffffff;
    color: #1e3a5f;
    padding: 14px 10px;
    font-size: 0.85em;
    font-weight: 600;
    transition: all 0.2s ease;
    height: 100%;
}
div[data-testid="column"] .stButton button:hover {
    border-color: #3b82f6;
    color: #1d4ed8;
    background: #eff6ff;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(59,130,246,0.15);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stMetric {
    background: #f1f5fb;
    border-radius: 10px;
    padding: 8px 12px;
    border: 1px solid #e2e8f0;
}

/* Reset button styling */
section[data-testid="stSidebar"] .stButton button {
    border: 1px solid #fca5a5;
    color: #dc2626;
    background: #fef2f2;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #fee2e2;
    border-color: #dc2626;
}

/* ===== Chat Bubbles ===== */
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 6px;
    margin-bottom: 6px;
    background: #ffffff;
    border: 1px solid #e7edf5;
}

/* ===== Status / Progress text ===== */
.progress-text {
    font-size: 0.88em;
    color: #475569;
}

/* ===== Footer ===== */
.footer-bar {
    margin-top: 10px;
    padding: 14px 0 4px 0;
    border-top: 1px solid #e2e8f0;
}
.footer-bar p {
    color: #94a3b8 !important;
    font-size: 0.8em;
}

/* ===== Empty state ===== */
.empty-state {
    text-align:center;
    padding: 30px 0;
    color:#94a3b8;
}
/* ===== Fix komponen yang ikut dark theme bawaan ===== */
header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stChatInput"] {
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-radius: 12px;
}
div[data-testid="stChatInput"] textarea {
    color: #1f2a44 !important;
}

section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: #f1f5fb;
    color: #1f2a44 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .streamlit-expanderContent {
    background: #ffffff;
    color: #1f2a44;
}

/* Pastikan teks tombol quick start tetap kontras */
div[data-testid="column"] .stButton button p {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
  <h1>StudyMate AI</h1>
  <p>Personal Multi-Agent Learning Assistant</p>
  <div class="stack-line">LangChain &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; LangSmith &nbsp;·&nbsp; Gemini AI</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## StudyMate AI")
    st.caption("Multi-Agent Learning Assistant")
    st.markdown("---")

    with st.expander("Tech Stack", expanded=True):
        st.markdown("""
| Library | Peran di Project |
|---------|-----------------|
| LangChain | LLM, Prompts, Chains |
| LangGraph | Multi-Agent Flow |
| LangSmith | Monitoring & Trace |
| Gemini Flash | LLM Model (Free) |
| Tavily | Web Search (Free) |
        """)

    with st.expander("5 Mode Agent", expanded=True):
        st.markdown("""
- **Research** — Riset + web search  
- **Analysis** — Analisis teks  
- **Writing** — Buat konten/esai  
- **Flashcard** — Kartu belajar  
- **Q&A** — Tanya jawab  
        """)

    with st.expander("LangGraph Diagram"):
        mermaid = get_mermaid_diagram()
        if mermaid:
            st.code(mermaid, language="text")
        else:
            st.caption("Diagram tidak tersedia.")

    st.markdown("---")

    if "messages" in st.session_state:
        n = sum(1 for m in st.session_state.messages if m["role"] == "user")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Pertanyaan", n)
        with col_b:
            st.metric("Agent Aktif", "5")

    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_input = None
        st.rerun()

    st.markdown("---")
    st.caption("Semua percakapan otomatis di-trace di [LangSmith](https://smith.langchain.com)")

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# Quick Start Buttons
st.markdown('<div class="qs-title">Coba Contoh Cepat</div>', unsafe_allow_html=True)
cols = st.columns(5)
examples = [
    ("Research", "Jelaskan tentang Large Language Models (LLM) dan perkembangannya"),
    ("Analisis", "Analisis teks ini: 'Perkembangan AI sangat pesat dalam 5 tahun terakhir. Meski membawa banyak manfaat seperti efisiensi kerja, ada kekhawatiran serius soal privasi data dan hilangnya lapangan pekerjaan.'"),
    ("Writing", "Buatkan artikel lengkap tentang pentingnya NLP dalam era digital"),
    ("Flashcard", "Buat flashcard untuk belajar konsep dasar Machine Learning"),
    ("Q&A", "Apa perbedaan LangChain, LangGraph, dan LangSmith?"),
]
for col, (label, prompt) in zip(cols, examples):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.pending_input = prompt
            st.rerun()

st.markdown("---")

# Tampilkan Chat History
intent_labels = {
    "research":  ("🔍", "RESEARCH",  "b-research"),
    "analysis":  ("📊", "ANALYSIS",  "b-analysis"),
    "writing":   ("✍️", "WRITING",   "b-writing"),
    "flashcard": ("🃏", "FLASHCARD", "b-flashcard"),
    "qa":        ("🧠", "Q&A",       "b-qa"),
}

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">Mulai percakapan dengan mengetik pertanyaan di bawah, '
        'atau coba salah satu contoh cepat di atas.</div>',
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "intent" in msg:
            icon, label, css = intent_labels.get(msg["intent"], ("🤖", "AI", "b-qa"))
            st.markdown(
                f'<span class="badge {css}">{icon} {label}</span>'
                f'<span class="trace-tag">· LangGraph Multi-Agent · Traced by LangSmith</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# Proses Input
user_input = None

if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

if chat_input := st.chat_input("Ketik pertanyaan, teks untuk dianalisis, atau topik untuk diriset..."):
    user_input = chat_input

# Run LangGraph Workflow
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        stages = [
            "Memulai LangGraph StateGraph...",
            "Router Agent menganalisis intent...",
            "Meneruskan ke agent yang sesuai...",
            "Generating response...",
        ]
        progress = st.progress(0)
        for i, stage in enumerate(stages):
            placeholder.markdown(f'<div class="progress-text">{stage}</div>', unsafe_allow_html=True)
            progress.progress(int((i + 1) / len(stages) * 100))
            time.sleep(0.35)
        placeholder.empty()
        progress.empty()

        try:
            result = studymate_graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "intent": "",
                "search_results": "",
                "final_response": "",
            })

            intent = result.get("intent", "qa")
            response = result.get("final_response", "Maaf, terjadi kesalahan.")

            icon, label, css = intent_labels.get(intent, ("🤖", "AI", "b-qa"))
            st.markdown(
                f'<span class="badge {css}">{icon} {label}</span>'
                f'<span class="trace-tag">· LangGraph Multi-Agent · Traced by LangSmith</span>',
                unsafe_allow_html=True
            )

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent,
            })

        except Exception as e:
            st.error(
                f"**Error:** {str(e)}\n\n"
                "Pastikan file `.env` sudah berisi API keys yang benar."
            )

# Footer
st.markdown('<div class="footer-bar"></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.caption("Built with [LangChain](https://python.langchain.com)")
c2.caption("Orchestrated by [LangGraph](https://langchain-ai.github.io/langgraph)")
c3.caption("Monitored by [LangSmith](https://smith.langchain.com)")