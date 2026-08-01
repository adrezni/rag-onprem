import threading

import streamlit as st
from rag_engine import index_pdfs, ensure_indexed, query_rag_stream, warm_up
from config import MODEL_NAME
from config import MODEL_BASE_URL

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="📄 PDF Chat RAG",
    page_icon="📄",
    layout="wide"
)

# ── Warm Up ────────────────────────────────────────────────────
# Load the model onto the GPU in the background as soon as the app opens,
# so the user's first question doesn't pay the ~several-second cold start.
# Guarded so it only fires once per session, and threaded so it never blocks
# the page from rendering.
if "warmed_up" not in st.session_state:
    st.session_state.warmed_up = True
    threading.Thread(target=warm_up, daemon=True).start()

# ── Auto-Index ─────────────────────────────────────────────────
# Index the PDFs automatically on first load so the app is ready to answer
# questions without a manual click. Runs once per session; when an index
# already exists this is a fast no-op (just a count check).
if "index_status" not in st.session_state:
    with st.spinner("⏳ Preparing document index..."):
        st.session_state.index_status = ensure_indexed()

# ── Title ──────────────────────────────────────────────────────
st.title("📄 Local PDF Chat — Powered by " + MODEL_NAME + "")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("""
    **Instructions:**
    1. 📁 Place your PDFs in the `pdfs/` folder
    2. 💬 Ask questions in the chat below

    PDFs are indexed automatically on startup, and added, changed, or
    removed files are re-indexed the next time the app loads.
    **Re-index PDFs** forces a manual rebuild.
    """)

    st.markdown("---")

    # Show the automatic startup indexing result.
    status = st.session_state.get("index_status", "")
    if status.startswith("❌"):
        st.error(status)
    else:
        st.caption(status)

    # Manual re-index for when PDFs are added or changed after startup.
    if st.button("🔄 Re-index PDFs", use_container_width=True):
        with st.spinner("⏳ Indexing PDFs... please wait..."):
            result = index_pdfs()
            st.session_state.index_status = result
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)

    st.markdown("---")
    st.markdown("**🤖 Model:** " + MODEL_NAME + " (via NIM)")
    st.markdown("**🗄️ Vector DB:** ChromaDB")
    st.markdown("**🔍 Embeddings:** all-MiniLM-L6-v2")

# ── Chat History ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ─────────────────────────────────────────────────
if prompt := st.chat_input("💬 Ask a question about your PDFs..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(query_rag_stream(prompt))
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.markdown("""
            **💡 Troubleshooting:**
            - ✅ Make sure NIM endpoint is reachable: `curl {MODEL_BASE_URL}/v1/models`
            - ✅ Make sure PDFs are indexed — click **Index PDFs**
            - ✅ Make sure Mistral is pulled: `ollama pull mistral`
            """)