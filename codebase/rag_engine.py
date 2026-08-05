__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from functools import lru_cache

# ── Document Loading ───────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

# ── Text Splitting ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings ─────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Vector Store ───────────────────────────────────────────────
from langchain_community.vectorstores import Chroma

# ── Chains & Prompts ───────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# ── LLM ───────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from config import (
    MODEL_BASE_URL,
    MODEL_NAME,
    MODEL_API_KEY,
    MODEL_TEMPERATURE,
    MODEL_MAX_TOKENS,
)


# ── Configuration ──────────────────────────────────────────────
CHROMA_DB_PATH = "./chroma_db"
PDF_FOLDER     = "./pdfs"
#OLLAMA_MODEL   = MODEL_NAME
# Records which PDFs (and their size/mtime) the current index was built from,
# so startup can detect added, changed, or removed files and re-index only when
# the folder actually differs. Kept inside the DB dir so it travels with it.
MANIFEST_PATH  = os.path.join(CHROMA_DB_PATH, "pdf_manifest.json")


# ── Embeddings Model ───────────────────────────────────────────
# Cached so the MiniLM model is only loaded into memory once per process,
# instead of on every single query.
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# ── PDF Folder Fingerprint ─────────────────────────────────────
# A manifest maps each PDF's filename to its [size, mtime]. Comparing the
# folder's current fingerprint against the one saved at index time tells us
# whether any file was added, edited, or removed since the last index.
def _pdf_manifest():
    manifest = {}
    if os.path.exists(PDF_FOLDER):
        for file in os.listdir(PDF_FOLDER):
            if file.endswith(".pdf"):
                stat = os.stat(os.path.join(PDF_FOLDER, file))
                manifest[file] = [stat.st_size, int(stat.st_mtime)]
    return manifest


def _load_manifest():
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _save_manifest(manifest):
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)


# ── Load PDFs ──────────────────────────────────────────────────
def load_pdfs():
    documents = []
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            filepath = os.path.join(PDF_FOLDER, file)
            loader   = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents


# ── Split Documents ────────────────────────────────────────────
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


# ── Create Vector Store ────────────────────────────────────────
def create_vector_store(chunks):
    embeddings   = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vector_store


# ── Load Vector Store ──────────────────────────────────────────
def load_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )


# ── Build RAG Chain ────────────────────────────────────────────
# Cached so the vector store, retriever, and LLM client are constructed
# once and reused across queries rather than rebuilt on every question.
# num_predict caps how many tokens the model generates. Generation is the
# slowest stage on this hardware, so bounding it keeps responses snappy and
# stops the model from rambling past a complete answer. 128 tokens is ~1-2
# short paragraphs, which suits the concise answers this prompt asks for.
#
# keep_alive keeps the 4.9 GB model resident on the GPU between questions.
# Ollama's default unloads it after 5 minutes idle, so a question asked after a
# short pause pays a multi-second reload before generating a single token.
# Holding it in memory removes that cold-start penalty entirely.
#
# num_ctx sizes the context window (and thus the KV cache). We only ever feed
# k=3 short chunks plus the prompt, so the default 4096 is oversized; 2048
# allocates less memory and loads faster with no loss of context.
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        base_url=f"{MODEL_BASE_URL}/v1",
        api_key=MODEL_API_KEY,
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
    )

@lru_cache(maxsize=1)
def get_rag_chain():
    vector_store = load_vector_store()
    retriever    = vector_store.as_retriever(search_kwargs={"k": 3})
    llm          = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    Answer concisely in a few sentences. Get straight to the point.
    If you don't know the answer, say you don't know.
    Do not make up information.

    Context: {context}
    Question: {input}

    Answer:""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain


# ── Warm Up ────────────────────────────────────────────────────
# Builds the chain and pings the model with a throwaway token so it is
# loaded onto the GPU before the user asks their first real question.
# Call this once at app startup to hide the initial ~several-second load.
def warm_up():
    try:
        get_llm().invoke("hi")
        return True
    except Exception:
        return False


# ── Index PDFs ─────────────────────────────────────────────────
def index_pdfs():
    try:
        documents = load_pdfs()
        if not documents:
            return "❌ No PDFs found in the pdfs/ folder!"
        chunks = split_documents(documents)
        # Drop any existing collection first. Chroma.from_documents appends, so
        # without this a rebuild would duplicate every chunk already indexed.
        _reset_vector_store()
        create_vector_store(chunks)
        # Record what we just indexed so ensure_indexed() can detect changes.
        _save_manifest(_pdf_manifest())
        # Invalidate cached chain so the new index is picked up.
        get_rag_chain.cache_clear()
        return f"✅ Successfully indexed {len(documents)} pages from {len(set(d.metadata.get('source', '') for d in documents))} PDF(s)!"
    except Exception as e:
        return f"❌ Error during indexing: {str(e)}"


# Drops the existing Chroma collection so index_pdfs() can rebuild from a
# clean slate rather than appending on top of the previous chunks.
def _reset_vector_store():
    try:
        load_vector_store().delete_collection()
    except Exception:
        pass


# ── Ensure Indexed ─────────────────────────────────────────────
# Indexes the PDFs automatically so questions work without a manual click.
# Re-indexes only when the pdfs/ folder differs from what was last indexed
# (a file was added, changed, or removed); otherwise the existing index is
# reused, since re-embedding every PDF is slow. This runs on every startup.
def ensure_indexed():
    try:
        store   = load_vector_store()
        current = _pdf_manifest()
        if store._collection.count() > 0 and _load_manifest() == current:
            return "ℹ️ Using existing index."
        if not current:
            return "❌ No PDFs found in the pdfs/ folder!"
        return index_pdfs()
    except Exception as e:
        return f"❌ Error preparing index: {str(e)}"


# ── Query ──────────────────────────────────────────────────────
def query_rag(question: str):
    chain    = get_rag_chain()
    response = chain.invoke({"input": question})
    return response["answer"]


# ── Streaming Query ────────────────────────────────────────────
# Yields answer tokens as they are generated so the UI can show the
# response progressively instead of waiting for the full answer.
def query_rag_stream(question: str):
    chain = get_rag_chain()
    for chunk in chain.stream({"input": question}):
        # create_retrieval_chain streams partial dicts as generation proceeds.
        # Early chunks may only contain "context" or "input" keys — skip those,
        # and yield only the "answer" pieces as they arrive.
        if "answer" in chunk:
            yield chunk["answer"]