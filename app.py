import hashlib
import os
import time
from pathlib import Path

# load .env file before importing langchain
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="PDF Talker", page_icon="📄", layout="wide")

# Streamlit Cloud exposes top-level secrets as env vars; copy explicitly in case they are nested.
if not os.getenv("COHERE_API_KEY"):
    try:
        os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
    except Exception:
        try:
            found = sorted(st.secrets.keys())
        except Exception:
            found = []
        st.error("COHERE_API_KEY is not set. Add it to `.env` locally or to the app's Secrets on Streamlit Cloud "
                 "(TOML format, e.g. `COHERE_API_KEY = \"...\"` with quotes), then reboot the app.")
        st.caption(f"Secret names the app can see: {found or 'none'}")
        st.stop()

from agent import MODES, build_graph, initial_state
from rag_pipeline import CHAT_MODEL, EMBED_MODEL, MAX_PAGES, RERANK_MODEL, build_knowledge_base, load_embeddings, load_llm, load_reranker


SAMPLE_PDF = Path(__file__).parent / "eval" / "sample_policy.pdf"
MODE_LABELS = {
    "agentic": "🧠 Agentic RAG (LangGraph)",
    "hybrid": "🔀 Hybrid search + rerank",
    "basic": "📐 Basic vector search",
}
MODE_HELP = """
**Agentic**: routes the message, rewrites follow-ups into standalone queries, grades retrieved chunks,
re-searches when nothing is relevant, and checks the answer is grounded before showing it.
**Hybrid**: FAISS + BM25 keyword search, fused and reranked by Cohere Rerank.
**Basic**: FAISS top-3 similarity search, the original pipeline, kept for comparison.
"""


@st.cache_resource
def get_models():
    return load_embeddings(), load_llm(), load_reranker()


def index_files(files: list[tuple[str, bytes]]):
    key = hashlib.sha256(b"".join(name.encode() + data for name, data in files)).hexdigest()
    if st.session_state.get("kb_key") == key:
        st.toast("These documents are already indexed.")
        return
    embeddings, llm, reranker = get_models()
    with st.spinner(f"Reading and embedding {len(files)} file(s)..."):
        start = time.time()
        try:
            kb, warnings = build_knowledge_base(files, embeddings)
        except ValueError as exc:
            st.error(str(exc))
            return
    st.session_state.update(kb=kb, kb_key=key, graph=build_graph(kb, llm, reranker), messages=[],
                            index_seconds=time.time() - start)
    for warning in warnings:
        st.warning(warning)


def render_details(msg: dict):
    """Sources, agent trace and metrics under an assistant answer."""
    if msg.get("grounded") is False:
        st.warning("The grounding check could not verify every claim in this answer against the sources.")

    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for i, src in enumerate(sources, start=1):
                score = {"relevance": f"relevance {src['score']:.2f}",
                         "distance": f"distance {src['score']:.3f}"}.get(src["kind"], "whole document")
                st.markdown(f"**[{i}] {src['label']}** · {score}")
                st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))

    with st.expander("🔍 Under the hood"):
        m = msg["metrics"]
        cols = st.columns(4)
        cols[0].metric("Total latency", f"{m['total']:.2f}s")
        cols[1].metric("First token", f"{m['first_token']:.2f}s" if m["first_token"] else "N/A")
        cols[2].metric("Sources used", len(sources))
        cols[3].metric("Tokens generated", m["tokens"] or "N/A")
        st.markdown(f"**Mode:** {MODE_LABELS[msg['mode']]}")
        for step, line in enumerate(msg.get("trace", []), start=1):
            st.markdown(f"{step}. {line}")


def answer(question: str, mode: str):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=mode == "agentic")
        placeholder = st.empty()
        final, streamed, trace = {}, "", []
        start, first_token = time.time(), None
        try:
            for kind, payload in st.session_state.graph.stream(
                initial_state(question, history, mode), stream_mode=["messages", "updates"]
            ):
                if kind == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") in ("generate", "converse") and isinstance(chunk.content, str):
                        first_token = first_token or time.time() - start
                        streamed += chunk.content
                        placeholder.markdown(streamed + "▌")
                    continue
                for node, update in payload.items():
                    update = update or {}
                    for line in update.get("trace", []):
                        trace.append(line)
                        status.write(line)
                    final.update({k: v for k, v in update.items() if k != "trace"})
                    if node == "check_grounding" and not update.get("grounded"):
                        streamed = ""  # a stricter regeneration may follow; replace the draft
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong while answering: {exc}")
            st.session_state.messages.pop()
            return

        status.update(label=f"Done in {time.time() - start:.1f}s · {len(trace)} steps", state="complete", expanded=False)
        placeholder.markdown(final.get("answer", ""))
        msg = {
            "role": "assistant",
            "content": final.get("answer", ""),
            "mode": mode,
            "grounded": final.get("grounded"),
            "trace": trace,
            "sources": [
                {"label": c.label, "score": c.score, "kind": c.score_kind, "text": c.doc.page_content}
                for c in final.get("chunks", [])
            ],
            "metrics": {"total": time.time() - start, "first_token": first_token,
                        "tokens": final.get("output_tokens", 0)},
        }
        render_details(msg)
    st.session_state.messages.append(msg)


# ----- Sidebar -----
with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", type="primary", width="stretch", disabled=not uploads):
        index_files([(f.name, f.getvalue()) for f in uploads])
    if st.button("Try the sample insurance policy", width="stretch"):
        index_files([(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())])

    if "kb" in st.session_state:
        kb = st.session_state.kb
        pages = len({(c.metadata["source"], c.metadata["page"]) for c in kb.chunks})
        st.success(f"{len(kb.sources)} file(s) · {pages} pages · {len(kb.chunks)} chunks "
                   f"· indexed in {st.session_state.index_seconds:.1f}s")
        for source in kb.sources:
            st.caption(f"• {source}")

    st.divider()
    mode = st.radio("Pipeline", MODES, format_func=MODE_LABELS.get, help=MODE_HELP)
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
    st.caption(f"LLM `{CHAT_MODEL}` · embeddings `{EMBED_MODEL}` · rerank `{RERANK_MODEL}` "
               f"· up to {MAX_PAGES} pages per session")


# ----- Main -----
st.title("📄 PDF Talker")
st.caption("Chat with your PDFs: agentic RAG with hybrid search, reranking, citations and grounding checks.")

if "kb" not in st.session_state:
    st.info("Upload one or more PDFs in the sidebar and click **Process PDFs**, "
            "or click **Try the sample insurance policy** to explore with a ready-made document.")
    st.stop()

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg)

if question := st.chat_input("Ask a question about your documents..."):
    answer(question, mode)
