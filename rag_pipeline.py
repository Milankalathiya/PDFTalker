"""Core RAG building blocks: PDF loading, chunking, hybrid retrieval and reranking.

Deliberately free of Streamlit so the app, the agent graph and evaluation.py
all share one code path.
"""

import io
import os
import re
from dataclasses import dataclass

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRerank


# Models are hosted by Cohere, so the app needs no torch / local model weights
# and stays well inside Streamlit Community Cloud's memory limit.
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus-08-2024")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PAGES = 300      # total pages per session, keeps indexing fast on free hosting
FETCH_K = 10         # candidates pulled from each retriever before reranking
TOP_K = 3            # chunks passed to the LLM


def load_embeddings():
    return CohereEmbeddings(model=EMBED_MODEL)


def load_llm(temperature: float = 0.2):
    return ChatCohere(model=CHAT_MODEL, temperature=temperature)


def load_reranker():
    return CohereRerank(model=RERANK_MODEL)


@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    score_kind: str  # "relevance" (rerank, higher is better) or "distance" (FAISS L2, lower is better)

    @property
    def label(self) -> str:
        return f"{self.doc.metadata['source']}, p. {self.doc.metadata['page']}"


def read_pdf(name: str, data: bytes) -> list[Document]:
    """One Document per page, keeping file name and 1-based page number for citations."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": name, "page": number}))
    return pages


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase words for BM25; keeps hyphenated terms and codes like 'cooling-off' or 'hs-ex-04' whole."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge several ranked lists; a chunk ranked high by any retriever rises to the top."""
    scores: dict[int, float] = {}
    by_id: dict[int, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            chunk_id = doc.metadata["chunk_id"]
            by_id[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_id[chunk_id] for chunk_id in ordered]


class KnowledgeBase:
    """FAISS (semantic) + BM25 (keyword) indexes over the same chunks."""

    def __init__(self, chunks: list[Document], embeddings):
        self.chunks = chunks
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.bm25 = BM25Retriever.from_documents(chunks, k=FETCH_K, preprocess_func=tokenize)

    @property
    def sources(self) -> list[str]:
        return sorted({c.metadata["source"] for c in self.chunks})

    def vector_search(self, query: str, k: int = TOP_K) -> list[RetrievedChunk]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [RetrievedChunk(doc, float(score), "distance") for doc, score in results]

    def hybrid_search(self, query: str, k: int = FETCH_K) -> list[Document]:
        """Up to 2k fused candidates; all go to the reranker so neither retriever's finds are cut early."""
        semantic = [c.doc for c in self.vector_search(query, k=k)]
        keyword = self.bm25.invoke(query)
        return reciprocal_rank_fusion([semantic, keyword])


def rerank(reranker, query: str, docs: list[Document], top_n: int = TOP_K) -> list[RetrievedChunk]:
    if not docs:
        return []
    results = reranker.rerank(documents=[d.page_content for d in docs], query=query, top_n=top_n)
    return [RetrievedChunk(docs[r["index"]], float(r["relevance_score"]), "relevance") for r in results]


def build_knowledge_base(files: list[tuple[str, bytes]], embeddings) -> tuple[KnowledgeBase, list[str]]:
    """Index (name, bytes) pairs. Returns the knowledge base and warnings for skipped files."""
    pages, warnings = [], []
    for name, data in files:
        try:
            file_pages = read_pdf(name, data)
        except Exception as exc:  # corrupt or encrypted PDF
            warnings.append(f"{name}: could not be read ({exc.__class__.__name__}).")
            continue
        if not file_pages:
            warnings.append(f"{name}: no extractable text (scanned images are not supported).")
            continue
        if len(pages) + len(file_pages) > MAX_PAGES:
            warnings.append(f"{name}: skipped, the {MAX_PAGES}-page limit per session was reached.")
            continue
        pages.extend(file_pages)

    if not pages:
        raise ValueError("None of the uploaded files contained readable text.\n" + "\n".join(warnings))
    return KnowledgeBase(split_pages(pages), embeddings), warnings
