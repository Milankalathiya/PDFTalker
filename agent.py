"""LangGraph pipeline. One graph serves all three modes so they can be compared fairly.

basic    FAISS top-3 -> answer                                   (the original PDFTalker)
hybrid   FAISS + BM25 -> Cohere Rerank -> answer
agentic  route -> rewrite follow-up into standalone queries (split multi-part questions)
         -> hybrid retrieve + rerank per query -> merge -> grade by rerank score
         -> (broaden the search once if evidence is weak or missing)
         -> answer with citations -> grounding check -> (regenerate if unsupported)
"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from rag_pipeline import TOP_K, KnowledgeBase, RetrievedChunk, rerank


MODES = ("agentic", "hybrid", "basic")
MIN_RELEVANCE = 0.1   # rerank score below which a chunk is treated as off-topic
CONFIDENT = 0.3       # best rerank score below this counts as weak evidence and triggers a broader search
MAX_REWRITES = 1      # extra retrieval rounds with reworded queries
MAX_QUERIES = 3       # sub-queries per question in agentic mode
MAX_SOURCES = 5       # merged chunks passed to the LLM in agentic mode
MAX_ATTEMPTS = 2      # answer generations (1 + one regeneration after a failed grounding check)
OVERVIEW_CHUNKS = 12  # whole-document questions read up to this many chunks, in document order
PREVIEW_CHARS = 400   # start of each document shown to the router so it knows what it is searching
HISTORY_TURNS = 6     # past messages sent to the LLM
NOT_FOUND = "I couldn't find this in the uploaded documents."


ANALYZE_SYSTEM = """You route messages for an assistant that answers questions about PDFs the user uploaded.
- route "conversation" only for greetings, thanks, or questions about the assistant itself.
- route "overview" when the message is about the document as a whole rather than a specific fact:
  summarize it, review or critique it, list its strengths and weaknesses, suggest improvements,
  or explain what it is about. "The file" or "this document" means the uploaded document.
- route "documents" for everything else; the user expects answers from their documents.
Rewrite the latest user message into standalone search queries, resolving pronouns and
references from the conversation so far. Use the vocabulary of the documents described below,
not generic words like "file". Keep key terms, numbers and codes exactly as written.
Usually one query is enough. If answering depends on several facts, such as whether a rule
applies in a described situation, add up to two more queries for the conditions, definitions
or exceptions that could decide it.

Uploaded documents:
{documents}"""

ANSWER_SYSTEM = """You answer questions using only the numbered sources below, which are excerpts from the user's PDFs.
- Cite the sources you use with bracketed numbers right after each claim, e.g. [1] or [2][3].
- If the sources do not contain the answer, say you couldn't find it in the documents. Never use outside knowledge.
- Be concise and direct.

Sources:
{context}"""

OVERVIEW_SYSTEM = """You help the user with a document they uploaded. Its content is given below as numbered excerpts.
Answer the user's request about the document as a whole: summarize, review, critique, point out strengths
and weaknesses, or suggest improvements, as asked.
- Base every observation on what the document actually says, and cite the excerpt it comes from, e.g. [1].
- You may use your own expertise to judge the document and recommend changes; make clear which parts
  are your assessment.
- Be specific and practical.

Document:
{context}"""

STRICT_NOTE = """

A reviewer found these claims unsupported by the sources: {unsupported}
Answer again using only what the sources state."""

REWRITE_SYSTEM = """A search of the user's documents found little or no relevant evidence for the question below.
Write up to two alternative search queries using different wording, synonyms or formal document
terminology. Think about which conditions, definitions or exceptions in such a document could
decide the answer, and search for those."""

GROUNDING_SYSTEM = """You check whether an answer is fully supported by its sources.
An answer that says the information could not be found counts as grounded."""

CONVERSATION_SYSTEM = """You are PDF Talker, an assistant that answers questions about PDFs the user uploads.
The user has already loaded these documents: {sources}.
Reply briefly and warmly, and invite the user to ask about them."""


class QueryAnalysis(BaseModel):
    route: Literal["documents", "overview", "conversation"] = Field(description="Where the answer should come from.")
    search_queries: list[str] = Field(description="1 to 3 standalone search queries for the document index.")


class AlternativeQueries(BaseModel):
    queries: list[str] = Field(description="1 or 2 alternative search queries.")


class GroundingCheck(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported by the sources.")
    unsupported_claims: str = Field(default="", description="Claims not supported by the sources, if any.")


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]          # [{"role": "user" | "assistant", "content": str}]
    mode: str
    route: str
    search_queries: list[str]
    chunks: list[RetrievedChunk]
    rewrites: int
    attempts: int
    answer: str
    grounded: bool | None
    unsupported: str
    output_tokens: int
    trace: Annotated[list[str], operator.add]


def initial_state(question: str, history: list[dict], mode: str) -> RAGState:
    return {
        "question": question,
        "history": history[-HISTORY_TURNS:],
        "mode": mode,
        "rewrites": 0,
        "attempts": 0,
        "grounded": None,
        "output_tokens": 0,
        "trace": [],
    }


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.label})\n{c.doc.page_content}" for i, c in enumerate(chunks, start=1))


def output_tokens(response) -> int:
    # Cohere reports usage in usage_metadata when invoked, but only in token_count when streamed.
    usage = response.usage_metadata or response.response_metadata.get("token_count") or {}
    return int(usage.get("output_tokens", 0))


def describe_documents(kb: KnowledgeBase) -> str:
    """File name plus the opening text of each document, so the router knows what it is searching."""
    first_chunk = {}
    for chunk in kb.chunks:
        first_chunk.setdefault(chunk.metadata["source"], chunk.page_content)
    return "\n".join(f"- {name}: {text[:PREVIEW_CHARS].replace(chr(10), ' ')}..." for name, text in first_chunk.items())


def history_messages(state: RAGState) -> list:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in state.get("history", [])
    ]


# ----- routing (pure functions, unit-tested) -----

def route_after_analyze(state: RAGState) -> str:
    return {"conversation": "converse", "overview": "overview"}.get(state["route"], "retrieve")


def route_after_retrieve(state: RAGState) -> str:
    if state["mode"] != "agentic":
        return "generate"
    chunks = state["chunks"]
    weak = not chunks or chunks[0].score < CONFIDENT
    if weak and state.get("rewrites", 0) < MAX_REWRITES:
        return "rewrite"
    return "generate" if chunks else "not_found"


def route_after_generate(state: RAGState) -> str:
    # an overview answer contains the model's own assessment, which a grounding check would reject
    return "check_grounding" if state["mode"] == "agentic" and state.get("route") != "overview" else END


def route_after_check(state: RAGState) -> str:
    if not state.get("grounded") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "generate"
    return END


def build_graph(kb: KnowledgeBase, llm, reranker):
    analyzer = llm.with_structured_output(QueryAnalysis)
    checker = llm.with_structured_output(GroundingCheck)
    rewriter = llm.with_structured_output(AlternativeQueries)
    analyze_system = ANALYZE_SYSTEM.format(documents=describe_documents(kb))

    def analyze(state: RAGState) -> RAGState:
        if state["mode"] != "agentic":
            return {"route": "documents", "search_queries": [state["question"]],
                    "trace": ["Search query: the question as typed (no routing or rewriting)"]}
        result = analyzer.invoke(
            [SystemMessage(analyze_system), *history_messages(state), HumanMessage(state["question"])]
        )
        queries = [q for q in (result.search_queries if result else []) if q.strip()][:MAX_QUERIES]
        route = result.route if result else "documents"
        queries = queries or [state["question"]]  # model skipped the tool call: plain document search
        shown = "; ".join(f"“{q}”" for q in queries)
        return {"route": route, "search_queries": queries, "trace": [f"Router → {route}; search: {shown}"]}

    def retrieve(state: RAGState) -> RAGState:
        queries = state["search_queries"]
        if state["mode"] == "basic":
            chunks = kb.vector_search(queries[0], k=TOP_K)
            return {"chunks": chunks, "trace": [f"FAISS semantic search → top {len(chunks)} chunks"]}

        if state["mode"] == "hybrid":
            candidates = kb.hybrid_search(queries[0])
            chunks = rerank(reranker, queries[0], candidates, top_n=TOP_K)
            best = chunks[0].score if chunks else 0.0
            return {"chunks": chunks, "trace": [
                f"FAISS + BM25 → {len(candidates)} candidates → Cohere Rerank top {len(chunks)} (best {best:.2f})"]}

        # agentic: rerank each sub-query separately, keep each chunk's best score, then grade
        # chunks from an earlier round are kept, so broadening the search never loses evidence
        best_by_id = {c.doc.metadata["chunk_id"]: c for c in state.get("chunks", [])}
        for query in queries:
            for chunk in rerank(reranker, query, kb.hybrid_search(query), top_n=TOP_K):
                chunk_id = chunk.doc.metadata["chunk_id"]
                if chunk_id not in best_by_id or chunk.score > best_by_id[chunk_id].score:
                    best_by_id[chunk_id] = chunk
        merged = sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)
        chunks = [c for c in merged if c.score >= MIN_RELEVANCE][:MAX_SOURCES]
        return {"chunks": chunks, "trace": [
            f"FAISS + BM25 + Cohere Rerank for {len(queries)} quer{'y' if len(queries) == 1 else 'ies'} "
            f"→ {len(merged)} distinct chunks; {len(chunks)} pass the relevance grade (≥ {MIN_RELEVANCE})"]}

    def rewrite(state: RAGState) -> RAGState:
        result = rewriter.invoke([
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(f"Question: {state['question']}\n"
                         f"Queries already tried: {'; '.join(state['search_queries'])}"),
        ])
        queries = [q for q in (result.queries if result else []) if q.strip()][:2] or [state["question"]]
        shown = "; ".join(f"“{q}”" for q in queries)
        return {"search_queries": queries, "rewrites": state.get("rewrites", 0) + 1,
                "trace": [f"Evidence weak; broadening search with {shown}"]}

    def overview(state: RAGState) -> RAGState:
        chunks = [RetrievedChunk(c, 1.0, "document") for c in kb.chunks[:OVERVIEW_CHUNKS]]
        note = "" if len(kb.chunks) <= OVERVIEW_CHUNKS else f" (first {OVERVIEW_CHUNKS} of {len(kb.chunks)} chunks)"
        return {"chunks": chunks, "trace": [f"Whole-document question -> reading {len(chunks)} chunks in order{note}"]}

    def not_found(state: RAGState) -> RAGState:
        return {"answer": NOT_FOUND, "grounded": True,
                "trace": ["Still nothing relevant → answered 'not found' without calling the LLM"]}

    def generate(state: RAGState) -> RAGState:
        template = OVERVIEW_SYSTEM if state.get("route") == "overview" else ANSWER_SYSTEM
        system = template.format(context=format_context(state["chunks"]))
        if state.get("unsupported"):
            system += STRICT_NOTE.format(unsupported=state["unsupported"])
        response = llm.invoke([SystemMessage(system), *history_messages(state), HumanMessage(state["question"])])
        attempts = state.get("attempts", 0) + 1
        return {"answer": response.content, "attempts": attempts, "output_tokens": output_tokens(response),
                "trace": [f"Answer generated from {len(state['chunks'])} sources (attempt {attempts})"]}

    def check_grounding(state: RAGState) -> RAGState:
        result = checker.invoke([
            SystemMessage(GROUNDING_SYSTEM),
            HumanMessage(f"Sources:\n{format_context(state['chunks'])}\n\nAnswer:\n{state['answer']}"),
        ])
        grounded = True if result is None else result.grounded
        unsupported = "" if result is None else result.unsupported_claims
        verdict = "supported by the sources ✓" if grounded else f"unsupported claims: {unsupported}"
        return {"grounded": grounded, "unsupported": unsupported, "trace": [f"Grounding check → {verdict}"]}

    def converse(state: RAGState) -> RAGState:
        system = CONVERSATION_SYSTEM.format(sources=", ".join(kb.sources))
        response = llm.invoke([SystemMessage(system), *history_messages(state), HumanMessage(state["question"])])
        return {"answer": response.content, "chunks": [], "output_tokens": output_tokens(response),
                "trace": ["Answered conversationally (no retrieval)"]}

    graph = StateGraph(RAGState)
    for name, node in [("analyze", analyze), ("retrieve", retrieve), ("rewrite", rewrite),
                       ("overview", overview), ("not_found", not_found), ("generate", generate),
                       ("check_grounding", check_grounding), ("converse", converse)]:
        graph.add_node(name, node)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, ["retrieve", "overview", "converse"])
    graph.add_edge("overview", "generate")
    graph.add_conditional_edges("retrieve", route_after_retrieve, ["generate", "rewrite", "not_found"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("generate", route_after_generate, ["check_grounding", END])
    graph.add_conditional_edges("check_grounding", route_after_check, ["generate", END])
    graph.add_edge("not_found", END)
    graph.add_edge("converse", END)
    return graph.compile()
