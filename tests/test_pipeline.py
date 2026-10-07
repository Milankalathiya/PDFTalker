"""Offline tests: no API key or network needed."""

from pathlib import Path

from langchain_core.documents import Document
from langgraph.graph import END

from agent import (
    MAX_ATTEMPTS, MAX_REWRITES, mentions_documents, route_after_analyze, route_after_check,
    route_after_generate, route_after_retrieve,
)
from rag_pipeline import RetrievedChunk, read_pdf, reciprocal_rank_fusion, split_pages, tokenize

SAMPLE_PDF = Path(__file__).parent.parent / "eval" / "sample_policy.pdf"


def doc(chunk_id):
    return Document(page_content=f"chunk {chunk_id}", metadata={"chunk_id": chunk_id})


def test_rrf_rewards_agreement_between_retrievers():
    semantic = [doc(1), doc(2), doc(3)]
    keyword = [doc(3), doc(4), doc(1)]
    fused = [d.metadata["chunk_id"] for d in reciprocal_rank_fusion([semantic, keyword])]
    assert fused[:2] == [1, 3]          # found by both retrievers
    assert sorted(fused) == [1, 2, 3, 4]  # no duplicates, nothing lost


def test_bm25_tokens_ignore_case_and_punctuation():
    # The default whitespace split made "cooling-off period?" miss "Cooling-off period."
    assert tokenize("How long is the Cooling-off period?") == ["how", "long", "is", "the", "cooling-off", "period"]
    assert tokenize("Exclusion HS-EX-04: vermin.") == ["exclusion", "hs-ex-04", "vermin"]


def test_pdf_pages_keep_source_and_page_numbers():
    pages = read_pdf("policy.pdf", SAMPLE_PDF.read_bytes())
    assert [p.metadata["page"] for p in pages] == list(range(1, len(pages) + 1))
    chunks = split_pages(pages)
    assert all(c.metadata["source"] == "policy.pdf" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert any("HS-EX-04" in c.page_content for c in chunks)


def test_conversation_skips_retrieval():
    assert route_after_analyze({"route": "conversation"}) == "converse"
    assert route_after_analyze({"route": "documents"}) == "retrieve"


def test_questions_about_the_file_are_never_small_talk():
    sources = ["Milan_Kalathiya_AI.pdf"]
    assert mentions_documents("can you tell me about file? pros and cons? how to correct it?", sources)
    assert mentions_documents("what do you think of Milan's CV?", sources)
    assert not mentions_documents("hi", sources)
    assert not mentions_documents("thanks, what can you do?", sources)


def test_overview_reads_whole_document_and_skips_grounding_check():
    assert route_after_analyze({"route": "overview"}) == "overview"
    assert route_after_generate({"mode": "agentic", "route": "overview"}) == END
    assert route_after_generate({"mode": "agentic", "route": "documents"}) == "check_grounding"


def chunk(score):
    return RetrievedChunk(doc(0), score, "relevance")


def test_agentic_broadens_search_when_evidence_is_weak_then_answers_or_gives_up():
    assert route_after_retrieve({"mode": "agentic", "chunks": [], "rewrites": 0}) == "rewrite"
    assert route_after_retrieve({"mode": "agentic", "chunks": [chunk(0.12)], "rewrites": 0}) == "rewrite"
    assert route_after_retrieve({"mode": "agentic", "chunks": [chunk(0.8)], "rewrites": 0}) == "generate"
    assert route_after_retrieve({"mode": "agentic", "chunks": [chunk(0.12)], "rewrites": MAX_REWRITES}) == "generate"
    assert route_after_retrieve({"mode": "agentic", "chunks": [], "rewrites": MAX_REWRITES}) == "not_found"


def test_simple_modes_always_answer_directly():
    for mode in ("basic", "hybrid"):
        assert route_after_retrieve({"mode": mode, "chunks": []}) == "generate"
        assert route_after_generate({"mode": mode}) == END
    assert route_after_generate({"mode": "agentic"}) == "check_grounding"


def test_ungrounded_answer_is_regenerated_once():
    assert route_after_check({"grounded": False, "attempts": 1}) == "generate"
    assert route_after_check({"grounded": False, "attempts": MAX_ATTEMPTS}) == END
    assert route_after_check({"grounded": True, "attempts": 1}) == END
