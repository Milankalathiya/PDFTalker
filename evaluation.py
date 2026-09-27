"""Evaluate the three pipeline modes on a labelled question set.

Metrics per mode
  hit@3         a retrieved chunk comes from an expected page      (retrieval, no LLM)
  MRR           1 / rank of the first chunk from an expected page  (retrieval, no LLM)
  correctness   LLM judge: answer matches the ground truth; for out-of-scope
                questions, the answer must decline instead of guessing
  faithfulness  LLM judge: every claim is supported by the retrieved chunks
  latency       seconds per question, end to end

Run:  python evaluation.py                     (sample policy, all modes)
      python evaluation.py --pdf my.pdf --set my_set.json --modes agentic basic
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from agent import MODES, build_graph, format_context, initial_state
from rag_pipeline import build_knowledge_base, load_embeddings, load_llm, load_reranker


class Verdict(BaseModel):
    passed: bool = Field(description="True if the answer meets the criterion.")
    reason: str = Field(description="One short sentence explaining the verdict.")


CORRECTNESS_PROMPT = """You grade a question-answering system. Compare the ANSWER with the GROUND TRUTH.
Pass if the answer gives the key fact that answers the question and does not contradict the
ground truth. Wording may differ, and missing secondary detail is fine; a wrong or opposite
conclusion fails.
If the ground truth says the information is not in the document, pass only if the answer says
it could not find it or that the document does not say, and fail if it invents an answer."""

FAITHFULNESS_PROMPT = """You check an answer against the SOURCES it was generated from.
Pass if every factual claim in the answer is supported by the sources, or if the answer
says the information could not be found. Fail if it contains any unsupported claim."""


rate_limit_wait = 0.0  # seconds slept on rate limits, excluded from reported latency


def with_backoff(fn, *args, attempts: int = 6):
    """Retry on rate limits; trial Cohere keys allow only a few calls per minute."""
    global rate_limit_wait
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:
            limited = "429" in str(exc) or "TooManyRequests" in type(exc).__name__
            if not limited or attempt == attempts - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s")
            time.sleep(wait)
            rate_limit_wait += wait


def retrieval_scores(chunks, expected_pages) -> tuple[float, float]:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.doc.metadata["page"] in expected_pages:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def evaluate(pdf: Path, eval_set: list[dict], modes: list[str], results_file: Path) -> dict:
    """Saves after every question and skips questions already in results_file, so a run
    interrupted by rate limits or timeouts resumes where it stopped."""
    print(f"Indexing {pdf.name} ...")
    kb, _ = build_knowledge_base([(pdf.name, pdf.read_bytes())], load_embeddings())
    graph = build_graph(kb, load_llm(), load_reranker())
    judge = load_llm(temperature=0.0).with_structured_output(Verdict)

    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}
    for mode in modes:
        print(f"\n== {mode} ==")
        rows = results.setdefault(mode, [])
        done = {r["question"] for r in rows}
        for item in eval_set:
            if item["question"] in done:
                continue
            start, waited_before = time.time(), rate_limit_wait
            state = with_backoff(graph.invoke, initial_state(item["question"], [], mode))
            latency = time.time() - start - (rate_limit_wait - waited_before)
            chunks = state.get("chunks", [])

            correct = with_backoff(judge.invoke, [
                SystemMessage(CORRECTNESS_PROMPT),
                HumanMessage(f"QUESTION: {item['question']}\nGROUND TRUTH: {item['ground_truth']}\n"
                             f"ANSWER: {state['answer']}"),
            ])
            faithful = with_backoff(judge.invoke, [
                SystemMessage(FAITHFULNESS_PROMPT),
                HumanMessage(f"SOURCES:\n{format_context(chunks) or '(none)'}\n\nANSWER: {state['answer']}"),
            ])

            row = {
                "question": item["question"],
                "answer": state["answer"],
                "pages_retrieved": [c.doc.metadata["page"] for c in chunks],
                "correct": bool(correct and correct.passed),
                "faithful": bool(faithful and faithful.passed),
                "judge_reason": correct.reason if correct else "",
                "latency": round(latency, 2),
            }
            if item["pages"]:
                row["hit"], row["rr"] = retrieval_scores(chunks, item["pages"])
            rows.append(row)
            results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  {'✓' if row['correct'] else '✗'} {item['question'][:70]}")
    return {mode: results[mode] for mode in modes}


def summarize(results: dict) -> str:
    lines = [
        "| Mode | hit@3 | MRR | Correctness | Faithfulness | Avg latency |",
        "|---|---|---|---|---|---|",
    ]
    for mode, rows in results.items():
        answerable = [r for r in rows if "hit" in r]
        mean = lambda values: sum(values) / len(values) if values else 0.0
        lines.append(
            f"| {mode} | {mean([r['hit'] for r in answerable]):.2f} | {mean([r['rr'] for r in answerable]):.2f} "
            f"| {mean([r['correct'] for r in rows]):.0%} | {mean([r['faithful'] for r in rows]):.0%} "
            f"| {mean([r['latency'] for r in rows]):.1f}s |"
        )
    return "\n".join(lines)


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=here / "eval" / "sample_policy.pdf")
    parser.add_argument("--set", type=Path, default=here / "eval" / "eval_set.json")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--limit", type=int, help="only the first N questions (quick smoke test)")
    parser.add_argument("--out", type=Path, default=here / "eval" / "results")
    parser.add_argument("--fresh", action="store_true", help="discard saved results instead of resuming")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

    eval_set = json.loads(args.set.read_text(encoding="utf-8"))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    results_file = args.out / "results.json"
    if args.fresh:
        results_file.unlink(missing_ok=True)
    results = evaluate(args.pdf, eval_set, args.modes, results_file)

    table = summarize(results)
    print("\n" + table)
    (args.out / "summary.md").write_text(
        f"Evaluated on `{args.pdf.name}` with {len(eval_set)} questions.\n\n{table}\n", encoding="utf-8"
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
