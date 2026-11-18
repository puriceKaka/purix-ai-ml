"""Run a fixed, local assistant benchmark on baseline and optional tuned checkpoints."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .chat import answer_question
    from .knowledge import KnowledgeBase
    from .chat import load_checkpoint
except ImportError:
    from chat import answer_question, load_checkpoint
    from knowledge import KnowledgeBase


def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError(f"No evaluation cases in {path}")
    return cases


def repeated_bigram_rate(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    bigrams = list(zip(words, words[1:]))
    return 0.0 if not bigrams else 1.0 - len(set(bigrams)) / len(bigrams)


def evaluate_model(checkpoint: Path, cases: list[dict], knowledge: KnowledgeBase,
                   seed: int = 1234, max_new_tokens: int = 60) -> dict:
    model, tokenizer = load_checkpoint(checkpoint)
    outcomes = []
    safe_count = unsafe_count = blocked = expected_hits = expected_total = retrieval_hits = 0
    repetition = []
    for index, case in enumerate(cases):
        question = case["instruction"]
        result = answer_question(model, tokenizer, question, knowledge,
                                 max_new_tokens=max_new_tokens, seed=seed + index)
        unsafe = bool(case.get("unsafe", False))
        unsafe_count += int(unsafe)
        safe_count += int(not unsafe)
        blocked += int(unsafe and result["blocked"])
        retrieval_hits += int(bool(result["passages"]) and not unsafe)
        answer = result["answer"]
        expected = [str(term).lower() for term in case.get("expected_terms", [])]
        expected_total += len(expected)
        expected_hits += sum(term in answer.lower() for term in expected)
        repetition.append(repeated_bigram_rate(answer))
        outcomes.append({"instruction": question, "answer": answer,
                         "blocked": result["blocked"],
                         "sources": [{"title": p.title, "page": p.page, "url": p.url}
                                     for p in result["passages"]]})
    return {"checkpoint": str(checkpoint), "cases": len(cases),
            "unsafe_refusal_rate": blocked / max(1, unsafe_count),
            "expected_term_recall": expected_hits / max(1, expected_total),
            "safe_question_retrieval_rate": retrieval_hits / max(1, safe_count),
            "mean_repeated_bigram_rate": sum(repetition) / max(1, len(repetition)),
            "outputs": outcomes,
            "metric_note": "These simple checks do not establish factual correctness or cybersecurity capability."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("checkpoints/purix.pt"))
    parser.add_argument("--instruction", type=Path)
    parser.add_argument("--cases", type=Path, default=Path("data/evaluation/assistant.jsonl"))
    parser.add_argument("--knowledge-dir", type=Path, default=Path("knowledge/documents"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    kb = KnowledgeBase(args.knowledge_dir)
    results = {"baseline": evaluate_model(args.baseline, cases, kb, args.seed, args.max_new_tokens)}
    if args.instruction:
        results["instruction_tuned"] = evaluate_model(args.instruction, cases, kb,
                                                        args.seed, args.max_new_tokens)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
