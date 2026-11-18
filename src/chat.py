"""Generate text from a trained PURIX checkpoint; never invokes system tools."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

try:
    from .knowledge import KnowledgeBase, Passage
    from .model import LanguageModel, ModelConfig
    from .tokenizer import BPETokenizer, Tokenizer, load_tokenizer
except ImportError:
    from knowledge import KnowledgeBase, Passage
    from model import LanguageModel, ModelConfig
    from tokenizer import BPETokenizer, Tokenizer, load_tokenizer


UNSAFE_PATTERNS = (
    re.compile(r"\b(?:write|create|build|deploy)\b.{0,50}\b(?:ransomware|malware|keylogger|credential stealer)\b", re.I),
    re.compile(r"\b(?:steal|exfiltrate|dump)\b.{0,40}\b(?:passwords?|credentials?|tokens?|secrets?)\b", re.I),
    re.compile(r"\b(?:break into|hack|access)\b.{0,50}\b(?:neighbor|someone else's|public|third.party)\b", re.I),
    re.compile(r"\b(?:evade|bypass|disable)\b.{0,40}\b(?:edr|antivirus|detection|security controls?)\b", re.I),
)

# High-frequency grammatical words are expected in normal English, but a long
# completion made almost entirely from them is often a fluent-looking token
# salad. This deliberately conservative guard only applies to long answers.
COMMON_FUNCTION_WORDS = frozenset("""
    a an the and or but if as at by for from in into of on onto to with without
    is am are was were be been being do does did have has had can could will
    would shall should may might must it its he she they we you i me him her
    them us my your our their this that these those what which who whom when
    where why how not no so than then there here
""".split())

DIAGNOSTIC_PROMPTS = (
    "Hello",
    "What is cybersecurity?",
    "Explain multi-factor authentication.",
    "In Python, what does len([3, 5, 7]) return?",
)
GREETING_RE = re.compile(
    r"^\s*(?:(?:hi|hello|hey|greetings)(?:\s+(?:there|purix))?|"
    r"how are (?:you|u)(?: doing)?|how r u|are you (?:good|okay|well))[!.?\s]*$", re.I)


def is_greeting(text: str) -> bool:
    """Recognize a short greeting for a fixed response that bypasses the LM and RAG."""
    return bool(GREETING_RE.fullmatch(text))


def is_unsafe_request(text: str) -> bool:
    """Block a small set of clearly harmful requests before model generation."""
    return any(pattern.search(text) for pattern in UNSAFE_PATTERNS)


def generation_is_usable(text: str, question: str) -> bool:
    """Apply conservative, explainable rejection checks to a completion.

    This is a presentation guard, not a model-quality detector.  It rejects
    empty, fragmentary, repetitive, or plainly off-topic completions; passing
    does not establish correctness.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3 or not text.strip():
        return False
    # The word tokenizer renders punctuation as separated tokens. Reject
    # structurally broken brackets rather than mistaking keyword overlap for
    # a well-formed response.
    delimiter_pairs = {')': '(', ']': '[', '}': '{'}
    delimiters: list[str] = []
    for char in text:
        if char in "([{":
            delimiters.append(char)
        elif char in ")]}":
            if not delimiters or delimiters.pop() != delimiter_pairs[char]:
                return False
    if delimiters:
        return False
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
    if punctuation / max(1, len(text)) > 0.25:
        return False
    if len(set(words)) / len(words) < 0.45:
        return False
    if len(words) >= 20:
        function_word_ratio = sum(word in COMMON_FUNCTION_WORDS for word in words) / len(words)
        if function_word_ratio >= 0.72:
            return False
    query_terms = {term for term in re.findall(r"[a-z]+", question.lower())
                   if term not in {"a", "an", "the", "is", "what", "does", "in", "explain", "return",
                                   "hello", "hi", "hey", "greetings", "how", "are", "you", "there"}}
    if not query_terms:
        return any(word in {"hello", "hi", "greetings", "welcome"} for word in words)
    return bool(query_terms & set(words))


def evidence_excerpt(text: str, question: str, limit: int = 280) -> str:
    """Show a compact window around a query term in a retrieved passage."""
    compact = " ".join(text.split())
    query_terms = sorted(set(re.findall(r"[a-z0-9]+", question.lower())), key=len, reverse=True)
    query_terms = [term for term in query_terms if term not in {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "the", "to", "what", "with",
    }]
    match = re.search(r"\b(?:" + "|".join(map(re.escape, query_terms)) + r")\b", compact, re.I) if query_terms else None
    if not match or len(compact) <= limit:
        return compact[:limit] + ("…" if len(compact) > limit else "")
    start = max(0, match.start() - limit // 3)
    end = min(len(compact), start + limit)
    if start:
        next_space = compact.find(" ", start)
        start = next_space + 1 if next_space >= 0 else start
    if end < len(compact):
        previous_space = compact.rfind(" ", start, end)
        end = previous_space if previous_space > start else end
    return ("…" if start else "") + compact[start:end] + ("…" if end < len(compact) else "")


def retrieved_evidence_answer(passages: list[Passage]) -> str | None:
    """Return a plain failure message; excerpts are not themselves an answer."""
    if not passages:
        return None
    return "I found local material, but I couldn't explain it clearly. I don't want to guess."


def build_grounded_prompt(question: str, passages: list[Passage], history: str = "",
                          context_word_limit: int = 28, *,
                          tokenizer: Tokenizer | BPETokenizer | None = None,
                          block_size: int | None = None,
                          max_new_tokens: int = 60) -> str:
    """Build a prompt, optionally fitting references to the model token window.

    The mandatory question is never clipped. A portion of the context window is
    reserved for generation; older history and retrieved excerpts are expendable.
    """
    if tokenizer is not None and block_size is not None:
        if block_size < 2 or max_new_tokens < 0:
            raise ValueError("block_size must be >= 2 and max_new_tokens nonnegative")
        response_reserve = min(max_new_tokens, max(1, block_size // 3))
        question_block = f"User: {question}\nAssistant:"

        def token_count(value: str) -> int:
            return len(tokenizer.encode(value, add_bos=True))

        mandatory_count = token_count(question_block)
        if mandatory_count + response_reserve > block_size:
            raise ValueError(
                "Question and response reserve exceed the checkpoint context window; "
                "shorten the question or use a model with a larger block size."
            )

        # Keep the newest history that fits after reserving the complete question.
        history_block = ""
        history_words = history.split()
        for start in range(0, len(history_words) + 1):
            candidate = " ".join(history_words[start:])
            candidate_block = f"{candidate}\n" if candidate else ""
            if token_count(candidate_block + question_block) + response_reserve <= block_size:
                history_block = candidate_block
                break

        prompt = history_block + question_block
        prompt_limit = block_size - response_reserve
        for item in passages:
            page = f", page {item.page}" if item.page is not None else ""
            # Citations retain full URL/source in Passage metadata and are shown
            # separately by chat; keep the in-window label compact for evidence.
            prefix = f"Context: [{item.title}{page}] "
            words = item.text.split()
            accepted = ""
            for end in range(1, len(words) + 1):
                excerpt = " ".join(words[:end])
                candidate = f"{prefix}{excerpt}\n" + prompt
                if token_count(candidate) > prompt_limit:
                    break
                accepted = excerpt
            if accepted:
                prompt = f"{prefix}{accepted}\n" + prompt
        if token_count(prompt) > prompt_limit:
            raise ValueError("Prompt exceeds the checkpoint context window")
        return prompt

    excerpt_parts, remaining = [], max(0, context_word_limit)
    for item in passages:
        if remaining <= 0:
            break
        excerpt = " ".join(item.text.split()[:remaining])
        if excerpt:
            page = f", page {item.page}" if item.page else ""
            excerpt_parts.append(f"[{item.title}{page}] {excerpt}")
            remaining -= len(excerpt.split())
    reference_block = ""
    if excerpt_parts:
        reference_block = ("Use these local reference excerpts as evidence. If they do not answer the question, "
                           "say so and do not invent citations.\nContext:\n" + "\n".join(excerpt_parts) + "\n")
    history_block = history.strip() + "\n" if history.strip() else ""
    return f"{reference_block}{history_block}User: {question}\nAssistant:"


def answer_question(model: LanguageModel, tokenizer: Tokenizer | BPETokenizer,
                    question: str,
                    knowledge: KnowledgeBase | None = None, top_k: int = 3,
                    max_new_tokens: int = 60, temperature: float = 0.7,
                    seed: int | None = None, history: str = "", *,
                    sampling_top_k: int | None = 20, repetition_penalty: float = 1.2,
                    no_repeat_ngram_size: int = 3) -> dict:
    if is_unsafe_request(question):
        return {"answer": "I can’t help with unauthorized access, credential theft, malware, or evasion. "
                          "I can help with an authorized defensive review or safer security guidance.",
                "passages": [], "blocked": True, "response_kind": "blocked"}
    passages = knowledge.search(question, top_k=top_k) if knowledge else []
    prompt = build_grounded_prompt(question, passages, history,
                                   context_word_limit=max(8, model.config.block_size // 2),
                                   tokenizer=tokenizer,
                                   block_size=model.config.block_size,
                                   max_new_tokens=max_new_tokens)
    if seed is not None:
        torch.manual_seed(seed)
    completion = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=sampling_top_k, return_prompt=False,
                          repetition_penalty=repetition_penalty,
                          no_repeat_ngram_size=no_repeat_ngram_size)
    completion = completion.strip()
    if generation_is_usable(completion, question):
        return {"answer": completion, "generated_answer": completion,
                "passages": passages, "blocked": False, "response_kind": "generated"}
    evidence = retrieved_evidence_answer(passages)
    if evidence:
        answer, response_kind = evidence, "unavailable"
    else:
        answer = "I couldn't produce a clear, reliable answer."
        if knowledge is not None:
            answer += " I also found no matching local reference."
        response_kind = "unavailable"
    return {"answer": answer, "generated_answer": completion, "passages": passages,
            "blocked": False, "response_kind": response_kind}


def load_checkpoint(path: Path, device: torch.device | None = None):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {path}. Run a training command that writes this exact path first; "
            "the preserved checkpoints/purix.pt is separate and is not replaced automatically.")
    device = device or torch.device("cpu")
    saved = torch.load(path, map_location=device, weights_only=False)
    vocab_path = path.parent / saved["tokenizer_path"]
    tokenizer = load_tokenizer(vocab_path, saved.get("tokenizer_type"),
                               saved.get("tokenizer_version"))
    saved_vocab_hash = saved.get("tokenizer_sha256")
    if saved_vocab_hash is not None:
        actual_vocab_hash = tokenizer.checkpoint_sha256()
        if actual_vocab_hash != saved_vocab_hash:
            raise ValueError("Checkpoint tokenizer vocabulary hash does not match its companion file")
    config = ModelConfig(**saved["model_config"])
    if len(tokenizer.vocab) != config.vocab_size:
        raise ValueError("Checkpoint tokenizer vocabulary does not match model configuration")
    model = LanguageModel(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model: LanguageModel, tokenizer: Tokenizer | BPETokenizer,
             prompt: str,
             max_new_tokens: int = 60, temperature: float = 1.0,
             top_k: int | None = 20, device: torch.device | None = None,
             return_prompt: bool = True, repetition_penalty: float = 1.2,
             no_repeat_ngram_size: int = 3) -> str:
    if (max_new_tokens < 0 or temperature <= 0 or
            (top_k is not None and top_k < 1) or repetition_penalty < 1 or
            no_repeat_ngram_size < 0):
        raise ValueError("max_new_tokens >= 0, temperature > 0, top_k >= 1, "
                         "repetition_penalty >= 1 and no_repeat_ngram_size >= 0 are required")
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_bos=True)
    if not ids:
        ids = [tokenizer.vocab[Tokenizer.BOS]]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_length = tokens.shape[1]
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size:]
        logits = model(context)[:, -1, :] / temperature
        # BOS, PAD and UNK are input/placeholder symbols rather than useful
        # user-facing continuations. EOS remains available as the stop token.
        for special in (Tokenizer.PAD, Tokenizer.BOS, Tokenizer.UNK):
            logits[:, tokenizer.vocab[special]] = float("-inf")
        if repetition_penalty > 1 and generated_ids:
            for token_id in set(generated_ids):
                score = logits[:, token_id]
                logits[:, token_id] = torch.where(
                    score < 0, score * repetition_penalty, score / repetition_penalty)
        if generated_ids:
            # Avoid obvious word-level loops such as "sample sample sample"
            # without globally banning a word from appearing again later.
            logits[:, generated_ids[-1]] = float("-inf")
        if no_repeat_ngram_size and len(generated_ids) >= no_repeat_ngram_size - 1:
            prefix = generated_ids[-(no_repeat_ngram_size - 1):] if no_repeat_ngram_size > 1 else []
            banned = set()
            for index in range(len(generated_ids) - no_repeat_ngram_size + 1):
                ngram = generated_ids[index:index + no_repeat_ngram_size]
                if ngram[:-1] == prefix:
                    banned.add(ngram[-1])
            if banned:
                logits[:, list(banned)] = float("-inf")
        if top_k is not None:
            threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        tokens = torch.cat((tokens, next_id), dim=1)
        next_token_id = next_id.item()
        if next_token_id == tokenizer.vocab[Tokenizer.EOS]:
            break
        generated_ids.append(next_token_id)
    output_ids = tokens[0].tolist() if return_prompt else tokens[0, prompt_length:].tolist()
    return tokenizer.decode(output_ids)


def main() -> None:
    try:
        from .agent.handlers import create_model_orchestrator
    except ImportError:
        from agent.handlers import create_model_orchestrator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/purix.pt"))
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3,
                        help="Block repeated generated n-grams; use 0 to disable")
    parser.add_argument("--knowledge-dir", type=Path, default=Path("knowledge/documents"))
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--top-k-passages", type=int, default=3)
    parser.add_argument("--show-sources", action="store_true",
                        help="Show local references and short excerpts after each response")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run fixed prompts with a fixed seed and print JSON results")
    args = parser.parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint)
    knowledge = None if args.no_rag else KnowledgeBase(args.knowledge_dir)
    agent = create_model_orchestrator(model, tokenizer, knowledge,
                                      max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature,
                                      top_k_passages=args.top_k_passages,
                                      sampling_top_k=args.top_k,
                                      repetition_penalty=args.repetition_penalty,
                                      no_repeat_ngram_size=args.no_repeat_ngram_size)
    conversation_id = "cli-session"
    def print_sources(result, question: str) -> None:
        if not result.sources:
            print("\nNo matching local references found.")
            return
        print("\nLocal references (optional evidence, not the answer):")
        for source in result.sources:
            location = f", page {source.page}" if source.page is not None else ""
            print(f"[{source.title}{location}] {source.url or source.source}")
            if source.excerpt:
                print(evidence_excerpt(source.excerpt, question))

    def respond(question: str) -> None:
        result = agent.handle(question, conversation_id=conversation_id)
        if result.status != "completed":
            print(f"PURIX error ({result.status}): {result.error or 'Request was not completed.'}")
            if args.show_sources and result.sources:
                print_sources(result, question)
            return
        labels = {
            "generated": "PURIX (experimental; generated text is unverified):",
            "greeting": "PURIX (preset greeting; not model-generated):",
            "smalltalk": "PURIX (preset small-talk response; not model-generated):",
            "calculation": "PURIX (calculated locally; not model-generated):",
            "retrieved_evidence": "PURIX (local search results; not model-generated):",
            "blocked": "PURIX (safety response; not model-generated):",
        }
        print(f"\n{labels.get(result.response_kind, 'PURIX response:')}")
        print(result.content)
        if args.show_sources and result.sources:
            print_sources(result, question)
    if args.diagnostic:
        import json
        results = []
        for index, question in enumerate(DIAGNOSTIC_PROMPTS):
            result = answer_question(model, tokenizer, question, knowledge,
                                     top_k=args.top_k_passages,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature, seed=1234 + index,
                                     sampling_top_k=args.top_k,
                                     repetition_penalty=args.repetition_penalty,
                                     no_repeat_ngram_size=args.no_repeat_ngram_size)
            results.append({"prompt": question, "answer": result["answer"],
                            "generated_answer": result["generated_answer"],
                            "response_kind": result["response_kind"],
                            "sources": [{"title": item.title, "url": item.url,
                                         "page": item.page} for item in result["passages"]]})
        print(json.dumps(results, indent=2))
        return
    if args.prompt is not None:
        respond(args.prompt)
        return
    print("PURIX local experimental assistant. Type /exit to quit. No prompts or files are uploaded.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if question:
            respond(question)


if __name__ == "__main__":
    main()
