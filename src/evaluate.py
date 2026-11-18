"""Reproducible held-out next-token evaluation and fixed-prompt sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
try:
    from .chat import generate, load_checkpoint
    from .train import evaluate_document
    from .tokenizer import Tokenizer
except ImportError:
    from chat import generate, load_checkpoint
    from train import evaluate_document
    from tokenizer import Tokenizer


@torch.no_grad()
def heldout_loss(model, tokenizer, text: str, batch_size: int = 16) -> float:
    """Mean next-token cross entropy, counting each target token exactly once."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return evaluate_document(model, text, tokenizer, batch_size, next(model.parameters()).device)


def evaluate_checkpoint(checkpoint: Path, validation_corpus: Path,
                        prompts: list[str], seed: int = 1234,
                        batch_size: int = 16,
                        training_corpus: Path = Path("data/training.txt"), *,
                        max_new_tokens: int = 40, temperature: float = 0.8,
                        top_k: int | None = 20) -> dict:
    text = validation_corpus.read_text(encoding="utf-8")
    model, tokenizer = load_checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validation_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    training_text = training_corpus.read_text(encoding="utf-8")
    training_hash = hashlib.sha256(training_text.encode("utf-8")).hexdigest()
    saved_training_hash = saved.get("training_corpus_sha256")
    saved_validation_hash = saved.get("validation_corpus_sha256")
    if saved_training_hash is None or saved_validation_hash is None:
        raise ValueError("Checkpoint lacks training/validation corpus hashes; provenance cannot be verified")
    if saved_training_hash != training_hash:
        raise ValueError("Training corpus hash does not match the checkpoint")
    if saved_validation_hash != validation_hash:
        raise ValueError("Validation corpus hash does not match the checkpoint")
    if training_hash == validation_hash:
        raise ValueError("Training and validation corpus hashes are identical")
    loss = heldout_loss(model, tokenizer, text, batch_size)
    token_ids = tokenizer.encode(text)
    unknown_count = sum(token_id == tokenizer.vocab[Tokenizer.UNK] for token_id in token_ids)
    samples = []
    for index, prompt in enumerate(prompts):
        torch.manual_seed(seed + index)
        samples.append({"prompt": prompt,
                        "completion": generate(model, tokenizer, prompt,
                                               max_new_tokens=max_new_tokens,
                                               temperature=temperature, top_k=top_k)})
    return {"checkpoint": str(checkpoint), "validation_corpus": str(validation_corpus),
            "validation_sha256": validation_hash,
            "training_corpus_sha256": training_hash,
            "heldout_provenance_verified": True,
            "heldout_next_token_loss": loss, "validation_token_count": len(token_ids),
            "unknown_token_count": unknown_count,
            "unknown_token_rate": unknown_count / max(1, len(token_ids)),
            "seed": seed, "tokenizer_sha256": tokenizer.checkpoint_sha256(),
            "model_config": saved["model_config"],
            "generation": {"max_new_tokens": max_new_tokens,
                           "temperature": temperature, "top_k": top_k},
            "samples": samples,
            "note": "Corpus file hashes match checkpoint metadata; this metric is not a capability claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/purix_best.pt"))
    parser.add_argument("--validation-corpus", type=Path, default=Path("data/validation.txt"))
    parser.add_argument("--training-corpus", type=Path, default=Path("data/training.txt"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--prompts-file", type=Path,
                        help="JSONL records containing a prompt or instruction field")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path,
                        help="New JSON file for raw responses and evaluation metadata")
    args = parser.parse_args()
    prompts = args.prompts or ["PURIX learns", "A decoder", "Defensive security"]
    if args.prompts_file:
        records = [json.loads(line) for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        prompts = [record.get("prompt", record.get("instruction", "")) for record in records]
        if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            parser.error("--prompts-file must contain non-empty prompt or instruction strings")
    result = evaluate_checkpoint(args.checkpoint, args.validation_corpus,
                                 prompts, args.seed, args.batch_size,
                                 args.training_corpus, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k)
    rendered = json.dumps(result, indent=2)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
