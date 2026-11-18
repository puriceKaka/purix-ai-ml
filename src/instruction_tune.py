"""Optional response-only instruction tuning for a PURIX checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .chat import load_checkpoint
    from .model import LanguageModel, ModelConfig
    from .tokenizer import BPETokenizer, Tokenizer
    from .train import save_checkpoint, set_seed
except ImportError:
    from chat import load_checkpoint
    from model import LanguageModel, ModelConfig
    from tokenizer import BPETokenizer, Tokenizer
    from train import save_checkpoint, set_seed

LOG = logging.getLogger("purix.instruction_tune")


def read_jsonl(path: Path) -> list[dict[str, str]]:
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(item.get("instruction"), str) or not item["instruction"].strip():
            raise ValueError(f"{path}:{line_number} needs a non-empty instruction")
        if not isinstance(item.get("response"), str) or not item["response"].strip():
            raise ValueError(f"{path}:{line_number} needs a non-empty response")
        context = item.get("context", "")
        if not isinstance(context, str):
            raise ValueError(f"{path}:{line_number} context must be text")
        examples.append({"instruction": item["instruction"], "context": context,
                         "response": item["response"]})
    if not examples:
        raise ValueError(f"No instruction examples found in {path}")
    return examples


def extend_tokenizer(base: Tokenizer, examples: list[dict[str, str]]) -> Tokenizer:
    """Append training-example tokens without changing any existing token IDs."""
    tokenizer = Tokenizer(base.vocab)
    for example in examples:
        prompt, answer = render_prompt(example), example["response"]
        for token in tokenizer.tokenize(prompt + " " + answer):
            if token not in tokenizer.vocab:
                tokenizer.vocab[token] = len(tokenizer.vocab)
    tokenizer.id_to_token = {index: token for token, index in tokenizer.vocab.items()}
    tokenizer._validate_vocab(tokenizer.vocab)
    return tokenizer


def render_prompt(example: dict[str, str]) -> str:
    context = example.get("context", "").strip()
    context_part = f"\nContext: {context}" if context else ""
    return f"User: {example['instruction'].strip()}{context_part}\nAssistant:"


def prepare_instruction_example(example: dict[str, str],
                               tokenizer: Tokenizer | BPETokenizer,
                                block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input IDs and labels, masking prompt tokens with ignore_index=-100."""
    answer_ids = [*tokenizer.encode(example["response"]), tokenizer.vocab[Tokenizer.EOS]]
    if len(answer_ids) >= block_size:
        raise ValueError("Instruction response plus EOS must be shorter than block_size")
    prompt_limit = block_size - len(answer_ids)
    context_words = example.get("context", "").split()

    def encode_prompt(context: str) -> list[int]:
        candidate = {**example, "context": context}
        return [tokenizer.vocab[Tokenizer.BOS], *tokenizer.encode(render_prompt(candidate))]

    prompt_ids = encode_prompt(" ".join(context_words))
    if len(prompt_ids) > prompt_limit and context_words:
        # Keep the instruction and assistant marker intact. Trim only optional
        # context, using the longest prefix that fits the available window.
        low, high = 0, len(context_words)
        while low < high:
            middle = (low + high + 1) // 2
            candidate_ids = encode_prompt(" ".join(context_words[:middle]))
            if len(candidate_ids) <= prompt_limit:
                low = middle
            else:
                high = middle - 1
        prompt_ids = encode_prompt(" ".join(context_words[:low]))
    if len(prompt_ids) > prompt_limit:
        raise ValueError(
            "Instruction prompt and response do not fit block_size; shorten the instruction or response")
    sequence = prompt_ids + answer_ids
    inputs = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor([
        token if position >= len(prompt_ids) - 1 else -100
        for position, token in enumerate(sequence[1:])
    ], dtype=torch.long)
    if not (labels != -100).any():
        raise ValueError("Instruction example produced no response targets")
    return inputs, labels


def expand_model_vocabulary(base_model: LanguageModel,
                            tokenizer: Tokenizer | BPETokenizer) -> LanguageModel:
    """Copy existing PURIX weights and initialize only newly added token rows."""
    old_config = base_model.config
    new_config = ModelConfig(**{**old_config.to_dict(), "vocab_size": len(tokenizer.vocab)})
    expanded = LanguageModel(new_config)
    old_parameters = dict(base_model.named_parameters())
    with torch.no_grad():
        for name, parameter in expanded.named_parameters():
            if name == "token_embedding.weight":
                rows = old_parameters[name].shape[0]
                parameter[:rows].copy_(old_parameters[name])
                if parameter.shape[0] > rows:
                    parameter[rows:].normal_(mean=0.0, std=0.02)
            elif name in old_parameters and parameter.shape == old_parameters[name].shape:
                parameter.copy_(old_parameters[name])
    return expanded


@torch.no_grad()
def response_loss(model: LanguageModel, examples: list[dict[str, str]],
                  tokenizer: Tokenizer | BPETokenizer,
                  block_size: int) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for example in examples:
        inputs, labels = prepare_instruction_example(example, tokenizer, block_size)
        logits = model(inputs.unsqueeze(0))
        total += F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                 labels, ignore_index=-100, reduction="sum").item()
        tokens += int((labels != -100).sum())
    return total / max(tokens, 1)


def tune(base_checkpoint: Path, train_path: Path, validation_path: Path,
         latest_path: Path, best_path: Path, steps: int = 100,
         learning_rate: float = 1e-4, seed: int = 42, block_size: int | None = None) -> dict:
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if latest_path.resolve() == best_path.resolve():
        raise ValueError("Latest and best instruction checkpoints must use separate paths")
    existing = [path for path in (latest_path, best_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite instruction checkpoint: {existing[0]}")
    set_seed(seed)
    train_examples, validation_examples = read_jsonl(train_path), read_jsonl(validation_path)
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Instruction training and validation files must be separate")
    base_model, base_tokenizer = load_checkpoint(base_checkpoint)
    tokenizer = (base_tokenizer if isinstance(base_tokenizer, BPETokenizer)
                 else extend_tokenizer(base_tokenizer, train_examples))
    model = expand_model_vocabulary(base_model, tokenizer)
    context_size = block_size or model.config.block_size
    if context_size > model.config.block_size:
        raise ValueError("Instruction block_size cannot exceed the base model context size")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    base_hash = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    metadata = {"task": "response_only_instruction_tuning", "base_checkpoint_sha256": base_hash,
                "instruction_format": "User/Context/Assistant; labels cover assistant response and EOS only"}
    best_val, best_step = float("inf"), None
    model.train()
    for step in range(1, steps + 1):
        example = random.choice(train_examples)
        inputs, labels = prepare_instruction_example(example, tokenizer, context_size)
        logits = model(inputs.unsqueeze(0))
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                               labels, ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % max(1, min(25, steps // 4)) == 0 or step == steps:
            val_loss = response_loss(model, validation_examples, tokenizer, context_size)
            improved = val_loss < best_val
            if improved:
                best_val, best_step = val_loss, step
            LOG.info("instruction step %d train_batch_loss=%.4f response_val_loss=%.4f",
                     step, loss.item(), val_loss)
            for path, kind in ((latest_path, "latest"), *(([(best_path, "best")] if improved else []))):
                save_checkpoint(path, model, tokenizer, optimizer, step, model.config,
                                seed, best_val, train_hash, validation_hash,
                                best_step, kind, metadata)
    return {"latest_checkpoint": str(latest_path), "best_checkpoint": str(best_path),
            "best_step": best_step, "best_response_validation_loss": best_val,
            "base_checkpoint_sha256": base_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/purix.pt"))
    parser.add_argument("--train", type=Path, default=Path("data/instructions/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/instructions/validation.jsonl"))
    parser.add_argument("--latest", type=Path, default=Path("checkpoints/purix_instruction_latest.pt"))
    parser.add_argument("--best", type=Path, default=Path("checkpoints/purix_instruction_best.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(tune(args.base_checkpoint, args.train, args.validation,
                          args.latest, args.best, args.steps, args.learning_rate,
                          args.seed, args.block_size), indent=2))


if __name__ == "__main__":
    main()
