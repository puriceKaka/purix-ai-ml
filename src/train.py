"""Train the PURIX decoder LM on a plain-text corpus (CPU by default)."""
from __future__ import annotations

import argparse
import bisect
import ctypes
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .model import LanguageModel, ModelConfig
    from .tokenizer import BPETokenizer, Tokenizer, load_tokenizer
    from .prepare_data import DOCUMENT_BOUNDARY
except ImportError:  # direct ``python src/train.py`` invocation
    from model import LanguageModel, ModelConfig
    from tokenizer import BPETokenizer, Tokenizer, load_tokenizer
    from prepare_data import DOCUMENT_BOUNDARY

LOG = logging.getLogger("purix.train")
TOKENIZER_TYPES = (Tokenizer.tokenizer_type, BPETokenizer.tokenizer_type)


def peak_memory_bytes() -> int | None:
    """Return process peak RSS where the platform exposes it."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, AttributeError):
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_tokenizer(tokenizer_type: str, training_corpus: Path,
                    training_text: str, vocab_size: int = 8192,
                    min_frequency: int = 2):
    """Create a tokenizer from training data only; validation is not accepted."""
    if tokenizer_type == Tokenizer.tokenizer_type:
        tokenizer = Tokenizer()
        tokenizer.build_vocab(training_text)
        return tokenizer
    if tokenizer_type == BPETokenizer.tokenizer_type:
        return BPETokenizer.train_from_file(
            training_corpus, vocab_size=vocab_size,
            min_frequency=min_frequency)
    raise ValueError(f"Unsupported tokenizer type {tokenizer_type!r}; choose from {TOKENIZER_TYPES}")


def prepare_data(text: str, tokenizer: Tokenizer | BPETokenizer,
                 block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed next-token windows without crossing document boundaries."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    documents = [document.strip() for document in text.split(DOCUMENT_BOUNDARY)
                 if document.strip()]
    windows_x, windows_y = [], []
    for document in documents:
        # Each document starts a new sequence. EOS remains the final target,
        # while BOS is never predicted from the preceding document.
        ids = tokenizer.encode(document, add_bos=True, add_eos=True)
        if len(ids) - 1 < block_size:
            continue
        values = torch.tensor(ids, dtype=torch.long)
        windows_x.append(values[:-1].unfold(0, block_size, 1))
        windows_y.append(values[1:].unfold(0, block_size, 1))
    if not windows_x:
        raise ValueError(
            f"No document has {block_size} next-token positions; add longer documents or lower block_size")
    return torch.cat(windows_x), torch.cat(windows_y)


def encode_documents(text: str, tokenizer: Tokenizer | BPETokenizer,
                     block_size: int) -> tuple[list[torch.Tensor], list[int]]:
    """Tokenize documents once; retain compact token arrays, not every window."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    documents, counts = [], []
    for document in text.split(DOCUMENT_BOUNDARY):
        document = document.strip()
        if not document:
            continue
        values = torch.tensor(tokenizer.encode(document, add_bos=True, add_eos=True),
                              dtype=torch.long)
        window_count = len(values) - block_size
        if window_count > 0:
            documents.append(values)
            counts.append(window_count)
    if not documents:
        raise ValueError(
            f"No document has {block_size} next-token positions; add longer documents or lower block_size")
    return documents, counts


def sample_batch(documents: list[torch.Tensor], window_counts: list[int],
                 batch_size: int, block_size: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent next-token windows without materializing the corpus windows."""
    cumulative = []
    total = 0
    for count in window_counts:
        total += count
        cumulative.append(total)
    if batch_size < 1 or total < 1:
        raise ValueError("batch_size and total training windows must be positive")
    sampled = torch.randint(total, (batch_size,)).tolist()
    xs, ys = [], []
    for global_index in sampled:
        doc_index = bisect.bisect_right(cumulative, global_index)
        previous = cumulative[doc_index - 1] if doc_index else 0
        offset = global_index - previous
        values = documents[doc_index]
        xs.append(values[offset:offset + block_size])
        ys.append(values[offset + 1:offset + block_size + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate(model: LanguageModel, xs: torch.Tensor, ys: torch.Tensor,
             batch_size: int, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for start in range(0, len(xs), batch_size):
        x, y = xs[start:start + batch_size].to(device), ys[start:start + batch_size].to(device)
        losses.append(F.cross_entropy(model(x).reshape(-1, model.config.vocab_size), y.reshape(-1)).item())
    model.train(was_training)
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_document(model: LanguageModel, text: str,
                      tokenizer: Tokenizer | BPETokenizer,
                      batch_size: int, device: torch.device) -> float:
    """Score each next token once, with context clipped to block_size."""
    documents = [document.strip() for document in text.split(DOCUMENT_BOUNDARY)
                 if document.strip()]
    if not documents:
        raise ValueError("Validation document must contain at least two tokens")
    short_contexts: dict[int, list[tuple[list[int], int]]] = {}
    full_window_documents: list[torch.Tensor] = []
    for document in documents:
        values = torch.tensor(tokenizer.encode(document, add_bos=True, add_eos=True),
                              dtype=torch.long)
        if len(values) <= 1:
            continue
        for target_pos in range(1, min(len(values), model.config.block_size)):
            short_contexts.setdefault(target_pos, []).append(
                (values[:target_pos].tolist(), int(values[target_pos])))
        if len(values) > model.config.block_size:
            full_window_documents.append(values)
    if not short_contexts and not full_window_documents:
        raise ValueError("Validation document must contain at least two tokens")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for examples in short_contexts.values():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            x = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            total_tokens += len(batch)
    for values in full_window_documents:
        # unfold returns a view, avoiding a corpus-sized copy of overlapping windows.
        windows = values.unfold(0, model.config.block_size + 1, 1)
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            x = batch[:, :-1].to(device)
            targets = batch[:, -1].to(device)
            logits = model(x)[:, -1, :]
            total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
            total_tokens += len(batch)
    model.train(was_training)
    return total_loss / total_tokens


def save_checkpoint(path: Path, model: LanguageModel,
                    tokenizer: Tokenizer | BPETokenizer,
                    optimizer: torch.optim.Optimizer, step: int, config: ModelConfig,
                    seed: int, best_val_loss: float,
                    training_sha256: str | None = None,
                    validation_sha256: str | None = None,
                    best_step: int | None = None,
                    checkpoint_kind: str = "latest",
                    extra_metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path.with_suffix(
        ".tokenizer.json" if getattr(tokenizer, "tokenizer_type", "word") == "bytelevel_bpe"
        else ".vocab.json")
    tokenizer.save(tokenizer_path)
    tokenizer_sha256 = tokenizer.checkpoint_sha256()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": config.to_dict(), "tokenizer_path": tokenizer_path.name,
                "tokenizer_type": getattr(tokenizer, "tokenizer_type", "word"),
                "tokenizer_version": getattr(tokenizer, "tokenizer_version", 1),
                "tokenizer_sha256": tokenizer_sha256,
                "training_corpus_sha256": training_sha256,
                "validation_corpus_sha256": validation_sha256,
                "step": step, "seed": seed, "best_val_loss": best_val_loss,
                "best_step": best_step, "checkpoint_kind": checkpoint_kind}
    if extra_metadata:
        payload.update(extra_metadata)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def train(corpus_path: Path, checkpoint_path: Path, steps: int = 500,
          batch_size: int = 16, block_size: int = 64, learning_rate: float = 3e-4,
          seed: int = 42, resume: bool = False, d_model: int = 128,
          n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1,
          validation_path: Path = Path("data/validation.txt"),
          best_checkpoint_path: Path | None = None,
          tokenizer_type: str = Tokenizer.tokenizer_type,
          tokenizer_vocab_size: int = 8192,
          tokenizer_min_frequency: int = 2,
          validation_interval: int | None = None,
          checkpoint_interval: int | None = None,
          weight_decay: float = 0.01,
          device_name: str = "cpu") -> dict:
    if best_checkpoint_path is None:
        best_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")
    if steps < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("steps, batch_size and learning_rate must be positive and weight_decay nonnegative")
    if validation_interval is not None and validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if device_name != "cpu":
        raise ValueError("PURIX Stage 1 training supports the explicit CPU device only")
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(f"Unsupported tokenizer type {tokenizer_type!r}; choose from {TOKENIZER_TYPES}")
    if tokenizer_type == BPETokenizer.tokenizer_type and (
            tokenizer_vocab_size < 260 or tokenizer_min_frequency < 1):
        raise ValueError("tokenizer_vocab_size must be >= 260 and tokenizer_min_frequency positive")
    if not resume and (checkpoint_path.exists() or best_checkpoint_path.exists()):
        existing = checkpoint_path if checkpoint_path.exists() else best_checkpoint_path
        raise FileExistsError(f"Refusing to overwrite existing experiment checkpoint: {existing}; use --resume")
    set_seed(seed)
    text = corpus_path.read_text(encoding="utf-8")
    if not validation_path.exists():
        raise FileNotFoundError(f"Separate validation corpus not found: {validation_path}")
    validation_text = validation_path.read_text(encoding="utf-8")
    if Tokenizer.tokenize(text) == Tokenizer.tokenize(validation_text):
        raise ValueError("Training and validation corpora must be different documents")
    training_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    validation_sha256 = hashlib.sha256(validation_text.encode("utf-8")).hexdigest()
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found at {checkpoint_path}")
        saved_for_tokenizer = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_type = saved_for_tokenizer.get("tokenizer_type", Tokenizer.tokenizer_type)
        if saved_type != tokenizer_type:
            raise ValueError(
                f"Resume tokenizer type {saved_type!r} does not match requested {tokenizer_type!r}"
            )
        tokenizer = load_tokenizer(
            checkpoint_path.parent / saved_for_tokenizer["tokenizer_path"],
            saved_type, saved_for_tokenizer.get("tokenizer_version"),
        )
    else:
        tokenizer = build_tokenizer(tokenizer_type, corpus_path, text,
                                    tokenizer_vocab_size, tokenizer_min_frequency)
    if (isinstance(tokenizer, BPETokenizer) and
            tokenizer.training_corpus_sha256 != hashlib.sha256(corpus_path.read_bytes()).hexdigest()):
        raise ValueError("BPE tokenizer was not fitted from the current training corpus file")
    tokenizer_sha256 = tokenizer.checkpoint_sha256()
    config = ModelConfig(len(tokenizer.vocab), block_size, d_model, n_heads, n_layers, dropout)
    config.validate()
    training_documents, training_window_counts = encode_documents(text, tokenizer, block_size)
    # Validate the held-out file's minimum usable length, but never mix its
    # windows into the optimization set.
    encode_documents(validation_text, tokenizer, block_size)
    device = torch.device(device_name)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_step, best_val, best_step = 0, float("inf"), None
    if resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("tokenizer_type", Tokenizer.tokenizer_type) != tokenizer.tokenizer_type:
            raise ValueError("Resume tokenizer type differs from checkpoint")
        if saved.get("tokenizer_version", Tokenizer.tokenizer_version) != tokenizer.tokenizer_version:
            raise ValueError("Resume tokenizer version differs from checkpoint")
        if saved["model_config"] != config.to_dict():
            raise ValueError("Resume configuration or corpus vocabulary differs from checkpoint")
        expected_hashes = {"training_corpus_sha256": training_sha256,
                           "validation_corpus_sha256": validation_sha256,
                           "tokenizer_sha256": tokenizer_sha256}
        for key, expected in expected_hashes.items():
            if saved.get(key) != expected:
                raise ValueError(f"Cannot resume: {key} differs from the latest checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        if best_checkpoint_path.exists():
            best_saved = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            if best_saved.get("tokenizer_type", Tokenizer.tokenizer_type) != tokenizer.tokenizer_type:
                raise ValueError("Best checkpoint tokenizer type differs from training tokenizer")
            if best_saved.get("tokenizer_version", Tokenizer.tokenizer_version) != tokenizer.tokenizer_version:
                raise ValueError("Best checkpoint tokenizer version differs from training tokenizer")
            if best_saved.get("model_config") != config.to_dict():
                raise ValueError("Best checkpoint configuration differs from training configuration")
            for key, expected in expected_hashes.items():
                if best_saved.get(key) != expected:
                    raise ValueError(f"Best checkpoint {key} differs from the current corpus")
            best_val = float(best_saved["best_val_loss"])
            best_step = int(best_saved["step"])
    last_val = float("nan")
    started_at = time.perf_counter()
    last_train_loss = float("nan")
    validation_interval = validation_interval or max(1, min(100, steps // 5))
    checkpoint_interval = checkpoint_interval or validation_interval
    for step in range(start_step, start_step + steps):
        x, y = sample_batch(training_documents, training_window_counts,
                            batch_size, block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_loss = float(loss.item())
        current_step = step + 1
        if current_step % validation_interval == 0 or current_step == start_step + steps:
            val_loss = evaluate_document(model, validation_text, tokenizer, batch_size, device)
            last_val = val_loss
            elapsed = max(time.perf_counter() - started_at, 1e-12)
            done_steps = step + 1 - start_step
            LOG.info("step %d train_batch_loss=%.4f heldout_val_loss=%.4f "
                     "steps_per_second=%.3f token_positions_processed=%d elapsed_seconds=%.1f peak_rss_bytes=%s",
                      current_step, loss.item(), val_loss, done_steps / elapsed,
                     done_steps * batch_size * block_size, elapsed, peak_memory_bytes())
            improved = val_loss < best_val
            if improved:
                best_val, best_step = val_loss, current_step
            if current_step % checkpoint_interval == 0 or current_step == start_step + steps:
                save_checkpoint(checkpoint_path, model, tokenizer, optimizer, current_step,
                                config, seed, best_val, training_sha256,
                                validation_sha256, best_step, "latest")
            if improved:
                save_checkpoint(best_checkpoint_path, model, tokenizer, optimizer,
                                current_step, config, seed, best_val, training_sha256,
                                validation_sha256, best_step, "best")
    elapsed_seconds = time.perf_counter() - started_at
    token_positions = steps * batch_size * block_size
    return {"checkpoint": str(checkpoint_path), "step": start_step + steps,
            "best_checkpoint": str(best_checkpoint_path), "best_step": best_step,
            "last_val_loss": last_val, "best_val_loss": best_val,
            "last_train_batch_loss": last_train_loss,
            "vocab_size": len(tokenizer.vocab),
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_version": tokenizer.tokenizer_version,
            "device": device_name, "weight_decay": weight_decay,
            "validation_interval": validation_interval,
            "checkpoint_interval": checkpoint_interval,
            "training_tokens_processed": token_positions,
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": steps / max(elapsed_seconds, 1e-12),
            "token_positions_per_second": token_positions / max(elapsed_seconds, 1e-12),
            "peak_rss_bytes": peak_memory_bytes()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/training.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/purix_latest.pt"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/purix_best.pt"))
    parser.add_argument("--validation-corpus", type=Path, default=Path("data/validation.txt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int,
                        help="Evaluate held-out loss every N optimizer steps")
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Write latest checkpoint every N validation steps")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--tokenizer-type", choices=TOKENIZER_TYPES,
                        default=Tokenizer.tokenizer_type,
                        help="Tokenizer for a new experiment; omitted means legacy word tokenizer")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=8192,
                        help="BPE vocabulary target; ignored by the word tokenizer")
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2,
                        help="BPE minimum merge frequency; ignored by the word tokenizer")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train(args.corpus, args.checkpoint, args.steps, args.batch_size,
                   args.block_size, args.learning_rate, args.seed, args.resume,
                   args.d_model, args.heads, args.layers, args.dropout,
                   args.validation_corpus, args.best_checkpoint,
                   tokenizer_type=args.tokenizer_type,
                   tokenizer_vocab_size=args.tokenizer_vocab_size,
                   tokenizer_min_frequency=args.tokenizer_min_frequency,
                   validation_interval=args.validation_interval,
                   checkpoint_interval=args.checkpoint_interval,
                   weight_decay=args.weight_decay, device_name=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
