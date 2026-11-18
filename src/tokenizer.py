"""A deterministic, word-level tokenizer for the PURIX language model."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Iterable


class Tokenizer:
    """Lowercase word/punctuation tokenizer with stable special token IDs."""

    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
    SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)
    tokenizer_type = "word"
    TOKENIZER_FORMAT_VERSION = 1
    tokenizer_version = TOKENIZER_FORMAT_VERSION
    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = dict(vocab) if vocab is not None else {
            token: index for index, token in enumerate(self.SPECIAL_TOKENS)
        }
        self._validate_vocab(self.vocab)
        self.id_to_token = {index: token for token, index in self.vocab.items()}

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Split text into normalized words and individual punctuation marks."""
        return cls._TOKEN_RE.findall(text.lower())

    @classmethod
    def _validate_vocab(cls, vocab: dict[str, int]) -> None:
        if any(vocab.get(token) != index for index, token in enumerate(cls.SPECIAL_TOKENS)):
            raise ValueError("Vocabulary must reserve IDs 0-3 for PAD, UNK, BOS, EOS")
        if len(vocab) != len(set(vocab.values())):
            raise ValueError("Vocabulary IDs must be unique")
        if any(not isinstance(i, int) or i < 0 for i in vocab.values()):
            raise ValueError("Vocabulary IDs must be non-negative integers")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("Vocabulary IDs must be contiguous starting at zero")

    def build_vocab(self, text: str | Iterable[str], min_frequency: int = 1) -> None:
        """Build vocabulary in first-seen order, retaining fixed special IDs."""
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        corpus = [text] if isinstance(text, str) else text
        counts: dict[str, int] = {}
        ordered: list[str] = []
        for sample in corpus:
            for token in self.tokenize(sample):
                if token not in counts:
                    counts[token] = 0
                    ordered.append(token)
                counts[token] += 1
        self.vocab = {token: i for i, token in enumerate(self.SPECIAL_TOKENS)}
        for token in ordered:
            if token not in self.vocab and counts[token] >= min_frequency:
                self.vocab[token] = len(self.vocab)
        self.id_to_token = {i: token for token, i in self.vocab.items()}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode normalized text, mapping absent words to UNK."""
        ids = [self.vocab.get(token, self.vocab[self.UNK]) for token in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.vocab[self.BOS])
        if add_eos:
            ids.append(self.vocab[self.EOS])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode IDs using spaces between tokens; optionally omit special tokens."""
        special = set(self.SPECIAL_TOKENS)
        tokens = []
        for index in ids:
            token = self.id_to_token.get(int(index), self.UNK)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        """Write versioned, deterministic JSON vocabulary."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": self.TOKENIZER_FORMAT_VERSION,
                   "vocab": sorted(self.vocab.items(), key=lambda item: item[1])}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def checkpoint_sha256(self) -> str:
        """Return the legacy vocabulary digest used by word-tokenizer checkpoints."""
        canonical_vocab = json.dumps(self.vocab, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_vocab.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load and validate a vocabulary saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.TOKENIZER_FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format version")
        return cls(dict(payload["vocab"]))


class BPETokenizer:
    """Byte-level BPE tokenizer with lossless UTF-8 coverage and stable specials.

    Train with :meth:`train_from_file` on a single explicitly named training
    corpus. This implementation does not lowercase, so case and spacing remain
    available to the model. Unknown ordinary Unicode text is represented by
    byte-level pieces rather than collapsed to UNK.
    """

    PAD, UNK, BOS, EOS = Tokenizer.SPECIAL_TOKENS
    SPECIAL_TOKENS = Tokenizer.SPECIAL_TOKENS
    tokenizer_type = "bytelevel_bpe"
    TOKENIZER_FORMAT_VERSION = 1
    tokenizer_version = TOKENIZER_FORMAT_VERSION
    _NON_TRAINING_PARTS = {
        "val", "valid", "validation", "dev", "eval", "evaluation", "test",
        "tests", "testing", "heldout", "holdout",
    }

    def __init__(self, backend, training_corpus_sha256: str | None = None) -> None:
        self._backend = backend
        self.training_corpus_sha256 = training_corpus_sha256
        self._validate_special_ids()

    @classmethod
    def train_from_file(cls, training_corpus: str | Path,
                        vocab_size: int = 8192,
                        min_frequency: int = 2) -> "BPETokenizer":
        """Fit BPE from only ``training_corpus``; validation paths are rejected."""
        try:
            from tokenizers import Tokenizer as BackendTokenizer
            from tokenizers import decoders, models, pre_tokenizers, trainers
        except ImportError as exc:
            raise RuntimeError(
                "Byte-level BPE requires the optional dependency; install with "
                "python -m pip install -r requirements-tokenizer.txt"
            ) from exc

        path = Path(training_corpus)
        path_words = {
            word for part in path.parts
            for word in re.split(r"[^a-z0-9]+", part.lower()) if word
        }
        if path_words & cls._NON_TRAINING_PARTS:
            raise ValueError("BPE training accepts a training corpus, not validation/evaluation data")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Training corpus must be a non-empty file: {path}")
        if vocab_size < 260:
            raise ValueError("Byte-level BPE vocab_size must be at least 260")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        backend = BackendTokenizer(models.BPE(unk_token=cls.UNK))
        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.pre_tokenizer = byte_level
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(cls.SPECIAL_TOKENS),
            initial_alphabet=sorted(byte_level.alphabet()),
            show_progress=False,
        )
        backend.train([str(path)], trainer)
        corpus_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(backend, training_corpus_sha256=corpus_hash)

    @classmethod
    def _from_serialized(cls, backend_json: str,
                         training_corpus_sha256: str | None = None) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer as BackendTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Loading this checkpoint requires tokenizers; install with "
                "python -m pip install -r requirements-tokenizer.txt"
            ) from exc
        return cls(BackendTokenizer.from_str(backend_json), training_corpus_sha256)

    def _validate_special_ids(self) -> None:
        expected = {token: index for index, token in enumerate(self.SPECIAL_TOKENS)}
        actual = {token: self._backend.token_to_id(token) for token in self.SPECIAL_TOKENS}
        if actual != expected:
            raise ValueError(f"BPE special-token IDs must be {expected}, got {actual}")
        ids = self._backend.get_vocab().values()
        if set(ids) != set(range(len(self._backend.get_vocab()))):
            raise ValueError("BPE vocabulary IDs must be contiguous starting at zero")

    @property
    def vocab(self) -> dict[str, int]:
        """Token-to-ID mapping compatible with PURIX checkpoint validation."""
        return self._backend.get_vocab()

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.vocab.items()}

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids = self._backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.vocab[self.BOS])
        if add_eos:
            ids.append(self.vocab[self.EOS])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._backend.decode([int(index) for index in ids],
                                    skip_special_tokens=skip_special_tokens)

    def _serialized_payload(self) -> dict:
        return {
            "purix_tokenizer_type": self.tokenizer_type,
            "purix_tokenizer_version": self.tokenizer_version,
            "training_corpus_sha256": self.training_corpus_sha256,
            "backend": json.loads(self._backend.to_str()),
        }

    def checkpoint_sha256(self) -> str:
        canonical = json.dumps(self._serialized_payload(), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self._serialized_payload(), ensure_ascii=False,
                                     sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("purix_tokenizer_type") != cls.tokenizer_type:
            raise ValueError("Tokenizer file is not a PURIX byte-level BPE tokenizer")
        if payload.get("purix_tokenizer_version") != cls.tokenizer_version:
            raise ValueError("Unsupported PURIX BPE tokenizer version")
        return cls._from_serialized(
            json.dumps(payload["backend"], ensure_ascii=False),
            payload.get("training_corpus_sha256"),
        )


def load_tokenizer(path: str | Path, tokenizer_type: str | None = None,
                   tokenizer_version: int | None = None):
    """Load the recorded tokenizer; absent metadata means legacy word format."""
    resolved_type = tokenizer_type or Tokenizer.tokenizer_type
    if resolved_type == Tokenizer.tokenizer_type:
        if tokenizer_version not in (None, Tokenizer.tokenizer_version):
            raise ValueError(f"Unsupported word tokenizer version: {tokenizer_version}")
        return Tokenizer.load(path)
    if resolved_type == BPETokenizer.tokenizer_type:
        if tokenizer_version not in (None, BPETokenizer.tokenizer_version):
            raise ValueError(f"Unsupported byte-level BPE tokenizer version: {tokenizer_version}")
        return BPETokenizer.load(path)
    raise ValueError(f"Unsupported checkpoint tokenizer type: {resolved_type}")
