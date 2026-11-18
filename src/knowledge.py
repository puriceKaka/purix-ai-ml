"""Local-only keyword retrieval over trusted text, Markdown, and PDF files."""
from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.IGNORECASE)
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "when", "with"}
INDEX_VERSION = 4
OSCAL_INSERT_RE = re.compile(
    r"\{\{\s*insert\s*:\s*[a-z][a-z0-9_-]*\s*,\s*[a-z0-9_.:-]+\s*\}\}",
    re.IGNORECASE,
)


def remove_unresolved_oscal_placeholders(text: str) -> str:
    """Remove only OSCAL ``{{ insert: type, identifier }}`` expressions."""
    return OSCAL_INSERT_RE.sub(" ", text)


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    url: str
    page: int | None
    text: str
    score: float


def terms(text: str) -> list[str]:
    """Return lexical search terms, excluding stopwords and bare numbers.

    Bare numbers made unrelated technical passages match programming questions
    solely because both contained a section or version number.
    """
    return [word.lower() for word in WORD_RE.findall(text)
            if word.lower() not in STOPWORDS and not word.isdecimal()]


def _normalized_passage(text: str) -> str:
    return " ".join(terms(text))


def _near_duplicate(left: Passage, right: Passage, threshold: float = 0.82) -> bool:
    """Identify substantially overlapping chunks without excluding same-source evidence."""
    left_terms, right_terms = set(terms(left.text)), set(terms(right.text))
    if not left_terms or not right_terms:
        return False
    return len(left_terms & right_terms) / len(left_terms | right_terms) >= threshold


def _source_family(passage: Passage) -> str:
    """Group alternate local copies of the same publication by canonical URL."""
    url = passage.url.strip().rstrip("/").casefold()
    return f"url:{url}" if url else f"file:{passage.source.casefold()}"


class KnowledgeBase:
    """Index files in a local directory; retrieval never makes network requests."""

    def __init__(self, root: str | Path, chunk_words: int = 140, overlap_words: int = 30):
        self.root = Path(root)
        if chunk_words < 20 or overlap_words < 0 or overlap_words >= chunk_words:
            raise ValueError("Require chunk_words >= 20 and 0 <= overlap_words < chunk_words")
        self.chunk_words, self.overlap_words = chunk_words, overlap_words
        self.metadata = self._load_metadata()
        self._chunks: list[Passage] = []
        self._terms: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._load_documents()

    def _load_metadata(self) -> dict:
        path = self.root / "sources.json"
        if not path.exists():
            path = self.root.parent / "sources.json"
        return json.loads(path.read_text(encoding="utf-8")).get("documents", {}) if path.exists() else {}

    @staticmethod
    def _read_pdf(path: Path) -> list[tuple[int | None, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading local PDFs requires pypdf; install it with python -m pip install pypdf") from exc
        return [(page_no, page.extract_text() or "")
                for page_no, page in enumerate(PdfReader(str(path)).pages, start=1)]

    @staticmethod
    def _read_oscal(path: Path) -> list[tuple[int | None, str]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalog = payload.get("catalog", {})
        passages = []

        def visit_parts(parts: list[dict]) -> None:
            for part in parts:
                if part.get("prose"):
                    passages.append(part["prose"])
                visit_parts(part.get("parts", []))

        def visit_controls(controls: list[dict], family: str) -> None:
            for control in controls:
                passages.append(f"{family} control {control.get('id', '')}: {control.get('title', '')}.")
                visit_parts(control.get("parts", []))
                visit_controls(control.get("controls", []), family)

        for group in catalog.get("groups", []):
            family = group.get("title", "Security")
            passages.append(f"NIST security and privacy control family: {family}.")
            visit_controls(group.get("controls", []), family)
        return [(None, "\n\n".join(passages))]

    def _load_documents(self) -> None:
        if not self.root.exists():
            return
        files = [path for path in sorted(self.root.rglob("*"))
                 if path.is_file() and path.name not in {"sources.json", "README.md"}
                 and path.suffix.lower() in {".pdf", ".txt", ".md", ".json"}]
        metadata_path = self.root.parent / "sources.json"
        fingerprint_data = [(path.relative_to(self.root).as_posix(), path.stat().st_size,
                             hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
        fingerprint_data.append(("sources.json", metadata_path.stat().st_size,
                                 hashlib.sha256(metadata_path.read_bytes()).hexdigest())
                                if metadata_path.exists() else ("sources.json", 0, ""))
        fingerprint_data.append(("index_version", INDEX_VERSION))
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        cache_path = self.root.parent / ".purix_knowledge_index.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    self._chunks = [Passage(**chunk) for chunk in cached["chunks"]]
                    self._terms = [Counter(terms(chunk.text)) for chunk in self._chunks]
                    self._document_frequency.update(
                        term for row in self._terms for term in row.keys())
                    return
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        exact_chunks: set[str] = set()
        for path in files:
            if path.suffix.lower() == ".pdf":
                pages = self._read_pdf(path)
            elif path.suffix.lower() == ".json":
                pages = self._read_oscal(path)
            elif path.suffix.lower() in {".txt", ".md"}:
                pages = [(None, path.read_text(encoding="utf-8", errors="replace"))]
            else:
                continue
            metadata = self.metadata.get(path.name, {})
            title = metadata.get("title", path.stem.replace("_", " "))
            url = metadata.get("url", "")
            for page, text in pages:
                # OSCAL catalogs may retain unresolved parameter insertions in prose.
                # Match only the formal insertion syntax; other braces and punctuation
                # are ordinary document content and remain untouched.
                text = remove_unresolved_oscal_placeholders(text)
                words = text.split()
                stride = self.chunk_words - self.overlap_words
                for start in range(0, len(words), stride):
                    part = words[start:start + self.chunk_words]
                    if len(part) < 12:
                        continue
                    chunk = Passage(path.relative_to(self.root).as_posix(), title,
                                    url, page, " ".join(part), 0.0)
                    indexed_terms = Counter(terms(chunk.text))
                    normalized = _normalized_passage(chunk.text)
                    if not indexed_terms or not normalized or normalized in exact_chunks:
                        continue
                    exact_chunks.add(normalized)
                    self._chunks.append(chunk)
                    self._terms.append(indexed_terms)
                    self._document_frequency.update(indexed_terms.keys())
                    if start + self.chunk_words >= len(words):
                        break
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {"fingerprint": fingerprint,
                         "chunks": [asdict(chunk) for chunk in self._chunks]}
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        temporary.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_path)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.05) -> list[Passage]:
        """Return BM25-like ranked passages with local source and page metadata."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        # Repeating a word in the question should not multiply its influence
        # and dominate the ranking (for example, "cyber cyber security").
        query_terms = Counter({term: 1 for term in terms(query)})
        if not query_terms or not self._chunks:
            return []
        # A longer query needs more than one lexical coincidence.  This avoids
        # treating a programming question as relevant merely because a NIST
        # passage also says "return" or contains a version number.
        required_matches = 1 if len(query_terms) <= 2 else 2
        count = len(self._chunks)
        average_length = sum(sum(row.values()) for row in self._terms) / count
        results = []
        for chunk, frequencies in zip(self._chunks, self._terms):
            length = sum(frequencies.values())
            score = 0.0
            matches = 0
            for term, qtf in query_terms.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                matches += 1
                df = self._document_frequency[term]
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                score += qtf * idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / average_length))
            if matches >= required_matches and score >= min_score:
                results.append(Passage(**{**asdict(chunk), "score": score}))
        selected: list[Passage] = []
        selected_sources: set[str] = set()
        for candidate in sorted(results, key=lambda item: item.score, reverse=True):
            # Alternate local copies (such as a PDF and OSCAL catalog) may
            # represent the same publication and URL. Keep one result per work.
            family = _source_family(candidate)
            if family in selected_sources:
                continue
            if any(_near_duplicate(candidate, prior) for prior in selected):
                continue
            selected.append(candidate)
            selected_sources.add(family)
            if len(selected) == top_k:
                break
        return selected


def format_sources(passages: list[Passage]) -> str:
    """Format source citations for display, keeping excerpts separate from model text."""
    if not passages:
        return "No relevant local passages found."
    lines = []
    for passage in passages:
        location = f", page {passage.page}" if passage.page is not None else ""
        source = passage.url or passage.source
        lines.append(f"[{passage.title}{location}] {source}\n{passage.text}")
    return "\n\n".join(lines)
