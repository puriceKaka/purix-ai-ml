"""Prepare a small, provenance-tracked text corpus from local documents.

No corpus is downloaded by default. Network imports require an explicit manifest
which records license and intended-use approval for every URL.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import html
import json
import logging
import posixpath
import re
import socket
import ssl
import time
import zipfile
import unicodedata
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

try:
    from .tokenizer import BPETokenizer, Tokenizer
except ImportError:
    from tokenizer import BPETokenizer, Tokenizer

LOG = logging.getLogger("purix.prepare_data")
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "Public-Domain-US", "PSF-2.0"}
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".json", ".epub"}
FORMAT_VERSION = 1
DOCUMENT_BOUNDARY = "\n<|PURIX_DOCUMENT_BOUNDARY|>\n"


@dataclass
class Document:
    source_path: str
    source_sha256: str
    cleaned_sha256: str
    text: str
    title: str
    source_url: str
    retrieved_at: str
    license: str
    license_url: str
    license_evidence_url: str
    attribution: str
    attribution_requirements: str
    document_id: str
    document_family_id: str
    category: str
    provenance: str
    source_bytes: int
    cleaned_bytes: int
    estimated_tokens: int


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_text(text: str, suffix: str = ".txt") -> str:
    """Normalize encoding artifacts/markup and whitespace without inventing text."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if suffix.lower() in {".html", ".htm"}:
        parser = _VisibleHTML()
        parser.feed(text)
        text = "\n".join(parser.parts)
    elif suffix.lower() in {".md", ".markdown"}:
        text = re.sub(r"!?(\[([^\]]*)\])\([^)]*\)", r"\2", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[`*_~]", "", text)
    text = html.unescape(text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = payload.get("query", {}).get("pages", {})
        if isinstance(pages, list):
            extracts = [page.get("extract", "") for page in pages]
        else:
            extracts = [page.get("extract", "") for page in pages.values()]
        if not extracts or not any(extracts):
            raise ValueError("JSON source has no MediaWiki query.pages[].extract text")
        return "\n\n".join(extracts)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF import requires pypdf in the existing environment") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        with zipfile.ZipFile(path) as package:
            xml = ElementTree.fromstring(package.read("word/document.xml"))
        return "\n".join("".join(node.itertext()) for node in xml.iter()
                          if node.tag.endswith("}p"))
    if suffix == ".epub":
        # EPUB is a ZIP container. Follow the package spine so chapters are
        # emitted in reading order instead of archive order.
        with zipfile.ZipFile(path) as package:
            container = ElementTree.fromstring(package.read("META-INF/container.xml"))
            rootfile = next((node for node in container.iter()
                             if node.tag.endswith("}rootfile")), None)
            if rootfile is None or not rootfile.get("full-path"):
                raise ValueError("EPUB has no package document path in container.xml")
            package_path = rootfile.get("full-path", "")
            opf = ElementTree.fromstring(package.read(package_path))
            manifest = {item.get("id"): item.get("href")
                        for item in opf.iter()
                        if item.tag.endswith("}item") and item.get("id") and item.get("href")
                        and item.get("media-type", "").lower() in {
                            "application/xhtml+xml", "text/html"}}
            spine = [node.get("idref") for node in opf.iter()
                     if node.tag.endswith("}itemref") and node.get("idref")]
            chapters = []
            base_dir = posixpath.dirname(package_path)
            for item_id in spine:
                href = manifest.get(item_id)
                if not href:
                    continue
                member = posixpath.normpath(posixpath.join(base_dir, href.split("#", 1)[0]))
                parser = _VisibleHTML()
                parser.feed(package.read(member).decode("utf-8", errors="replace"))
                chapter = "\n".join(parser.parts).strip()
                if chapter:
                    chapters.append(chapter)
        if not chapters:
            raise ValueError("EPUB package spine contains no readable XHTML or HTML text")
        return "\n\n".join(chapters)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_source_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("documents", payload)


def metadata_for(path: Path, input_roots: list[Path], sources: dict) -> dict:
    candidates = [path.name, path.as_posix()]
    for root in input_roots:
        try:
            candidates.insert(0, path.relative_to(root).as_posix())
        except ValueError:
            pass
    for key in candidates:
        if key in sources:
            return sources[key]
    return {}


def download_sources(manifest_path: Path, raw_dir: Path, max_total_bytes: int,
                     dry_run: bool = False) -> list[dict]:
    """Fetch only explicitly approved text or EPUB URLs; never exceed the aggregate cap."""
    if max_total_bytes < 1 or max_total_bytes > 10 * 1024 * 1024:
        raise ValueError("Download limit must be between 1 byte and 10 MiB")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("sources", [])
    if not isinstance(entries, list):
        raise ValueError("Download manifest must contain a 'sources' list")
    planned, total = [], 0
    filenames = set()
    for entry in entries:
        url = entry.get("url", "")
        license_id = entry.get("license", "")
        if not url.startswith("https://"):
            raise ValueError("Dataset URLs must use HTTPS")
        if license_id not in ALLOWED_LICENSES or not entry.get("intended_use_permitted"):
            raise ValueError(f"Source {url} needs an allowed license and intended_use_permitted=true")
        if not entry.get("license_url"):
            raise ValueError(f"Source {url} needs a license_url")
        from urllib.parse import urlparse
        filename = entry.get("filename") or Path(urlparse(url).path).name
        if Path(filename).suffix.lower() not in {".txt", ".md", ".markdown", ".html", ".htm", ".json", ".epub"}:
            raise ValueError(f"Download source must be text, HTML, MediaWiki JSON, or EPUB: {url}")
        if filename in filenames or not filename:
            raise ValueError(f"Download manifest contains an empty or duplicate filename: {filename!r}")
        filenames.add(filename)
        planned.append((entry, filename))
    if dry_run:
        return [{"url": e["url"], "filename": n, "license": e["license"],
                 "would_overwrite": (raw_dir / n).exists(), "dry_run": True}
                for e, n in planned]
    existing_metadata = load_source_metadata(raw_dir / "sources.json")
    verified_existing = set()
    for entry, filename in planned:
        destination = raw_dir / filename
        recorded = existing_metadata.get(filename, {})
        if destination.exists():
            actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            if Path(filename).suffix.lower() == ".epub" and not zipfile.is_zipfile(destination):
                raise FileExistsError(
                    f"Refusing to reuse invalid EPUB file {destination}; its bytes are not a ZIP archive. "
                    "Use a new raw directory to preserve it and download a verified copy."
                )
            if (recorded.get("sha256") == actual_hash
                    and recorded.get("source_url") == entry["url"]):
                entry.update(retrieved_at=recorded.get("retrieved_at", ""),
                             downloaded_bytes=destination.stat().st_size,
                             sha256=actual_hash, local_file=str(destination))
                verified_existing.add(filename)
            else:
                raise FileExistsError(
                    f"Refusing to overwrite unverified existing raw source {destination}. "
                    "Move it aside after checking it, then rerun; completed files with matching "
                    "sources.json hashes are resumed automatically."
                )
    raw_dir.mkdir(parents=True, exist_ok=True)
    for entry, filename in planned:
        if filename in verified_existing:
            continue
        destination = raw_dir / filename
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "PURIX-local-corpus-preparer/1.0"})
        temporary = destination.with_name(destination.name + ".part")
        for attempt in range(3):
            downloaded = 0
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    if not response.geturl().startswith("https://"):
                        raise ValueError(f"Refusing download redirected away from HTTPS: {entry['url']}")
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > max_total_bytes - total:
                        raise ValueError(f"Download exceeds remaining byte limit: {entry['url']}")
                    while block := response.read(64 * 1024):
                        downloaded += len(block)
                        if total + downloaded > max_total_bytes:
                            raise ValueError(f"Aggregate download exceeds --max-download-bytes ({max_total_bytes})")
                        output.write(block)
                        digest.update(block)
                if Path(filename).suffix.lower() == ".epub" and not zipfile.is_zipfile(temporary):
                    raise ValueError(
                        f"Expected a valid EPUB ZIP archive for {entry['url']}; "
                        "the server returned another format (possibly an HTML landing page)."
                    )
                break
            except (urllib.error.URLError, TimeoutError, socket.timeout,
                    http.client.HTTPException, ssl.SSLError) as exc:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError(
                        f"Download failed after 3 attempts for {entry['url']}: {exc}. "
                        "Rerun the same command to resume files already verified by sources.json."
                    ) from exc
                time.sleep(2 ** attempt)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(destination)
        total += downloaded
        entry["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        entry["downloaded_bytes"] = downloaded
        entry["sha256"] = digest.hexdigest()
        entry["local_file"] = str(destination)
        # Persist each completed source so an interrupted run can safely resume.
        current_metadata = load_source_metadata(raw_dir / "sources.json")
        current_metadata[filename] = {
            "title": entry.get("title", filename), "source_url": entry["url"],
            "retrieved_at": entry["retrieved_at"], "license": entry["license"],
            "license_url": entry["license_url"],
            "license_evidence_url": entry.get("license_evidence_url", ""),
            "attribution": entry.get("attribution", ""),
            "attribution_requirements": entry.get("attribution_requirements", ""),
            "category": entry.get("category", "general_english"),
            "document_family_id": entry.get("document_family_id", filename),
            "document_id": entry.get("document_id", entry.get("document_family_id", filename)),
            "provenance": entry.get("provenance", "Fetched through explicit license-gated manifest."),
            "sha256": entry["sha256"],
        }
        atomic_json(raw_dir / "sources.json", {"documents": current_metadata})
        payload["retrieval_date"] = datetime.now(timezone.utc).date().isoformat()
        payload["status"] = "retrieval_in_progress"
        atomic_json(manifest_path, payload)
    if planned:
        metadata_path = raw_dir / "sources.json"
        additions = {
            name: {"title": entry.get("title", name), "source_url": entry["url"],
                  "retrieved_at": entry["retrieved_at"], "license": entry["license"],
                  "license_url": entry["license_url"],
                  "license_evidence_url": entry.get("license_evidence_url", ""),
                  "attribution": entry.get("attribution", ""),
                  "attribution_requirements": entry.get("attribution_requirements", ""),
                  "category": entry.get("category", "general_english"),
                  "document_family_id": entry.get("document_family_id", name),
                  "document_id": entry.get("document_id", entry.get("document_family_id", name)),
                  "provenance": entry.get("provenance", "Fetched through explicit license-gated manifest."),
                  "sha256": entry["sha256"]}
            for entry, name in planned}
        atomic_json(metadata_path, {"documents": existing_metadata | additions})
        payload["retrieval_date"] = datetime.now(timezone.utc).date().isoformat()
        payload["status"] = "retrieved"
        atomic_json(manifest_path, payload)
    return entries


def prepare(input_roots: Iterable[Path], output_root: Path,
            source_manifest: Path | list[Path] | None = None, target_bytes: int = 5 * 1024 * 1024,
            max_bytes: int = 10 * 1024 * 1024, validation_fraction: float = 0.05,
            seed: int = 42, min_chars: int = 300, dry_run: bool = False,
            resume: bool = False, max_file_bytes: int = 20 * 1024 * 1024,
            fit_bpe: bool = False, bpe_vocab_size: int = 8192,
            bpe_min_frequency: int = 2) -> dict:
    """Clean, deduplicate and document-split sources, then write corpora atomically."""
    roots = [Path(root).resolve() for root in input_roots]
    if not roots:
        raise ValueError("Provide at least one --input directory")
    output_resolved = output_root.resolve()
    checkpoint_root = (Path(__file__).resolve().parents[1] / "checkpoints").resolve()
    if output_resolved == checkpoint_root or output_resolved.is_relative_to(checkpoint_root):
        raise ValueError("Dataset outputs may not be written inside checkpoints/")
    if any(output_resolved.is_relative_to(root) or root.is_relative_to(output_resolved)
           for root in roots):
        raise ValueError("Output directory and input directories must not overlap")
    if not dry_run and output_resolved.exists() and any(
            path.is_file() for path in output_resolved.rglob("*")):
        if not resume:
            raise FileExistsError("Output directory is non-empty; use --resume to continue that experiment")
        if not (output_resolved / "prepare_state.json").is_file():
            raise FileExistsError("Cannot resume a non-empty output directory without its prepare_state.json")
    if target_bytes < 1 or max_bytes < target_bytes or max_bytes > 10 * 1024 * 1024:
        raise ValueError("Require 0 < target bytes <= max bytes <= 10 MiB")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if min_chars < 1 or max_file_bytes < 1:
        raise ValueError("min_chars and max_file_bytes must be positive")
    source_manifests = (source_manifest if isinstance(source_manifest, list)
                        else [source_manifest] if source_manifest else [])
    sources: dict = {}
    for manifest_path in source_manifests:
        sources.update(load_source_metadata(manifest_path))
    files = sorted({path.resolve() for root in roots if root.exists()
                    for path in root.rglob("*") if path.is_file()
                    and not path.name.startswith(".")
                    and path.name.lower() not in {"readme.md", "sources.json"}
                    and path.suffix.lower() in SUPPORTED_SUFFIXES})
    state_path = output_root / "prepare_state.json"
    resume_docs: dict[str, dict] = {}
    if resume and state_path.exists():
        old_state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {"seed": seed, "target_cleaned_bytes": target_bytes,
                    "max_cleaned_bytes": max_bytes,
                    "validation_fraction": validation_fraction}
        if any(old_state.get(key) != value for key, value in expected.items()):
            raise ValueError("Cannot resume with changed seed, size limits, or validation fraction")
        resume_docs = {item["source_identity"]: item for item in old_state.get("documents", [])}
    excluded: list[dict] = []
    records: list[Document] = []
    source_paths: dict[str, Path] = {}
    seen_content: dict[str, str] = {}
    used_bytes = 0
    for file_index, path in enumerate(files):
        size = path.stat().st_size
        rel = next((path.relative_to(root).as_posix() for root in roots
                    if path.is_relative_to(root)), path.name)
        if size > max_file_bytes:
            excluded.append({"path": rel, "reason": "file_exceeds_max_file_bytes", "bytes": size})
            continue
        raw_bytes = path.read_bytes()
        source_sha = hashlib.sha256(raw_bytes).hexdigest()
        prior = resume_docs.get(str(path))
        reused = False
        if prior and prior.get("source_sha256") == source_sha:
            cleaned_path = output_root / "cleaned" / prior.get("cleaned_file", "")
            if cleaned_path.is_file() and hashlib.sha256(cleaned_path.read_bytes()).hexdigest() == prior.get("cleaned_sha256"):
                text = cleaned_path.read_text(encoding="utf-8")
                reused = True
        if not reused:
            try:
                raw = extract_document(path)
                text = clean_text(raw, path.suffix)
            except Exception as exc:
                excluded.append({"path": rel, "reason": f"extraction_error: {type(exc).__name__}: {exc}", "bytes": size})
                continue
        encoded = text.encode("utf-8")
        if len(text) < min_chars:
            excluded.append({"path": rel, "reason": "empty_or_too_short", "bytes": size})
            continue
        normalized_hash = hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()
        if normalized_hash in seen_content:
            excluded.append({"path": rel, "reason": "duplicate_document", "duplicate_of": seen_content[normalized_hash], "bytes": size})
            continue
        if used_bytes + len(encoded) > max_bytes:
            excluded.append({"path": rel, "reason": "corpus_size_limit", "bytes": size,
                             "cleaned_bytes": len(encoded)})
            continue
        seen_content[normalized_hash] = rel
        used_bytes += len(encoded)
        metadata = metadata_for(path, roots, sources)
        retrieved = metadata.get("retrieved_at") or metadata.get("retrieval_date") or ""
        records.append(Document(
            source_path=rel, source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            cleaned_sha256=hashlib.sha256(encoded).hexdigest(), text=text,
            title=metadata.get("title", path.stem), source_url=metadata.get("source_url", metadata.get("url", "")),
            retrieved_at=retrieved, license=metadata.get("license", "unspecified-local-source"),
            license_url=metadata.get("license_url", ""),
            license_evidence_url=metadata.get("license_evidence_url", ""),
            attribution=metadata.get("attribution", ""),
            attribution_requirements=metadata.get("attribution_requirements", ""),
            document_id=metadata.get("document_id", ""),
            document_family_id=metadata.get("document_family_id") or metadata.get("document_id") or rel,
            category=metadata.get("category", "uncategorized"),
            provenance=metadata.get("provenance", "Imported from local file; verify rights before use."),
            source_bytes=size, cleaned_bytes=len(encoded),
            estimated_tokens=len(Tokenizer.tokenize(text))))
        source_paths[records[-1].cleaned_sha256] = path
        if used_bytes >= target_bytes:
            for remaining in files[file_index + 1:]:
                excluded.append({"path": str(remaining), "reason": "target_bytes_reached_not_processed",
                                 "bytes": remaining.stat().st_size})
            break
    if not records and not dry_run:
        searched = ", ".join(str(root) for root in roots) or "(no valid input directories)"
        if files:
            reasons = "; ".join(
                f"{row['path']}: {row['reason']}" for row in excluded[:5]
            ) or "all candidate files were rejected during cleaning"
            found = f"Found {len(files)} supported file(s), but accepted none ({reasons})."
        else:
            found = "Found no supported document files."
        raise ValueError(
            f"No eligible documents found under: {searched}. {found} "
            "A sources.json manifest only adds provenance metadata; it does not provide document text. "
            "Place actual .txt, .md, .markdown, .html, .htm, .pdf, .docx, or .epub files in the input "
            "directory, check --input and --min-chars, then rerun preparation."
        )
    if 0 < len(records) < 2 and not dry_run:
        raise ValueError(
            f"Only {len(records)} distinct eligible document was found under "
            f"{', '.join(str(root) for root in roots)}. At least two are required "
            "for a document-level held-out split; add another source document and retry."
        )

    # Group chapters/pages from one work before assigning a deterministic split.
    families: dict[str, list[Document]] = {}
    for doc in records:
        families.setdefault(doc.document_family_id, []).append(doc)
    ordered_families = sorted(families, key=lambda family: hashlib.sha256(
        f"{seed}:{family}".encode()).hexdigest())
    validation_family_count = (max(1, round(len(ordered_families) * validation_fraction))
                               if len(ordered_families) > 1 else 0)
    validation_families = set(ordered_families[:validation_family_count])
    validation_set = {doc.cleaned_sha256 for doc in records
                      if doc.document_family_id in validation_families}
    training_docs = [doc for doc in records if doc.document_family_id not in validation_families]
    validation_docs = [doc for doc in records if doc.document_family_id in validation_families]
    train_hashes = {doc.cleaned_sha256 for doc in training_docs}
    valid_hashes = {doc.cleaned_sha256 for doc in validation_docs}
    if train_hashes & valid_hashes:
        raise AssertionError("Exact document duplicate crossed the train/validation split")

    tokenizer = Tokenizer()
    tokenizer.build_vocab(doc.text for doc in training_docs)
    vocab_size = len(tokenizer.vocab)
    validation_ids = tokenizer.encode("\n\n".join(doc.text for doc in validation_docs))
    validation_unknown_count = sum(token_id == tokenizer.vocab[Tokenizer.UNK]
                                   for token_id in validation_ids)
    def split_category_summary(documents: list[Document]) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for document in documents:
            row = summary.setdefault(document.category, {"documents": 0, "cleaned_bytes": 0,
                                                          "estimated_tokens": 0})
            row["documents"] += 1
            row["cleaned_bytes"] += document.cleaned_bytes
            row["estimated_tokens"] += document.estimated_tokens
        return summary

    report = {
        "format_version": FORMAT_VERSION, "seed": seed,
        "split_method": "deterministic assignment by document_family_id before corpus/window generation",
        "document_family_count": len(families),
        "training_document_family_count": len(families) - len(validation_families),
        "validation_document_family_count": len(validation_families),
        "target_cleaned_bytes": target_bytes, "max_cleaned_bytes": max_bytes,
        "cleaned_bytes": sum(d.cleaned_bytes for d in records),
        "source_bytes": sum(d.source_bytes for d in records),
        "document_count": len(records), "training_document_count": len(training_docs),
        "validation_document_count": len(validation_docs),
        "training_bytes": sum(d.cleaned_bytes for d in training_docs),
        "validation_bytes": sum(d.cleaned_bytes for d in validation_docs),
        "training_estimated_tokens": sum(d.estimated_tokens for d in training_docs),
        "validation_estimated_tokens": sum(d.estimated_tokens for d in validation_docs),
        "validation_unknown_token_count": validation_unknown_count,
        "validation_unknown_token_rate": validation_unknown_count / max(1, len(validation_ids)),
        "estimated_word_punctuation_vocab_size": vocab_size,
        "duplicate_count": sum(row["reason"] == "duplicate_document" for row in excluded),
        "excluded_count": len(excluded), "excluded_files": excluded,
        "train_validation_exact_overlap": len(train_hashes & valid_hashes),
        "category_summary": {"training": split_category_summary(training_docs),
                             "validation": split_category_summary(validation_docs)},
        "documents": [{key: value for key, value in asdict(doc).items() if key != "text"}
                      | {"split": "validation" if doc.cleaned_sha256 in validation_set else "training"}
                      for doc in records],
    }
    if not dry_run:
        raw_dir = output_root / "raw"
        clean_dir = output_root / "cleaned"
        corpora_dir = output_root / "datasets"
        reports_dir = output_root / "reports"
        for directory in (raw_dir, clean_dir, corpora_dir / "train", corpora_dir / "validation", reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        state_docs = []
        run_state = {"format_version": FORMAT_VERSION, "seed": seed,
                     "target_cleaned_bytes": target_bytes,
                     "max_cleaned_bytes": max_bytes,
                     "validation_fraction": validation_fraction,
                     "status": "in_progress", "documents": state_docs}
        atomic_json(state_path, run_state)
        for doc in records:
            name = doc.cleaned_sha256 + ".txt"
            cleaned_path = clean_dir / name
            if not (resume and cleaned_path.exists() and
                    hashlib.sha256(cleaned_path.read_bytes()).hexdigest() == doc.cleaned_sha256):
                temporary = cleaned_path.with_suffix(".txt.tmp")
                temporary.write_text(doc.text, encoding="utf-8")
                temporary.replace(cleaned_path)
            source_file = source_paths[doc.cleaned_sha256]
            raw_copy = raw_dir / f"{doc.source_sha256}{source_file.suffix.lower()}"
            if not raw_copy.exists():
                temporary_raw = raw_copy.with_name(raw_copy.name + ".tmp")
                temporary_raw.write_bytes(source_file.read_bytes())
                temporary_raw.replace(raw_copy)
            split = "validation" if doc.cleaned_sha256 in validation_set else "train"
            state_docs.append({"source_path": doc.source_path, "cleaned_file": name,
                               "source_identity": str(source_paths[doc.cleaned_sha256]),
                               "source_sha256": doc.source_sha256,
                               "cleaned_sha256": doc.cleaned_sha256,
                               "split": split, "sha256": doc.cleaned_sha256})
            atomic_json(state_path, run_state)
        atomic_text(corpora_dir / "train" / "training.txt",
                    DOCUMENT_BOUNDARY.join(doc.text for doc in training_docs) + "\n")
        atomic_text(corpora_dir / "validation" / "validation.txt",
                    DOCUMENT_BOUNDARY.join(doc.text for doc in validation_docs) + "\n")
        if fit_bpe:
            training_path = corpora_dir / "train" / "training.txt"
            bpe = BPETokenizer.train_from_file(training_path, bpe_vocab_size,
                                               bpe_min_frequency)
            bpe_path = output_root / "purix.tokenizer.json"
            bpe.save(bpe_path)
            validation_ids = [token_id for doc in validation_docs
                              for token_id in bpe.encode(doc.text)]
            unknown_count = sum(token_id == bpe.vocab[BPETokenizer.UNK]
                                for token_id in validation_ids)
            report.update({
                "tokenizer_type": BPETokenizer.tokenizer_type,
                "tokenizer_file": bpe_path.name,
                "tokenizer_vocab_size": len(bpe.vocab),
                "tokenizer_requested_vocab_size": bpe_vocab_size,
                "tokenizer_training_corpus_sha256": bpe.training_corpus_sha256,
                "validation_unknown_token_count": unknown_count,
                "validation_unknown_token_rate": unknown_count / max(1, len(validation_ids)),
            })
        atomic_json(output_root / "sources.json", {"documents": [
            {key: value for key, value in asdict(doc).items() if key != "text"} for doc in records]})
        atomic_json(reports_dir / "dataset_report.json", report)
        run_state["status"] = "complete"
        run_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, run_state)
    return report


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/v1"))
    parser.add_argument("--sources", type=Path, action="append",
                        help="Provenance JSON keyed by relative path or filename; may be repeated")
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--download-only", action="store_true",
                        help="Download approved sources and return without preparing a dataset")
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw"),
                        help="Directory for files explicitly listed in --download-manifest")
    parser.add_argument("--max-download-mb", type=float, default=5.0)
    parser.add_argument("--target-mb", type=float, default=5.0)
    parser.add_argument("--max-clean-mb", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fit-bpe", action="store_true",
                        help="Fit a byte-level BPE tokenizer using training corpus only")
    parser.add_argument("--bpe-vocab-size", type=int, default=8192)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.max_download_mb <= 10:
        parser.error("--max-download-mb must be > 0 and <= 10")
    inputs = args.input or [Path("data/raw")]
    if args.download_manifest:
        download_plan = download_sources(args.download_manifest, args.download_dir,
                                         int(args.max_download_mb * 1024 * 1024), args.dry_run)
        if args.dry_run:
            print(json.dumps({"planned_downloads": download_plan,
                              "note": "No network request or file write was performed."}, indent=2))
            return
        if args.download_only:
            print(json.dumps({"completed_downloads": download_plan,
                              "download_dir": str(args.download_dir),
                              "note": "Downloads were verified; no dataset was prepared."}, indent=2))
            return
        inputs = list(dict.fromkeys([args.download_dir, *args.input]))
    source_manifest = args.sources or (args.download_dir / "sources.json" if args.download_manifest else None)
    result = prepare(inputs, args.output_root, source_manifest,
                     int(args.target_mb * 1024 * 1024),
                     int(args.max_clean_mb * 1024 * 1024),
                     args.validation_fraction, args.seed, args.min_chars,
                     args.dry_run, args.resume,
                     int(args.max_file_mb * 1024 * 1024), args.fit_bpe,
                     args.bpe_vocab_size, args.bpe_min_frequency)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
