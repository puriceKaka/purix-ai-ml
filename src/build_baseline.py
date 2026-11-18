"""Extract a document-separated baseline from the bundled official PDFs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - environment-specific message
    raise SystemExit("PDF extraction requires the small optional dependency: python -m pip install pypdf") from exc

try:
    from .tokenizer import Tokenizer
except ImportError:
    from tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "training": {
        "path": ROOT / "knowledge/documents/NIST_SP_800-53r5_catalog.json",
        "title": "NIST SP 800-53 Release 5.2.0 Security and Privacy Controls (OSCAL)",
        "url": "https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final",
        "published": "Release 5.2.0; source catalog last modified 2026-05",
        "note": "Official machine-readable control catalog; retrieve implementation details from current NIST sources.",
    },
    "validation": {
        "path": ROOT / "knowledge/documents/NIST_SP_800-61r3.pdf",
        "title": "NIST SP 800-61 Rev. 3, Incident Response Recommendations and Considerations",
        "url": "https://doi.org/10.6028/NIST.SP.800-61r3",
        "published": "2025-04",
        "note": "Separate current publication held out by document; this is a next-token language baseline, not a capability test.",
    },
}


def extract_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Source PDF is missing: {path}")
    pages = []
    for page in PdfReader(str(path)).pages:
        content = page.extract_text() or ""
        content = content.replace("\x00", "").replace("\u00ad", "")
        content = re.sub(r"-\s*\n\s*", "", content)
        content = re.sub(r"(?m)^\s*(?:NIST SP 800-[^\n]+|GUIDE TO COMPUTER SECURITY LOG MANAGEMENT)\s*$", "", content)
        content = re.sub(r"(?m)^\s*(?:April 2025|\d{1,3}|ES-\d+)\s*$", "", content)
        pages.append(content)
    text = "\n\n".join(pages)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_oscal(path: Path) -> str:
    """Extract human-readable catalog headings and prose from NIST OSCAL JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    catalog = payload.get("catalog")
    if not catalog:
        raise ValueError(f"No OSCAL catalog found in {path}")
    paragraphs: list[str] = []

    def visit_parts(parts: list[dict]) -> None:
        for part in parts:
            prose = part.get("prose")
            if prose:
                paragraphs.append(prose)
            for subpart in part.get("parts", []):
                visit_parts([subpart])

    def visit_controls(controls: list[dict], family: str) -> None:
        for control in controls:
            paragraphs.append(f"{family} control {control.get('id', '')}: {control.get('title', '')}.")
            visit_parts(control.get("parts", []))
            visit_controls(control.get("controls", []), family)

    for group in catalog.get("groups", []):
        family = group.get("title", "Security").strip()
        paragraphs.append(f"NIST security and privacy control family: {family}.")
        visit_controls(group.get("controls", []), family)
    text = "\n\n".join(paragraphs)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_source(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".json":
        return extract_oscal(path)
    raise ValueError(f"Unsupported baseline source format: {path.suffix}")


def build(output_dir: Path = ROOT / "data/baseline/v1") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "split_method": "separate source documents; no random or window-level split", "splits": {}}
    extracted: dict[str, str] = {}
    for split, source in SOURCES.items():
        text = extract_source(source["path"])
        extracted[split] = text
        target = output_dir / f"{split}.txt"
        target.write_text(text + "\n", encoding="utf-8")
        manifest["splits"][split] = {
            "file": target.name,
            "source_file": str(source["path"].relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source["path"].read_bytes()).hexdigest(),
            "text_sha256": hashlib.sha256((text + "\n").encode("utf-8")).hexdigest(),
            "title": source["title"], "url": source["url"],
            "published": source["published"], "note": source["note"],
            "characters": len(text), "whitespace_tokens": len(text.split()),
        }
    tokenizer = Tokenizer()
    tokenizer.build_vocab(extracted["training"])
    validation_ids = tokenizer.encode(extracted["validation"])
    unknown = sum(index == tokenizer.vocab[Tokenizer.UNK] for index in validation_ids)
    manifest["tokenizer_diagnostics"] = {
        "tokenizer": "PURIX word-and-punctuation tokenizer, lowercase",
        "training_token_count": len(tokenizer.tokenize(extracted["training"])),
        "training_vocab_size_including_specials": len(tokenizer.vocab),
        "validation_token_count": len(validation_ids),
        "validation_unknown_count": unknown,
        "validation_unknown_rate": unknown / max(1, len(validation_ids)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
