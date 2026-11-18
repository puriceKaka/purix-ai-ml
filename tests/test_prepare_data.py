import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.prepare_data import clean_text, download_sources, extract_document, prepare
from src.tokenizer import BPETokenizer, Tokenizer


class PrepareDataTests(unittest.TestCase):
    def write_docs(self, root: Path) -> list[str]:
        bodies = [
            "General English introduces a familiar subject with readable examples. " * 8,
            "Science writing explains observations, methods, and conclusions clearly. " * 8,
            "Programming guides describe functions, tests, and debugging steps. " * 9,
            "Defensive security guidance covers evidence handling and incident response. " * 8,
        ]
        for name, text in zip(("a.txt", "b.md", "c.txt", "d.md"), bodies):
            (root / name).write_text(text, encoding="utf-8")
        (root / "duplicate.txt").write_text(bodies[0].upper(), encoding="utf-8")
        (root / "short.txt").write_text("tiny", encoding="utf-8")
        return bodies

    def test_cleaning_normalizes_markup_and_whitespace(self):
        cleaned = clean_text("## Hello **world**\r\n\r\n[manual](https://example.test)\u00ad\x00", ".md")
        self.assertEqual(cleaned, "Hello world\n\nmanual")

    def test_epub_extraction_follows_spine_order_and_ignores_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            epub = Path(directory) / "sample.epub"
            with zipfile.ZipFile(epub, "w") as package:
                package.writestr("META-INF/container.xml", """
                    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                      <rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles>
                    </container>""")
                package.writestr("OPS/package.opf", """
                    <package xmlns="http://www.idpf.org/2007/opf">
                      <manifest>
                        <item id="second" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
                        <item id="first" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
                        <item id="cover" href="cover.jpg" media-type="image/jpeg"/>
                      </manifest>
                      <spine><itemref idref="first"/><itemref idref="second"/></spine>
                    </package>""")
                package.writestr("OPS/chapter1.xhtml", "<html><body>First chapter<script>noise()</script></body></html>")
                package.writestr("OPS/chapter2.xhtml", "<html><body>Second chapter</body></html>")
            text = extract_document(epub)
            self.assertLess(text.index("First chapter"), text.index("Second chapter"))
            self.assertNotIn("noise()", text)

    def test_epub_is_an_explicitly_supported_download_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "downloads.json"
            manifest.write_text(json.dumps({"sources": [{
                "url": "https://example.test/book.epub", "filename": "book.epub",
                "license": "Public-Domain-US", "license_url": "https://example.test/license",
                "intended_use_permitted": True
            }]}), encoding="utf-8")
            plan = download_sources(manifest, root / "raw", 1024, dry_run=True)
            self.assertEqual(plan[0]["filename"], "book.epub")
            self.assertFalse((root / "raw").exists())

    def test_empty_input_error_explains_manifest_is_not_document_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            source.mkdir()
            manifest = source / "sources.json"
            manifest.write_text(json.dumps({"documents": {
                "manual.txt": {"title": "Metadata only", "license": "CC0-1.0"}
            }}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sources.json manifest only adds provenance metadata") as error:
                prepare([source], root / "experiment", manifest)
            self.assertIn("Found no supported document files", str(error.exception))
            self.assertIn(str(source), str(error.exception))
            self.assertFalse((root / "experiment").exists())

    def test_no_eligible_document_reports_rejection_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw"
            source.mkdir()
            (source / "tiny.txt").write_text("tiny", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tiny.txt: empty_or_too_short"):
                prepare([source], root / "experiment", min_chars=50)

    def test_dedup_split_report_and_raw_cleaned_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "experiment"
            source.mkdir()
            docs = self.write_docs(source)
            (source / "sources.json").write_text(json.dumps({"documents": {
                "a.txt": {"title": "General English", "url": "https://example.test/a",
                          "retrieved_at": "2026-01-01", "license": "CC-BY-4.0",
                          "license_url": "https://example.test/license", "category": "general_english"}
            }}), encoding="utf-8")
            report = prepare([source], output, source / "sources.json", target_bytes=5000,
                             max_bytes=5000, validation_fraction=0.25, seed=17, min_chars=80)
            self.assertEqual(report["duplicate_count"], 1)
            self.assertEqual(report["excluded_count"], 2)
            self.assertGreater(report["training_document_count"], 0)
            self.assertGreater(report["validation_document_count"], 0)
            self.assertEqual(report["train_validation_exact_overlap"], 0)
            self.assertIn("training", report["category_summary"])
            self.assertIn("validation", report["category_summary"])
            self.assertGreaterEqual(report["validation_unknown_token_rate"], 0.0)
            self.assertLessEqual(report["validation_unknown_token_rate"], 1.0)
            self.assertTrue((output / "raw").is_dir())
            self.assertTrue((output / "cleaned").is_dir())
            self.assertTrue((output / "datasets/train/training.txt").exists())
            self.assertTrue((output / "datasets/validation/validation.txt").exists())
            self.assertTrue((output / "reports/dataset_report.json").exists())
            training = (output / "datasets/train/training.txt").read_text(encoding="utf-8")
            validation = (output / "datasets/validation/validation.txt").read_text(encoding="utf-8")
            for body in docs:
                self.assertNotEqual(body.strip() in training, body.strip() in validation)
            source_record = next(item for item in report["documents"] if item["source_path"] == "a.txt")
            self.assertEqual(source_record["license"], "CC-BY-4.0")
            self.assertTrue(source_record["retrieved_at"])
            self.assertIn("attribution", source_record)
            self.assertIn("document_id", source_record)
            with patch("src.prepare_data.extract_document", wraps=extract_document) as extractor:
                resumed = prepare([source], output, source / "sources.json", target_bytes=5000,
                                   max_bytes=5000, validation_fraction=0.25, seed=17,
                                   min_chars=80, resume=True)
            self.assertEqual(resumed["document_count"], report["document_count"])
            self.assertEqual(extractor.call_count, 2)  # only duplicate and too-short files are re-extracted

    def test_related_documents_stay_in_same_deterministic_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (source / "chapter-a.txt").write_text("Alpha chapter. " * 40, encoding="utf-8")
            (source / "chapter-b.txt").write_text("Beta chapter. " * 40, encoding="utf-8")
            (source / "other.txt").write_text("Separate work. " * 40, encoding="utf-8")
            manifest = {"documents": {
                "chapter-a.txt": {"document_family_id": "same-book", "category": "science"},
                "chapter-b.txt": {"document_family_id": "same-book", "category": "science"},
                "other.txt": {"document_family_id": "another-book", "category": "math"},
            }}
            (source / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = prepare([source], root / "out", source / "sources.json",
                             target_bytes=10000, max_bytes=10000,
                             validation_fraction=0.4, seed=91, min_chars=10)
            splits = {item["source_path"]: item["split"] for item in report["documents"]}
            self.assertEqual(splits["chapter-a.txt"], splits["chapter-b.txt"])
            self.assertEqual(report["document_family_count"], 2)

    def test_multiple_provenance_manifests_are_merged_for_multiple_input_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            (first / "one.txt").write_text("First licensed source. " * 30, encoding="utf-8")
            (second / "two.txt").write_text("Second licensed source. " * 30, encoding="utf-8")
            manifests = []
            for folder, name, family in ((first, "one.txt", "first-family"),
                                         (second, "two.txt", "second-family")):
                manifest = folder / "sources.json"
                manifest.write_text(json.dumps({"documents": {name: {
                    "title": name, "license": "CC0-1.0", "category": "general_english",
                    "document_family_id": family, "source_url": "https://example.test/" + name
                }}}), encoding="utf-8")
                manifests.append(manifest)
            report = prepare([first, second], root / "out", manifests,
                             target_bytes=10000, max_bytes=10000,
                             validation_fraction=0.4, seed=4, min_chars=10)
            self.assertEqual(report["document_family_count"], 2)
            self.assertEqual({item["license"] for item in report["documents"]}, {"CC0-1.0"})
            self.assertEqual({item["document_family_id"] for item in report["documents"]},
                             {"first-family", "second-family"})

    def test_bpe_fit_uses_training_corpus_and_reports_real_vocab_and_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            for name, text in (("a.txt", "training science facts. " * 40),
                               ("b.txt", "training math examples. " * 40),
                               ("c.txt", "heldout programming details. " * 40),
                               ("d.txt", "heldout english reading. " * 40)):
                (source / name).write_text(text, encoding="utf-8")
            manifest = {"documents": {
                "a.txt": {"document_family_id": "train-science", "category": "science"},
                "b.txt": {"document_family_id": "train-math", "category": "math"},
                "c.txt": {"document_family_id": "valid-code", "category": "code"},
                "d.txt": {"document_family_id": "valid-english", "category": "english"},
            }}
            (source / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "out"
            report = prepare([source], output, source / "sources.json", target_bytes=100000,
                             max_bytes=100000, validation_fraction=0.4, seed=2,
                             min_chars=10, fit_bpe=True, bpe_vocab_size=300,
                             bpe_min_frequency=1)
            training_path = output / "datasets/train/training.txt"
            tokenizer = BPETokenizer.load(output / "purix.tokenizer.json")
            self.assertEqual(report["tokenizer_vocab_size"], len(tokenizer.vocab))
            self.assertEqual(report["tokenizer_training_corpus_sha256"],
                             hashlib.sha256(training_path.read_bytes()).hexdigest())
            self.assertEqual(report["validation_unknown_token_count"], 0)
            self.assertEqual(report["validation_unknown_token_rate"], 0.0)
            self.assertEqual(report["tokenizer_type"], BPETokenizer.tokenizer_type)

    def test_size_cap_and_hard_ten_mib_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            self.write_docs(source)
            report = prepare([source], root / "out", target_bytes=1200,
                             max_bytes=1300, validation_fraction=0.25, min_chars=80)
            self.assertLessEqual(report["cleaned_bytes"], 1300)
            self.assertTrue(any(row["reason"] == "corpus_size_limit"
                                for row in report["excluded_files"]))
            with self.assertRaisesRegex(ValueError, "10 MiB"):
                prepare([source], root / "out2", target_bytes=1,
                        max_bytes=10 * 1024 * 1024 + 1, min_chars=80)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "out"
            source.mkdir()
            self.write_docs(source)
            report = prepare([source], output, target_bytes=5000,
                             max_bytes=5000, min_chars=80, dry_run=True)
            self.assertGreater(report["document_count"], 0)
            self.assertFalse(output.exists())

    def test_tokenizer_vocab_estimate_and_unknown_handling_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            self.write_docs(source)
            report = prepare([source], root / "out", target_bytes=5000,
                             max_bytes=5000, validation_fraction=0.25, min_chars=80)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("known word")
            self.assertEqual(tokenizer.encode("unseen")[0], tokenizer.vocab[Tokenizer.UNK])
            self.assertEqual(tokenizer.vocab[Tokenizer.PAD], 0)
            self.assertGreaterEqual(report["estimated_word_punctuation_vocab_size"], 4)

    def test_pipeline_never_changes_existing_checkpoint_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            self.write_docs(source)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "purix.pt"
            vocab = checkpoint_dir / "purix.vocab.json"
            checkpoint.write_bytes(b"preserve original checkpoint")
            vocab.write_bytes(b"preserve original vocabulary")
            before = (hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      hashlib.sha256(vocab.read_bytes()).hexdigest())
            prepare([source], root / "experiment", target_bytes=5000,
                    max_bytes=5000, min_chars=80)
            with self.assertRaisesRegex(ValueError, "inside checkpoints"):
                prepare([source], Path(__file__).resolve().parents[1] / "checkpoints",
                        target_bytes=5000, max_bytes=5000, min_chars=80)
            after = (hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     hashlib.sha256(vocab.read_bytes()).hexdigest())
            self.assertEqual(after, before)

    def test_download_manifest_requires_approved_license_and_is_dry_run_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "downloads.json"
            manifest.write_text(json.dumps({"sources": [{
                "url": "https://example.test/public.txt", "license": "CC-BY-4.0",
                "license_url": "https://example.test/license", "intended_use_permitted": True
            }]}), encoding="utf-8")
            plan = download_sources(manifest, root / "raw", 2048, dry_run=True)
            self.assertTrue(plan[0]["dry_run"])
            self.assertFalse((root / "raw").exists())
            (root / "raw").mkdir()
            existing = root / "raw/public.txt"
            existing.write_text("keep existing local document", encoding="utf-8")
            with patch("src.prepare_data.urllib.request.urlopen") as network:
                with self.assertRaises(FileExistsError):
                    download_sources(manifest, root / "raw", 2048)
                network.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep existing local document")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["sources"][0]["license"] = "unknown"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allowed license"):
                download_sources(manifest, root / "raw", 2048, dry_run=True)

    def test_download_enforces_size_limit_without_leaving_partial_file(self):
        class FakeResponse:
            headers = {"Content-Length": "10"}
            def geturl(self): return "https://example.test/public.txt"
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1): return b"1234567890"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "downloads.json"
            manifest.write_text(json.dumps({"sources": [{
                "url": "https://example.test/public.txt", "license": "CC0-1.0",
                "license_url": "https://example.test/license", "intended_use_permitted": True
            }]}), encoding="utf-8")
            with patch("src.prepare_data.urllib.request.urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    download_sources(manifest, root / "raw", 5)
            self.assertFalse((root / "raw/public.txt").exists())
            self.assertFalse((root / "raw/public.txt.part").exists())

    def test_download_rejects_html_disguised_as_epub(self):
        class FakeResponse:
            headers = {"Content-Length": "46"}
            def geturl(self): return "https://example.test/book.epub"
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1):
                if not hasattr(self, "returned"):
                    self.returned = True
                    return b"<!doctype html><html><body>download started</body></html>"
                return b""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "downloads.json"
            manifest.write_text(json.dumps({"sources": [{
                "url": "https://example.test/book.epub?source=download",
                "filename": "book.epub", "license": "CC0-1.0",
                "license_url": "https://example.test/license",
                "intended_use_permitted": True,
            }]}), encoding="utf-8")
            with patch("src.prepare_data.urllib.request.urlopen", return_value=FakeResponse()):
                with self.assertRaisesRegex(ValueError, "valid EPUB ZIP archive"):
                    download_sources(manifest, root / "raw", 2048)
            self.assertFalse((root / "raw/book.epub").exists())
            self.assertFalse((root / "raw/book.epub.part").exists())

    def test_download_retries_transient_read_failure_and_resumes_verified_files(self):
        class FakeResponse:
            headers = {"Content-Length": "5"}
            def geturl(self): return "https://example.test/public.txt"
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size=-1):
                if not hasattr(self, "read_once"):
                    self.read_once = True
                    raise TimeoutError("simulated socket read timeout")
                if not hasattr(self, "returned"):
                    self.returned = True
                    return b"hello"
                return b""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "downloads.json"
            manifest.write_text(json.dumps({"sources": [{
                "url": "https://example.test/public.txt", "filename": "public.txt",
                "license": "CC0-1.0", "license_url": "https://example.test/license",
                "intended_use_permitted": True,
            }]}), encoding="utf-8")
            with patch("src.prepare_data.urllib.request.urlopen", return_value=FakeResponse()) as network, \
                    patch("src.prepare_data.time.sleep"):
                result = download_sources(manifest, root / "raw", 2048)
                self.assertEqual(network.call_count, 2)
                self.assertEqual(result[0]["sha256"], hashlib.sha256(b"hello").hexdigest())
            with patch("src.prepare_data.urllib.request.urlopen") as network:
                resumed = download_sources(manifest, root / "raw", 2048)
                network.assert_not_called()
                self.assertEqual(resumed[0]["sha256"], result[0]["sha256"])


if __name__ == "__main__":
    unittest.main()
