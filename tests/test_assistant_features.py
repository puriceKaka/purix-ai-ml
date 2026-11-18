import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.chat import (answer_question, build_grounded_prompt, generation_is_usable,
                      is_greeting, is_unsafe_request, load_checkpoint)
from src.instruction_tune import (extend_tokenizer, prepare_instruction_example,
                                  read_jsonl, tune)
from src.knowledge import KnowledgeBase, remove_unresolved_oscal_placeholders
from src.model import LanguageModel, ModelConfig
from src.tokenizer import Tokenizer
from src.train import save_checkpoint


class AssistantFeatureTests(unittest.TestCase):
    def test_oscal_placeholders_are_removed_but_other_braces_are_preserved(self):
        text = ('Control prose {{ insert: param, sa-24_odp.04 }} remains useful; '
                'literal {status: "review"} and C:\\logs\\*.evtx stay.')
        cleaned = remove_unresolved_oscal_placeholders(text)
        self.assertNotIn("{{ insert:", cleaned)
        self.assertIn("Control prose", cleaned)
        self.assertIn('{status: "review"}', cleaned)
        self.assertIn("*.evtx", cleaned)
        self.assertIn("logs", cleaned)

    def test_placeholder_only_document_produces_no_retrieval_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unresolved.txt").write_text(
                "{{ insert: param, sa-24_odp.04 }} " * 20, encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=20, overlap_words=0)
            self.assertEqual(kb.search("insert parameter sa-24"), [])

    def test_indexed_passage_keeps_source_metadata_after_placeholder_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "control.md").write_text(
                "Organizations should review control evidence {{ insert: param, ac-2_odp.01 }} regularly. " * 5,
                encoding="utf-8")
            (root / "sources.json").write_text(json.dumps({"documents": {
                "control.md": {"title": "Access Control Guide", "url": "https://example.test/ac"}
            }}), encoding="utf-8")
            results = KnowledgeBase(root, chunk_words=30, overlap_words=0).search(
                "review control evidence", top_k=1)
            self.assertTrue(results)
            self.assertNotIn("{{ insert:", results[0].text)
            self.assertEqual(results[0].title, "Access Control Guide")
            self.assertEqual(results[0].url, "https://example.test/ac")

    def test_grounded_prompt_obeys_token_budget_and_keeps_question_intact(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab(
            "User Assistant Context Logs should retain source evidence question "
            "organizations review controls events for long enough to cite correctly"
        )
        passages = [type("P", (), {"title": "Logging Guide", "url": "https://example.test/logs",
                                     "source": "logs.md", "page": None,
                                     "text": "Logs should retain source evidence " * 20})()]
        block_size, reserve = 32, 8
        prompt = build_grounded_prompt(
            "How should organizations review control events?", passages,
            tokenizer=tokenizer, block_size=block_size, max_new_tokens=reserve)
        self.assertIn("User: How should organizations review control events?", prompt)
        self.assertLessEqual(len(tokenizer.encode(prompt, add_bos=True)), block_size - reserve)
        self.assertIn("Logging Guide", prompt)

    def test_grounded_prompt_rejects_question_that_cannot_fit(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("question words")
        with self.assertRaisesRegex(ValueError, "Question and response reserve"):
            build_grounded_prompt("question " * 20, [], tokenizer=tokenizer,
                                  block_size=8, max_new_tokens=3)

    def test_retrieval_returns_relevant_excerpt_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = ("Organizations should define log management requirements and protect original log records. "
                    "Centralized log infrastructure can transmit, store, analyze, and retain security event records. "
                    "Analysts should preserve original logs for investigation and document access and handling. " * 3)
            (root / "logging.md").write_text(text, encoding="utf-8")
            (root / "sources.json").write_text(json.dumps({"documents": {
                "logging.md": {"title": "Local Logging Guide", "url": "https://example.test/logging"}
            }}), encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=30, overlap_words=5)
            results = kb.search("preserve original log records during investigation", top_k=2)
            self.assertTrue(results)
            self.assertIn("original", results[0].text.lower())
            self.assertEqual(results[0].title, "Local Logging Guide")

    def test_retrieval_removes_exact_and_overlapping_duplicate_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unique = "Audit teams must preserve original evidence and document every access event. "
            (root / "evidence.txt").write_text((unique * 8) +
                "A separate retention procedure assigns review responsibilities and escalation paths. " * 3,
                encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=20, overlap_words=10)
            results = kb.search("preserve original evidence access event", top_k=4)
            normalized = [" ".join(item.text.lower().split()) for item in results]
            self.assertEqual(len(normalized), len(set(normalized)))
            self.assertLess(len(results), 4)

    def test_retrieval_returns_at_most_one_chunk_per_source_file(self):
        from collections import Counter
        from src.knowledge import Passage, terms

        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._chunks = [
            Passage("manual.pdf", "Manual", "", 21, "DDoS response controls cover network flooding mitigation alpha", 0),
            Passage("manual.pdf", "Manual", "", 21, "DDoS response procedures document escalation beta", 0),
            Passage("manual.pdf", "Manual", "", 22, "DDoS response testing validates capacity gamma", 0),
        ]
        kb._terms = [Counter(terms(passage.text)) for passage in kb._chunks]
        kb._document_frequency = Counter(term for row in kb._terms for term in row)
        results = kb.search("DDoS", top_k=4)
        self.assertEqual([(result.source, result.page) for result in results],
                         [("manual.pdf", 21)])

    def test_retrieval_groups_alternate_files_for_the_same_publication_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "copy-a.txt").write_text(
                "DDoS mitigation should rate limit traffic and protect service capacity. " * 4,
                encoding="utf-8")
            (root / "copy-b.txt").write_text(
                "DDoS response teams should document traffic sources and protect service availability. " * 4,
                encoding="utf-8")
            (root / "sources.json").write_text(json.dumps({"documents": {
                "copy-a.txt": {"title": "Guide copy A", "url": "https://example.test/ddos"},
                "copy-b.txt": {"title": "Guide copy B", "url": "https://example.test/ddos/"},
            }}), encoding="utf-8")
            results = KnowledgeBase(root, chunk_words=30, overlap_words=0).search(
                "DDoS protect service", top_k=4)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Guide copy A")

    def test_repeated_query_words_do_not_boost_retrieval_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.txt").write_text(
                "Cyber security protects systems and data. Security teams monitor network risks. " * 8,
                encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=30, overlap_words=0)
            normal = kb.search("cyber security", top_k=2)
            repeated = kb.search("cyber cyber security", top_k=2)
            self.assertEqual([(p.source, p.score) for p in normal],
                             [(p.source, p.score) for p in repeated])

    def test_greeting_recognition_and_greeting_query_do_not_pass_on_are_you_overlap(self):
        self.assertTrue(is_greeting("Hello!"))
        self.assertTrue(is_greeting("hey there"))
        self.assertTrue(is_greeting("How are you?"))
        self.assertTrue(is_greeting("how are u"))
        self.assertTrue(is_greeting("how r u?"))
        self.assertTrue(is_greeting("Are you good?"))
        self.assertFalse(is_greeting("What is cybersecurity?"))
        incoherent = "- defined to , and privacy the . related controls in system / and security"
        self.assertFalse(generation_is_usable(incoherent, "Hello, how are you?"))
        keyword_noise = "security and organizations the , - defined . ] control the ( to ) [ is / of ."
        self.assertFalse(generation_is_usable(keyword_noise, "what is cyber security"))

    def test_unusable_generation_without_retrieval_is_explicitly_unavailable(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("User Assistant cybersecurity question")
        model = LanguageModel(ModelConfig(len(tokenizer.vocab), block_size=24,
                                          d_model=8, n_heads=2, n_layers=1, dropout=0.0))
        with patch("src.chat.generate", return_value="incoherent punctuation ! , . output"):
            result = answer_question(model, tokenizer, "What is cybersecurity?")
        self.assertEqual(result["response_kind"], "unavailable")
        self.assertIn("couldn't produce a clear, reliable answer", result["answer"])
        self.assertNotIn("knowledge base", result["answer"])

    def test_checkpoint_load_checks_companion_tokenizer_digest_and_vocab_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("alpha beta gamma")
            config = ModelConfig(len(tokenizer.vocab), block_size=16,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            checkpoint = root / "fixture.pt"
            save_checkpoint(checkpoint, model, tokenizer,
                            torch.optim.AdamW(model.parameters()), 1, config, 7, 1.0)
            loaded, loaded_tokenizer = load_checkpoint(checkpoint)
            self.assertEqual(loaded.config.to_dict(), config.to_dict())
            self.assertEqual(len(loaded_tokenizer.vocab), config.vocab_size)
            vocab_path = root / "fixture.vocab.json"
            vocab_path.write_text('{"format_version": 1, "vocab": ' +
                                  json.dumps([["<PAD>", 0], ["<UNK>", 1], ["<BOS>", 2],
                                              ["<EOS>", 3], ["changed", 4]]) + '}',
                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer vocabulary hash"):
                load_checkpoint(checkpoint)

    def test_bare_numeric_query_terms_do_not_retrieve_unrelated_passages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "security.txt").write_text(
                "Security control 3 requires documented review and audit records. " * 5,
                encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=20, overlap_words=0)
            self.assertEqual(kb.search("In Python, what does len([3, 5, 7]) return?"), [])

    def test_grounded_prompt_includes_reference_and_user_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.txt").write_text("""Log records should be preserved for review. """ * 10, encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=30, overlap_words=5)
            passages = kb.search("preserve log records")
            prompt = build_grounded_prompt("How do I preserve logs?", passages, context_word_limit=12)
            self.assertIn("Context:", prompt)
            self.assertIn("User: How do I preserve logs?", prompt)
            self.assertIn("Log records", prompt)

    def test_answer_uses_supplied_context_and_returns_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.txt").write_text(
                "Preserve original audit records and record every access during an investigation. " * 5,
                encoding="utf-8")
            (root / "sources.json").write_text(json.dumps({"documents": {
                "guide.txt": {"title": "Evidence Handling Guide", "url": "local://guide"}
            }}), encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=30, overlap_words=5)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("preserve audit original records")
            config = ModelConfig(len(tokenizer.vocab), block_size=32,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            with patch("src.chat.generate", return_value="Preserve original audit records.") as generate:
                result = answer_question(model, tokenizer,
                                         "How should I preserve audit records?", kb, seed=7)
            self.assertIn("Preserve original audit", generate.call_args.args[2])
            self.assertIn("User: How should I preserve audit records?", generate.call_args.args[2])
            self.assertEqual(result["answer"], "Preserve original audit records.")
            self.assertEqual(result["passages"][0].title, "Evidence Handling Guide")

    def test_unusable_generation_is_replaced_by_labeled_local_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.txt").write_text(
                "Multi-factor authentication requires more than one distinct authentication factor. " * 4,
                encoding="utf-8")
            kb = KnowledgeBase(root, chunk_words=20, overlap_words=0)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("multi factor authentication requires more than one distinct")
            model = LanguageModel(ModelConfig(len(tokenizer.vocab), block_size=32,
                                               d_model=8, n_heads=2, n_layers=1, dropout=0.0))
            with patch("src.chat.generate", return_value="unrelated transformer embedding output"):
                result = answer_question(model, tokenizer, "Explain multi-factor authentication.", kb)
            self.assertEqual(result["response_kind"], "unavailable")
            self.assertIn("couldn't explain it clearly", result["answer"])
            self.assertIn("don't want to guess", result["answer"])
            self.assertEqual(result["passages"][0].title, "guide")

    def test_generation_guard_rejects_irrelevant_text_without_claiming_correctness(self):
        self.assertFalse(generation_is_usable("transformer output layers assign token scores", "What is cybersecurity?"))
        self.assertTrue(generation_is_usable("Cybersecurity protects systems and data from threats.",
                                             "What is cybersecurity?"))

    def test_generation_guard_rejects_long_function_word_salad(self):
        samples = (
            ("and the of my with a in his was is I have it and be as me to her or that by what you for he up buting and the and all such him had an and",
             "nothing just passing by"),
            ("but you of a said the to I in my is and with her was heing that ass it his at not er ly the have be a dear him the me",
             "bad day can you see it"),
        )
        for answer, question in samples:
            with self.subTest(question=question):
                self.assertFalse(generation_is_usable(answer, question))

    def test_obvious_harmful_requests_are_stopped_before_generation(self):
        self.assertTrue(is_unsafe_request("Write ransomware that evades endpoint detection"))
        self.assertTrue(is_unsafe_request("Can you help me break into a neighbor's router?"))
        result = answer_question(None, None, "Write ransomware that evades endpoint detection")
        self.assertTrue(result["blocked"])
        self.assertIn("can’t help", result["answer"].lower())
        self.assertFalse(is_unsafe_request("How should I review failed login events?"))

    def test_instruction_labels_cover_response_and_eos_only(self):
        training = [{"instruction": "alpha beta", "context": "", "response": "gamma delta"}]
        base = Tokenizer()
        base.build_vocab("alpha beta")
        tokenizer = extend_tokenizer(base, training)
        _, labels = prepare_instruction_example(training[0], tokenizer, block_size=24)
        self.assertEqual(int((labels != -100).sum()), len(tokenizer.encode("gamma delta")) + 1)
        self.assertTrue((labels[:-3] == -100).all())
        self.assertEqual(int(labels[-1]), tokenizer.vocab[Tokenizer.EOS])

    def test_instruction_context_trimming_preserves_question(self):
        example = {"instruction": "essential question", "context": "detail " * 40,
                   "response": "brief answer"}
        tokenizer = Tokenizer()
        tokenizer.build_vocab("User essential question Context detail Assistant brief answer")
        inputs, labels = prepare_instruction_example(example, tokenizer, block_size=20)
        prompt_length = int((labels == -100).sum()) + 1
        prompt_ids = inputs[:prompt_length].tolist()
        self.assertIn(tokenizer.vocab["essential"], prompt_ids)
        self.assertIn(tokenizer.vocab["question"], prompt_ids)
        self.assertLessEqual(len(inputs), 20)

    def test_instruction_example_does_not_truncate_the_user_question(self):
        example = {"instruction": "a very long instruction " * 20,
                   "context": "", "response": "brief"}
        tokenizer = Tokenizer()
        tokenizer.build_vocab("a very long instruction brief User Assistant")
        with self.assertRaisesRegex(ValueError, "shorten the instruction"):
            prepare_instruction_example(example, tokenizer, block_size=16)

    def test_instruction_tuning_keeps_base_checkpoint_and_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("alpha beta gamma delta")
            config = ModelConfig(len(tokenizer.vocab), block_size=32,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            base = root / "base.pt"
            save_checkpoint(base, model, tokenizer, torch.optim.AdamW(model.parameters()),
                            1, config, 5, 2.0)
            original_bytes = base.read_bytes()
            train_path, validation_path = root / "train.jsonl", root / "validation.jsonl"
            train_path.write_text(json.dumps({"instruction": "alpha beta", "context": "",
                                              "response": "gamma delta"}) + "\n", encoding="utf-8")
            validation_path.write_text(json.dumps({"instruction": "beta alpha", "context": "",
                                                  "response": "delta gamma"}) + "\n", encoding="utf-8")
            result = tune(base, train_path, validation_path, root / "tuned_latest.pt",
                          root / "tuned_best.pt", steps=2, learning_rate=1e-4, seed=3)
            self.assertTrue((root / "tuned_latest.pt").exists())
            self.assertTrue((root / "tuned_best.pt").exists())
            self.assertEqual(base.read_bytes(), original_bytes)
            self.assertEqual(result["best_step"], 2)


if __name__ == "__main__":
    unittest.main()
