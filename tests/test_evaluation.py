import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from src.chat import generate, load_checkpoint
from src.evaluate import evaluate_checkpoint, heldout_loss
from src.model import LanguageModel, ModelConfig
from src.tokenizer import BPETokenizer, Tokenizer
from src.train import prepare_data, save_checkpoint


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.tokenizer = Tokenizer()
        self.tokenizer.build_vocab("alpha beta gamma delta")
        self.config = ModelConfig(len(self.tokenizer.vocab), block_size=4,
                                  d_model=8, n_heads=2, n_layers=1, dropout=0.0)
        self.model = LanguageModel(self.config).eval()

    def _checkpoint(self, directory: Path, training_text: str, validation_text: str) -> Path:
        optimizer = torch.optim.AdamW(self.model.parameters())
        path = directory / "model.pt"
        training_path = directory / "training.txt"
        training_path.write_text(training_text, encoding="utf-8")
        save_checkpoint(path, self.model, self.tokenizer, optimizer, 3,
                        self.config, 19, 1.0,
                        hashlib.sha256(training_text.encode()).hexdigest(),
                        hashlib.sha256(validation_text.encode()).hexdigest())
        return path

    def test_checkpoint_loads_matching_config_weights_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(Path(directory), "alpha beta gamma", "gamma delta alpha")
            loaded, tokenizer = load_checkpoint(path)
            self.assertEqual(loaded.config.to_dict(), self.config.to_dict())
            self.assertEqual(tokenizer.vocab, self.tokenizer.vocab)
            self.assertTrue(torch.equal(loaded.token_embedding.weight,
                                        self.model.token_embedding.weight))

    def test_legacy_checkpoint_without_tokenizer_metadata_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._checkpoint(root, "alpha beta gamma", "gamma delta alpha")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload.pop("tokenizer_type", None)
            payload.pop("tokenizer_version", None)
            torch.save(payload, path)
            loaded, tokenizer = load_checkpoint(path)
            self.assertIs(type(tokenizer), Tokenizer)
            self.assertTrue(torch.equal(loaded.token_embedding.weight,
                                        self.model.token_embedding.weight))

    def test_bpe_checkpoint_selects_and_verifies_matching_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "training.txt"
            corpus.write_text("alpha beta, Python 3.13!\n" * 30, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_file(corpus, vocab_size=512,
                                                     min_frequency=1)
            config = ModelConfig(len(tokenizer.vocab), block_size=8,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            validation_text = "Beta, 17! heldout phrase.\n" * 4
            validation = root / "validation.txt"
            validation.write_text(validation_text, encoding="utf-8")
            checkpoint = root / "bpe.pt"
            save_checkpoint(checkpoint, model, tokenizer,
                            torch.optim.AdamW(model.parameters()), 1,
                            config, 42, 1.0,
                            hashlib.sha256(corpus.read_text(encoding="utf-8").encode()).hexdigest(),
                            hashlib.sha256(validation_text.encode()).hexdigest())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            loaded, loaded_tokenizer = load_checkpoint(checkpoint)
            self.assertEqual(saved["tokenizer_type"], "bytelevel_bpe")
            self.assertEqual(saved["tokenizer_version"], 1)
            self.assertTrue((root / "bpe.tokenizer.json").exists())
            self.assertIs(type(loaded_tokenizer), BPETokenizer)
            self.assertEqual(loaded_tokenizer.vocab, tokenizer.vocab)
            self.assertTrue(torch.equal(loaded.token_embedding.weight,
                                        model.token_embedding.weight))
            completion = generate(loaded, loaded_tokenizer, "Hello,", max_new_tokens=2)
            self.assertIsInstance(completion, str)
            metrics = evaluate_checkpoint(checkpoint, validation, ["Hello,"],
                                          training_corpus=corpus, seed=11)
            self.assertTrue(metrics["heldout_provenance_verified"])
            self.assertGreater(metrics["heldout_next_token_loss"], 0.0)

    def test_heldout_evaluation_is_deterministic(self):
        text = "gamma alpha delta beta gamma alpha delta beta"
        first = heldout_loss(self.model, self.tokenizer, text, batch_size=2)
        second = heldout_loss(self.model, self.tokenizer, text, batch_size=2)
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)

    def test_generation_is_autoregressive_and_seed_reproducible(self):
        torch.manual_seed(321)
        first = generate(self.model, self.tokenizer, "alpha", max_new_tokens=6, top_k=3)
        torch.manual_seed(321)
        second = generate(self.model, self.tokenizer, "alpha", max_new_tokens=6, top_k=3)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("alpha"))

    def test_evaluation_command_result_repeats_with_fixed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "heldout.txt"
            validation_text = "gamma alpha delta beta gamma alpha delta beta"
            validation.write_text(validation_text, encoding="utf-8")
            checkpoint = self._checkpoint(root, "alpha beta delta gamma", validation_text)
            training = root / "training.txt"
            one = evaluate_checkpoint(checkpoint, validation, ["alpha"], seed=7,
                                      training_corpus=training)
            two = evaluate_checkpoint(checkpoint, validation, ["alpha"], seed=7,
                                      training_corpus=training)
            self.assertEqual(one["heldout_next_token_loss"], two["heldout_next_token_loss"])
            self.assertEqual(one["samples"], two["samples"])
            self.assertEqual(one["validation_sha256"], two["validation_sha256"])

    def test_evaluation_records_generation_and_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_text = "gamma alpha delta beta gamma alpha delta beta"
            checkpoint = self._checkpoint(root, "alpha beta delta gamma", validation_text)
            validation = root / "heldout.txt"
            validation.write_text(validation_text, encoding="utf-8")
            result = evaluate_checkpoint(checkpoint, validation, ["alpha", "beta"], seed=7,
                                         training_corpus=root / "training.txt",
                                         max_new_tokens=3, temperature=0.6, top_k=2)
            self.assertEqual(result["generation"],
                             {"max_new_tokens": 3, "temperature": 0.6, "top_k": 2})
            self.assertEqual(len(result["samples"]), 2)
            self.assertEqual(result["tokenizer_sha256"], self.tokenizer.checkpoint_sha256())

    def test_evaluation_rejects_changed_training_or_validation_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_text = "gamma alpha delta beta gamma alpha delta beta"
            checkpoint = self._checkpoint(root, "alpha beta delta gamma", validation_text)
            validation = root / "heldout.txt"
            validation.write_text(validation_text, encoding="utf-8")
            training = root / "training.txt"
            training.write_text("modified training source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Training corpus hash"):
                evaluate_checkpoint(checkpoint, validation, ["alpha"], training_corpus=training)


if __name__ == "__main__":
    unittest.main()
