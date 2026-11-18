import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.chat import generate, load_checkpoint
from src.model import LanguageModel, ModelConfig
from src.tokenizer import BPETokenizer, Tokenizer
from src.prepare_data import DOCUMENT_BOUNDARY
from src.train import (build_tokenizer, encode_documents, evaluate_document,
                       main as train_main, peak_memory_bytes, prepare_data,
                       sample_batch, train)


class ModelTests(unittest.TestCase):
    def test_train_cli_selects_bpe_explicitly(self):
        with patch.object(sys, "argv", ["src.train", "--tokenizer-type", "bytelevel_bpe",
                                         "--tokenizer-vocab-size", "384",
                                         "--tokenizer-min-frequency", "1"]), \
                patch("src.train.train", return_value={"ok": True}) as mocked_train, \
                contextlib.redirect_stdout(io.StringIO()):
            train_main()
        self.assertEqual(mocked_train.call_args.kwargs["tokenizer_type"], "bytelevel_bpe")
        self.assertEqual(mocked_train.call_args.kwargs["tokenizer_vocab_size"], 384)
        self.assertEqual(mocked_train.call_args.kwargs["tokenizer_min_frequency"], 1)

    def test_train_cli_defaults_to_legacy_word_tokenizer(self):
        with patch.object(sys, "argv", ["src.train"]), \
                patch("src.train.train", return_value={"ok": True}) as mocked_train, \
                contextlib.redirect_stdout(io.StringIO()):
            train_main()
        self.assertEqual(mocked_train.call_args.kwargs["tokenizer_type"], "word")

    def test_bpe_builder_uses_only_training_file_and_matches_model_vocab_size(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.txt"
            validation = root / "validation.txt"
            train_text = "hello world Python print value 42!\n" * 20
            validation_text = "validationonlyphrase unseen heldout material\n" * 20
            training.write_text(train_text, encoding="utf-8")
            validation.write_text(validation_text, encoding="utf-8")
            training_hash = hashlib.sha256(training.read_bytes()).hexdigest()
            tokenizer = build_tokenizer("bytelevel_bpe", training, train_text,
                                        vocab_size=512, min_frequency=1)
            model_config = ModelConfig(len(tokenizer.vocab), block_size=16,
                                       d_model=16, n_heads=4, n_layers=1)
        self.assertIsInstance(tokenizer, BPETokenizer)
        self.assertEqual(tokenizer.training_corpus_sha256, training_hash)
        self.assertNotIn("validationonlyphrase", tokenizer.vocab)
        self.assertEqual(model_config.vocab_size, len(tokenizer.vocab))
        self.assertEqual([tokenizer.vocab[token] for token in Tokenizer.SPECIAL_TOKENS],
                         [0, 1, 2, 3])

    def test_peak_memory_probe_when_supported(self):
        peak = peak_memory_bytes()
        self.assertTrue(peak is None or peak > 0)

    def test_shape_and_future_tokens_are_masked(self):
        torch.manual_seed(7)
        model = LanguageModel(ModelConfig(12, block_size=8, d_model=16, n_heads=4, n_layers=2, dropout=0.0)).eval()
        first = torch.tensor([[2, 4, 5, 6]])
        second = torch.tensor([[2, 4, 9, 10]])
        with torch.no_grad():
            out_a, out_b = model(first), model(second)
        self.assertEqual(tuple(out_a.shape), (1, 4, 12))
        self.assertTrue(torch.allclose(out_a[:, :2], out_b[:, :2], atol=1e-6))

    def test_each_document_window_starts_with_bos_and_can_target_eos(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("alpha beta gamma delta")
        xs, ys = prepare_data(f"alpha beta{DOCUMENT_BOUNDARY}gamma delta", tokenizer, block_size=2)
        pairs = list(zip(xs.tolist(), ys.tolist()))
        eos, bos = tokenizer.vocab[Tokenizer.EOS], tokenizer.vocab[Tokenizer.BOS]
        self.assertEqual(sum(source[0] == bos for source, _ in pairs), 2)
        self.assertTrue(any(target[-1] == eos for _, target in pairs))

    def test_document_windows_never_cross_a_boundary(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("alpha beta gamma delta epsilon zeta eta theta iota kappa")
        first = "alpha beta gamma delta epsilon"
        second = "zeta eta theta iota kappa"
        xs, ys = prepare_data(f"{first}{DOCUMENT_BOUNDARY}{second}", tokenizer, block_size=3)
        first_ids, second_ids = set(tokenizer.encode(first)), set(tokenizer.encode(second))
        for source, target in zip(xs.tolist(), ys.tolist()):
            window_tokens = set(source + target)
            self.assertFalse(window_tokens & first_ids and window_tokens & second_ids)

    def test_sampled_training_windows_preserve_document_boundaries_and_shift(self):
        tokenizer = Tokenizer()
        first, second = "alpha beta gamma delta epsilon", "zeta eta theta iota kappa"
        tokenizer.build_vocab(f"{first} {second}")
        documents, counts = encode_documents(
            f"{first}{DOCUMENT_BOUNDARY}{second}", tokenizer, block_size=3)
        torch.manual_seed(9)
        xs, ys = sample_batch(documents, counts, batch_size=128,
                              block_size=3, device=torch.device("cpu"))
        first_ids = set(tokenizer.encode(first)) - {tokenizer.vocab[Tokenizer.BOS],
                                                     tokenizer.vocab[Tokenizer.EOS]}
        second_ids = set(tokenizer.encode(second)) - {tokenizer.vocab[Tokenizer.BOS],
                                                      tokenizer.vocab[Tokenizer.EOS]}
        for source, target in zip(xs.tolist(), ys.tolist()):
            window = set(source + target)
            self.assertFalse(window & first_ids and window & second_ids)
            self.assertEqual(source[1:], target[:-1])

    def test_streaming_validation_loss_matches_each_next_token_reference(self):
        text = f"alpha beta gamma delta{DOCUMENT_BOUNDARY}epsilon zeta eta theta"
        tokenizer = Tokenizer()
        tokenizer.build_vocab("alpha beta gamma delta epsilon zeta eta theta")
        model = LanguageModel(ModelConfig(len(tokenizer.vocab), block_size=3,
                                          d_model=8, n_heads=2, n_layers=1,
                                          dropout=0.0)).eval()
        actual = evaluate_document(model, text, tokenizer, batch_size=2,
                                   device=torch.device("cpu"))
        losses = []
        with torch.no_grad():
            for document in text.split(DOCUMENT_BOUNDARY):
                ids = tokenizer.encode(document.strip(), add_bos=True, add_eos=True)
                for target_pos in range(1, len(ids)):
                    context = ids[max(0, target_pos - model.config.block_size):target_pos]
                    logits = model(torch.tensor([context]))[:, -1, :]
                    target = torch.tensor([ids[target_pos]])
                    losses.append(torch.nn.functional.cross_entropy(logits, target).item())
        self.assertAlmostEqual(actual, sum(losses) / len(losses), places=5)

    def test_end_to_end_training_checkpoint_and_generation(self):
        corpus = "alpha beta gamma delta. " * 16
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, checkpoint = root / "training.txt", root / "model.pt"
            validation_path, best_checkpoint = root / "validation.txt", root / "model_best.pt"
            corpus_path.write_text(corpus, encoding="utf-8")
            validation_path.write_text("delta gamma beta alpha. " * 8, encoding="utf-8")
            result = train(corpus_path, checkpoint, steps=2, batch_size=2,
                           block_size=8, d_model=16, n_heads=4, n_layers=1, dropout=0.0,
                           validation_path=validation_path, best_checkpoint_path=best_checkpoint)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(best_checkpoint.exists())
            self.assertEqual(result["step"], 2)
            self.assertEqual(result["training_tokens_processed"], 2 * 2 * 8)
            self.assertGreater(result["steps_per_second"], 0)
            self.assertGreater(result["token_positions_per_second"], 0)
            self.assertGreaterEqual(result["peak_rss_bytes"] or 0, 0)
            latest = torch.load(checkpoint, map_location="cpu", weights_only=False)
            best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(latest["checkpoint_kind"], "latest")
            self.assertEqual(latest["tokenizer_type"], "word")
            self.assertEqual(best["checkpoint_kind"], "best")
            self.assertEqual(best["step"], best["best_step"])
            model, tokenizer = load_checkpoint(checkpoint)
            generated = generate(model, tokenizer, "alpha", max_new_tokens=3, top_k=2)
            self.assertTrue(generated.startswith("alpha"))

    def test_training_records_explicit_cpu_intervals_and_optimizer_settings(self):
        corpus = "alpha beta gamma delta. " * 16
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, validation = root / "training.txt", root / "validation.txt"
            training.write_text(corpus, encoding="utf-8")
            validation.write_text("delta gamma beta alpha. " * 8, encoding="utf-8")
            result = train(training, root / "latest.pt", steps=3, batch_size=2,
                           block_size=8, d_model=16, n_heads=4, n_layers=1, dropout=0.0,
                           validation_path=validation, best_checkpoint_path=root / "best.pt",
                           validation_interval=1, checkpoint_interval=2,
                           weight_decay=0.02, device_name="cpu")
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["validation_interval"], 1)
            self.assertEqual(result["checkpoint_interval"], 2)
            self.assertEqual(result["weight_decay"], 0.02)

    def test_new_training_refuses_to_overwrite_an_experiment_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, validation, checkpoint = root / "training.txt", root / "validation.txt", root / "latest.pt"
            training.write_text("alpha beta gamma delta. " * 16, encoding="utf-8")
            validation.write_text("delta gamma beta alpha. " * 8, encoding="utf-8")
            checkpoint.write_bytes(b"preserve me")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                train(training, checkpoint, steps=1, batch_size=2, block_size=8,
                      d_model=16, n_heads=4, n_layers=1, dropout=0.0,
                      validation_path=validation)


if __name__ == "__main__":
    unittest.main()
