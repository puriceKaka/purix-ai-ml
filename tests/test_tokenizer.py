import tempfile
import unittest
from pathlib import Path

from src.tokenizer import BPETokenizer, Tokenizer, load_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_special_ids_and_build(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("Hello, hello world!")
        self.assertEqual([tokenizer.vocab[t] for t in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")], [0, 1, 2, 3])
        self.assertEqual(tokenizer.vocab["hello"], 4)
        self.assertIn(",", tokenizer.vocab)

    def test_encode_decode_normalization_punctuation_and_empty(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("Hello, World!")
        encoded = tokenizer.encode("HELLO, world!", add_bos=True, add_eos=True)
        self.assertEqual(tokenizer.decode(encoded), "hello , world !")
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.encode("hello?"), [tokenizer.vocab["hello"], 1])

    def test_unknown_and_explicit_specials(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("known")
        self.assertEqual(tokenizer.encode("missing"), [tokenizer.vocab[Tokenizer.UNK]])
        self.assertEqual(tokenizer.decode([2, 4, 3], skip_special_tokens=False), "<BOS> known <EOS>")

    def test_save_load_round_trip(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["one two", "two, three"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            tokenizer.save(path)
            restored = Tokenizer.load(path)
        self.assertEqual(restored.vocab, tokenizer.vocab)
        self.assertEqual(restored.encode("TWO, one"), tokenizer.encode("TWO, one"))

    def test_bpe_round_trips_english_numbers_punctuation_and_python(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "training.txt"
            corpus.write_text(
                ("Hello, world! Python uses print('hello') and values like 42. "
                 "A beginner can write for item in items: print(item).\n") * 30,
                encoding="utf-8")
            tokenizer = BPETokenizer.train_from_file(corpus, vocab_size=512,
                                                     min_frequency=1)
        examples = [
            "Hello, how are you?",
            "unseenword xyzzy Ω 🛡️!",
            "The value is 12,345.67; is that correct?",
            "for item in items:\n    print(item)  # beginner Python",
        ]
        for text in examples:
            with self.subTest(text=text):
                ids = tokenizer.encode(text)
                self.assertTrue(ids)
                self.assertEqual(tokenizer.decode(ids), text)
                self.assertNotIn(tokenizer.vocab[Tokenizer.UNK], ids)

    def test_bpe_special_ids_and_explicit_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "training.txt"
            corpus.write_text("hello world\n" * 30, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_file(corpus, vocab_size=300,
                                                     min_frequency=1)
        self.assertEqual([tokenizer.vocab[token] for token in Tokenizer.SPECIAL_TOKENS],
                         [0, 1, 2, 3])
        ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tokenizer.vocab[Tokenizer.BOS])
        self.assertEqual(ids[-1], tokenizer.vocab[Tokenizer.EOS])
        self.assertEqual(tokenizer.decode(ids), "hello")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False),
                         "<BOS>hello<EOS>")

    def test_bpe_training_rejects_validation_and_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation" / "validation.txt"
            validation.parent.mkdir()
            validation.write_text("must not fit", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not validation/evaluation"):
                BPETokenizer.train_from_file(validation, vocab_size=300)
            empty = root / "training.txt"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                BPETokenizer.train_from_file(empty, vocab_size=300)

    def test_bpe_persistence_preserves_mapping_and_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "training.txt"
            corpus.write_text("Hello, BPE! Python 3.13.\n" * 30, encoding="utf-8")
            tokenizer = BPETokenizer.train_from_file(corpus, vocab_size=512,
                                                     min_frequency=1)
            path = root / "tokenizer.json"
            tokenizer.save(path)
            restored = load_tokenizer(path, "bytelevel_bpe", 1)
        self.assertEqual(restored.vocab, tokenizer.vocab)
        self.assertEqual(restored.encode("Hello, new_word!"),
                         tokenizer.encode("Hello, new_word!"))
        self.assertEqual(restored.checkpoint_sha256(), tokenizer.checkpoint_sha256())
        self.assertEqual(restored.training_corpus_sha256,
                         tokenizer.training_corpus_sha256)


if __name__ == "__main__":
    unittest.main()
