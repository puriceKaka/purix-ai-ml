import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.agent.handlers import create_model_orchestrator
from src.agent.orchestrator import HandlerResponse, Intent
from src.agent.routing import KeywordRequestRouter
from src.chat import main as chat_main
from src.knowledge import Passage
from src.model import LanguageModel, ModelConfig
from src.tokenizer import Tokenizer
from src.train import save_checkpoint


class AgentIntegrationTests(unittest.TestCase):
    def test_normal_message_reaches_existing_model_conversation_path(self):
        tokenizer = Tokenizer()
        with patch("src.agent.handlers.answer_question", return_value={"answer": "Hello there!"}) as answer:
            agent = create_model_orchestrator(object(), tokenizer)
            result = agent.handle("hey, could you tell me a story?", conversation_id="one")
        self.assertEqual(result.intent, Intent.CONVERSATION)
        self.assertEqual(result.content, "Hello there!")
        self.assertFalse(result.tool_action_occurred)
        answer.assert_called_once()
        self.assertEqual(len(answer.call_args.args), 3)  # ordinary conversation does not pass retrieval

    def test_defensive_request_uses_retrieval_and_preserves_source_metadata(self):
        tokenizer = Tokenizer()
        passage = Passage("manual.pdf", "Defensive Logging Manual", "https://example.test/logs",
                          7, "Review failed login events and preserve the audit trail.", 2.3)
        with patch("src.agent.handlers.answer_question", return_value={
                "answer": "Review the relevant logs.", "passages": [passage], "blocked": False}) as answer:
            agent = create_model_orchestrator(object(), tokenizer, knowledge=object())
            result = agent.handle("How should I review failed login security logs?", conversation_id="soc")
        self.assertEqual(result.intent, Intent.CYBERSECURITY)
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.tool_action_occurred)
        self.assertEqual(result.sources[0].title, passage.title)
        self.assertEqual(result.sources[0].url, passage.url)
        self.assertEqual(result.sources[0].page, 7)
        self.assertEqual(result.sources[0].excerpt, passage.text)
        self.assertIn("no live system access", answer.call_args.kwargs["history"])

    def test_followup_keeps_only_its_conversation_context(self):
        seen = []

        class ContextHandler:
            def handle_request(self, message, context):
                seen.append((message, context.conversation_id, context.history))
                return HandlerResponse("Acknowledged.")

        from src.agent.orchestrator import AgentOrchestrator
        agent = AgentOrchestrator(KeywordRequestRouter())
        handler = ContextHandler()
        agent.register(Intent.CONVERSATION, handler)
        agent.register(Intent.CYBERSECURITY, handler)
        first = agent.handle("Review the security log events", conversation_id="case-a")
        followup = agent.handle("What does that mean?", conversation_id="case-a")
        other = agent.handle("Hey", conversation_id="case-b")
        self.assertEqual(first.intent, Intent.CYBERSECURITY)
        self.assertEqual(followup.intent, Intent.CYBERSECURITY)
        self.assertEqual(seen[1][2], ("User: Review the security log events", "Assistant: Acknowledged."))
        self.assertEqual(other.intent, Intent.CONVERSATION)
        self.assertEqual(seen[2][2], ())

    def test_handler_failure_is_error_without_fake_content(self):
        agent = create_model_orchestrator(object(), Tokenizer())
        with patch("src.agent.handlers.answer_question", side_effect=RuntimeError("model unavailable")):
            result = agent.handle("Can you explain a poem?", conversation_id="failure")
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.content)
        self.assertIn("model unavailable", result.error)

    def test_greeting_is_a_preset_and_unusable_generation_is_not_completed(self):
        tokenizer = Tokenizer()
        tokenizer.build_vocab("hello user assistant")
        agent = create_model_orchestrator(object(), tokenizer)
        with patch("src.agent.handlers.answer_question") as answer:
            greeting = agent.handle("Hello!", conversation_id="greeting")
            answer.assert_not_called()
        self.assertEqual(greeting.status, "completed")
        self.assertEqual(greeting.response_kind, "greeting")
        self.assertTrue(greeting.content)

        with patch("src.agent.handlers.answer_question", return_value={
                "answer": "PURIX could not produce a usable answer.",
                "response_kind": "unavailable", "passages": []}):
            failed = agent.handle("Tell me a story", conversation_id="generation-failure")
        self.assertEqual(failed.status, "unavailable")
        self.assertIsNone(failed.content)
        self.assertEqual(failed.response_kind, "unavailable")

    def test_common_smalltalk_is_acknowledged_without_asking_the_model(self):
        agent = create_model_orchestrator(object(), Tokenizer())
        with patch("src.agent.handlers.answer_question") as answer:
            for prompt in ("what't up", "what's up", "am good"):
                with self.subTest(prompt=prompt):
                    result = agent.handle(prompt)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.response_kind, "smalltalk")
                    self.assertTrue(result.content)
            answer.assert_not_called()

    def test_simple_arithmetic_is_calculated_locally_without_model_generation(self):
        agent = create_model_orchestrator(object(), Tokenizer())
        with patch("src.agent.handlers.answer_question") as answer:
            for prompt, expected in (("2+3", "2+3 = 5"), ("5-7", "5-7 = -2"),
                                    ("2+5(6*8)-7", "2+5(6*8)-7 = 235"),
                                    ("(2+3)(4)", "(2+3)(4) = 20")):
                with self.subTest(prompt=prompt):
                    result = agent.handle(prompt)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.response_kind, "calculation")
                    self.assertEqual(result.content, expected)
            answer.assert_not_called()

    def test_ddos_and_denial_of_service_are_routed_to_local_retrieval(self):
        router = KeywordRequestRouter()
        for prompt in ("what DDOS", "distributed denial of service", "denial-of-service"):
            with self.subTest(prompt=prompt):
                self.assertEqual(router.route(prompt).intent, Intent.CYBERSECURITY)

    def test_existing_checkpoint_format_and_chat_cli_remain_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("hello user assistant")
            config = ModelConfig(len(tokenizer.vocab), block_size=16,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            checkpoint = root / "purix.pt"
            save_checkpoint(checkpoint, model, tokenizer, torch.optim.AdamW(model.parameters()),
                            1, config, 9, 1.0)
            stdout = io.StringIO()
            with patch("sys.argv", ["purix-chat", "--checkpoint", str(checkpoint),
                                     "--no-rag", "--prompt", "hello", "--max-new-tokens", "1"]), \
                    contextlib.redirect_stdout(stdout):
                chat_main()
            self.assertIn("PURIX (preset greeting; not model-generated):", stdout.getvalue())
            self.assertTrue((root / "purix.vocab.json").exists())

    def test_cli_marks_unusable_generation_as_error_not_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer()
            tokenizer.build_vocab("user assistant cybersecurity question")
            config = ModelConfig(len(tokenizer.vocab), block_size=16,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            checkpoint = root / "purix.pt"
            save_checkpoint(checkpoint, model, tokenizer, torch.optim.AdamW(model.parameters()),
                            1, config, 9, 1.0)
            stdout = io.StringIO()
            with patch("sys.argv", ["purix-chat", "--checkpoint", str(checkpoint),
                                     "--no-rag", "--prompt", "What is cybersecurity?"]), \
                    patch("src.chat.generate", return_value="incoherent punctuation ! , . output"), \
                    contextlib.redirect_stdout(stdout):
                chat_main()
            self.assertIn("PURIX error (unavailable)", stdout.getvalue())
            self.assertNotIn("generated text is unverified", stdout.getvalue())

    def test_cli_hides_reference_dump_when_model_cannot_explain_retrieved_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            passage = ("DDoS response teams should monitor traffic volume, protect service availability, "
                       "and document mitigation actions for authorized incident response. " * 15)
            (documents / "ddos.txt").write_text(passage, encoding="utf-8")
            tokenizer = Tokenizer()
            tokenizer.build_vocab("user assistant DDoS response teams monitor traffic volume protect service availability")
            config = ModelConfig(len(tokenizer.vocab), block_size=32,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            checkpoint = root / "purix.pt"
            save_checkpoint(checkpoint, model, tokenizer,
                            torch.optim.AdamW(model.parameters()), 1, config, 9, 1.0)
            stdout = io.StringIO()
            with patch("sys.argv", ["purix-chat", "--checkpoint", str(checkpoint),
                                     "--knowledge-dir", str(documents), "--prompt",
                                     "What should I do about DDoS traffic?", "--top-k-passages", "1"]), \
                    patch("src.chat.generate", return_value="unrelated punctuation ! , noise"), \
                    contextlib.redirect_stdout(stdout):
                chat_main()
            output = stdout.getvalue()
            self.assertIn("PURIX error (unavailable)", output)
            self.assertIn("couldn't explain it clearly", output)
            self.assertNotIn("Local references", output)
            self.assertNotIn("DDoS response teams should monitor traffic volume", output)

    def test_cli_shows_references_only_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            passage = ("DDoS response teams should monitor traffic volume, protect service availability, "
                       "and document mitigation actions for authorized incident response. " * 15)
            (documents / "ddos.txt").write_text(passage, encoding="utf-8")
            tokenizer = Tokenizer()
            tokenizer.build_vocab("user assistant DDoS response teams monitor traffic volume protect service availability")
            config = ModelConfig(len(tokenizer.vocab), block_size=32,
                                 d_model=8, n_heads=2, n_layers=1, dropout=0.0)
            model = LanguageModel(config)
            checkpoint = root / "purix.pt"
            save_checkpoint(checkpoint, model, tokenizer,
                            torch.optim.AdamW(model.parameters()), 1, config, 9, 1.0)
            stdout = io.StringIO()
            with patch("sys.argv", ["purix-chat", "--checkpoint", str(checkpoint),
                                     "--knowledge-dir", str(documents), "--prompt",
                                     "What should I do about DDoS traffic?", "--top-k-passages", "1",
                                     "--show-sources"]), \
                    patch("src.chat.generate", return_value="unrelated punctuation ! , noise"), \
                    contextlib.redirect_stdout(stdout):
                chat_main()
            output = stdout.getvalue()
            self.assertIn("Local references (optional evidence, not the answer)", output)
            self.assertIn("DDoS response teams should monitor traffic volume", output)


if __name__ == "__main__":
    unittest.main()
