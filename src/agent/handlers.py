"""Adapters from agent capabilities to PURIX's existing model chat and retrieval."""
from __future__ import annotations

import ast
import operator
import re

try:
    from ..chat import answer_question, is_greeting
    from ..knowledge import KnowledgeBase, Passage
    from ..model import LanguageModel
    from ..tokenizer import BPETokenizer, Tokenizer
    from .orchestrator import AgentOrchestrator, HandlerResponse, Intent, RequestContext, SourceReference
    from .routing import KeywordRequestRouter
except ImportError:  # direct ``python src/chat.py`` invocation
    from chat import answer_question, is_greeting
    from knowledge import KnowledgeBase, Passage
    from model import LanguageModel
    from tokenizer import BPETokenizer, Tokenizer
    from agent.orchestrator import AgentOrchestrator, HandlerResponse, Intent, RequestContext, SourceReference
    from agent.routing import KeywordRequestRouter


_ARITHMETIC = re.compile(r"\s*[\d.+*/()%\-\s]+\s*\Z")
_BINARY_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                     ast.Mod: operator.mod, ast.Pow: operator.pow}
_SMALLTALK_PATTERNS = (
    (re.compile(r"\s*(?:what(?:['’](?:s|t)|s)?\s+up|what\s+is\s+up|sup)[!.?\s]*", re.I),
     "I'm here and ready to help. What are you working on?"),
    (re.compile(r"\s*(?:(?:i['’]?m|i\s+am|am)\s+)?(?:good|fine|great|okay|ok|well)(?:\s*,?\s*thanks)?[!.?\s]*", re.I),
     "Glad to hear it. What would you like to talk about?"),
)


def smalltalk_response(message: str) -> str | None:
    """Handle a few explicit social turns without presenting them as LM output."""
    for pattern, response in _SMALLTALK_PATTERNS:
        if pattern.fullmatch(message):
            return response
    return None


def calculate_simple_expression(message: str) -> str | None:
    """Evaluate a bounded arithmetic expression without invoking the language model."""
    expression = message.strip()
    if len(expression) > 80 or not _ARITHMETIC.fullmatch(expression):
        return None
    # Accept ordinary implicit multiplication inside the same strict
    # character allowlist and AST evaluator.
    expression = re.sub(r"(?<=[0-9)])\s*(?=\()", "*", expression)
    expression = re.sub(r"(?<=\))\s*(?=[0-9(])", "*", expression)
    try:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 8:
                    raise ValueError("exponent out of range")
                value = _BINARY_OPERATORS[type(node.op)](left, right)
                if abs(value) > 1e100:
                    raise ValueError("result out of range")
                return value
            raise ValueError("unsupported arithmetic")

        value = evaluate(tree)
        rendered = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return f"{message.strip()} = {rendered}"
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


class ConversationHandler:
    """Generate a normal conversational response with the supplied model instance."""

    def __init__(self, model: LanguageModel, tokenizer: Tokenizer | BPETokenizer,
                 max_new_tokens: int = 60, temperature: float = 0.7,
                 sampling_top_k: int | None = 20, repetition_penalty: float = 1.2,
                 no_repeat_ngram_size: int = 3) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.sampling_top_k = sampling_top_k
        self.repetition_penalty, self.no_repeat_ngram_size = repetition_penalty, no_repeat_ngram_size

    def handle_request(self, message: str, context: RequestContext) -> HandlerResponse:
        if is_greeting(message):
            return HandlerResponse("Hello! I'm PURIX, an experimental local assistant. What would you like help with?",
                                   response_kind="greeting")
        smalltalk = smalltalk_response(message)
        if smalltalk is not None:
            return HandlerResponse(smalltalk, response_kind="smalltalk")
        calculation = calculate_simple_expression(message)
        if calculation is not None:
            return HandlerResponse(calculation, response_kind="calculation")
        result = answer_question(self.model, self.tokenizer, message, max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature, history="\n".join(context.history),
                                 sampling_top_k=self.sampling_top_k,
                                 repetition_penalty=self.repetition_penalty,
                                 no_repeat_ngram_size=self.no_repeat_ngram_size)
        return HandlerResponse(result["answer"], response_kind=result.get("response_kind", "generated"))


class DefensiveAnalysisHandler:
    """Answer defensively framed questions using local passages, never live tools."""

    _scope_note = ("Capability limit: you have no live system access and have not run a scan or taken an action. "
                   "Do not claim otherwise. Use local reference excerpts when available.")

    def __init__(self, model: LanguageModel, tokenizer: Tokenizer | BPETokenizer,
                 knowledge: KnowledgeBase | None, max_new_tokens: int = 60,
                 temperature: float = 0.7, top_k_passages: int = 3,
                 sampling_top_k: int | None = 20, repetition_penalty: float = 1.2,
                 no_repeat_ngram_size: int = 3) -> None:
        self.model, self.tokenizer, self.knowledge = model, tokenizer, knowledge
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.top_k_passages, self.sampling_top_k = top_k_passages, sampling_top_k
        self.repetition_penalty, self.no_repeat_ngram_size = repetition_penalty, no_repeat_ngram_size

    @staticmethod
    def _source_reference(passage: Passage) -> SourceReference:
        return SourceReference(title=passage.title, source=passage.source, url=passage.url,
                               page=passage.page, excerpt=passage.text)

    def handle_request(self, message: str, context: RequestContext) -> HandlerResponse:
        history = "\n".join((*context.history, self._scope_note))
        result = answer_question(self.model, self.tokenizer, message, self.knowledge,
                                 top_k=self.top_k_passages, max_new_tokens=self.max_new_tokens,
                                 temperature=self.temperature, history=history,
                                 sampling_top_k=self.sampling_top_k,
                                 repetition_penalty=self.repetition_penalty,
                                 no_repeat_ngram_size=self.no_repeat_ngram_size)
        sources = tuple(self._source_reference(passage) for passage in result["passages"])
        return HandlerResponse(result["answer"], sources=sources, tool_action_occurred=False,
                               response_kind=result.get("response_kind", "generated"))


def create_model_orchestrator(model: LanguageModel, tokenizer: Tokenizer | BPETokenizer,
                              knowledge: KnowledgeBase | None = None, *,
                              router: KeywordRequestRouter | None = None,
                              max_new_tokens: int = 60, temperature: float = 0.7,
                              top_k_passages: int = 3, sampling_top_k: int | None = 20,
                              repetition_penalty: float = 1.2,
                              no_repeat_ngram_size: int = 3) -> AgentOrchestrator:
    """Inject an already-loaded checkpoint/model/tokenizer into both local handlers."""
    orchestrator = AgentOrchestrator(router or KeywordRequestRouter())
    orchestrator.register(Intent.CONVERSATION,
                          ConversationHandler(model, tokenizer, max_new_tokens, temperature,
                                              sampling_top_k, repetition_penalty, no_repeat_ngram_size))
    orchestrator.register(Intent.CYBERSECURITY,
                          DefensiveAnalysisHandler(model, tokenizer, knowledge,
                                                   max_new_tokens, temperature, top_k_passages,
                                                   sampling_top_k, repetition_penalty,
                                                   no_repeat_ngram_size))
    return orchestrator
