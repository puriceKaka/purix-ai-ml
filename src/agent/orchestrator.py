"""Small request-routing core; capabilities are injected by the application."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class Intent(str, Enum):
    CONVERSATION = "conversation"
    CYBERSECURITY = "cybersecurity"


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    reason: str


@dataclass(frozen=True)
class SourceReference:
    title: str
    source: str
    url: str
    page: int | None = None
    excerpt: str | None = None


@dataclass(frozen=True)
class HandlerResponse:
    content: str | None = None
    sources: tuple[SourceReference, ...] = ()
    tool_action_occurred: bool = False
    response_kind: str = "generated"


@dataclass(frozen=True)
class AgentResult:
    intent: Intent
    status: str
    content: str | None = None
    error: str | None = None
    sources: tuple[SourceReference, ...] = ()
    tool_action_occurred: bool = False
    response_kind: str = "generated"


@dataclass(frozen=True)
class RequestContext:
    conversation_id: str | None
    history: tuple[str, ...]


class ContextualHandler(Protocol):
    def handle_request(self, message: str, context: RequestContext) -> HandlerResponse: ...


class RequestRouter(Protocol):
    def route(self, message: str) -> RouteDecision: ...


RequestHandler = Callable[[str], str | HandlerResponse] | ContextualHandler


class AgentOrchestrator:
    """Routes a request to an explicitly registered local handler."""

    def __init__(self, router: RequestRouter) -> None:
        self.router = router
        self._handlers: dict[Intent, RequestHandler] = {}
        self._history: dict[str, list[str]] = {}
        self._last_intent: dict[str, Intent] = {}

    def register(self, intent: Intent, handler: RequestHandler) -> None:
        self._handlers[intent] = handler

    def handle(self, message: str, conversation_id: str | None = None) -> AgentResult:
        if not isinstance(message, str) or not message.strip():
            return AgentResult(Intent.CONVERSATION, "invalid", error="Message must not be empty.")
        history = self._history.get(conversation_id, []) if conversation_id else []
        previous = self._last_intent.get(conversation_id) if conversation_id else None
        contextual_route = getattr(self.router, "route_with_context", None)
        decision = contextual_route(message, previous) if contextual_route else self.router.route(message)
        handler = self._handlers.get(decision.intent)
        if handler is None:
            return AgentResult(decision.intent, "unavailable", error=f"No handler registered for {decision.intent.value}.")
        try:
            if hasattr(handler, "handle_request"):
                handled = handler.handle_request(message, RequestContext(conversation_id, tuple(history[-6:])))
            else:
                handled = handler(message)
            response = handled if isinstance(handled, HandlerResponse) else HandlerResponse(content=handled)
            if not isinstance(response.content, str):
                raise TypeError("Handler response content must be text")
            if response.response_kind == "unavailable":
                return AgentResult(decision.intent, "unavailable",
                                   error=response.content or "I couldn't produce a clear, reliable answer.",
                                   sources=response.sources,
                                   response_kind=response.response_kind)
            if conversation_id:
                turns = self._history.setdefault(conversation_id, [])
                turns.extend((f"User: {message}", f"Assistant: {response.content}"))
                del turns[:-12]
                self._last_intent[conversation_id] = decision.intent
            return AgentResult(decision.intent, "completed", response.content, sources=response.sources,
                               tool_action_occurred=response.tool_action_occurred,
                               response_kind=response.response_kind)
        except Exception as exc:
            return AgentResult(decision.intent, "error", error=f"Handler failed: {exc}")

    def clear_conversation(self, conversation_id: str) -> None:
        """Forget context for one conversation identifier."""
        self._history.pop(conversation_id, None)
        self._last_intent.pop(conversation_id, None)
