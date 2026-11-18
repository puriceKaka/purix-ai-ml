"""Conservative deterministic routing; this is not an intent-understanding model."""
from __future__ import annotations

import re
from .orchestrator import Intent, RouteDecision


class KeywordRequestRouter:
    """Route clearly defensive-security language; default to conversation."""

    _security = re.compile(
        r"\b(ddos|distributed denial(?:[- ]of[- ])?service|denial(?:[- ]of[- ])?service|"
        r"cyb{1,2}er\s?security|security incident|incident response|security log|"
        r"failed login|phishing|malware|vulnerabilit(?:y|ies)|firewall|"
        r"endpoint detection|security alert|suspicious activity|security control)\b",
        re.IGNORECASE,
    )
    _bare_security = re.compile(r"^\s*security\s*[?.!]*\s*$", re.IGNORECASE)
    _follow_up = re.compile(
        r"^\s*(?:(?:and|so|then)\s+)?(?:what does that mean|what does it mean|"
        r"what about that|tell me more|explain that|why|how so|what should i do next|"
        r"can you elaborate|then what)\b",
        re.IGNORECASE,
    )

    def route(self, message: str) -> RouteDecision:
        if self._security.search(message) or self._bare_security.fullmatch(message):
            return RouteDecision(Intent.CYBERSECURITY, "matched defensive-security terminology")
        return RouteDecision(Intent.CONVERSATION, "default route")

    def route_with_context(self, message: str, previous_intent: Intent | None) -> RouteDecision:
        decision = self.route(message)
        if (decision.intent == Intent.CONVERSATION and previous_intent == Intent.CYBERSECURITY
                and self._follow_up.search(message)):
            return RouteDecision(Intent.CYBERSECURITY, "short follow-up to a cybersecurity conversation")
        return decision
