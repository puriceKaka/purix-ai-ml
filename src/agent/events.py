"""Typed local events and an append-only SQLite event log."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"


@dataclass(frozen=True)
class AgentEvent:
    event_type: str
    source: str
    simulated: bool
    severity: EventSeverity = EventSeverity.INFO
    data: dict[str, object] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    dedupe_key: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventStore:
    """SQLite event store; defaults to memory to avoid creating workspace state."""

    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.execute("""CREATE TABLE IF NOT EXISTS agent_events (
            event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, source TEXT NOT NULL,
            simulated INTEGER NOT NULL, severity TEXT NOT NULL, data_json TEXT NOT NULL,
            correlation_id TEXT, dedupe_key TEXT UNIQUE, created_at TEXT NOT NULL)""")
        self._db.commit()

    def append(self, event: AgentEvent) -> bool:
        with self._lock:
            try:
                self._db.execute("INSERT INTO agent_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    event.event_id, event.event_type, event.source, int(event.simulated),
                    event.severity.value, json.dumps(event.data, sort_keys=True),
                    event.correlation_id, event.dedupe_key, event.created_at))
                self._db.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def list_events(self, *, simulated: bool | None = None) -> list[AgentEvent]:
        query = "SELECT * FROM agent_events"
        args: tuple[object, ...] = ()
        if simulated is not None:
            query += " WHERE simulated = ?"
            args = (int(simulated),)
        query += " ORDER BY rowid"
        with self._lock:
            rows = self._db.execute(query, args).fetchall()
        return [AgentEvent(row["event_type"], row["source"], bool(row["simulated"]),
                           EventSeverity(row["severity"]), json.loads(row["data_json"]),
                           row["event_id"], row["correlation_id"], row["dedupe_key"], row["created_at"])
                for row in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class EventBus:
    """Dispatch each event once per bus and persist before notifying listeners."""

    def __init__(self, store: EventStore | None = None) -> None:
        self.store = store
        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> None:
        self._listeners.append(listener)

    def publish(self, event: AgentEvent) -> bool:
        with self._lock:
            key = event.dedupe_key or event.event_id
            if key in self._seen:
                return False
            if self.store is not None and not self.store.append(event):
                self._seen.add(key)
                return False
            self._seen.add(key)
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # A failing observer cannot undo the durable event or block other observers.
                continue
        return True
