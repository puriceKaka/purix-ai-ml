"""Deterministic simulated monitoring countdown; performs no operating-system action."""
from __future__ import annotations

import uuid
from enum import Enum

from .events import AgentEvent, EventBus, EventSeverity


class SimulationState(str, Enum):
    IDLE = "idle"
    NORMAL = "normal"
    COUNTDOWN = "countdown"
    COMPLETED = "completed"
    RECOVERED = "recovered"


class SimulatedMonitoringProvider:
    """Emits explicitly simulated events. Countdown completion is only an event."""

    def __init__(self, event_bus: EventBus, countdown_seconds: int = 200) -> None:
        if countdown_seconds < 11:
            raise ValueError("countdown_seconds must be at least 11 to emit the 10-second announcement window")
        self.event_bus = event_bus
        self.countdown_seconds = countdown_seconds
        self.state = SimulationState.IDLE
        self._scenario: str | None = None
        self._remaining = 0

    def report_normal(self) -> bool:
        if self.state in (SimulationState.COUNTDOWN, SimulationState.COMPLETED):
            return False
        self.state = SimulationState.NORMAL
        return self.event_bus.publish(AgentEvent("system.status.normal", "simulation", True,
                                                  data={"message": "Simulated system status is normal."},
                                                  dedupe_key="simulation:normal"))

    def simulate_problem(self, countdown_seconds: int | None = None) -> bool:
        seconds = self.countdown_seconds if countdown_seconds is None else countdown_seconds
        if seconds < 11:
            raise ValueError("countdown_seconds must be at least 11")
        if self.state == SimulationState.COUNTDOWN:
            return False
        self._scenario = str(uuid.uuid4())
        self._remaining = seconds
        self.state = SimulationState.COUNTDOWN
        self.event_bus.publish(AgentEvent("simulation.problem.detected", "simulation", True,
            EventSeverity.WARNING, {"message": "Simulated problem detected."}, correlation_id=self._scenario))
        self.event_bus.publish(AgentEvent("simulation.countdown.started", "simulation", True,
            data={"seconds": seconds}, correlation_id=self._scenario))
        return True

    def tick(self) -> int | None:
        if self.state != SimulationState.COUNTDOWN:
            return None
        self._remaining -= 1
        if 1 <= self._remaining <= 10:
            self.event_bus.publish(AgentEvent("announcement.requested", "simulation", True,
                data={"text": f"Simulation: {self._remaining} seconds remaining", "seconds": self._remaining},
                correlation_id=self._scenario,
                dedupe_key=f"{self._scenario}:announcement:{self._remaining}"))
        if self._remaining == 0:
            self.state = SimulationState.COMPLETED
            self.event_bus.publish(AgentEvent("simulation.shutdown.completed", "simulation", True,
                data={"system_action_taken": False, "message": "Simulated countdown completed; no system action was taken."},
                correlation_id=self._scenario))
        return self._remaining

    def recover(self) -> bool:
        if self.state == SimulationState.COUNTDOWN:
            self.event_bus.publish(AgentEvent("simulation.countdown.cancelled", "simulation", True,
                data={"message": "Simulated countdown cancelled."}, correlation_id=self._scenario))
        elif self.state != SimulationState.COMPLETED:
            return False
        self.event_bus.publish(AgentEvent("simulation.recovered", "simulation", True,
            data={"message": "Simulation recovered; no system action was taken."}, correlation_id=self._scenario))
        self.state = SimulationState.RECOVERED
        self._scenario = None
        self._remaining = 0
        return True
