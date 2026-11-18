import tempfile
import unittest
from pathlib import Path

from src.agent.events import AgentEvent, EventBus, EventStore
from src.agent.orchestrator import AgentOrchestrator, Intent
from src.agent.routing import KeywordRequestRouter
from src.agent.simulation import SimulatedMonitoringProvider, SimulationState


class AgentFoundationTests(unittest.TestCase):
    def test_routing_is_conservative(self):
        router = KeywordRequestRouter()
        self.assertEqual(router.route("hey, tell me a joke").intent, Intent.CONVERSATION)
        self.assertEqual(router.route("How do I review failed login security logs?").intent, Intent.CYBERSECURITY)
        self.assertEqual(router.route("cybber security").intent, Intent.CYBERSECURITY)
        self.assertEqual(router.route("security").intent, Intent.CYBERSECURITY)
        self.assertEqual(router.route("Explain a Python function").intent, Intent.CONVERSATION)

    def test_orchestrator_requires_handlers_and_reports_failures(self):
        agent = AgentOrchestrator(KeywordRequestRouter())
        self.assertEqual(agent.handle("hello").status, "unavailable")
        agent.register(Intent.CONVERSATION, lambda message: "handled: " + message)
        self.assertEqual(agent.handle("hello").content, "handled: hello")
        agent.register(Intent.CONVERSATION, lambda _: (_ for _ in ()).throw(RuntimeError("failure")))
        self.assertIn("failure", agent.handle("hello").error)
        self.assertEqual(agent.handle("  ").status, "invalid")

    def test_event_store_preserves_simulated_and_actual_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(str(Path(directory) / "events.sqlite"))
            bus = EventBus(store)
            bus.publish(AgentEvent("test.simulated", "test", True, dedupe_key="sim"))
            bus.publish(AgentEvent("test.actual", "test", False, dedupe_key="actual"))
            self.assertEqual([event.simulated for event in store.list_events()], [True, False])
            self.assertEqual(len(store.list_events(simulated=True)), 1)
            store.close()

    def test_duplicate_event_is_not_dispatched_twice(self):
        seen = []
        bus = EventBus()
        bus.subscribe(seen.append)
        event = AgentEvent("test.event", "test", True, dedupe_key="same")
        self.assertTrue(bus.publish(event))
        self.assertFalse(bus.publish(event))
        self.assertEqual(seen, [event])

    def test_simulated_countdown_and_recovery_never_take_system_action(self):
        store = EventStore()
        self.addCleanup(store.close)
        bus = EventBus(store)
        provider = SimulatedMonitoringProvider(bus, countdown_seconds=12)
        self.assertTrue(provider.report_normal())
        self.assertFalse(provider.report_normal())
        self.assertTrue(provider.simulate_problem())
        self.assertFalse(provider.simulate_problem())
        for _ in range(2):
            provider.tick()
        self.assertEqual(provider.state, SimulationState.COUNTDOWN)
        self.assertTrue(provider.recover())
        self.assertEqual(provider.state, SimulationState.RECOVERED)
        self.assertFalse(any(e.event_type == "simulation.shutdown.completed" for e in store.list_events()))

    def test_announcements_are_once_per_second_and_completion_is_simulated(self):
        store = EventStore()
        self.addCleanup(store.close)
        provider = SimulatedMonitoringProvider(EventBus(store), countdown_seconds=11)
        provider.simulate_problem()
        for _ in range(11):
            provider.tick()
        events = store.list_events()
        announcements = [e for e in events if e.event_type == "announcement.requested"]
        self.assertEqual([e.data["seconds"] for e in announcements], list(range(10, 0, -1)))
        completed = [e for e in events if e.event_type == "simulation.shutdown.completed"]
        self.assertEqual(len(completed), 1)
        self.assertTrue(completed[0].simulated)
        self.assertFalse(completed[0].data["system_action_taken"])


if __name__ == "__main__":
    unittest.main()
