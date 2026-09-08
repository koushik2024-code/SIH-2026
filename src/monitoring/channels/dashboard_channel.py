import json
import queue
import logging
from typing import List, Dict, Any, Generator

from src.monitoring.models import AlertEvent
from src.monitoring.channels.base import BaseAlertChannel

logger = logging.getLogger(__name__)

class DashboardAlertChannel(BaseAlertChannel):
    """
    Delivers real-time alerts to the Interactive Web Dashboard.
    Maintains active SSE (Server-Sent Events) listener queues and in-memory caches.
    """

    def __init__(self, max_history: int = 100):
        super().__init__(name="dashboard")
        self.max_history = max_history
        self._history: List[AlertEvent] = []
        self._subscribers: List[queue.Queue] = []

    def is_enabled(self) -> bool:
        return True

    def send(self, alert: AlertEvent) -> bool:
        """Broadcast alert to all active dashboard SSE queues and store in history."""
        # 1. Update in-memory history (latest first)
        self._history.insert(0, alert)
        if len(self._history) > self.max_history:
            self._history.pop()

        # 2. Push to all active browser SSE streams
        payload = json.dumps(alert.to_dict())
        dead_queues = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead_queues.append(q)

        for dq in dead_queues:
            if dq in self._subscribers:
                self._subscribers.remove(dq)

        logger.info(f"Dashboard alert published to {len(self._subscribers)} active stream listener(s).")
        return True

    def register_listener(self) -> queue.Queue:
        """Register a new client listener queue for Server-Sent Events."""
        q = queue.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unregister_listener(self, q: queue.Queue):
        """Unregister a client listener queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most recent alert event dictionaries."""
        return [a.to_dict() for a in self._history[:limit]]

    def sse_event_stream(self) -> Generator[str, None, None]:
        """Yield Server-Sent Events (SSE) formatted data chunks for Flask streaming."""
        client_q = self.register_listener()
        try:
            # Yield initial connection message
            yield "event: connected\ndata: {\"status\": \"connected\"}\n\n"
            while True:
                try:
                    data = client_q.get(timeout=25.0)
                    yield f"event: alert\ndata: {data}\n\n"
                except queue.Empty:
                    # Keep-alive heartbeat ping every 25 seconds
                    yield ": ping\n\n"
        finally:
            self.unregister_listener(client_q)
