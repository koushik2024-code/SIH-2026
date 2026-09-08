from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from src.monitoring.models import AlertEvent

logger = logging.getLogger(__name__)

class BaseAlertChannel(ABC):
    """Abstract base class for alert delivery channels."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if channel is configured and enabled."""
        pass

    @abstractmethod
    def send(self, alert: AlertEvent) -> bool:
        """Deliver alert through this channel. Returns True if successful."""
        pass
