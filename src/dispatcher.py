import logging
from typing import List, Optional, Dict, Any

from config import settings
from src.monitoring.models import AlertEvent, AlertSeverity
from src.monitoring.channels.base import BaseAlertChannel
from src.monitoring.channels.email_channel import EmailAlertChannel
from src.monitoring.channels.sms_channel import SMSAlertChannel
from src.monitoring.channels.dashboard_channel import DashboardAlertChannel
from src.monitoring.channels.log_channel import LogAlertChannel
from src.monitoring.channels.webhook_channel import WebhookAlertChannel

logger = logging.getLogger(__name__)

class AlertDispatcher:
    """
    Coordinates multi-channel alert delivery, deduplication, cooldown suppression,
    and audit trail logging.
    """

    def __init__(
        self,
        cooldown_minutes: Optional[float] = None,
        db: Optional[Any] = None,
        custom_channels: Optional[List[BaseAlertChannel]] = None,
    ):
        self.cooldown_minutes = (
            cooldown_minutes
            if cooldown_minutes is not None
            else settings.ALERT_COOLDOWN_MINUTES
        )
        self.db = db

        # Shared singleton-style channels
        if custom_channels is not None:
            self.channels = custom_channels
        else:
            self.dashboard_channel = DashboardAlertChannel()
            self.log_channel = LogAlertChannel()
            self.email_channel = EmailAlertChannel()
            self.sms_channel = SMSAlertChannel()
            self.webhook_channel = WebhookAlertChannel()

            self.channels: List[BaseAlertChannel] = [
                self.dashboard_channel,
                self.log_channel,
                self.email_channel,
                self.sms_channel,
                self.webhook_channel,
            ]

    def set_database(self, db: Any):
        """Bind persistent SQLite database to dispatcher."""
        self.db = db

    def dispatch_alert(self, alert: AlertEvent, check_cooldown: bool = True) -> bool:
        """
        Dispatch a single alert across all configured channels with cooldown checking.
        Returns True if alert was dispatched, False if suppressed or failed.
        """
        # 1. Cooldown & Deduplication check
        if check_cooldown and self.db is not None:
            if hasattr(self.db, "has_recent_alert"):
                # Always allow CRITICAL alerts through immediately
                if alert.severity != AlertSeverity.CRITICAL.value:
                    if self.db.has_recent_alert(
                        alert.facility_name, alert.trigger_type, self.cooldown_minutes
                    ):
                        logger.info(
                            f"Suppressing duplicate alert for {alert.facility_name} "
                            f"({alert.trigger_type}) within {self.cooldown_minutes}min cooldown window."
                        )
                        return False

        # 2. Dispatch across active channels
        dispatched_channels: List[str] = []
        for channel in self.channels:
            try:
                if channel.is_enabled():
                    success = channel.send(alert)
                    if success:
                        dispatched_channels.append(channel.name)
            except Exception as e:
                logger.error(f"Error dispatching alert {alert.alert_id} via channel {channel.name}: {e}")

        alert.channels_dispatched = dispatched_channels

        # 3. Persist to SQLite Database
        if self.db is not None and hasattr(self.db, "insert_alert"):
            try:
                self.db.insert_alert(alert.to_dict())
            except Exception as e:
                logger.error(f"Failed to record alert {alert.alert_id} in database: {e}")

        logger.info(
            f"Alert [{alert.severity}] '{alert.title}' dispatched to channels: {dispatched_channels}"
        )
        return True

    def dispatch_batch(self, alerts: List[AlertEvent], check_cooldown: bool = True) -> List[AlertEvent]:
        """Dispatch a list of alerts and return those successfully sent."""
        dispatched = []
        for a in alerts:
            if self.dispatch_alert(a, check_cooldown=check_cooldown):
                dispatched.append(a)
        return dispatched


# Module-level shared singleton dispatcher
_global_dispatcher: Optional[AlertDispatcher] = None

def get_alert_dispatcher() -> AlertDispatcher:
    """Obtain or initialize the global shared AlertDispatcher instance."""
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = AlertDispatcher()
    return _global_dispatcher
