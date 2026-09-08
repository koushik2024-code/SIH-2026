"""
Monitoring & Alert System (Part 5.2)
=====================================
Multi-channel alert dispatch (Email/SMTP, SMS/Twilio, Dashboard SSE/WebSockets,
Audit Logs, Webhooks) and trigger condition evaluation for industrial fire monitoring.
"""

from src.monitoring.models import AlertSeverity, TriggerType, AlertEvent
from src.monitoring.trigger_engine import TriggerConditionEngine
from src.monitoring.dispatcher import AlertDispatcher, get_alert_dispatcher
from src.monitoring.alert_engine import AlertEngine
from src.monitoring.channels import (
    BaseAlertChannel,
    EmailAlertChannel,
    SMSAlertChannel,
    DashboardAlertChannel,
    LogAlertChannel,
    WebhookAlertChannel,
)

__all__ = [
    "AlertSeverity",
    "TriggerType",
    "AlertEvent",
    "TriggerConditionEngine",
    "AlertDispatcher",
    "get_alert_dispatcher",
    "AlertEngine",
    "BaseAlertChannel",
    "EmailAlertChannel",
    "SMSAlertChannel",
    "DashboardAlertChannel",
    "LogAlertChannel",
    "WebhookAlertChannel",
]
