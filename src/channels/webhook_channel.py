import logging
import requests
from typing import Optional

from config import settings
from src.monitoring.models import AlertEvent, AlertSeverity
from src.monitoring.channels.base import BaseAlertChannel

logger = logging.getLogger(__name__)

class WebhookAlertChannel(BaseAlertChannel):
    """
    Dispatches alerts to Slack, Discord, Microsoft Teams, or custom HTTP webhooks.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        super().__init__(name="webhook")
        self.webhook_url = webhook_url or settings.ALERT_WEBHOOK_URL

    def is_enabled(self) -> bool:
        return bool(self.webhook_url and str(self.webhook_url).startswith("http"))

    def send(self, alert: AlertEvent) -> bool:
        """Post alert JSON payload to configured webhook endpoint."""
        if not self.is_enabled():
            logger.debug("Webhook channel is not configured or disabled.")
            return False

        # Build Slack/Discord-compatible card payload
        severity_colors = {
            AlertSeverity.CRITICAL.value: "#ef4444",
            AlertSeverity.HIGH.value: "#f97316",
            AlertSeverity.WARNING.value: "#eab308",
            AlertSeverity.INFO.value: "#3b82f6",
        }
        color = severity_colors.get(alert.severity, "#f97316")

        slack_payload = {
            "text": f"🚨 *[{alert.severity}] {alert.title}*",
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Facility", "value": alert.facility_name, "short": True},
                        {"title": "Category", "value": alert.facility_type, "short": True},
                        {"title": "Distance", "value": f"{alert.distance_km:.2f} km", "short": True},
                        {"title": "FRP", "value": f"{alert.frp:.1f} MW", "short": True},
                        {"title": "Confidence", "value": f"{alert.confidence:.0f}%", "short": True},
                        {"title": "Coordinates", "value": f"{alert.latitude:.4f}, {alert.longitude:.4f}", "short": True},
                    ],
                    "footer": "SIH 2026 Industrial Thermal Anomaly Monitor",
                    "ts": alert.timestamp,
                }
            ],
            "raw_alert": alert.to_dict(),
        }

        try:
            resp = requests.post(self.webhook_url, json=slack_payload, timeout=5.0)
            if resp.status_code in [200, 201, 204]:
                logger.info(f"Webhook alert dispatched to {self.webhook_url[:30]}...")
                return True
            else:
                logger.warning(f"Webhook endpoint returned status code {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to post alert to webhook: {e}")
            return False
