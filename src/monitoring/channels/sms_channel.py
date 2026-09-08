import logging
import requests
from typing import List, Optional

from config import settings
from src.monitoring.models import AlertEvent
from src.monitoring.channels.base import BaseAlertChannel

logger = logging.getLogger(__name__)

class SMSAlertChannel(BaseAlertChannel):
    """
    Delivers urgent operational alerts via SMS (Twilio REST API)
    with concise, mission-critical messages formatted for emergency first responders.
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        phone_numbers: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        simulation_mode: Optional[bool] = None,
    ):
        super().__init__(name="sms")
        self.account_sid = account_sid or settings.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or settings.TWILIO_AUTH_TOKEN
        self.from_number = from_number or settings.TWILIO_FROM_NUMBER
        self.phone_numbers = phone_numbers or settings.ALERT_PHONE_NUMBERS
        self.enabled = enabled if enabled is not None else settings.ALERT_SMS_ENABLED
        self.simulation_mode = (
            simulation_mode if simulation_mode is not None else settings.ALERT_SIMULATION_MODE
        )

    def is_enabled(self) -> bool:
        return self.enabled or self.simulation_mode

    def _format_sms_body(self, alert: AlertEvent) -> str:
        """Format a concise emergency SMS string (< 160 characters when possible)."""
        return (
            f"[SIH ALERT] {alert.severity}: {alert.title}. "
            f"Near: {alert.facility_name} ({alert.distance_km:.1f}km). "
            f"FRP: {alert.frp:.0f}MW, Conf: {alert.confidence:.0f}%. "
            f"Coords: {alert.latitude:.3f},{alert.longitude:.3f}. ID: {alert.alert_id}"
        )

    def send(self, alert: AlertEvent) -> bool:
        """Send SMS via Twilio API or record in simulation log."""
        if not self.is_enabled():
            logger.debug("SMS alert channel is disabled.")
            return False

        message_body = self._format_sms_body(alert)

        # 1. Live Twilio API dispatch if credentials configured
        if self.enabled and self.account_sid and self.auth_token and self.from_number and self.phone_numbers:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            success = True
            for phone in self.phone_numbers:
                try:
                    resp = requests.post(
                        url,
                        data={
                            "From": self.from_number,
                            "To": phone,
                            "Body": message_body,
                        },
                        auth=(self.account_sid, self.auth_token),
                        timeout=8.0,
                    )
                    if resp.status_code in [200, 201]:
                        logger.info(f"SMS alert dispatched to {phone}: {resp.json().get('sid')}")
                    else:
                        logger.warning(f"Twilio SMS delivery returned status {resp.status_code}: {resp.text}")
                        success = False
                except Exception as e:
                    logger.error(f"Twilio SMS request failed for {phone}: {e}")
                    success = False
            return success

        # 2. Simulation / Dry-run Mode
        logger.info(f"[SIMULATION SMS] Dispatching to {len(self.phone_numbers) or 1} recipient(s): '{message_body}'")
        return True
