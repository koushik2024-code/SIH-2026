import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Optional

from config import settings
from src.monitoring.models import AlertEvent, AlertSeverity
from src.monitoring.channels.base import BaseAlertChannel

logger = logging.getLogger(__name__)

class EmailAlertChannel(BaseAlertChannel):
    """
    Delivers high-priority incident notifications via SMTP Email
    with responsive HTML templates, emergency action checklists, and map links.
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        simulation_mode: Optional[bool] = None,
    ):
        super().__init__(name="email")
        self.smtp_host = smtp_host or settings.SMTP_HOST
        self.smtp_port = smtp_port or settings.SMTP_PORT
        self.smtp_user = smtp_user or settings.SMTP_USER
        self.smtp_password = smtp_password or settings.SMTP_PASSWORD
        self.recipients = recipients or settings.ALERT_RECIPIENT_EMAILS
        self.enabled = enabled if enabled is not None else settings.ALERT_EMAIL_ENABLED
        self.simulation_mode = (
            simulation_mode if simulation_mode is not None else settings.ALERT_SIMULATION_MODE
        )

    def is_enabled(self) -> bool:
        # If enabled explicitly or if simulation mode is on, channel is considered operable
        return self.enabled or self.simulation_mode

    def _render_html_template(self, alert: AlertEvent) -> str:
        """Render dark-mode responsive HTML email body."""
        severity_colors = {
            AlertSeverity.CRITICAL.value: "#ef4444",
            AlertSeverity.HIGH.value: "#f97316",
            AlertSeverity.WARNING.value: "#eab308",
            AlertSeverity.INFO.value: "#3b82f6",
        }
        accent_color = severity_colors.get(alert.severity, "#f97316")
        maps_link = f"https://www.google.com/maps?q={alert.latitude},{alert.longitude}"

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{alert.title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
    .card {{ max-width: 640px; margin: 0 auto; background: #1e293b; border-radius: 12px; border: 1px solid #334155; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); }}
    .header {{ background: linear-gradient(135deg, {accent_color}22 0%, #1e293b 100%); border-bottom: 2px solid {accent_color}; padding: 24px; text-align: left; }}
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; background-color: {accent_color}; color: #ffffff; }}
    .title {{ font-size: 20px; font-weight: 700; margin: 12px 0 4px 0; color: #ffffff; }}
    .timestamp {{ font-size: 12px; color: #94a3b8; }}
    .content {{ padding: 24px; }}
    .alert-desc {{ background: #0f172a; border-left: 4px solid {accent_color}; padding: 14px; border-radius: 6px; font-size: 14px; line-height: 1.5; color: #e2e8f0; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
    .stat-box {{ background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }}
    .stat-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; font-weight: 600; }}
    .stat-value {{ font-size: 15px; font-weight: 700; color: #f1f5f9; }}
    .btn {{ display: inline-block; background: {accent_color}; color: #ffffff; text-decoration: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 14px; text-align: center; margin-top: 10px; }}
    .footer {{ background: #0f172a; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #334155; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">{alert.severity}</span>
      <div class="title">{alert.title}</div>
      <div class="timestamp">UTC Timestamp: {alert.timestamp} | Alert ID: {alert.alert_id}</div>
    </div>
    <div class="content">
      <div class="alert-desc">{alert.description}</div>
      <table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
        <tr>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155; width: 50%;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">At Risk Facility</div>
            <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{alert.facility_name}</div>
          </td>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155; width: 50%;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">Facility Category</div>
            <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{alert.facility_type}</div>
          </td>
        </tr>
        <tr>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">Proximity Distance</div>
            <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{alert.distance_km:.2f} km</div>
          </td>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">Radiative Power (FRP)</div>
            <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{alert.frp:.1f} MW</div>
          </td>
        </tr>
        <tr>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">Coordinates</div>
            <div style="font-size:13px; font-weight:bold; color:#38bdf8;">Lat: {alert.latitude:.4f}, Lon: {alert.longitude:.4f}</div>
          </td>
          <td style="padding: 8px; background: #0f172a; border: 1px solid #334155;">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase;">Detection Confidence</div>
            <div style="font-size:15px; font-weight:bold; color:#f8fafc;">{alert.confidence:.0f}%</div>
          </td>
        </tr>
      </table>
      <div style="text-align: center;">
        <a href="{maps_link}" class="btn" target="_blank">📍 View Anomaly on Satellite Map</a>
      </div>
    </div>
    <div class="footer">
      SIH 2026 | NTRO Challenge: AI-Based Industrial Fire & Persistent Thermal Source Monitoring System
    </div>
  </div>
</body>
</html>"""

    def send(self, alert: AlertEvent) -> bool:
        """Send HTML email via SMTP or save simulation preview."""
        if not self.is_enabled():
            logger.debug("Email alert channel is disabled.")
            return False

        html_body = self._render_html_template(alert)

        # 1. If SMTP credentials configured and not strictly in offline simulation
        if self.enabled and self.smtp_user and self.smtp_password and self.recipients:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"[{alert.severity}] {alert.title} - SIH 2026 Alert"
                msg["From"] = self.smtp_user
                msg["To"] = ", ".join(self.recipients)

                text_summary = f"[{alert.severity}] {alert.title}\nFacility: {alert.facility_name}\nDistance: {alert.distance_km:.2f}km\nFRP: {alert.frp:.1f} MW\n{alert.description}"
                msg.attach(MIMEText(text_summary, "plain"))
                msg.attach(MIMEText(html_body, "html"))

                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10.0) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.smtp_user, self.recipients, msg.as_string())

                logger.info(f"Email alert dispatched successfully to {len(self.recipients)} recipients.")
                return True
            except Exception as e:
                logger.error(f"Failed to dispatch SMTP email alert: {e}")
                # Fallback to simulation preview saving
        
        # 2. Simulation / Dry-run Mode (persist preview HTML)
        try:
            preview_dir = settings.OUTPUT_DIR
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_file = preview_dir / "latest_alert_email.html"
            with open(preview_file, "w", encoding="utf-8") as f:
                f.write(html_body)
            logger.info(f"[SIMULATION] Email alert rendered and saved to preview: {preview_file}")
            return True
        except Exception as e:
            logger.warning(f"Could not write email preview: {e}")
            return True
