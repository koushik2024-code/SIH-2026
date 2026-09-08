import os
import json
import tempfile
import unittest
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.monitoring.models import AlertEvent, AlertSeverity, TriggerType
from src.monitoring.trigger_engine import TriggerConditionEngine, haversine_distance
from src.monitoring.channels.email_channel import EmailAlertChannel
from src.monitoring.channels.sms_channel import SMSAlertChannel
from src.monitoring.channels.dashboard_channel import DashboardAlertChannel
from src.monitoring.channels.log_channel import LogAlertChannel
from src.monitoring.channels.webhook_channel import WebhookAlertChannel
from src.monitoring.dispatcher import AlertDispatcher
from src.monitoring.alert_engine import AlertEngine
from src.pipeline_automation.database import FireMonitoringDatabase
from src.web.app import create_app

class TestPart52AlertSystem(unittest.TestCase):
    """Comprehensive test suite for Part 5.2 Multi-Channel Alert System."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_alerts.db")
        self.log_path = os.path.join(self.temp_dir.name, "test_alerts.log")
        self.json_path = os.path.join(self.temp_dir.name, "test_alerts.json")

        self.db = FireMonitoringDatabase(db_path=self.db_path)
        self.engine = TriggerConditionEngine()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1. Trigger Conditions Tests
    # -------------------------------------------------------------------------

    def test_trigger_condition_1_industrial_fire(self):
        """Condition 1: New industrial fire with confidence > 80%."""
        df = pd.DataFrame([
            {
                "detection_id": "DET-IND-01",
                "latitude": 22.47,
                "longitude": 70.05,
                "fire_type": "Industrial Fire",
                "fire_type_id": 0,
                "confidence": 92.0,
                "classification_confidence": 0.92,
                "frp": 38.5,
                "brightness": 350.0,
                "nearest_facility_name": "Jamnagar Refinery",
                "nearest_facility_type": "oil_refinery",
                "distance_to_nearest_industrial": 0.8,
            },
            {
                "detection_id": "DET-FOR-01",
                "latitude": 30.0,
                "longitude": 78.0,
                "fire_type": "Forest Fire",
                "fire_type_id": 2,
                "confidence": 95.0,
                "classification_confidence": 0.95,
                "frp": 50.0,
                "brightness": 340.0,
                "nearest_facility_name": "Forest Reserve",
                "nearest_facility_type": "none",
                "distance_to_nearest_industrial": 15.0,
            }
        ])

        alerts = self.engine.check_new_industrial_fires(df)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].trigger_type, TriggerType.NEW_INDUSTRIAL_FIRE.value)
        self.assertEqual(alerts[0].facility_name, "Jamnagar Refinery")
        self.assertIn(alerts[0].severity, [AlertSeverity.CRITICAL.value, AlertSeverity.HIGH.value])

    def test_trigger_condition_2_thermal_spike(self):
        """Condition 2: Unusual thermal spike at known facility."""
        df = pd.DataFrame([
            {
                "detection_id": "DET-SPIKE-01",
                "latitude": 24.18,
                "longitude": 82.66,
                "is_near_industrial": 1,
                "distance_to_nearest_industrial": 1.1,
                "nearest_facility_name": "Singrauli Super Thermal",
                "nearest_facility_type": "thermal_power_plant",
                "frp": 68.0,  # Spike >= 40 MW
                "brightness": 375.0,  # Spike >= 365 K
                "classification_confidence": 0.88,
            },
            {
                "detection_id": "DET-NORMAL-01",
                "latitude": 24.20,
                "longitude": 82.70,
                "is_near_industrial": 1,
                "distance_to_nearest_industrial": 1.2,
                "nearest_facility_name": "Singrauli Super Thermal",
                "nearest_facility_type": "thermal_power_plant",
                "frp": 15.0,  # Normal
                "brightness": 320.0,  # Normal
                "classification_confidence": 0.85,
            }
        ])

        alerts = self.engine.check_thermal_spikes(df)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].trigger_type, TriggerType.FACILITY_THERMAL_SPIKE.value)
        self.assertEqual(alerts[0].severity, AlertSeverity.CRITICAL.value)
        self.assertEqual(alerts[0].facility_name, "Singrauli Super Thermal")

    def test_trigger_condition_3_cluster_formation(self):
        """Condition 3: New fire cluster formation near industrial zone."""
        # 3 nearby fires near industrial site (<5km)
        df = pd.DataFrame([
            {
                "detection_id": "DET-CLUST-01",
                "latitude": 22.300,
                "longitude": 82.700,
                "distance_to_nearest_industrial": 2.2,
                "nearest_facility_name": "Korba Super Thermal",
                "nearest_facility_type": "thermal_power_plant",
                "frp": 25.0,
                "brightness": 330.0,
                "fire_cluster_id": 101,
            },
            {
                "detection_id": "DET-CLUST-02",
                "latitude": 22.305,
                "longitude": 82.703,
                "distance_to_nearest_industrial": 2.3,
                "nearest_facility_name": "Korba Super Thermal",
                "nearest_facility_type": "thermal_power_plant",
                "frp": 30.0,
                "brightness": 335.0,
                "fire_cluster_id": 101,
            },
            {
                "detection_id": "DET-CLUST-03",
                "latitude": 22.302,
                "longitude": 82.705,
                "distance_to_nearest_industrial": 2.4,
                "nearest_facility_name": "Korba Super Thermal",
                "nearest_facility_type": "thermal_power_plant",
                "frp": 28.0,
                "brightness": 332.0,
                "fire_cluster_id": 101,
            }
        ])

        alerts = self.engine.check_cluster_formation(df)
        self.assertGreaterEqual(len(alerts), 1)
        self.assertEqual(alerts[0].trigger_type, TriggerType.INDUSTRIAL_CLUSTER_FORMATION.value)
        self.assertEqual(alerts[0].facility_name, "Korba Super Thermal")

    def test_trigger_condition_4_persistent_fires(self):
        """Condition 4: Persistent fire burning for > 48 hours."""
        hist_df = pd.DataFrame([{
            "detection_id": "HIST-01",
            "latitude": 22.470,
            "longitude": 70.050,
            "acq_date": "2026-09-01",
            "nearest_facility_name": "Jamnagar Flare Vent",
            "nearest_facility_type": "gas_flare",
            "frp": 20.0,
            "brightness": 340.0,
        }])

        current_df = pd.DataFrame([{
            "detection_id": "CURR-01",
            "latitude": 22.471,
            "longitude": 70.051,
            "acq_date": "2026-09-05",  # 4 days later (> 48h)
            "nearest_facility_name": "Jamnagar Flare Vent",
            "nearest_facility_type": "gas_flare",
            "distance_to_nearest_industrial": 0.5,
            "frp": 22.0,
            "brightness": 342.0,
        }])

        alerts = self.engine.check_persistent_fires(current_df, hist_df)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].trigger_type, TriggerType.PERSISTENT_FIRE_48H.value)
        self.assertEqual(alerts[0].facility_name, "Jamnagar Flare Vent")

    # -------------------------------------------------------------------------
    # 2. Alert Channels Tests
    # -------------------------------------------------------------------------

    def test_email_alert_channel(self):
        """Test HTML email generation and preview output."""
        email_channel = EmailAlertChannel(simulation_mode=True)
        self.assertTrue(email_channel.is_enabled())

        alert = AlertEvent(
            title="Test Critical Industrial Fire",
            severity=AlertSeverity.CRITICAL.value,
            facility_name="IOCL Koyali Refinery",
            facility_type="oil_refinery",
            distance_km=0.35,
            frp=45.0,
            latitude=22.36,
            longitude=73.12,
            confidence=90.0,
            description="Testing email rendering."
        )

        sent = email_channel.send(alert)
        self.assertTrue(sent)

        # Check HTML template rendering
        html = email_channel._render_html_template(alert)
        self.assertIn("IOCL Koyali Refinery", html)
        self.assertIn("CRITICAL", html)
        self.assertIn("45.0 MW", html)

    def test_sms_alert_channel(self):
        """Test SMS body formatting and simulation dispatch."""
        sms_channel = SMSAlertChannel(simulation_mode=True)
        self.assertTrue(sms_channel.is_enabled())

        alert = AlertEvent(
            title="Thermal Flare Event",
            severity=AlertSeverity.HIGH.value,
            facility_name="Dahej LNG Terminal",
            distance_km=0.6,
            frp=32.0,
            confidence=85.0,
            latitude=21.70,
            longitude=72.58,
        )

        body = sms_channel._format_sms_body(alert)
        self.assertIn("SIH ALERT", body)
        self.assertIn("Dahej LNG Terminal", body)
        self.assertTrue(sms_channel.send(alert))

    def test_dashboard_alert_channel_sse(self):
        """Test Dashboard SSE channel streaming and in-memory history."""
        dash_channel = DashboardAlertChannel(max_history=10)
        self.assertTrue(dash_channel.is_enabled())

        alert = AlertEvent(
            title="Dashboard Broadcast Test",
            severity=AlertSeverity.WARNING.value,
            facility_name="TATA Steel Jamshedpur",
        )

        q = dash_channel.register_listener()
        dash_channel.send(alert)

        self.assertFalse(q.empty())
        raw_msg = q.get_nowait()
        data = json.loads(raw_msg)
        self.assertEqual(data["facility_name"], "TATA Steel Jamshedpur")

        recent = dash_channel.get_recent_alerts(limit=5)
        self.assertEqual(len(recent), 1)
        dash_channel.unregister_listener(q)

    def test_log_alert_channel(self):
        """Test LogAlertChannel writes to log file and output JSON."""
        log_channel = LogAlertChannel(log_path=self.log_path, json_path=self.json_path)
        alert = AlertEvent(
            title="Log Audit Test",
            facility_name="Singrauli Power",
            severity=AlertSeverity.HIGH.value,
        )

        sent = log_channel.send(alert)
        self.assertTrue(sent)
        self.assertTrue(os.path.exists(self.log_path))
        self.assertTrue(os.path.exists(self.json_path))

        with open(self.json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["facility_name"], "Singrauli Power")

    # -------------------------------------------------------------------------
    # 3. Dispatcher, Cooldown & Database Tests
    # -------------------------------------------------------------------------

    def test_cooldown_suppression(self):
        """Test that non-critical duplicate alerts for the same facility are suppressed."""
        dispatcher = AlertDispatcher(cooldown_minutes=60.0, db=self.db)

        alert1 = AlertEvent(
            alert_id="ALT-COOL-1",
            title="Initial Warning",
            severity=AlertSeverity.WARNING.value,
            trigger_type=TriggerType.NEW_INDUSTRIAL_FIRE.value,
            facility_name="Vizag Steel Plant",
        )

        # First alert should dispatch and record
        dispatched1 = dispatcher.dispatch_alert(alert1, check_cooldown=True)
        self.assertTrue(dispatched1)

        # Second alert with same facility within 60 mins should be suppressed
        alert2 = AlertEvent(
            alert_id="ALT-COOL-2",
            title="Duplicate Warning",
            severity=AlertSeverity.WARNING.value,
            trigger_type=TriggerType.NEW_INDUSTRIAL_FIRE.value,
            facility_name="Vizag Steel Plant",
        )
        dispatched2 = dispatcher.dispatch_alert(alert2, check_cooldown=True)
        self.assertFalse(dispatched2)

        # CRITICAL alert should bypass cooldown
        alert3 = AlertEvent(
            alert_id="ALT-COOL-3",
            title="Escalated Critical Event",
            severity=AlertSeverity.CRITICAL.value,
            trigger_type=TriggerType.NEW_INDUSTRIAL_FIRE.value,
            facility_name="Vizag Steel Plant",
        )
        dispatched3 = dispatcher.dispatch_alert(alert3, check_cooldown=True)
        self.assertTrue(dispatched3)

    def test_sqlite_alert_persistence_and_stats(self):
        """Test SQLite alert storage, retrieval, acknowledgment, and stats."""
        alert = AlertEvent(
            alert_id="ALT-DB-1",
            title="DB Test Fire",
            severity=AlertSeverity.CRITICAL.value,
            trigger_type=TriggerType.NEW_INDUSTRIAL_FIRE.value,
            facility_name="Talcher Power Plant",
            status="ACTIVE",
            channels_dispatched=["email", "sms", "dashboard", "log"],
        )

        inserted = self.db.insert_alert(alert.to_dict())
        self.assertTrue(inserted)

        # Retrieve
        alerts = self.db.get_alerts(limit=10, severity="CRITICAL")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["facility_name"], "Talcher Power Plant")
        self.assertEqual(alerts[0]["status"], "ACTIVE")

        # Stats
        stats = self.db.get_alert_statistics()
        self.assertEqual(stats["total_alerts"], 1)
        self.assertEqual(stats["active_alerts"], 1)
        self.assertEqual(stats["critical_active_alerts"], 1)

        # Acknowledge
        acked = self.db.acknowledge_alert("ALT-DB-1", acknowledged_by="unit_tester")
        self.assertTrue(acked)

        updated_stats = self.db.get_alert_statistics()
        self.assertEqual(updated_stats["active_alerts"], 0)

    # -------------------------------------------------------------------------
    # 4. Web Dashboard & API Endpoints Tests
    # -------------------------------------------------------------------------

    def test_flask_alert_endpoints(self):
        """Test Flask REST & streaming endpoints for alerts."""
        app = create_app()
        app.config['TESTING'] = True
        client = app.test_client()

        # 1. Test alert dispatch endpoint
        test_res = client.post('/api/alerts/test')
        self.assertEqual(test_res.status_code, 200)
        test_data = test_res.get_json()
        self.assertEqual(test_data["status"], "dispatched")
        alert_id = test_data["alert"]["alert_id"]

        # 2. Get alerts endpoint
        get_res = client.get('/api/alerts')
        self.assertEqual(get_res.status_code, 200)
        alerts_list = get_res.get_json()
        self.assertGreaterEqual(len(alerts_list), 1)

        # 3. Get alert stats endpoint
        stats_res = client.get('/api/alerts/stats')
        self.assertEqual(stats_res.status_code, 200)
        stats_data = stats_res.get_json()
        self.assertIn("total_alerts", stats_data)

        # 4. Acknowledge alert endpoint
        ack_res = client.post(f'/api/alerts/{alert_id}/ack', json={"user": "test_agent"})
        self.assertEqual(ack_res.status_code, 200)
        self.assertEqual(ack_res.get_json()["status"], "success")

        # 5. Dedicated /alerts page route
        page_res = client.get('/alerts')
        self.assertEqual(page_res.status_code, 200)
        self.assertIn(b"INCIDENT OPERATIONS CENTER", page_res.data)

if __name__ == "__main__":
    unittest.main()
