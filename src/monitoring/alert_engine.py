import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import geopandas as gpd

from src.monitoring.models import AlertEvent
from src.monitoring.trigger_engine import TriggerConditionEngine
from src.monitoring.dispatcher import AlertDispatcher, get_alert_dispatcher

logger = logging.getLogger(__name__)

class AlertEngine:
    """
    Master Part 5.2 Alert System orchestrator:
    - Runs TriggerConditionEngine against active detections
    - Coordinates AlertDispatcher multi-channel notifications
    - Manages SQLite persistence and dashboard stream broadcasts
    """

    def __init__(
        self,
        dispatcher: Optional[AlertDispatcher] = None,
        trigger_engine: Optional[TriggerConditionEngine] = None,
        db: Optional[Any] = None,
    ):
        self.dispatcher = dispatcher or get_alert_dispatcher()
        self.trigger_engine = trigger_engine or TriggerConditionEngine()
        self.db = db
        if self.db is not None:
            self.dispatcher.set_database(self.db)

    def evaluate_and_dispatch(
        self,
        df: pd.DataFrame,
        facilities_gdf: Optional[gpd.GeoDataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None,
        check_cooldown: bool = True,
    ) -> List[AlertEvent]:
        """
        Evaluate detections against all 4 trigger conditions and immediately
        dispatch qualified alerts across all operational channels.
        """
        if df is None or df.empty:
            logger.info("No detections provided for alert evaluation.")
            return []

        # 1. Evaluate triggers
        triggered_alerts = self.trigger_engine.evaluate_all(
            df=df,
            facilities_gdf=facilities_gdf,
            historical_df=historical_df,
        )

        if not triggered_alerts:
            logger.info("No alert trigger conditions matched in current batch.")
            return []

        # 2. Dispatch alerts
        dispatched_alerts = self.dispatcher.dispatch_batch(
            triggered_alerts, check_cooldown=check_cooldown
        )

        logger.info(
            f"Alert Engine finished: {len(triggered_alerts)} condition(s) detected, "
            f"{len(dispatched_alerts)} alert(s) dispatched to operational channels."
        )
        return dispatched_alerts

    def trigger_test_alert(self) -> AlertEvent:
        """Generate and dispatch an immediate diagnostic test alert to all channels."""
        test_alert = self.trigger_engine.create_diagnostic_test_alert()
        self.dispatcher.dispatch_alert(test_alert, check_cooldown=False)
        return test_alert
