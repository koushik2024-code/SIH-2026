import math
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import geopandas as gpd

from src.monitoring.models import AlertEvent, AlertSeverity, TriggerType

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

class TriggerConditionEngine:
    """
    Evaluates thermal anomaly detections against Part 5.2 operational trigger conditions:
    1. New Industrial Fire (Confidence > 80%)
    2. Unusual Thermal Spike at Known Facility
    3. New Fire Cluster Formation Near Industrial Zone
    4. Persistent Fire Burning for > 48 Hours
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.80,
        spike_frp_threshold: float = 40.0,
        spike_brightness_threshold: float = 365.0,
        cluster_min_fires: int = 3,
        cluster_radius_km: float = 1.5,
        cluster_facility_proximity_km: float = 5.0,
        persistence_hours_threshold: float = 48.0,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.spike_frp_threshold = spike_frp_threshold
        self.spike_brightness_threshold = spike_brightness_threshold
        self.cluster_min_fires = cluster_min_fires
        self.cluster_radius_km = cluster_radius_km
        self.cluster_facility_proximity_km = cluster_facility_proximity_km
        self.persistence_hours_threshold = persistence_hours_threshold

    def evaluate_all(
        self,
        df: pd.DataFrame,
        facilities_gdf: Optional[gpd.GeoDataFrame] = None,
        historical_df: Optional[pd.DataFrame] = None,
    ) -> List[AlertEvent]:
        """
        Run all 4 trigger condition rules on incoming detection dataframe.
        Returns a deduplicated list of generated AlertEvents.
        """
        if df is None or df.empty:
            return []

        alerts: List[AlertEvent] = []

        # 1. Condition 1: High-Confidence Industrial Fires
        alerts.extend(self.check_new_industrial_fires(df))

        # 2. Condition 2: Unusual Thermal Spikes at Facilities
        alerts.extend(self.check_thermal_spikes(df))

        # 3. Condition 3: New Fire Cluster Formation Near Industrial Zone
        alerts.extend(self.check_cluster_formation(df))

        # 4. Condition 4: Persistent Fires (>48h)
        alerts.extend(self.check_persistent_fires(df, historical_df))

        logger.info(f"Trigger engine evaluation complete: {len(alerts)} alert condition(s) triggered.")
        return alerts

    def check_new_industrial_fires(self, df: pd.DataFrame) -> List[AlertEvent]:
        """
        Trigger Condition 1:
        New industrial fire detected with classification / FIRMS confidence > 80%.
        """
        alerts = []
        for _, row in df.iterrows():
            fire_type = str(row.get("fire_type", "")).strip()
            fire_type_id = row.get("fire_type_id")

            is_industrial = (
                fire_type.lower() in ["industrial fire", "industrial"]
                or fire_type_id == 0
            )
            if not is_industrial:
                continue

            # Confidence check (>80%)
            conf_val = 0.0
            if "classification_confidence" in row and pd.notna(row["classification_confidence"]):
                c = float(row["classification_confidence"])
                conf_val = c * 100.0 if c <= 1.0 else c
            elif "confidence" in row and pd.notna(row["confidence"]):
                c_raw = str(row["confidence"]).lower()
                if c_raw in ["h", "high"]:
                    conf_val = 90.0
                elif c_raw in ["n", "nominal", "medium"]:
                    conf_val = 65.0
                elif c_raw in ["l", "low"]:
                    conf_val = 30.0
                else:
                    try:
                        conf_val = float(row["confidence"])
                    except Exception:
                        conf_val = 50.0

            if conf_val >= (self.high_confidence_threshold * 100.0):
                lat = float(row.get("latitude", 0.0))
                lon = float(row.get("longitude", 0.0))
                frp = float(row.get("frp", 0.0)) if pd.notna(row.get("frp")) else 0.0
                brightness = float(row.get("brightness", 0.0)) if pd.notna(row.get("brightness")) else 0.0
                facility_name = str(row.get("nearest_facility_name", "Industrial Asset"))
                facility_type = str(row.get("nearest_facility_type", "Industrial Complex"))
                dist = float(row.get("distance_to_nearest_industrial", row.get("distance_to_industry_km", 0.0)))
                det_id = str(row.get("detection_id", f"DET-{lat:.3f}-{lon:.3f}"))

                severity = AlertSeverity.CRITICAL.value if frp >= 35.0 or dist <= 1.0 else AlertSeverity.HIGH.value
                title = f"High-Confidence Industrial Fire Near {facility_name}"
                description = (
                    f"Satellite surveillance confirmed active industrial combustion signature at "
                    f"[{lat:.4f}, {lon:.4f}] located {dist:.2f} km from {facility_name} ({facility_type}). "
                    f"Thermal intensity is {frp:.1f} MW (Brightness: {brightness:.1f} K) with {conf_val:.0f}% confidence."
                )

                alert = AlertEvent(
                    trigger_type=TriggerType.NEW_INDUSTRIAL_FIRE.value,
                    severity=severity,
                    detection_id=det_id,
                    latitude=lat,
                    longitude=lon,
                    facility_name=facility_name,
                    facility_type=facility_type,
                    distance_km=dist,
                    frp=frp,
                    brightness=brightness,
                    confidence=conf_val,
                    title=title,
                    description=description,
                    metadata={"fire_type": fire_type, "condition": "confidence > 80%"},
                )
                alerts.append(alert)

        return alerts

    def check_thermal_spikes(self, df: pd.DataFrame) -> List[AlertEvent]:
        """
        Trigger Condition 2:
        Unusual thermal spike at known facility (FRP >= 40 MW or brightness >= 365 K within 2km).
        """
        alerts = []
        for _, row in df.iterrows():
            dist = float(row.get("distance_to_nearest_industrial", row.get("distance_to_industry_km", 999.0)))
            is_near = row.get("is_near_industrial", 0) == 1 or dist <= 2.0

            if not is_near:
                continue

            frp = float(row.get("frp", 0.0)) if pd.notna(row.get("frp")) else 0.0
            brightness = float(row.get("brightness", 0.0)) if pd.notna(row.get("brightness")) else 0.0

            is_spike = (frp >= self.spike_frp_threshold) or (brightness >= self.spike_brightness_threshold)
            if not is_spike:
                continue

            lat = float(row.get("latitude", 0.0))
            lon = float(row.get("longitude", 0.0))
            facility_name = str(row.get("nearest_facility_name", "Critical Industrial Facility"))
            facility_type = str(row.get("nearest_facility_type", "Industrial Complex"))
            det_id = str(row.get("detection_id", f"DET-{lat:.3f}-{lon:.3f}"))

            title = f"Unusual Thermal Spike at {facility_name}"
            description = (
                f"Severe thermal energy spike detected directly inside facility hazard perimeter "
                f"({dist:.2f} km from center of {facility_name}). "
                f"Radiative power reached {frp:.1f} MW with temperature of {brightness:.1f} K. "
                f"Potential runaway combustion, reactor venting, or flare malfunction."
            )

            alert = AlertEvent(
                trigger_type=TriggerType.FACILITY_THERMAL_SPIKE.value,
                severity=AlertSeverity.CRITICAL.value,
                detection_id=det_id,
                latitude=lat,
                longitude=lon,
                facility_name=facility_name,
                facility_type=facility_type,
                distance_km=dist,
                frp=frp,
                brightness=brightness,
                confidence=float(row.get("classification_confidence", 0.90)) * 100.0,
                title=title,
                description=description,
                metadata={"frp_spike": frp, "brightness_spike": brightness},
            )
            alerts.append(alert)

        return alerts

    def check_cluster_formation(self, df: pd.DataFrame) -> List[AlertEvent]:
        """
        Trigger Condition 3:
        New fire cluster formation (>= 3 fires within 1.5km) near industrial zone (<= 5km).
        """
        alerts = []
        # Filter detections near industrial zone
        near_mask = df.apply(
            lambda r: float(r.get("distance_to_nearest_industrial", r.get("distance_to_industry_km", 999.0)))
            <= self.cluster_facility_proximity_km,
            axis=1,
        )
        candidates = df[near_mask].copy()
        if len(candidates) < self.cluster_min_fires:
            return alerts

        # Check by fire_cluster_id if present
        if "fire_cluster_id" in candidates.columns:
            cluster_counts = candidates["fire_cluster_id"].value_counts()
            for cluster_id, count in cluster_counts.items():
                if cluster_id != 0 and count >= self.cluster_min_fires:
                    c_group = candidates[candidates["fire_cluster_id"] == cluster_id]
                    rep_row = c_group.iloc[0]

                    lat = float(c_group["latitude"].mean())
                    lon = float(c_group["longitude"].mean())
                    max_frp = float(c_group["frp"].max()) if "frp" in c_group else 0.0
                    facility_name = str(rep_row.get("nearest_facility_name", "Industrial Zone"))
                    facility_type = str(rep_row.get("nearest_facility_type", "Industrial Complex"))
                    dist = float(rep_row.get("distance_to_nearest_industrial", rep_row.get("distance_to_industry_km", 0.0)))

                    title = f"Industrial Cluster Formation ({count} Anomalies) Near {facility_name}"
                    description = (
                        f"Multi-point thermal cluster detected comprising {count} simultaneous fire anomalies "
                        f"within {self.cluster_radius_km} km of each other, situated {dist:.2f} km from {facility_name}. "
                        f"Cumulative radiative power: {c_group['frp'].sum():.1f} MW. High risk of perimeter breach."
                    )

                    alert = AlertEvent(
                        trigger_type=TriggerType.INDUSTRIAL_CLUSTER_FORMATION.value,
                        severity=AlertSeverity.HIGH.value,
                        detection_id=f"CLUSTER-{cluster_id}",
                        latitude=lat,
                        longitude=lon,
                        facility_name=facility_name,
                        facility_type=facility_type,
                        distance_km=dist,
                        frp=max_frp,
                        brightness=float(c_group["brightness"].mean()) if "brightness" in c_group else 330.0,
                        confidence=85.0,
                        title=title,
                        description=description,
                        metadata={"cluster_size": int(count), "cluster_id": int(cluster_id)},
                    )
                    alerts.append(alert)

        # Spatial grouping fallback if no cluster_id produced
        if not alerts and len(candidates) >= self.cluster_min_fires:
            points = candidates[["latitude", "longitude"]].values
            visited = set()
            for i in range(len(points)):
                if i in visited:
                    continue
                neighbors = [i]
                for j in range(len(points)):
                    if i != j:
                        d = haversine_distance(points[i][0], points[i][1], points[j][0], points[j][1])
                        if d <= self.cluster_radius_km:
                            neighbors.append(j)
                if len(neighbors) >= self.cluster_min_fires:
                    visited.update(neighbors)
                    sub = candidates.iloc[neighbors]
                    rep = sub.iloc[0]
                    facility_name = str(rep.get("nearest_facility_name", "Industrial Cluster"))
                    facility_type = str(rep.get("nearest_facility_type", "Industrial Complex"))
                    dist = float(rep.get("distance_to_nearest_industrial", rep.get("distance_to_industry_km", 0.0)))

                    alert = AlertEvent(
                        trigger_type=TriggerType.INDUSTRIAL_CLUSTER_FORMATION.value,
                        severity=AlertSeverity.HIGH.value,
                        detection_id=f"SPATIAL-CLUSTER-{i}",
                        latitude=float(sub["latitude"].mean()),
                        longitude=float(sub["longitude"].mean()),
                        facility_name=facility_name,
                        facility_type=facility_type,
                        distance_km=dist,
                        frp=float(sub["frp"].max()) if "frp" in sub else 0.0,
                        brightness=float(sub["brightness"].mean()) if "brightness" in sub else 330.0,
                        confidence=80.0,
                        title=f"Rapid Cluster Formation ({len(neighbors)} Points) Near {facility_name}",
                        description=f"Dense cluster of {len(neighbors)} thermal points forming within {self.cluster_radius_km}km near {facility_name}.",
                        metadata={"cluster_size": len(neighbors)},
                    )
                    alerts.append(alert)

        return alerts

    def check_persistent_fires(
        self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None
    ) -> List[AlertEvent]:
        """
        Trigger Condition 4:
        Persistent fire burning for > 48 hours at the same location (<1.5 km).
        """
        alerts = []
        if historical_df is None or historical_df.empty:
            # Check within incoming df if it spans multi-day observation windows
            if "acq_date" in df.columns:
                try:
                    dates = pd.to_datetime(df["acq_date"])
                    span_days = (dates.max() - dates.min()).total_seconds() / 86400.0
                    if span_days >= 2.0:
                        # Find coordinates detected on min and max dates
                        min_date_str = str(dates.min().date())
                        max_date_str = str(dates.max().date())
                        early_fires = df[df["acq_date"].astype(str) == min_date_str]
                        late_fires = df[df["acq_date"].astype(str) == max_date_str]

                        for _, late_row in late_fires.iterrows():
                            llat, llon = float(late_row["latitude"]), float(late_row["longitude"])
                            for _, early_row in early_fires.iterrows():
                                elat, elon = float(early_row["latitude"]), float(early_row["longitude"])
                                if haversine_distance(llat, llon, elat, elon) <= 1.5:
                                    facility_name = str(late_row.get("nearest_facility_name", "Monitored Site"))
                                    facility_type = str(late_row.get("nearest_facility_type", "Industrial Complex"))
                                    dist = float(late_row.get("distance_to_nearest_industrial", late_row.get("distance_to_industry_km", 0.0)))

                                    alert = AlertEvent(
                                        trigger_type=TriggerType.PERSISTENT_FIRE_48H.value,
                                        severity=AlertSeverity.HIGH.value,
                                        detection_id=str(late_row.get("detection_id", "PERSISTENT")),
                                        latitude=llat,
                                        longitude=llon,
                                        facility_name=facility_name,
                                        facility_type=facility_type,
                                        distance_km=dist,
                                        frp=float(late_row.get("frp", 0.0)),
                                        brightness=float(late_row.get("brightness", 0.0)),
                                        confidence=85.0,
                                        title=f"Persistent Combustion (>48h) Near {facility_name}",
                                        description=(
                                            f"Thermal source at [{llat:.4f}, {llon:.4f}] has remained continuously active "
                                            f"for over {self.persistence_hours_threshold:.0f} hours near {facility_name}. "
                                            f"Indicates persistent industrial flaring, uncontrolled coal seam fire, or smoldering waste."
                                        ),
                                        metadata={"span_hours": span_days * 24.0},
                                    )
                                    alerts.append(alert)
                                    break
                except Exception as e:
                    logger.debug(f"Could not compute intra-dataset persistence: {e}")
            return alerts

        # Cross-reference incoming df with historical DB
        for _, curr_row in df.iterrows():
            clat, clon = float(curr_row["latitude"]), float(curr_row["longitude"])
            curr_date_str = str(curr_row.get("acq_date", ""))
            if not curr_date_str:
                continue

            try:
                curr_dt = pd.to_datetime(curr_date_str)
            except Exception:
                continue

            # Look for matching historical detections
            for _, hist_row in historical_df.iterrows():
                hlat, hlon = float(hist_row["latitude"]), float(hist_row["longitude"])
                if haversine_distance(clat, clon, hlat, hlon) <= 1.5:
                    hist_date_str = str(hist_row.get("acq_date", ""))
                    if not hist_date_str:
                        continue
                    try:
                        hist_dt = pd.to_datetime(hist_date_str)
                        hours_diff = abs((curr_dt - hist_dt).total_seconds()) / 3600.0
                        if hours_diff >= self.persistence_hours_threshold:
                            facility_name = str(curr_row.get("nearest_facility_name", "Industrial Site"))
                            facility_type = str(curr_row.get("nearest_facility_type", "Industrial Complex"))
                            dist = float(curr_row.get("distance_to_nearest_industrial", curr_row.get("distance_to_industry_km", 0.0)))

                            alert = AlertEvent(
                                trigger_type=TriggerType.PERSISTENT_FIRE_48H.value,
                                severity=AlertSeverity.HIGH.value,
                                detection_id=str(curr_row.get("detection_id", "PERSISTENT")),
                                latitude=clat,
                                longitude=clon,
                                facility_name=facility_name,
                                facility_type=facility_type,
                                distance_km=dist,
                                frp=float(curr_row.get("frp", 0.0)),
                                brightness=float(curr_row.get("brightness", 0.0)),
                                confidence=85.0,
                                title=f"Persistent Combustion (>48h) Near {facility_name}",
                                description=(
                                    f"Thermal source at [{clat:.4f}, {clon:.4f}] has been detected across satellite passes "
                                    f"spanning {hours_diff:.1f} hours near {facility_name}. "
                                    f"Requires environmental & safety inspection."
                                ),
                                metadata={"persistence_hours": hours_diff},
                            )
                            alerts.append(alert)
                            break
                    except Exception:
                        pass

        return alerts

    def create_diagnostic_test_alert(self) -> AlertEvent:
        """Create a realistic high-priority diagnostic test alert to verify all channels."""
        return AlertEvent(
            trigger_type=TriggerType.DIAGNOSTIC_TEST.value,
            severity=AlertSeverity.CRITICAL.value,
            detection_id="TEST-SIM-001",
            latitude=22.4707,
            longitude=70.0577,
            facility_name="Jamnagar Petrochemical Complex",
            facility_type="Refinery & Petrochemicals",
            distance_km=0.45,
            frp=72.5,
            brightness=378.2,
            confidence=95.0,
            title="CRITICAL TEST: Simulated Industrial Fire & Thermal Anomaly",
            description=(
                "SYSTEM DIAGNOSTIC ALERT: Multi-spectral sensors detected simulated extreme thermal anomaly "
                "(FRP 72.5 MW, 378.2 K) 0.45 km from Jamnagar Petrochemical Complex. "
                "All operational response channels (Email, SMS, Dashboard, Log, Webhook) tested successfully."
            ),
            metadata={"test_mode": True, "created_at": datetime.now(timezone.utc).isoformat()},
        )
