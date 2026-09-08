import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0
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


class HistoricalAnalyzer:
    """
    Longitudinal and spatial analytics engine for thermal anomalies.
    Computes monthly/quarterly aggregations, regional trends, and dynamic facility risk scores.
    """

    def __init__(self):
        pass

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure date, month, quarter, and numeric columns are populated."""
        if df.empty:
            return df

        df = df.copy()

        # Date normalization
        if "acq_date" in df.columns:
            df["parsed_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
        elif "datetime" in df.columns:
            df["parsed_date"] = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            df["parsed_date"] = pd.Timestamp.now()

        # Fill missing dates with current timestamp
        df["parsed_date"] = df["parsed_date"].fillna(pd.Timestamp.now())

        df["year_month"] = df["parsed_date"].dt.strftime("%Y-%m")
        df["year_quarter"] = df["parsed_date"].apply(lambda d: f"{d.year}-Q{(d.month - 1) // 3 + 1}")

        # Ensure numeric fields
        for col in ["frp", "brightness", "is_near_industrial", "is_daytime", "distance_to_nearest_industrial"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if "fire_type" not in df.columns:
            df["fire_type"] = "Other/Unknown"

        return df

    def compute_monthly_report(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Aggregate fire statistics grouped by year-month with MoM growth velocity.
        """
        df = self._prepare_dataframe(df)
        if df.empty:
            return []

        months = sorted(df["year_month"].unique())
        reports = []
        prev_count = None

        for ym in months:
            m_df = df[df["year_month"] == ym]
            count = len(m_df)
            avg_b = round(float(m_df["brightness"].mean()), 2) if "brightness" in m_df.columns else 0.0
            avg_f = round(float(m_df["frp"].mean()), 2) if "frp" in m_df.columns else 0.0
            total_f = round(float(m_df["frp"].sum()), 2) if "frp" in m_df.columns else 0.0
            max_f = round(float(m_df["frp"].max()), 2) if "frp" in m_df.columns and not m_df["frp"].empty else 0.0

            dist = m_df["fire_type"].value_counts().to_dict() if "fire_type" in m_df.columns else {}
            day_c = int((m_df["is_daytime"] == 1).sum()) if "is_daytime" in m_df.columns else 0
            night_c = count - day_c
            near_ind = int((m_df["is_near_industrial"] == 1).sum()) if "is_near_industrial" in m_df.columns else 0

            # Month-over-Month growth %
            mom_growth = 0.0
            if prev_count is not None and prev_count > 0:
                mom_growth = round(((count - prev_count) / prev_count) * 100.0, 1)
            prev_count = count

            reports.append({
                "period": ym,
                "total_fires": count,
                "avg_brightness_k": avg_b,
                "avg_frp_mw": avg_f,
                "total_frp_energy_mwh": total_f,
                "max_frp_mw": max_f,
                "day_fires": day_c,
                "night_fires": night_c,
                "near_industrial_count": near_ind,
                "fire_type_distribution": dist,
                "mom_growth_percent": mom_growth,
            })

        return reports

    def compute_quarterly_report(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Aggregate fire statistics grouped by calendar quarter with QoQ growth metrics.
        """
        df = self._prepare_dataframe(df)
        if df.empty:
            return []

        quarters = sorted(df["year_quarter"].unique())
        reports = []
        prev_count = None

        for yq in quarters:
            q_df = df[df["year_quarter"] == yq]
            count = len(q_df)
            avg_b = round(float(q_df["brightness"].mean()), 2) if "brightness" in q_df.columns else 0.0
            avg_f = round(float(q_df["frp"].mean()), 2) if "frp" in q_df.columns else 0.0
            total_f = round(float(q_df["frp"].sum()), 2) if "frp" in q_df.columns else 0.0
            max_f = round(float(q_df["frp"].max()), 2) if "frp" in q_df.columns and not q_df["frp"].empty else 0.0

            dist = q_df["fire_type"].value_counts().to_dict() if "fire_type" in q_df.columns else {}
            day_c = int((q_df["is_daytime"] == 1).sum()) if "is_daytime" in q_df.columns else 0
            night_c = count - day_c
            near_ind = int((q_df["is_near_industrial"] == 1).sum()) if "is_near_industrial" in q_df.columns else 0

            qoq_growth = 0.0
            if prev_count is not None and prev_count > 0:
                qoq_growth = round(((count - prev_count) / prev_count) * 100.0, 1)
            prev_count = count

            reports.append({
                "quarter": yq,
                "total_fires": count,
                "avg_brightness_k": avg_b,
                "avg_frp_mw": avg_f,
                "total_frp_energy_mwh": total_f,
                "max_frp_mw": max_f,
                "day_fires": day_c,
                "night_fires": night_c,
                "near_industrial_count": near_ind,
                "fire_type_distribution": dist,
                "qoq_growth_percent": qoq_growth,
            })

        return reports

    def compute_regional_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze thermal anomalies across key geographic surveillance zones of India:
        Northern Agricultural/Industrial, Western Refineries, Central/Eastern Mining, Southern Coastal.
        """
        df = self._prepare_dataframe(df)
        if df.empty:
            return []

        regions_def = [
            {
                "region_id": "REG-NORTH",
                "name": "Northern Agro-Industrial Zone",
                "states": "Punjab, Haryana, UP, NCR",
                "condition": lambda d: (d["latitude"] >= 25.0) & (d["longitude"] <= 84.0),
            },
            {
                "region_id": "REG-WEST",
                "name": "Western Energy & Petrochemical Corridor",
                "states": "Gujarat, Maharashtra, Offshore Bombay High",
                "condition": lambda d: (d["latitude"] < 25.0) & (d["longitude"] <= 76.0),
            },
            {
                "region_id": "REG-EAST-CENTRAL",
                "name": "Central-Eastern Mining & Steel Belt",
                "states": "Odisha, Jharkhand, Chhattisgarh, MP",
                "condition": lambda d: (d["longitude"] > 76.0) & (d["latitude"] >= 18.0),
            },
            {
                "region_id": "REG-SOUTH",
                "name": "Southern Coastal Infrastructure Zone",
                "states": "Karnataka, Kerala, Tamil Nadu, Andhra Pradesh",
                "condition": lambda d: (d["latitude"] < 18.0),
            },
        ]

        regional_results = []
        for r in regions_def:
            mask = r["condition"](df)
            r_df = df[mask]
            count = len(r_df)

            if count > 0:
                avg_b = round(float(r_df["brightness"].mean()), 2)
                avg_f = round(float(r_df["frp"].mean()), 2)
                ind_c = int((r_df["fire_type"] == "Industrial Fire").sum())
                flare_c = int((r_df["fire_type"] == "Gas Flare").sum())
                forest_c = int((r_df["fire_type"] == "Forest Fire").sum())
                agri_c = int((r_df["fire_type"] == "Agricultural Burning").sum())
                mine_c = int((r_df["fire_type"] == "Mining Activity").sum())

                # Estimate trend direction based on time distribution
                if len(r_df["year_month"].unique()) > 1:
                    monthly_counts = r_df["year_month"].value_counts().sort_index()
                    trend_slope = np.polyfit(range(len(monthly_counts)), monthly_counts.values, 1)[0]
                    if trend_slope > 0.5:
                        trend_dir = "Rising (▲)"
                    elif trend_slope < -0.5:
                        trend_dir = "Declining (▼)"
                    else:
                        trend_dir = "Stable (▬)"
                else:
                    trend_dir = "Stable (▬)"
            else:
                avg_b, avg_f = 0.0, 0.0
                ind_c = flare_c = forest_c = agri_c = mine_c = 0
                trend_dir = "No Data"

            regional_results.append({
                "region_id": r["region_id"],
                "region_name": r["name"],
                "coverage_states": r["states"],
                "total_fires": count,
                "avg_brightness_k": avg_b,
                "avg_frp_mw": avg_f,
                "industrial_fires": ind_c,
                "gas_flares": flare_c,
                "forest_fires": forest_c,
                "agricultural_burning": agri_c,
                "mining_fires": mine_c,
                "trend_status": trend_dir,
            })

        return regional_results

    def compute_facility_risk_scores(
        self,
        df: pd.DataFrame,
        facilities: Any,
        window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Compute longitudinal risk index R(f, t) for each monitored industrial asset.
        Combines spatial proximity, detection frequency, peak thermal power, and persistence duration.
        """
        df = self._prepare_dataframe(df)
        fac_list = []

        # Parse facilities into list of dictionaries
        if hasattr(facilities, "iterrows"):
            for idx, row in facilities.iterrows():
                fac_list.append({
                    "facility_id": str(row.get("facility_id") or f"FAC-{idx+1:04d}"),
                    "name": str(row.get("name", "Unknown Facility")),
                    "facility_type": str(row.get("facility_type") or row.get("type", "industrial")),
                    "latitude": float(row.geometry.y) if hasattr(row, "geometry") else float(row.get("lat", 0.0)),
                    "longitude": float(row.geometry.x) if hasattr(row, "geometry") else float(row.get("lon", 0.0)),
                })
        elif isinstance(facilities, list):
            fac_list = facilities
        else:
            return []

        facility_risks = []

        for fac in fac_list:
            f_lat = float(fac.get("latitude") or fac.get("lat", 0.0))
            f_lon = float(fac.get("longitude") or fac.get("lon", 0.0))
            f_name = str(fac.get("name", "Unknown"))
            f_type = str(fac.get("facility_type") or fac.get("type", "industrial"))
            fac_id = str(fac.get("facility_id", f_name))

            if df.empty:
                nearby_fires = pd.DataFrame()
            else:
                # Fast coordinate bounding box pre-filter (+/- 0.08 deg ~ 9km)
                bbox_fires = df[
                    (df["latitude"].between(f_lat - 0.08, f_lat + 0.08)) &
                    (df["longitude"].between(f_lon - 0.08, f_lon + 0.08))
                ]

                if not bbox_fires.empty:
                    # Precise haversine calculation
                    distances = [
                        haversine_km(f_lat, f_lon, r["latitude"], r["longitude"])
                        for _, r in bbox_fires.iterrows()
                    ]
                    nearby_mask = [d <= 5.0 for d in distances]
                    nearby_fires = bbox_fires[nearby_mask].copy()
                    nearby_fires["dist_km"] = [d for d in distances if d <= 5.0]
                else:
                    nearby_fires = pd.DataFrame()

            fire_count = len(nearby_fires)
            if fire_count > 0:
                peak_frp = float(nearby_fires["frp"].max())
                avg_dist = float(nearby_fires["dist_km"].mean())
                min_dist = float(nearby_fires["dist_km"].min())
                active_days = nearby_fires["year_month"].nunique()

                # Dynamic Multi-Factor Risk Formula (0 to 100)
                # w1: Frequency (0-35 pts)
                freq_score = min(35.0, fire_count * 3.5)
                # w2: Peak FRP Intensity (0-30 pts)
                frp_score = min(30.0, (peak_frp / 60.0) * 30.0)
                # w3: Proximity to perimeter (0-20 pts)
                dist_score = max(0.0, min(20.0, (5.0 - min_dist) * 4.0))
                # w4: Persistence duration (0-15 pts)
                pers_score = min(15.0, active_days * 5.0)

                total_risk = round(min(100.0, freq_score + frp_score + dist_score + pers_score), 1)

                if total_risk >= 75.0:
                    tier = "EXTREME RISK"
                    badge_color = "#e74c3c"
                elif total_risk >= 50.0:
                    tier = "HIGH RISK"
                    badge_color = "#e67e22"
                elif total_risk >= 25.0:
                    tier = "MODERATE RISK"
                    badge_color = "#f1c40f"
                else:
                    tier = "LOW RISK"
                    badge_color = "#27ae60"

                # Trend trajectory
                if fire_count >= 5:
                    trajectory = "Escalating (▲)"
                elif fire_count >= 2:
                    trajectory = "Stable (▬)"
                else:
                    trajectory = "De-escalating (▼)"
            else:
                peak_frp = 0.0
                avg_dist = -1.0
                min_dist = -1.0
                total_risk = 5.0
                tier = "LOW RISK"
                badge_color = "#27ae60"
                trajectory = "Nominal (▬)"

            facility_risks.append({
                "facility_id": fac_id,
                "facility_name": f_name,
                "facility_type": f_type,
                "latitude": f_lat,
                "longitude": f_lon,
                "nearby_fire_count_5km": fire_count,
                "peak_frp_mw": round(peak_frp, 1),
                "closest_fire_dist_km": round(min_dist, 2) if min_dist >= 0 else None,
                "risk_score": total_risk,
                "risk_tier": tier,
                "badge_color": badge_color,
                "risk_trajectory": trajectory,
            })

        # Sort descending by risk score
        facility_risks.sort(key=lambda r: r["risk_score"], reverse=True)
        return facility_risks

    def generate_executive_summary(self, df: pd.DataFrame, facilities: Any) -> Dict[str, Any]:
        """Generate high-level executive statistics and key takeaway metrics."""
        df = self._prepare_dataframe(df)
        monthly = self.compute_monthly_report(df)
        quarterly = self.compute_quarterly_report(df)
        regional = self.compute_regional_trends(df)
        facility_risks = self.compute_facility_risk_scores(df, facilities)

        total_fires = len(df)
        total_energy = round(float(df["frp"].sum()), 1) if not df.empty and "frp" in df.columns else 0.0
        extreme_risk_facs = [f for f in facility_risks if f["risk_tier"] == "EXTREME RISK"]
        high_risk_facs = [f for f in facility_risks if f["risk_tier"] == "HIGH RISK"]

        dominant_type = "Industrial Fire"
        if not df.empty and "fire_type" in df.columns:
            dominant_type = df["fire_type"].mode()[0] if not df["fire_type"].empty else "Industrial Fire"

        peak_record = None
        if not df.empty and "frp" in df.columns:
            max_idx = df["frp"].idxmax()
            peak_row = df.loc[max_idx]
            peak_record = {
                "detection_id": str(peak_row.get("detection_id", "")),
                "frp": float(peak_row.get("frp", 0.0)),
                "brightness": float(peak_row.get("brightness", 0.0)),
                "date": str(peak_row.get("acq_date", "")),
                "facility": str(peak_row.get("nearest_facility_name", "none")),
            }

        return {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_fires_analyzed": total_fires,
            "total_thermal_energy_mwh": total_energy,
            "monitored_facilities_count": len(facility_risks),
            "extreme_risk_facilities": len(extreme_risk_facs),
            "high_risk_facilities": len(high_risk_facs),
            "dominant_fire_type": dominant_type,
            "peak_thermal_event": peak_record,
            "monthly_periods": len(monthly),
            "quarterly_periods": len(quarterly),
            "regional_zones": len(regional),
        }
