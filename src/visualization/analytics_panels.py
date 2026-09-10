"""
Analytics Panels Engine (Part 4.4).

Generates mission-critical spatial intelligence and visual analytics for
thermal anomaly monitoring and industrial infrastructure safety:
1. Fire Type Distribution (Pie / Donut Chart)
2. Time Series of Fire Detections (Dual-axis Frequency & FRP Energy)
3. Top Facilities by Nearby Fire Count (Proximity Impact Leaderboard)
4. Regional Summary Statistics & Risk Matrix (Comprehensive KPIs & Spatial Quadrants)

Provides both interactive Chart.js HTML dashboards and publication-grade Matplotlib PNG exports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics and reporting engine for Part 4.4.
    """

    FIRE_TYPE_COLORS: Dict[str, str] = {
        "Industrial Fire": "#e74c3c",       # Vibrant Red
        "Gas Flare": "#e67e22",             # High-vis Orange
        "Forest Fire": "#27ae60",           # Forest Green
        "Agricultural Burning": "#f1c40f",  # Amber Yellow
        "Mining Activity": "#7f8c8d",       # Slate Gray
        "Other/Unknown": "#95a5a6",         # Light Cool Gray
    }

    FACILITY_COLORS: Dict[str, str] = {
        "oil_refinery": "#e74c3c",
        "thermal_power_plant": "#e67e22",
        "steel_plant": "#c0392b",
        "mining": "#7f8c8d",
        "petrochemical": "#9b59b6",
        "gas_flare": "#d35400",
        "lng_terminal": "#2980b9",
        "petroleum_well": "#34495e",
    }

    # ------------------------------------------------------------------
    # 1. Fire Type Distribution
    # ------------------------------------------------------------------
    def compute_fire_type_distribution(self, fire_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate classification proportions, detection counts, aggregate FRP energy,
        and temperature metrics per category.
        """
        if fire_df.empty:
            return {
                "labels": list(self.FIRE_TYPE_COLORS.keys()),
                "counts": [0] * len(self.FIRE_TYPE_COLORS),
                "percentages": [0.0] * len(self.FIRE_TYPE_COLORS),
                "total_frp": [0.0] * len(self.FIRE_TYPE_COLORS),
                "avg_frp": [0.0] * len(self.FIRE_TYPE_COLORS),
                "avg_brightness": [0.0] * len(self.FIRE_TYPE_COLORS),
                "colors": list(self.FIRE_TYPE_COLORS.values()),
                "table": [],
            }

        total_fires = len(fire_df)
        labels = list(self.FIRE_TYPE_COLORS.keys())

        # Include any custom categories in dataframe
        if "fire_type" in fire_df.columns:
            for cat in fire_df["fire_type"].dropna().unique():
                if cat not in labels:
                    labels.append(str(cat))

        counts = []
        percentages = []
        total_frps = []
        avg_frps = []
        avg_brights = []
        colors = []
        table = []

        for cat in labels:
            cat_df = fire_df[fire_df.get("fire_type", "") == cat] if "fire_type" in fire_df.columns else pd.DataFrame()
            cnt = len(cat_df)
            pct = round((cnt / total_fires) * 100.0, 1) if total_fires > 0 else 0.0
            t_frp = round(float(cat_df["frp"].sum()), 1) if ("frp" in cat_df.columns and not cat_df.empty) else 0.0
            a_frp = round(float(cat_df["frp"].mean()), 1) if ("frp" in cat_df.columns and not cat_df.empty) else 0.0
            a_bri = round(float(cat_df["brightness"].mean()), 1) if ("brightness" in cat_df.columns and not cat_df.empty) else 0.0
            c = self.FIRE_TYPE_COLORS.get(cat, "#95a5a6")

            counts.append(cnt)
            percentages.append(pct)
            total_frps.append(t_frp)
            avg_frps.append(a_frp)
            avg_brights.append(a_bri)
            colors.append(c)

            table.append({
                "category": cat,
                "count": cnt,
                "percentage": pct,
                "total_frp": t_frp,
                "avg_frp": a_frp,
                "avg_brightness": a_bri,
                "color": c,
            })

        return {
            "labels": labels,
            "counts": counts,
            "percentages": percentages,
            "total_frp": total_frps,
            "avg_frp": avg_frps,
            "avg_brightness": avg_brights,
            "colors": colors,
            "table": table,
        }

    # ------------------------------------------------------------------
    # 2. Time Series of Fire Detections
    # ------------------------------------------------------------------
    def compute_time_series(self, fire_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Chronological binning across satellite pass dates:
        Tracks daily detection counts, cumulative FRP energy, and industrial vs wildfire splits.
        """
        if fire_df.empty or "acq_date" not in fire_df.columns:
            return {
                "dates": [],
                "counts": [],
                "total_frp": [],
                "avg_frp": [],
                "industrial_counts": [],
                "vegetation_counts": [],
                "hourly_labels": [f"{h:02d}:00" for h in range(24)],
                "hourly_counts": [0] * 24,
            }

        df = fire_df.copy()
        df["acq_date"] = df["acq_date"].astype(str)
        daily_group = df.groupby("acq_date")

        dates = sorted(daily_group.groups.keys())
        counts = [int(len(daily_group.get_group(d))) for d in dates]
        
        total_frps = []
        avg_frps = []
        industrial_counts = []
        vegetation_counts = []

        for d in dates:
            sub = daily_group.get_group(d)
            t_frp = float(sub["frp"].sum()) if "frp" in sub.columns else 0.0
            a_frp = float(sub["frp"].mean()) if "frp" in sub.columns else 0.0
            total_frps.append(round(t_frp, 1))
            avg_frps.append(round(a_frp, 1))

            if "fire_type" in sub.columns:
                ind_cnt = len(sub[sub["fire_type"].isin(["Industrial Fire", "Gas Flare"])])
                veg_cnt = len(sub[sub["fire_type"].isin(["Forest Fire", "Agricultural Burning"])])
            else:
                ind_cnt = 0
                veg_cnt = 0
            industrial_counts.append(ind_cnt)
            vegetation_counts.append(veg_cnt)

        # Diurnal Hourly Analysis (0-23 UTC)
        hourly_labels = [f"{h:02d}:00" for h in range(24)]
        hourly_counts = [0] * 24
        if "acq_time" in df.columns:
            for _, row in df.iterrows():
                time_str = str(row["acq_time"]).zfill(4)
                try:
                    hour = int(time_str[:2])
                    if 0 <= hour < 24:
                        hourly_counts[hour] += 1
                except ValueError:
                    pass

        # Cumulative counts
        cum_counts = []
        c_acc = 0
        for c in counts:
            c_acc += c
            cum_counts.append(c_acc)

        return {
            "dates": dates,
            "counts": counts,
            "cumulative_counts": cum_counts,
            "total_frp": total_frps,
            "avg_frp": avg_frps,
            "industrial_counts": industrial_counts,
            "vegetation_counts": vegetation_counts,
            "hourly_labels": hourly_labels,
            "hourly_counts": hourly_counts,
        }

    # ------------------------------------------------------------------
    # 3. Top Facilities by Nearby Fire Count
    # ------------------------------------------------------------------
    def compute_top_facilities(
        self,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        max_dist_km: float = 5.0,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Identifies and ranks industrial facilities by nearby thermal anomalies,
        hazard buffer containment (500m, 1km, 2km, 5km), and peak thermal intensity.
        """
        if fire_df.empty:
            return []

        facility_stats: Dict[str, Dict[str, Any]] = {}

        # 1. Primary: Use pre-computed nearest facility metadata if present
        if "nearest_facility_name" in fire_df.columns and "distance_to_nearest_industrial" in fire_df.columns:
            valid_fires = fire_df[
                fire_df["distance_to_nearest_industrial"].notna() &
                (fire_df["distance_to_nearest_industrial"] <= max_dist_km)
            ]

            for _, row in valid_fires.iterrows():
                fac_name = str(row.get("nearest_facility_name", "Unknown Facility")).strip()
                if not fac_name or fac_name in ["None / Rural Area", "None / Rural", "nan", "None"]:
                    continue

                dist = float(row.get("distance_to_nearest_industrial", 99.0))
                frp = float(row.get("frp", 5.0))
                fac_type = str(row.get("nearest_facility_type", "industrial")).replace("_", " ").title()

                if fac_name not in facility_stats:
                    facility_stats[fac_name] = {
                        "name": fac_name,
                        "facility_type": fac_type,
                        "fire_count": 0,
                        "count_500m": 0,
                        "count_1km": 0,
                        "count_2km": 0,
                        "count_5km": 0,
                        "min_distance_km": dist,
                        "max_frp": frp,
                        "total_frp": 0.0,
                    }

                st = facility_stats[fac_name]
                st["fire_count"] += 1
                st["total_frp"] += frp
                st["min_distance_km"] = min(st["min_distance_km"], dist)
                st["max_frp"] = max(st["max_frp"], frp)

                if dist <= 0.5:
                    st["count_500m"] += 1
                elif dist <= 1.0:
                    st["count_1km"] += 1
                elif dist <= 2.0:
                    st["count_2km"] += 1
                else:
                    st["count_5km"] += 1

        # 2. Secondary fallback: Use spatial coordinates if nearest metadata missing
        elif facilities_gdf is not None and not (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
            for _, fac in facilities_gdf.iterrows():
                name = str(fac.get("name", "")).strip()
                fac_type = str(fac.get("facility_type", "industrial")).replace("_", " ").title()
                lat = fac.get("latitude", None)
                lon = fac.get("longitude", None)
                if (lat is None or lon is None) and hasattr(fac, "geometry") and fac.geometry:
                    lat = fac.geometry.y
                    lon = fac.geometry.x

                if lat is None or lon is None or pd.isna(lat) or pd.isna(lon) or not name:
                    continue

                # Haversine distance to all fires
                lat_rad = np.radians(lat)
                lon_rad = np.radians(lon)
                f_lat_rad = np.radians(fire_df["latitude"].values)
                f_lon_rad = np.radians(fire_df["longitude"].values)

                dlat = f_lat_rad - lat_rad
                dlon = f_lon_rad - lon_rad
                a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad) * np.cos(f_lat_rad) * np.sin(dlon / 2.0)**2
                c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                dists_km = 6371.0 * c

                nearby_mask = dists_km <= max_dist_km
                cnt = int(np.sum(nearby_mask))

                if cnt > 0:
                    nearby_dists = dists_km[nearby_mask]
                    nearby_frp = fire_df.loc[nearby_mask, "frp"].values if "frp" in fire_df.columns else np.array([5.0] * cnt)

                    facility_stats[name] = {
                        "name": name,
                        "facility_type": fac_type,
                        "fire_count": cnt,
                        "count_500m": int(np.sum(nearby_dists <= 0.5)),
                        "count_1km": int(np.sum((nearby_dists > 0.5) & (nearby_dists <= 1.0))),
                        "count_2km": int(np.sum((nearby_dists > 1.0) & (nearby_dists <= 2.0))),
                        "count_5km": int(np.sum(nearby_dists > 2.0)),
                        "min_distance_km": float(np.min(nearby_dists)),
                        "max_frp": float(np.max(nearby_frp)),
                        "total_frp": float(np.sum(nearby_frp)),
                    }

        # Convert to sorted list and assign risk levels
        results = list(facility_stats.values())
        results.sort(key=lambda x: (x["count_500m"] * 4 + x["count_1km"] * 2 + x["fire_count"]), reverse=True)

        # Attach exact facility coordinates if available
        fac_coords = {}
        if facilities_gdf is not None and not (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
            for _, f_row in facilities_gdf.iterrows():
                f_n = str(f_row.get("name", "")).strip()
                f_lat = f_row.get("latitude", None)
                f_lon = f_row.get("longitude", None)
                if (f_lat is None or f_lon is None) and hasattr(f_row, "geometry") and f_row.geometry:
                    f_lat = f_row.geometry.y
                    f_lon = f_row.geometry.x
                if f_n and f_lat is not None and f_lon is not None and not pd.isna(f_lat) and not pd.isna(f_lon):
                    fac_coords[f_n] = (round(float(f_lat), 5), round(float(f_lon), 5))

        for item in results:
            c = fac_coords.get(item["name"])
            if c:
                item["latitude"] = c[0]
                item["longitude"] = c[1]

            item["avg_frp"] = round(item["total_frp"] / item["fire_count"], 1) if item["fire_count"] > 0 else 0.0
            item["min_distance_km"] = round(item["min_distance_km"], 2)
            item["max_frp"] = round(item["max_frp"], 1)
            item["total_frp"] = round(item["total_frp"], 1)

            # Assign Criticality Tier
            if item["count_500m"] > 0 or item["fire_count"] >= 10:
                item["risk_level"] = "CRITICAL"
                item["risk_badge_class"] = "bg-danger"
            elif item["count_1km"] > 0 or item["fire_count"] >= 5:
                item["risk_level"] = "HIGH"
                item["risk_badge_class"] = "bg-warning text-dark"
            elif item["count_2km"] > 0:
                item["risk_level"] = "MODERATE"
                item["risk_badge_class"] = "bg-info text-dark"
            else:
                item["risk_level"] = "SURVEILLANCE"
                item["risk_badge_class"] = "bg-secondary"

        return results[:top_n]

    # ------------------------------------------------------------------
    # 4. Regional Summary Statistics & Risk Matrix
    # ------------------------------------------------------------------
    def compute_regional_summary(
        self, fire_df: pd.DataFrame, facilities_gdf: Any
    ) -> Dict[str, Any]:
        """
        Computes high-level surveillance KPIs and spatial quadrant risk breakdown.
        """
        total_fires = len(fire_df)
        if total_fires == 0:
            return {
                "total_detections": 0,
                "industrial_fire_count": 0,
                "industrial_exposure_rate": 0.0,
                "avg_frp": 0.0,
                "peak_frp": 0.0,
                "avg_brightness": 0.0,
                "peak_brightness": 0.0,
                "high_confidence_count": 0,
                "high_confidence_rate": 0.0,
                "daytime_count": 0,
                "nighttime_count": 0,
                "day_percentage": 0.0,
                "quadrants": [],
            }

        # KPIs
        avg_frp = round(float(fire_df["frp"].mean()), 1) if "frp" in fire_df.columns else 0.0
        peak_frp = round(float(fire_df["frp"].max()), 1) if "frp" in fire_df.columns else 0.0
        avg_bright = round(float(fire_df["brightness"].mean()), 1) if "brightness" in fire_df.columns else 0.0
        peak_bright = round(float(fire_df["brightness"].max()), 1) if "brightness" in fire_df.columns else 0.0

        # Industrial fires
        if "fire_type" in fire_df.columns:
            ind_count = len(fire_df[fire_df["fire_type"].isin(["Industrial Fire", "Gas Flare"])])
        elif "is_near_industrial" in fire_df.columns:
            ind_count = len(fire_df[fire_df["is_near_industrial"] == 1])
        else:
            ind_count = 0
        ind_rate = round((ind_count / total_fires) * 100.0, 1)

        # High Confidence
        if "confidence" in fire_df.columns:
            high_conf = len(fire_df[
                fire_df["confidence"].astype(str).str.lower().isin(["high", "h", "100", "90", "80"])
            ])
        else:
            high_conf = 0
        high_rate = round((high_conf / total_fires) * 100.0, 1)

        # Day vs Night
        day_count = 0
        night_count = 0
        if "daynight" in fire_df.columns:
            day_count = len(fire_df[fire_df["daynight"].astype(str).str.upper() == "D"])
            night_count = len(fire_df[fire_df["daynight"].astype(str).str.upper() == "N"])
        elif "is_daytime" in fire_df.columns:
            day_count = len(fire_df[fire_df["is_daytime"] == 1])
            night_count = len(fire_df[fire_df["is_daytime"] == 0])
        day_pct = round((day_count / total_fires) * 100.0, 1) if total_fires > 0 else 0.0

        # Spatial Quadrants across India / Bounding Area
        quadrants = []
        if "latitude" in fire_df.columns and "longitude" in fire_df.columns:
            lat_mid = fire_df["latitude"].median()
            lon_mid = fire_df["longitude"].median()

            quad_defs = [
                ("North-East Zone", fire_df["latitude"] >= lat_mid, fire_df["longitude"] >= lon_mid),
                ("North-West Zone", fire_df["latitude"] >= lat_mid, fire_df["longitude"] < lon_mid),
                ("South-East Zone", fire_df["latitude"] < lat_mid, fire_df["longitude"] >= lon_mid),
                ("South-West Zone", fire_df["latitude"] < lat_mid, fire_df["longitude"] < lon_mid),
            ]

            for name, lat_cond, lon_cond in quad_defs:
                q_df = fire_df[lat_cond & lon_cond]
                q_cnt = len(q_df)
                if q_cnt == 0:
                    continue
                q_ind = len(q_df[q_df["fire_type"].isin(["Industrial Fire", "Gas Flare"])]) if "fire_type" in q_df.columns else 0
                q_frp = round(float(q_df["frp"].mean()), 1) if "frp" in q_df.columns else 0.0
                q_top_type = q_df["fire_type"].mode().iloc[0] if ("fire_type" in q_df.columns and not q_df.empty) else "Other"

                quadrants.append({
                    "region": name,
                    "fire_count": q_cnt,
                    "industrial_count": q_ind,
                    "avg_frp": q_frp,
                    "dominant_type": q_top_type,
                    "risk_status": "CRITICAL" if q_ind >= 5 else ("ELEVATED" if q_ind >= 2 else "NORMAL"),
                })

        return {
            "total_detections": total_fires,
            "industrial_fire_count": ind_count,
            "industrial_exposure_rate": ind_rate,
            "avg_frp": avg_frp,
            "peak_frp": peak_frp,
            "avg_brightness": avg_bright,
            "peak_brightness": peak_bright,
            "high_confidence_count": high_conf,
            "high_confidence_rate": high_rate,
            "daytime_count": day_count,
            "nighttime_count": night_count,
            "day_percentage": day_pct,
            "quadrants": quadrants,
        }

    # ------------------------------------------------------------------
    # 5. Full Analytics Master Payload
    # ------------------------------------------------------------------
    def generate_full_analytics(
        self, fire_df: pd.DataFrame, facilities_gdf: Any
    ) -> Dict[str, Any]:
        """
        Generate complete structured analytics dataset for API endpoints and web dashboards.
        """
        dist = self.compute_fire_type_distribution(fire_df)
        ts = self.compute_time_series(fire_df)
        top_facs = self.compute_top_facilities(fire_df, facilities_gdf, max_dist_km=5.0, top_n=10)
        regional = self.compute_regional_summary(fire_df, facilities_gdf)

        return {
            "distribution": dist,
            "fire_type_distribution": dist,
            "time_series": ts,
            "top_facilities": top_facs,
            "regional": regional,
            "regional_summary": regional,
        }

    # ------------------------------------------------------------------
    # 6. Standalone HTML Report Generation
    # ------------------------------------------------------------------
    def generate_standalone_report_html(
        self,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        output_path: Path,
    ) -> Path:
        """
        Export a standalone, responsive, dark-mode visual intelligence report
        with embedded Chart.js interactive charts.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.generate_full_analytics(fire_df, facilities_gdf)
        dist = data["fire_type_distribution"]
        ts = data["time_series"]
        top_facs = data["top_facilities"]
        reg = data["regional_summary"]

        fac_rows_html = ""
        for i, fac in enumerate(top_facs, 1):
            badge = fac.get("risk_level", "NORMAL")
            badge_color = "#cf222e" if badge == "CRITICAL" else ("#d29922" if badge == "HIGH" else "#388bfd")
            fac_rows_html += f"""
            <tr>
                <td><b>#{i}</b></td>
                <td><strong>{fac['name']}</strong></td>
                <td><span class="facility-badge">{fac['facility_type']}</span></td>
                <td class="text-center font-bold text-danger">{fac['fire_count']}</td>
                <td class="text-center text-warning">{fac['min_distance_km']} km</td>
                <td class="text-center">{fac['max_frp']} MW</td>
                <td class="text-center">
                    <span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">
                        {badge}
                    </span>
                </td>
            </tr>
            """

        cat_rows_html = ""
        for item in dist["table"]:
            cat_rows_html += f"""
            <tr>
                <td>
                    <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{item['color']}; margin-right:8px;"></span>
                    {item['category']}
                </td>
                <td class="text-center">{item['count']}</td>
                <td class="text-center">{item['percentage']}%</td>
                <td class="text-center">{item['avg_frp']} MW</td>
                <td class="text-center">{item['avg_brightness']} K</td>
            </tr>
            """

        quad_rows_html = ""
        for q in reg["quadrants"]:
            quad_rows_html += f"""
            <tr>
                <td><strong>{q['region']}</strong></td>
                <td class="text-center">{q['fire_count']}</td>
                <td class="text-center text-danger font-bold">{q['industrial_count']}</td>
                <td class="text-center">{q['avg_frp']} MW</td>
                <td class="text-center">{q['dominant_type']}</td>
                <td class="text-center font-bold" style="color: {'#cf222e' if q['risk_status'] == 'CRITICAL' else '#3fb950'};">{q['risk_status']}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 GIS Analytics & Intelligence Report | Part 4.4</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --card-border: #30363d;
            --accent-red: #e74c3c;
            --text-main: #e6edf3;
            --text-muted: #8b949e;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            padding: 24px 0;
            margin: 0;
        }}
        .report-header {{
            background: rgba(22, 27, 34, 0.95);
            border-bottom: 1px solid var(--card-border);
            padding: 20px 0;
            margin-bottom: 24px;
        }}
        .report-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .report-card:hover {{
            border-color: #58a6ff;
            box-shadow: 0 6px 24px rgba(88, 166, 255, 0.15);
        }}
        .card-header-custom {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .card-title-custom {{
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .kpi-card {{
            background: rgba(28, 35, 51, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-val {{
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 4px;
        }}
        .kpi-lbl {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .table-dark-custom {{
            width: 100%;
            color: var(--text-main);
            font-size: 0.85rem;
            border-collapse: collapse;
        }}
        .table-dark-custom th {{
            background: #21262d;
            padding: 10px;
            border-bottom: 2px solid var(--card-border);
            font-size: 0.78rem;
            text-transform: uppercase;
            color: var(--text-muted);
        }}
        .table-dark-custom td {{
            padding: 10px;
            border-bottom: 1px solid var(--card-border);
        }}
        .table-dark-custom tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        .facility-badge {{
            background: #21262d;
            border: 1px solid #30363d;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            color: #8b949e;
        }}
    </style>
</head>
<body>

    <div class="report-header">
        <div class="container">
            <div class="d-flex align-items-center justify-content-between">
                <div>
                    <h2 class="mb-1 fw-bold">🔥 Spatial Analytics & Intelligence Report</h2>
                    <p class="text-muted mb-0 font-monospace small">SIH-2026 | NTRO Challenge | Part 4.4 Analytics Panels Engine</p>
                </div>
                <div class="d-flex gap-2">
                    <a href="map.html" class="btn btn-outline-primary btn-sm"><i class="fas fa-map-marked-alt me-1"></i> Interactive GIS Map</a>
                    <a href="/" class="btn btn-outline-danger btn-sm"><i class="fas fa-tachometer-alt me-1"></i> Live Web App</a>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- KPI Summary Cards Row -->
        <div class="row g-3 mb-4">
            <div class="col-md-3 col-6">
                <div class="kpi-card" style="border-left: 4px solid #58a6ff;">
                    <div class="kpi-val text-primary">{reg['total_detections']}</div>
                    <div class="kpi-lbl">Total Thermal Events</div>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="kpi-card" style="border-left: 4px solid #e74c3c;">
                    <div class="kpi-val text-danger">{reg['industrial_fire_count']} <span style="font-size:1rem;">({reg['industrial_exposure_rate']}%)</span></div>
                    <div class="kpi-lbl">Industrial Anomaly Risk</div>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="kpi-card" style="border-left: 4px solid #f1c40f;">
                    <div class="kpi-val text-warning">{reg['avg_frp']} <span style="font-size:0.9rem;">MW</span></div>
                    <div class="kpi-lbl">Average Fire Power</div>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="kpi-card" style="border-left: 4px solid #2ea043;">
                    <div class="kpi-val text-success">{reg['high_confidence_rate']}%</div>
                    <div class="kpi-lbl">High Confidence Rate</div>
                </div>
            </div>
        </div>

        <!-- ROW 1: Distribution Chart & Category Breakdown Table -->
        <div class="row">
            <div class="col-lg-5">
                <div class="report-card">
                    <div class="card-header-custom">
                        <div class="card-title-custom"><i class="fas fa-chart-pie text-danger"></i> Fire Type Distribution</div>
                        <span class="badge bg-secondary">Part 4.4.1</span>
                    </div>
                    <div style="height: 280px; position: relative;">
                        <canvas id="fireTypeDonut"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-lg-7">
                <div class="report-card">
                    <div class="card-header-custom">
                        <div class="card-title-custom"><i class="fas fa-table text-primary"></i> Category Metrics Breakdown</div>
                        <span class="badge bg-secondary">Detailed Breakdown</span>
                    </div>
                    <div class="table-responsive">
                        <table class="table-dark-custom">
                            <thead>
                                <tr>
                                    <th>Classification</th>
                                    <th class="text-center">Detections</th>
                                    <th class="text-center">Share</th>
                                    <th class="text-center">Avg FRP</th>
                                    <th class="text-center">Brightness</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cat_rows_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- ROW 2: Time Series Analysis -->
        <div class="row">
            <div class="col-12">
                <div class="report-card">
                    <div class="card-header-custom">
                        <div class="card-title-custom"><i class="fas fa-chart-line text-warning"></i> Time Series of Thermal Anomaly Detections</div>
                        <span class="badge bg-secondary">Part 4.4.2</span>
                    </div>
                    <p class="text-muted small mb-3">Chronological surveillance displaying daily fire detection counts against total Fire Radiative Power (MW).</p>
                    <div style="height: 320px; position: relative;">
                        <canvas id="timeSeriesChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- ROW 3: Top Facilities Leaderboard -->
        <div class="row">
            <div class="col-12">
                <div class="report-card">
                    <div class="card-header-custom">
                        <div class="card-title-custom"><i class="fas fa-industry text-danger"></i> Top Facilities by Nearby Fire Count Leaderboard</div>
                        <span class="badge bg-secondary">Part 4.4.3</span>
                    </div>
                    <p class="text-muted small mb-3">Ranking of critical industrial assets with fire anomalies detected within hazardous buffer zones (&le; 5.0 km).</p>
                    <div class="table-responsive">
                        <table class="table-dark-custom">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Industrial Facility</th>
                                    <th>Type</th>
                                    <th class="text-center">Nearby Fires</th>
                                    <th class="text-center">Closest Proximity</th>
                                    <th class="text-center">Max FRP</th>
                                    <th class="text-center">Risk Level</th>
                                </tr>
                            </thead>
                            <tbody>
                                {fac_rows_html if fac_rows_html else '<tr><td colspan="7" class="text-center text-muted">No facility proximity alerts detected.</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- ROW 4: Regional Summary Statistics -->
        <div class="row">
            <div class="col-12">
                <div class="report-card">
                    <div class="card-header-custom">
                        <div class="card-title-custom"><i class="fas fa-compass text-success"></i> Regional Surveillance Risk Matrix</div>
                        <span class="badge bg-secondary">Part 4.4.4</span>
                    </div>
                    <div class="table-responsive">
                        <table class="table-dark-custom">
                            <thead>
                                <tr>
                                    <th>Surveillance Zone</th>
                                    <th class="text-center">Active Anomalies</th>
                                    <th class="text-center">Industrial Threat Count</th>
                                    <th class="text-center">Mean FRP</th>
                                    <th class="text-center">Dominant Profile</th>
                                    <th class="text-center">Threat Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {quad_rows_html if quad_rows_html else '<tr><td colspan="6" class="text-center text-muted">Regional partitioning data unavailable.</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 1. Fire Type Donut Chart
        const distData = {json.dumps(dist)};
        new Chart(document.getElementById('fireTypeDonut'), {{
            type: 'doughnut',
            data: {{
                labels: distData.labels,
                datasets: [{{
                    data: distData.counts,
                    backgroundColor: distData.colors,
                    borderWidth: 0,
                    hoverOffset: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#e6edf3', font: {{ size: 11 }} }} }},
                    tooltip: {{
                        backgroundColor: 'rgba(22, 27, 34, 0.95)',
                        borderColor: '#30363d',
                        borderWidth: 1
                    }}
                }}
            }}
        }});

        // 2. Time Series Dual-Axis Chart
        const tsData = {json.dumps(ts)};
        new Chart(document.getElementById('timeSeriesChart'), {{
            type: 'bar',
            data: {{
                labels: tsData.dates,
                datasets: [
                    {{
                        label: 'Total Detection Count',
                        data: tsData.counts,
                        backgroundColor: 'rgba(88, 166, 255, 0.65)',
                        borderColor: '#58a6ff',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Cumulative FRP (MW)',
                        data: tsData.total_frp,
                        type: 'line',
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.2)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 4,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: {{ color: '#58a6ff' }},
                        title: {{ display: true, text: 'Detection Count', color: '#58a6ff' }},
                        grid: {{ color: '#21262d' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: {{ color: '#e74c3c' }},
                        title: {{ display: true, text: 'Total FRP (MW)', color: '#e74c3c' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#e6edf3' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        output_path.write_text(html_content, encoding="utf-8")
        logger.info(f"Analytics intelligence dashboard exported successfully to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 7. High-Resolution Matplotlib PNG Export (Optional / Production)
    # ------------------------------------------------------------------
    def export_matplotlib_charts(
        self,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Render publication-quality static chart figures (PNG) using Matplotlib.
        Returns dictionary mapping chart identifiers to output file paths.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is not installed. Skipping static PNG figure generation.")
            return {}

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: Dict[str, Path] = {}

        data = self.generate_full_analytics(fire_df, facilities_gdf)
        dist = data["fire_type_distribution"]
        ts = data["time_series"]
        top_facs = data["top_facilities"]
        reg = data["regional_summary"]

        # 1. Fire Type Distribution Pie Chart
        try:
            fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")

            non_zero = [(l, c, col) for l, c, col in zip(dist["labels"], dist["counts"], dist["colors"]) if c > 0]
            if non_zero:
                labels, counts, colors = zip(*non_zero)
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=140,
                    textprops=dict(color="#e6edf3", fontsize=10),
                )
                for autotext in autotexts:
                    autotext.set_color("#ffffff")
                    autotext.set_weight("bold")
            else:
                ax.text(0.5, 0.5, "No Fire Detections", color="#8b949e", ha="center", va="center")

            ax.set_title("Thermal Anomaly Classification Distribution (Part 4.4.1)", color="#ffffff", fontsize=13, pad=15)
            pie_path = output_dir / "fire_type_distribution.png"
            plt.tight_layout()
            plt.savefig(pie_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            generated_files["fire_type_distribution"] = pie_path
        except Exception as e:
            logger.warning(f"Error generating fire_type_distribution.png: {e}")

        # 2. Time Series Dual Axis Chart
        try:
            fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
            ax1.set_facecolor("#161b22")

            dates = ts["dates"]
            counts = ts["counts"]
            frps = ts["total_frp"]

            if dates:
                x = np.arange(len(dates))
                ax1.bar(x, counts, color="#58a6ff", alpha=0.7, label="Detection Count")
                ax1.set_xlabel("Acquisition Date", color="#8b949e", fontsize=10)
                ax1.set_ylabel("Detection Count", color="#58a6ff", fontsize=10)
                ax1.tick_params(axis="y", labelcolor="#58a6ff")
                ax1.set_xticks(x)
                ax1.set_xticklabels(dates, rotation=30, color="#8b949e", fontsize=9)

                ax2 = ax1.twinx()
                ax2.plot(x, frps, color="#e74c3c", linewidth=2.5, marker="o", label="Total FRP (MW)")
                ax2.set_ylabel("Total FRP (MW)", color="#e74c3c", fontsize=10)
                ax2.tick_params(axis="y", labelcolor="#e74c3c")

                ax1.grid(True, linestyle="--", alpha=0.2, color="#30363d")
            else:
                ax1.text(0.5, 0.5, "No Temporal Observations", color="#8b949e", ha="center", va="center")

            plt.title("Time Series of Fire Detections & Radiative Power (Part 4.4.2)", color="#ffffff", fontsize=13, pad=15)
            ts_path = output_dir / "time_series_detections.png"
            plt.tight_layout()
            plt.savefig(ts_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            generated_files["time_series"] = ts_path
        except Exception as e:
            logger.warning(f"Error generating time_series_detections.png: {e}")

        # 3. Top Facilities Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d1117")
            ax.set_facecolor("#161b22")

            if top_facs:
                fac_names = [f['name'][:24] for f in reversed(top_facs)]
                fac_counts = [f['fire_count'] for f in reversed(top_facs)]
                y_pos = np.arange(len(fac_names))

                bars = ax.barh(y_pos, fac_counts, color="#e74c3c", alpha=0.85)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(fac_names, color="#e6edf3", fontsize=9)
                ax.set_xlabel("Nearby Thermal Anomalies (within 5km)", color="#8b949e", fontsize=10)
                ax.tick_params(axis="x", labelcolor="#8b949e")
                ax.grid(True, linestyle="--", alpha=0.2, color="#30363d")

                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                            va="center", color="#ffffff", fontweight="bold", fontsize=9)
            else:
                ax.text(0.5, 0.5, "No Nearby Industrial Fires", color="#8b949e", ha="center", va="center")

            plt.title("Top Facilities by Nearby Fire Count (Part 4.4.3)", color="#ffffff", fontsize=13, pad=15)
            fac_path = output_dir / "top_facilities_fires.png"
            plt.tight_layout()
            plt.savefig(fac_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            generated_files["top_facilities"] = fac_path
        except Exception as e:
            logger.warning(f"Error generating top_facilities_fires.png: {e}")

        # 4. Regional Surveillance Quadrants Chart
        try:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d1117")
            ax.set_facecolor("#161b22")

            quads = reg.get("quadrants", [])
            if quads:
                q_names = [q["region"] for q in quads]
                total_counts = [q["fire_count"] for q in quads]
                ind_counts = [q["industrial_count"] for q in quads]
                x = np.arange(len(q_names))
                width = 0.35

                ax.bar(x - width/2, total_counts, width, label="Total Detections", color="#58a6ff", alpha=0.85)
                ax.bar(x + width/2, ind_counts, width, label="Industrial Fires", color="#e74c3c", alpha=0.85)

                ax.set_xticks(x)
                ax.set_xticklabels(q_names, color="#e6edf3", fontsize=9)
                ax.tick_params(axis="y", labelcolor="#8b949e")
                ax.set_ylabel("Detection Count", color="#8b949e", fontsize=10)
                ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
                ax.grid(True, linestyle="--", alpha=0.2, color="#30363d")
            else:
                ax.text(0.5, 0.5, "No Regional Data", color="#8b949e", ha="center", va="center")

            plt.title("Regional Surveillance Quadrant Statistics (Part 4.4.4)", color="#ffffff", fontsize=13, pad=15)
            quad_path = output_dir / "regional_quadrants.png"
            plt.tight_layout()
            plt.savefig(quad_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            generated_files["regional_quadrants"] = quad_path
        except Exception as e:
            logger.warning(f"Error generating regional_quadrants.png: {e}")

        logger.info(f"Generated {len(generated_files)} high-res Matplotlib chart figures in {output_dir}")
        return generated_files
