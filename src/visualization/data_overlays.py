"""
Data Overlay Layers Engine (Part 4.2).

Provides specialized geospatial overlay layers for:
1. Fire Detection Sub-Layers:
   - Category-specific FeatureGroupSubGroup tied to parent MarkerCluster
   - Color-coded by the 6 standard classifications (Industrial, Flare, Forest, Agri, Mining, Unknown)
   - Dynamic FRP-proportional marker scaling (4px to 22px)
   - Rich interactive popup detail cards with facility proximity alerts
2. Industrial Infrastructure & Hazard Buffers:
   - FontAwesome 6 category icons and metadata popups
   - Multi-ring concentric hazard buffer zones (500m, 1km, 2km, 5km)
3. Dual-Mode Thermal Heatmaps:
   - Static FRP-weighted thermal intensity surface (HeatMap)
   - Multi-temporal time-lapse animated slider (HeatMapWithTime)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import folium
from folium.plugins import (
    FeatureGroupSubGroup,
    HeatMap,
    HeatMapWithTime,
    MarkerCluster,
)
import pandas as pd

logger = logging.getLogger(__name__)


class OverlayManager:
    """
    Manages all Part 4.2 data overlay layers for the GIS map.
    """

    # 6 Standard Fire Classification Categories & Color Codes
    FIRE_TYPE_COLORS: Dict[str, str] = {
        "Industrial Fire": "#e74c3c",       # Vibrant Red
        "Gas Flare": "#e67e22",             # High-vis Orange
        "Forest Fire": "#27ae60",           # Forest Green
        "Agricultural Burning": "#f1c40f",  # Amber Yellow
        "Mining Activity": "#7f8c8d",       # Slate Gray
        "Other/Unknown": "#95a5a6",         # Light Cool Gray
    }

    # FontAwesome Facility Icons & Colors
    FACILITY_ICONS: Dict[str, Tuple[str, str]] = {
        "oil_refinery": ("tint", "red"),
        "thermal_power_plant": ("bolt", "orange"),
        "steel_plant": ("industry", "darkred"),
        "mining": ("gem", "gray"),
        "petrochemical": ("flask", "purple"),
        "gas_flare": ("fire", "orange"),
        "lng_terminal": ("ship", "blue"),
        "petroleum_well": ("oil-can", "black"),
    }

    # Concentric Hazard Buffer Rings (radius_meters, color, opacity, label)
    HAZARD_BUFFER_RINGS = [
        (500, "#e74c3c", 0.20, "500m - Immediate Danger Zone"),
        (1000, "#e67e22", 0.12, "1km - Critical Impact Zone"),
        (2000, "#f39c12", 0.07, "2km - Thermal Exposure Zone"),
        (5000, "#3498db", 0.03, "5km - Regional Surveillance Perimeter"),
    ]

    def __init__(self):
        self.marker_registry: List[Dict[str, Any]] = []

    def _normalize_confidence(self, conf: Any) -> int:
        """Convert confidence value (low/nominal/high or 0-100) to integer 0-100."""
        if conf is None or pd.isna(conf):
            return 50
        try:
            val_float = float(conf)
            return int(max(0, min(100, val_float)))
        except (ValueError, TypeError):
            pass
        s = str(conf).strip().lower()
        if s in ["h", "high"]:
            return 85
        elif s in ["n", "nominal", "medium", "med"]:
            return 60
        elif s in ["l", "low"]:
            return 30
        return 50

    # ------------------------------------------------------------------
    # 1. Fire Detection Sub-Layers (FeatureGroupSubGroup)
    # ------------------------------------------------------------------
    def add_category_subgroups(
        self,
        m: folium.Map,
        parent_cluster: MarkerCluster,
        fire_df: pd.DataFrame,
    ) -> Dict[str, FeatureGroupSubGroup]:
        """
        Create individual toggleable FeatureGroupSubGroups for each fire category
        bound to the parent MarkerCluster.
        """
        if fire_df.empty:
            return {}

        self.marker_registry.clear()
        subgroups = {}

        # Canonical categories
        all_categories = list(self.FIRE_TYPE_COLORS.keys())

        # Also inspect any extra categories in the dataframe
        if "fire_type" in fire_df.columns:
            present_categories = [c for c in fire_df["fire_type"].unique() if pd.notna(c)]
            for c in present_categories:
                if c not in all_categories:
                    all_categories.append(c)

        for cat in all_categories:
            cat_df = fire_df[fire_df.get("fire_type", "") == cat] if "fire_type" in fire_df.columns else pd.DataFrame()
            count = len(cat_df)

            # Subgroup icon
            icon_symbol = "🔥"
            if cat == "Industrial Fire":
                icon_symbol = "🏭"
            elif cat == "Gas Flare":
                icon_symbol = "🟠"
            elif cat == "Forest Fire":
                icon_symbol = "🌲"
            elif cat == "Agricultural Burning":
                icon_symbol = "🌾"
            elif cat == "Mining Activity":
                icon_symbol = "⛏️"

            group_name = f"{icon_symbol} {cat} ({count})"
            # Industrial and Gas Flare shown by default, others toggleable
            show_default = cat in ["Industrial Fire", "Gas Flare", "Forest Fire"] if count > 0 else False

            subgroup = FeatureGroupSubGroup(
                parent_cluster,
                name=group_name,
                show=show_default if count > 0 else False,
            )

            color = self.FIRE_TYPE_COLORS.get(cat, "#95a5a6")

            for idx, row in cat_df.iterrows():
                frp = float(row.get("frp", 5.0))
                brightness = float(row.get("brightness", 320.0))
                confidence = str(row.get("confidence", "Nominal"))
                acq_date = str(row.get("acq_date", "N/A"))
                acq_time = str(row.get("acq_time", "0000")).zfill(4)
                time_formatted = f"{acq_time[:2]}:{acq_time[2:]} UTC"
                daynight = "☀️ Day" if str(row.get("daynight", "D")).upper() == "D" else "🌙 Night"
                nearest_fac = str(row.get("nearest_facility_name", "None / Rural Area"))
                dist_km = row.get("distance_to_nearest_industrial", None)
                dist_str = f"{dist_km:.2f} km" if dist_km is not None and not pd.isna(dist_km) else "N/A"
                det_id = str(row.get("detection_id", f"DET-{idx:04d}"))[:16]

                # Scaled marker radius
                marker_radius = min(max(frp / 8.0, 4.0), 22.0)

                # High hazard banner if within 2km of industrial facility
                hazard_badge = ""
                if dist_km is not None and not pd.isna(dist_km) and dist_km <= 2.0:
                    hazard_badge = f"""
                    <div style='background-color: #ffebe9; color: #cf222e; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 6px; font-size: 11px; border: 1px solid #ff8182;'>
                        ⚠️ INDUSTRIAL RISK: Within {dist_str} of {nearest_fac}
                    </div>
                    """

                popup_html = f"""
                <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; min-width: 250px; font-size: 13px; line-height: 1.4; color: #24292f;'>
                    <div style='display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 8px;'>
                        <span style='font-weight: 700; color: {color}; font-size: 14px;'>{icon_symbol} {cat}</span>
                        <span style='font-size: 10px; background: #f6f8fa; padding: 2px 6px; border-radius: 10px; border: 1px solid #d0d7de;'>{det_id}</span>
                    </div>
                    {hazard_badge}
                    <table style='width: 100%; font-size: 12px; border-collapse: collapse;'>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Coordinates:</td><td style='text-align: right; font-weight: 600;'>{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Fire Radiative Power:</td><td style='text-align: right; font-weight: 600; color: #cf222e;'>{frp:.1f} MW</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Brightness Temp:</td><td style='text-align: right; font-weight: 600;'>{brightness:.1f} K</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Detection Confidence:</td><td style='text-align: right; font-weight: 600;'>{confidence.title()}</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Acquisition Time:</td><td style='text-align: right;'>{acq_date} {time_formatted}</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Diurnal Cycle:</td><td style='text-align: right;'>{daynight}</td></tr>
                        <tr style='border-top: 1px dashed #d0d7de;'><td style='color: #57606a; padding: 4px 0 2px 0;'>Nearest Infrastructure:</td><td style='text-align: right; font-weight: 600; padding-top: 4px;'>{nearest_fac}</td></tr>
                        <tr><td style='color: #57606a; padding: 2px 0;'>Proximity Distance:</td><td style='text-align: right; font-weight: 600;'>{dist_str}</td></tr>
                    </table>
                </div>
                """

                circle = folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=marker_radius,
                    color=color,
                    weight=1.5,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.75,
                    popup=folium.Popup(popup_html, max_width=320),
                    tooltip=f"{cat} | FRP: {frp:.1f} MW | {acq_date}",
                )
                circle.add_to(subgroup)

                # Record for client-side interactive filtering
                self.marker_registry.append({
                    "marker_var": circle.get_name(),
                    "parent_var": subgroup.get_name(),
                    "data": {
                        "id": det_id,
                        "fire_type": cat,
                        "acq_date": acq_date,
                        "confidence_num": self._normalize_confidence(confidence),
                        "frp": round(frp, 1),
                    },
                })

            subgroup.add_to(m)
            subgroups[cat] = subgroup

        return subgroups

    def generate_marker_registry_script(self) -> str:
        """
        Generate JavaScript block registering all markers and their parent subgroups
        into window._fireMarkerRegistry for client-side filtering.
        """
        import json
        if not self.marker_registry:
            return ""

        lines = [
            "<script>",
            "(function() {",
            "  function registerAllMarkers() {",
            "    window._fireMarkerRegistry = [",
        ]

        for item in self.marker_registry:
            m_var = item["marker_var"]
            p_var = item["parent_var"]
            d_json = json.dumps(item["data"])
            lines.append(f"      {{ marker: {m_var}, parent: {p_var}, data: {d_json} }},")

        lines.extend([
            "    ];",
            "  }",
            "  if (document.readyState === 'loading') {",
            "    document.addEventListener('DOMContentLoaded', registerAllMarkers);",
            "  } else {",
            "    registerAllMarkers();",
            "  }",
            "})();",
            "</script>",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2. Industrial Infrastructure & Multi-Ring Buffers
    # ------------------------------------------------------------------
    def add_facility_layers(
        self,
        m: folium.Map,
        facilities_gdf: Any,
        show_facilities: bool = True,
        show_buffers: bool = False,
    ) -> Tuple[Optional[folium.FeatureGroup], Optional[folium.FeatureGroup]]:
        """
        Add industrial facilities with custom category icons and concentric hazard buffer rings.
        """
        if facilities_gdf is None or (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
            return None, None

        facility_group = folium.FeatureGroup(
            name="🏭 Industrial Infrastructure (OSM)",
            show=show_facilities,
        )

        for _, row in facilities_gdf.iterrows():
            ftype = str(row.get("facility_type", "other"))
            fname = str(row.get("name", "Industrial Facility"))
            icon_name, icon_color = self.FACILITY_ICONS.get(ftype, ("industry", "gray"))
            tags = row.get("tags", {})
            operator = ""
            if isinstance(tags, dict):
                operator = tags.get("operator", "")
            elif isinstance(tags, str) and "operator" in tags:
                operator = "Commercial Operator"

            popup_html = f"""
            <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; min-width: 220px; font-size: 13px; line-height: 1.4;'>
                <div style='font-weight: 700; color: #0969da; border-bottom: 2px solid #0969da; padding-bottom: 4px; margin-bottom: 6px; font-size: 14px;'>
                    🏭 {fname}
                </div>
                <table style='width: 100%; font-size: 12px;'>
                    <tr><td style='color: #57606a;'>Category:</td><td style='text-align: right; font-weight: 600;'>{ftype.replace('_', ' ').title()}</td></tr>
                    <tr><td style='color: #57606a;'>Coordinates:</td><td style='text-align: right;'>{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
                    {f"<tr><td style='color: #57606a;'>Operator:</td><td style='text-align: right;'>{operator}</td></tr>" if operator else ""}
                    <tr><td style='color: #57606a;'>Hazard Protection:</td><td style='text-align: right; color: #cf222e; font-weight: 600;'>500m - 5km active</td></tr>
                </table>
            </div>
            """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"🏭 {fname} ({ftype.replace('_', ' ').title()})",
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
            ).add_to(facility_group)

        facility_group.add_to(m)

        # Concentric Multi-Ring Buffer Zones
        buffer_group = folium.FeatureGroup(
            name="⭕ Multi-Ring Hazard Buffers (500m-5km)",
            show=show_buffers,
        )

        for _, row in facilities_gdf.iterrows():
            lat, lon = row["latitude"], row["longitude"]
            for radius_m, color, opacity, label in self.HAZARD_BUFFER_RINGS:
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_m,
                    color=color,
                    weight=1.2,
                    dash_array="4, 4",
                    fill=True,
                    fillColor=color,
                    fillOpacity=opacity,
                    tooltip=f"{row.get('name', 'Facility')} | {label}",
                ).add_to(buffer_group)

        buffer_group.add_to(m)

        return facility_group, buffer_group

    # ------------------------------------------------------------------
    # 3. Dual-Mode Thermal Heatmaps
    # ------------------------------------------------------------------
    def add_static_heatmap(
        self,
        m: folium.Map,
        fire_df: pd.DataFrame,
        radius: int = 20,
        blur: int = 15,
        name: str = "🌡️ Static FRP Thermal Heatmap",
        show: bool = False,
    ) -> Optional[HeatMap]:
        """
        Add static continuous FRP-weighted thermal intensity heatmap.
        """
        if fire_df.empty or "latitude" not in fire_df.columns:
            return None

        heat_points = []
        for _, row in fire_df.iterrows():
            frp = float(row.get("frp", 10.0))
            weight = min(max(frp / 60.0, 0.2), 1.0)
            heat_points.append([row["latitude"], row["longitude"], weight])

        heatmap = HeatMap(
            heat_points,
            name=name,
            min_opacity=0.35,
            radius=radius,
            blur=blur,
            gradient={
                0.2: "#0000ff",   # Blue
                0.4: "#00ffff",   # Cyan
                0.6: "#00ff00",   # Green
                0.8: "#ffff00",   # Yellow
                1.0: "#ff0000",   # Red
            },
            show=show,
        )
        heatmap.add_to(m)
        return heatmap

    def add_temporal_heatmap(
        self,
        m: folium.Map,
        fire_df: pd.DataFrame,
        radius: int = 22,
        blur: int = 16,
        auto_play: bool = False,
        name: str = "⏳ Multi-Temporal Heatmap (Time-Lapse Slider)",
        show: bool = False,
    ) -> Optional[HeatMapWithTime]:
        """
        Add interactive HeatMapWithTime animated time-lapse slider.
        Groups fire anomaly detections chronologically by acquisition date or timestamp.
        """
        if fire_df.empty or "latitude" not in fire_df.columns:
            return None

        # Build chronological groups
        df_copy = fire_df.copy()

        # Identify date column
        date_col = None
        if "acq_date" in df_copy.columns:
            date_col = "acq_date"
        elif "datetime" in df_copy.columns:
            date_col = "datetime"

        if date_col is None:
            df_copy["_step"] = "Observation Period"
            date_col = "_step"

        df_copy[date_col] = df_copy[date_col].astype(str)
        unique_dates = sorted(df_copy[date_col].unique())

        time_data = []
        time_index = []

        if len(unique_dates) > 1:
            for d in unique_dates:
                sub = df_copy[df_copy[date_col] == d]
                points = []
                for _, row in sub.iterrows():
                    frp = float(row.get("frp", 10.0))
                    weight = min(max(frp / 60.0, 0.2), 1.0)
                    points.append([row["latitude"], row["longitude"], weight])
                if points:
                    time_data.append(points)
                    time_index.append(str(d))
        else:
            # If only a single day, partition into diurnal / 4-hour time steps
            time_slices = ["00:00-06:00 UTC", "06:00-12:00 UTC", "12:00-18:00 UTC", "18:00-24:00 UTC"]
            for i, ts in enumerate(time_slices):
                # Partition by index or hour if present
                if "acq_time" in df_copy.columns:
                    hour_min = i * 600
                    hour_max = (i + 1) * 600
                    acq_series = pd.to_numeric(df_copy["acq_time"], errors="coerce").fillna(0)
                    sub = df_copy[(acq_series >= hour_min) & (acq_series < hour_max)]
                else:
                    sub = df_copy.iloc[i::len(time_slices)]

                if sub.empty:
                    sub = df_copy

                points = []
                for _, row in sub.iterrows():
                    frp = float(row.get("frp", 10.0))
                    weight = min(max(frp / 60.0, 0.2), 1.0)
                    points.append([row["latitude"], row["longitude"], weight])
                time_data.append(points)
                time_index.append(f"Pass {i+1} ({ts})")

        if not time_data:
            return None

        temporal_hm = HeatMapWithTime(
            data=time_data,
            index=time_index,
            radius=radius,
            blur=blur,
            auto_play=auto_play,
            min_opacity=0.35,
            max_opacity=0.85,
            name=name,
            show=show,
            gradient={
                0.2: "#0000ff",
                0.4: "#00ffff",
                0.6: "#00ff00",
                0.8: "#ffff00",
                1.0: "#ff0000",
            },
        )
        temporal_hm.add_to(m)
        logger.info(f"Added Temporal HeatMapWithTime with {len(time_index)} time steps.")
        return temporal_hm
