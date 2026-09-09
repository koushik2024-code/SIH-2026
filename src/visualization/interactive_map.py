"""
Interactive GIS Map Engine (Part 4.1).

Production-grade web-based GIS interface using Folium/Leaflet.js.
Features:
- 4 Base Map Tile Layers: OpenStreetMap, ESRI Satellite, CartoDB Dark Matter, Topographic/Terrain
- Interactive GIS Controls: LayerControl, Fullscreen, MeasureControl, MousePosition HUD, MiniMap
- Data Overlays:
    * Clustered, FRP-proportional, color-coded fire anomaly markers
    * OpenStreetMap industrial infrastructure with category icons
    * Multi-ring facility hazard buffer zones (500m, 1km, 2km, 5km)
    * FRP-weighted thermal intensity heatmap
    * Floating glassmorphic GIS legend panel
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import folium
from folium.plugins import (
    HeatMap,
    MarkerCluster,
    MiniMap,
    Fullscreen,
    MeasureControl,
    MousePosition,
)
import branca.element
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from config import settings
from .data_overlays import OverlayManager
from .dashboard_controls import DashboardControlManager

logger = logging.getLogger(__name__)


class InteractiveGISMap:
    """
    Production-grade Folium/Leaflet GIS visualization engine.
    Implements Part 4.1 & Part 4.2 requirements for SIH-2026.
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

    def __init__(
        self,
        center: Tuple[float, float] = (22.5, 78.5),
        zoom_start: int = 5,
        enable_temporal_slider: bool = False,
        enable_controls: bool = True,
    ):
        self.center = center
        self.zoom_start = zoom_start
        self.enable_temporal_slider = enable_temporal_slider
        self.enable_controls = enable_controls
        self.overlay_manager = OverlayManager()
        self.dashboard_controls = DashboardControlManager()

    # ------------------------------------------------------------------
    # 1. Base Map Tile Layers (Part 4.1 Requirement)
    # ------------------------------------------------------------------
    def _add_base_layers(self, m: folium.Map) -> None:
        """
        Add the 4 required base map layers:
        1. OpenStreetMap (Default cartographic view)
        2. ESRI Satellite Imagery (High-resolution optical inspection)
        3. CartoDB Dark Matter (High-contrast thermal dark basemap)
        4. Topographic / Terrain (Elevation contours & land relief)
        """
        # 1. OpenStreetMap (Default)
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="🗺️ OpenStreetMap (Default)",
            control=True,
            show=True,
        ).add_to(m)

        # 2. ESRI World Imagery (Satellite)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics, USDA FSA, USGS, Aerogrid, IGN, IGP, and the GIS User Community",
            name="🛰️ ESRI Satellite Imagery",
            control=True,
            show=False,
            max_zoom=19,
        ).add_to(m)

        # 3. Dark Basemap (High-contrast night & thermal contrast - No API key watermark)
        carto_key = getattr(settings, "CARTO_API_KEY", "") or os.getenv("CARTO_API_KEY", "")
        if carto_key:
            dark_tiles = f"https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png?api_key={carto_key}"
            dark_attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            dark_name = "🌙 CartoDB Dark Matter"
            dark_sub = "abcd"
            dark_zoom = 20
        else:
            dark_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            dark_attr = "&copy; <a href='https://www.esri.com/'>Esri</a> &mdash; Esri, DeLorme, NAVTEQ"
            dark_name = "🌙 ESRI Dark Canvas (No Key Required)"
            dark_sub = "abc"
            dark_zoom = 18

        folium.TileLayer(
            tiles=dark_tiles,
            attr=dark_attr,
            name=dark_name,
            control=True,
            show=False,
            subdomains=dark_sub,
            max_zoom=dark_zoom,
        ).add_to(m)

        # 4. Topographic / Terrain View (OpenTopoMap with ESRI Topo fallback)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, HERE, Garmin, Intermap, increment P Corp., GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), (c) OpenStreetMap contributors, and the GIS User Community",
            name="🏔️ Topographic / Terrain",
            control=True,
            show=False,
            max_zoom=18,
        ).add_to(m)

    # ------------------------------------------------------------------
    # 2. Advanced Interactive GIS Controls
    # ------------------------------------------------------------------
    def _add_gis_controls(self, m: folium.Map, enable_layer_control: bool = False) -> None:
        """
        Add advanced GIS navigation, inspection, and measurement controls.
        """
        # Fullscreen toggle for command centers
        Fullscreen(
            position="topleft",
            title="Expand to Fullscreen",
            title_cancel="Exit Fullscreen",
            force_separate_button=True,
        ).add_to(m)

        # Interactive distance and area measurement tool
        MeasureControl(
            position="topleft",
            primary_length_unit="kilometers",
            secondary_length_unit="meters",
            primary_area_unit="sqkilometers",
            secondary_area_unit="sqmeters",
            active_color="#e74c3c",
            completed_color="#27ae60",
        ).add_to(m)

        # Real-time Mouse Coordinate HUD (Latitude & Longitude)
        MousePosition(
            position="bottomright",
            separator="  |  ",
            empty_string="Move cursor on map",
            lng_first=False,
            num_digits=4,
            prefix="Lat/Lon: ",
        ).add_to(m)

        # Overview MiniMap (start minimized so it doesn't block map canvas)
        MiniMap(
            tile_layer=folium.TileLayer("openstreetmap"),
            position="bottomright",
            width=150,
            height=110,
            collapsed_width=25,
            collapsed_height=25,
            zoom_level_offset=-5,
            toggle_display=True,
            minimized=True,
        ).add_to(m)

        # Inject styling for pinned coordinates and global map helpers
        nav_helpers = """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        <style>
        /* Suppress default bottom-left scale bar (200 mi) to avoid collision with status bar */
        .leaflet-control-scale,
        .leaflet-control-scale-line {
            display: none !important;
        }

        /* Pinned, high-visibility coordinates badge in bottom-right corner */
        .leaflet-control-mouseposition {
            position: fixed !important;
            bottom: 14px !important;
            right: 58px !important;
            background: rgba(14, 21, 36, 0.94) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid #223048 !important;
            border-radius: 8px !important;
            color: #38bdf8 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            padding: 5px 12px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
            z-index: 999 !important;
            pointer-events: auto !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
        }
        .leaflet-control-mouseposition:before {
            content: "📍";
            font-size: 12px;
        }
        /* Dark glassmorphic styling for Leaflet left toolbar controls */
        .leaflet-left .leaflet-control {
            border: 1px solid #30363d !important;
            background: rgba(22, 27, 34, 0.95) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
            overflow: hidden !important;
            margin-left: 14px !important;
        }
        .leaflet-left .leaflet-control-zoom a {
            background-color: rgba(22, 27, 34, 0.95) !important;
            color: #38bdf8 !important;
            border-bottom: 1px solid #30363d !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            width: 32px !important;
            height: 32px !important;
            text-decoration: none !important;
        }
        .leaflet-left .leaflet-control-zoom a:hover {
            background-color: #21262d !important;
            color: #ffffff !important;
        }

        /* 1. Fullscreen Button: Crisp SVG vector icon (Bright Cyan Blue / White) */
        .leaflet-control-fullscreen a,
        .leaflet-control-fullscreen a.leaflet-control-fullscreen-button,
        .leaflet-control-fullscreen .fullscreen-icon,
        .fullscreen-icon,
        .leaflet-touch .fullscreen-icon,
        .leaflet-touch .leaflet-control-fullscreen a {
            background-color: rgba(22, 27, 34, 0.95) !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2338bdf8' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-decoration: none !important;
            width: 32px !important;
            height: 32px !important;
            border-bottom: 1px solid #30363d !important;
            cursor: pointer !important;
        }
        .leaflet-control-fullscreen a:hover,
        .leaflet-control-fullscreen .fullscreen-icon:hover,
        .fullscreen-icon:hover {
            background-color: #21262d !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3'/%3E%3C/svg%3E") !important;
        }
        .leaflet-fullscreen-on .leaflet-control-fullscreen a,
        .leaflet-fullscreen-on .fullscreen-icon {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f97316' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14h6m0 0v6m0-6L3 21m17-7h-6m0 0v6m0-6l7 7M10 4v6m0 0H4m6 0L3 3m10 7h6m-6 0V4m0 6l7-7'/%3E%3C/svg%3E") !important;
        }

        /* 2. Measure Tool: Toolbar toggle button */
        .leaflet-control-measure > a.leaflet-control-measure-toggle,
        .leaflet-control-measure-toggle {
            background-color: rgba(22, 27, 34, 0.95) !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2338bdf8' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.3 8.7 8.7 21.3c-.4.4-1 .4-1.4 0l-4.6-4.6c-.4-.4-.4-1 0-1.4L15.3 2.7c.4-.4 1-.4 1.4 0l4.6 4.6c.4.4.4 1 0 1.4z'/%3E%3Cpath d='m14.5 3.5 1.5 1.5'/%3E%3Cpath d='m11.5 6.5 2 2'/%3E%3Cpath d='m8.5 9.5 1.5 1.5'/%3E%3Cpath d='m5.5 12.5 2 2'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-decoration: none !important;
            width: 32px !important;
            height: 32px !important;
            border-bottom: 1px solid #30363d !important;
            cursor: pointer !important;
        }
        .leaflet-control-measure > a.leaflet-control-measure-toggle:hover,
        .leaflet-control-measure-toggle:hover {
            background-color: #21262d !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.3 8.7 8.7 21.3c-.4.4-1 .4-1.4 0l-4.6-4.6c-.4-.4-.4-1 0-1.4L15.3 2.7c.4-.4 1-.4 1.4 0l4.6 4.6c.4.4.4 1 0 1.4z'/%3E%3Cpath d='m14.5 3.5 1.5 1.5'/%3E%3Cpath d='m11.5 6.5 2 2'/%3E%3Cpath d='m8.5 9.5 1.5 1.5'/%3E%3Cpath d='m5.5 12.5 2 2'/%3E%3C/svg%3E") !important;
        }

        /* Measure Tool: Interaction Popup Dialog (Clean typography, properly sized) */
        .leaflet-control-measure .leaflet-control-measure-interaction {
            background: rgba(22, 27, 34, 0.98) !important;
            color: #f0f6fc !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
            min-width: 220px !important;
            max-width: 250px !important;
        }
        .leaflet-control-measure-interaction h3 {
            color: #f0f6fc !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            margin: 0 0 8px 0 !important;
            padding-bottom: 6px !important;
            border-bottom: 1px solid #30363d !important;
        }
        .leaflet-control-measure-interaction p {
            color: #8b949e !important;
            font-size: 11px !important;
            margin: 4px 0 !important;
        }
        .leaflet-control-measure-interaction ul.tasks {
            list-style: none !important;
            margin: 6px 0 !important;
            padding: 0 !important;
        }
        .leaflet-control-measure-interaction .tasks-item {
            margin: 4px 0 !important;
        }
        .leaflet-control-measure-interaction a.js-start,
        .leaflet-control-measure-interaction a.js-cancel,
        .leaflet-control-measure-interaction a.js-finish {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            background: rgba(56, 189, 248, 0.12) !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            border-radius: 6px !important;
            color: #38bdf8 !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            padding: 6px 12px !important;
            width: auto !important;
            height: auto !important;
            text-decoration: none !important;
            transition: all 0.2s ease !important;
            line-height: 1.3 !important;
        }
        .leaflet-control-measure-interaction a.js-start:hover,
        .leaflet-control-measure-interaction a.js-finish:hover {
            background: #0284c7 !important;
            color: #ffffff !important;
            border-color: #38bdf8 !important;
        }
        .leaflet-control-measure-interaction a.js-cancel {
            background: rgba(239, 68, 68, 0.12) !important;
            border-color: rgba(239, 68, 68, 0.35) !important;
            color: #f87171 !important;
        }
        .leaflet-control-measure-interaction a.js-cancel:hover {
            background: #dc2626 !important;
            color: #ffffff !important;
        }

        /* 3. Geocoder Search Control: Icon button by default; expands only on click */
        .leaflet-control-geocoder {
            background: rgba(22, 27, 34, 0.95) !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
            overflow: hidden !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            margin-top: 6px !important;
            width: 32px !important;
            height: 32px !important;
            display: block !important;
        }
        .leaflet-control-geocoder button,
        .leaflet-control-geocoder-icon {
            background-color: rgba(22, 27, 34, 0.95) !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2338bdf8' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            border: none !important;
            width: 32px !important;
            height: 32px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            outline: none !important;
            filter: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .leaflet-control-geocoder button:hover,
        .leaflet-control-geocoder-icon:hover {
            background-color: #21262d !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
        }
        /* Hide form when collapsed - bar only shows when clicked */
        .leaflet-control-geocoder .leaflet-control-geocoder-form {
            display: none !important;
        }
        .leaflet-control-geocoder.leaflet-control-geocoder-expanded {
            width: auto !important;
            height: 32px !important;
            border-color: #38bdf8 !important;
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.25) !important;
            display: flex !important;
            align-items: center !important;
        }
        .leaflet-control-geocoder.leaflet-control-geocoder-expanded .leaflet-control-geocoder-form {
            display: flex !important;
            align-items: center !important;
            padding: 0 6px 0 0 !important;
        }
        .leaflet-control-geocoder.leaflet-control-geocoder-expanded input {
            color: #f0f6fc !important;
            background: transparent !important;
            border: none !important;
            font-size: 12px !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            padding: 4px 8px !important;
            outline: none !important;
            width: 240px !important;
            caret-color: #38bdf8 !important;
        }
        .leaflet-control-geocoder.leaflet-control-geocoder-expanded input::placeholder {
            color: #8b949e !important;
            font-size: 11px !important;
        }
        .leaflet-control-geocoder-alternatives {
            background: rgba(14, 21, 36, 0.98) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            margin-top: 6px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.65) !important;
            max-width: 300px !important;
            list-style: none !important;
            padding: 4px 0 !important;
            overflow: hidden !important;
        }
        .leaflet-control-geocoder-alternatives li {
            padding: 8px 12px !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
            color: #c9d1d9 !important;
            font-size: 11px !important;
            cursor: pointer !important;
            line-height: 1.4 !important;
        }
        .leaflet-control-geocoder-alternatives li:last-child {
            border-bottom: none !important;
        }
        .leaflet-control-geocoder-alternatives li:hover,
        .leaflet-control-geocoder-selected {
            background-color: rgba(56, 189, 248, 0.15) !important;
            color: #38bdf8 !important;
        }
        .leaflet-control-geocoder-address-context {
            color: #8b949e !important;
            font-size: 10px !important;
            display: block !important;
            margin-top: 2px !important;
        }

        /* MiniMap in the bottom-right corner */
        .leaflet-bottom.leaflet-right .leaflet-control-minimap {
            margin-bottom: 48px !important;
            margin-right: 16px !important;
            border: 1px solid #223048 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5) !important;
            background: rgba(14, 21, 36, 0.95) !important;
        }
        /* MiniMap Toggle Button: Solid Deep Black Background with Crisp White Map Logo */
        .leaflet-control-minimap-toggle-display,
        .leaflet-control-minimap-toggle {
            background-color: #0d1117 !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21'/%3E%3Cline x1='9' y1='3' x2='9' y2='18'/%3E%3Cline x1='15' y1='6' x2='15' y2='21'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 16px 16px !important;
            border: 1.5px solid #30363d !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6) !important;
            filter: none !important;
            width: 32px !important;
            height: 32px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            position: fixed !important;
            bottom: 14px !important;
            right: 16px !important;
            z-index: 1000 !important;
        }
        .leaflet-control-minimap-toggle-display:hover,
        .leaflet-control-minimap-toggle:hover {
            background-color: #161b22 !important;
            border-color: #38bdf8 !important;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.45) !important;
        }
        .leaflet-control-minimap:not(.minimized-bottomright) .leaflet-control-minimap-toggle-display {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'/%3E%3Cline x1='6' y1='6' x2='18' y2='18'/%3E%3C/svg%3E") !important;
        }
        </style>
        <script>
        window.resetMapView = function() {
            for (var key in window) {
                if (window[key] && window[key]._layers && typeof window[key].setView === 'function') {
                    window[key].setView([22.5, 78.5], 5);
                    break;
                }
            }
        };
        window.toggleMiniMap = function() {
            var toggleBtn = document.querySelector('.leaflet-control-minimap-toggle-display') || document.querySelector('.leaflet-control-minimap-toggle');
            if (toggleBtn) toggleBtn.click();
        };

        // Ensure proper tooltips & placeholders for toolbar controls
        function setupToolbarTooltips() {
            var geocoderInput = document.querySelector('.leaflet-control-geocoder-form input');
            if (geocoderInput) {
                geocoderInput.setAttribute('placeholder', 'Search city, district, coordinates or address...');
                geocoderInput.setAttribute('title', 'OSM Location Geocoder Search');
            }
            var geocoderBtn = document.querySelector('.leaflet-control-geocoder button') || document.querySelector('.leaflet-control-geocoder-icon');
            if (geocoderBtn) {
                geocoderBtn.setAttribute('title', 'Search location / address across India');
            }
            var fsBtn = document.querySelector('.leaflet-control-fullscreen a');
            if (fsBtn) {
                fsBtn.setAttribute('title', 'Toggle Fullscreen Mode');
            }
            var measureBtn = document.querySelector('.leaflet-control-measure-toggle') || document.querySelector('.leaflet-control-measure a');
            if (measureBtn) {
                measureBtn.setAttribute('title', 'Measure Distance & Area Tool');
            }
            var zoomIn = document.querySelector('.leaflet-control-zoom-in');
            if (zoomIn) zoomIn.setAttribute('title', 'Zoom In');
            var zoomOut = document.querySelector('.leaflet-control-zoom-out');
            if (zoomOut) zoomOut.setAttribute('title', 'Zoom Out');
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupToolbarTooltips);
        } else {
            setTimeout(setupToolbarTooltips, 400);
        }
        </script>
        """
        m.get_root().html.add_child(branca.element.Element(nav_helpers))

        # Layer switcher only added if explicitly requested (never when unified GIS dock is present)
        if enable_layer_control:
            folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # ------------------------------------------------------------------
    # 3. Data Overlays (Fires, Facilities, Buffers, Heatmap)
    # ------------------------------------------------------------------
    def _add_fire_layer(
        self, m: folium.Map, fire_df: pd.DataFrame, selected_types: Optional[List[str]] = None
    ) -> None:
        """Add clustered, color-coded fire anomaly markers."""
        if fire_df.empty:
            return

        display_df = fire_df
        if selected_types and "fire_type" in fire_df.columns:
            display_df = fire_df[fire_df["fire_type"].isin(selected_types)]

        fire_cluster = MarkerCluster(
            name="🔥 Thermal Anomaly Detections",
            show=True,
            options={
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
                "zoomToBoundsOnClick": True,
                "maxClusterRadius": 50,
            },
        )

        for idx, row in display_df.iterrows():
            fire_type = str(row.get("fire_type", "Other/Unknown"))
            color = self.FIRE_TYPE_COLORS.get(fire_type, "#95a5a6")
            frp = float(row.get("frp", 5.0))
            brightness = float(row.get("brightness", 320.0))
            confidence = str(row.get("confidence", "Nominal"))
            acq_date = str(row.get("acq_date", "N/A"))
            acq_time = str(row.get("acq_time", "0000")).zfill(4)
            time_formatted = f"{acq_time[:2]}:{acq_time[2:]} UTC"
            daynight = "☀️ Day" if str(row.get("daynight", "D")).upper() == "D" else "🌙 Night"
            nearest_fac = str(row.get("nearest_facility_name", "None / Rural"))
            dist_km = row.get("distance_to_nearest_industrial", None)
            dist_str = f"{dist_km:.2f} km" if dist_km is not None and not pd.isna(dist_km) else "N/A"
            det_id = str(row.get("detection_id", f"DET-{idx:04d}"))[:16]

            # FRP-proportional marker radius (min 4px, max 20px)
            marker_radius = min(max(frp / 8.0, 4.0), 20.0)

            # High hazard alert pill if near facility
            hazard_badge = ""
            if dist_km is not None and not pd.isna(dist_km) and dist_km <= 2.0:
                hazard_badge = f"""
                <div style='background-color: #ffebe9; color: #cf222e; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-bottom: 6px; font-size: 11px; border: 1px solid #ff8182;'>
                    ⚠️ CRITICAL: Within {dist_str} of industrial facility
                </div>
                """

            popup_html = f"""
            <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; min-width: 240px; font-size: 13px; line-height: 1.4; color: #24292f;'>
                <div style='display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid {color}; padding-bottom: 4px; margin-bottom: 8px;'>
                    <span style='font-weight: 700; color: {color}; font-size: 14px;'>🔥 {fire_type}</span>
                    <span style='font-size: 10px; background: #f6f8fa; padding: 2px 6px; border-radius: 10px; border: 1px solid #d0d7de;'>{det_id}</span>
                </div>
                {hazard_badge}
                <table style='width: 100%; font-size: 12px; border-collapse: collapse;'>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Coordinates:</td><td style='text-align: right; font-weight: 600;'>{row['latitude']:.4f}, {row['longitude']:.4f}</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Fire Radiative Power:</td><td style='text-align: right; font-weight: 600; color: #cf222e;'>{frp:.1f} MW</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Brightness Temp:</td><td style='text-align: right; font-weight: 600;'>{brightness:.1f} K</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Confidence:</td><td style='text-align: right; font-weight: 600;'>{confidence.title()}</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Detection Time:</td><td style='text-align: right;'>{acq_date} {time_formatted}</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Solar Period:</td><td style='text-align: right;'>{daynight}</td></tr>
                    <tr style='border-top: 1px dashed #d0d7de;'><td style='color: #57606a; padding: 4px 0 2px 0;'>Nearest Facility:</td><td style='text-align: right; font-weight: 600; padding-top: 4px;'>{nearest_fac}</td></tr>
                    <tr><td style='color: #57606a; padding: 2px 0;'>Facility Proximity:</td><td style='text-align: right; font-weight: 600;'>{dist_str}</td></tr>
                </table>
            </div>
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=marker_radius,
                color=color,
                weight=1.5,
                fill=True,
                fillColor=color,
                fillOpacity=0.75,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{fire_type} | FRP: {frp:.1f} MW | {acq_date}",
            ).add_to(fire_cluster)

        fire_cluster.add_to(m)

    def _add_facility_layers(
        self, m: folium.Map, facilities_gdf: Any
    ) -> None:
        """Add industrial facilities and multi-ring hazard buffer zones."""
        if facilities_gdf is None or (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
            return

        # 1. Industrial Facilities FeatureGroup
        facility_group = folium.FeatureGroup(name="🏭 Industrial Infrastructure (OSM)", show=True)

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
                    <tr><td style='color: #57606a;'>Buffer Protection:</td><td style='text-align: right; color: #cf222e; font-weight: 600;'>500m - 5km active</td></tr>
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

        # 2. Multi-Ring Concentric Hazard Buffers (500m, 1km, 2km, 5km)
        buffer_group = folium.FeatureGroup(name="⭕ Facility Hazard Buffers (500m-5km)", show=False)

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

    def _add_heatmap_layer(self, m: folium.Map, fire_df: pd.DataFrame) -> None:
        """Add continuous FRP-weighted thermal intensity heatmap."""
        if fire_df.empty or "latitude" not in fire_df.columns:
            return

        heat_points = []
        for _, row in fire_df.iterrows():
            frp = float(row.get("frp", 10.0))
            weight = min(max(frp / 60.0, 0.2), 1.0)
            heat_points.append([row["latitude"], row["longitude"], weight])

        HeatMap(
            heat_points,
            name="🌡️ FRP Thermal Intensity Heatmap",
            min_opacity=0.35,
            radius=20,
            blur=16,
            gradient={
                0.2: "#0000ff",   # Blue (Low anomaly)
                0.4: "#00ffff",   # Cyan
                0.6: "#00ff00",   # Green
                0.8: "#ffff00",   # Yellow (High thermal)
                1.0: "#ff0000",   # Red (Critical combustion)
            },
            show=False,
        ).add_to(m)

    # ------------------------------------------------------------------
    # 4. Floating Glassmorphic GIS Legend Panel
    # ------------------------------------------------------------------
    def _add_floating_legend(self, m: folium.Map, fire_count: int, facility_count: int) -> None:
        """Inject a floating responsive dark-glass GIS legend into the map."""
        legend_html = f"""
        {{% macro html(this, kwargs) %}}
        <div id="gis-floating-legend" style="
            position: fixed;
            bottom: 54px;
            left: 16px;
            z-index: 998;
            background: rgba(14, 21, 36, 0.94);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(48, 54, 61, 0.85);
            border-radius: 10px;
            padding: 10px 14px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.55);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #e6edf3;
            max-width: 230px;
            font-size: 11px;
            overflow: hidden;
            pointer-events: auto;
        ">
            <div style="display: flex; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-bottom: 8px;">
                <div style="font-weight: 700; font-size: 11px; display: flex; align-items: center; gap: 6px;">
                    <span style="color: #e74c3c;">🔥</span> SIH GIS Legend (Part 4.1)
                </div>
            </div>

            <div id="legend-content" style="display: block;">
                <!-- Status Pills -->
                <div style="display: flex; gap: 8px; margin-bottom: 10px; font-size: 11px;">
                    <span style="background: rgba(231, 76, 60, 0.2); color: #ff7b72; padding: 2px 8px; border-radius: 12px; border: 1px solid rgba(231, 76, 60, 0.4);">
                        🔥 {fire_count} Fires
                    </span>
                    <span style="background: rgba(56, 139, 253, 0.2); color: #79c0ff; padding: 2px 8px; border-radius: 12px; border: 1px solid rgba(56, 139, 253, 0.4);">
                        🏭 {facility_count} Facilities
                    </span>
                </div>

                <!-- Fire Classifications -->
                <div style="font-weight: 600; color: #8b949e; margin-bottom: 6px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;">
                    AI Classified Fire Sources
                </div>
                <div style="display: grid; grid-template-columns: 1fr; gap: 4px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: #e74c3c; display: inline-block; box-shadow: 0 0 6px #e74c3c;"></span>
                        <span>Industrial Fire (Near Plant)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: #e67e22; display: inline-block; box-shadow: 0 0 6px #e67e22;"></span>
                        <span>Gas Flare (Combustion Vent)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: #27ae60; display: inline-block;"></span>
                        <span>Forest Fire (Wildfire)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: #f1c40f; display: inline-block;"></span>
                        <span>Agricultural (Stubble)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: #7f8c8d; display: inline-block;"></span>
                        <span>Mining Activity (Thermal)</span>
                    </div>
                </div>

                <!-- Hazard Buffer Rings -->
                <div style="font-weight: 600; color: #8b949e; margin-bottom: 6px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;">
                    Facility Hazard Buffers
                </div>
                <div style="font-size: 11px; color: #c9d1d9; line-height: 1.5; margin-bottom: 8px;">
                    <span style="color: #ff7b72; font-weight: 600;">⭕ 500m</span> Immediate Danger &nbsp;|&nbsp;
                    <span style="color: #ffa657; font-weight: 600;">⭕ 1km</span> High Risk<br>
                    <span style="color: #e3b341; font-weight: 600;">⭕ 2km</span> Exposure Band &nbsp;|&nbsp;
                    <span style="color: #79c0ff; font-weight: 600;">⭕ 5km</span> Perimeter
                </div>

                <!-- Footer Hint -->
                <div style="border-top: 1px solid #30363d; padding-top: 6px; color: #8b949e; font-size: 10px;">
                    💡 Use <b>Layers Control</b> (top-right) to switch satellite & dark mode basemaps.
                </div>
            </div>
        </div>
        {{% endmacro %}}
        """
        macro = branca.element.MacroElement()
        macro._template = branca.element.Template(legend_html)
        m.get_root().add_child(macro)

    # ------------------------------------------------------------------
    # 5. Map Construction & Export Engine
    # ------------------------------------------------------------------
    def create_interactive_map(
        self,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        selected_types: Optional[List[str]] = None,
        enable_temporal_slider: Optional[bool] = None,
        enable_controls: Optional[bool] = None,
    ) -> folium.Map:
        """
        Construct and return the complete Interactive GIS Map instance (Part 4.1, 4.2 & 4.3).
        """
        if enable_temporal_slider is None:
            enable_temporal_slider = self.enable_temporal_slider
        if enable_controls is None:
            enable_controls = self.enable_controls

        # Determine initial center
        center_lat, center_lon = self.center
        if not fire_df.empty and "latitude" in fire_df.columns:
            center_lat = float(fire_df["latitude"].mean())
            center_lon = float(fire_df["longitude"].mean())

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.zoom_start,
            tiles=None,
            prefer_canvas=True,
            control_scale=False,
        )

        # 1. Base map tile layers (4 layers)
        self._add_base_layers(m)

        # 2. Data Overlays (Part 4.2)
        # 2a. Industrial facilities & multi-ring hazard buffers
        self.overlay_manager.add_facility_layers(m, facilities_gdf, show_facilities=True, show_buffers=False)

        # 2b. Static FRP-weighted thermal intensity heatmap
        self.overlay_manager.add_static_heatmap(m, fire_df, show=False)

        # 2c. Multi-temporal time-lapse animated slider heatmap (HeatMapWithTime)
        if enable_temporal_slider:
            self.overlay_manager.add_temporal_heatmap(m, fire_df, show=False)

        # 2d. Clustered Fire Detections with category subgroups
        parent_cluster = MarkerCluster(
            name="🔥 All Fire Detections (Clustered)",
            control=False,
            show=True,
            options={
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
                "zoomToBoundsOnClick": True,
                "maxClusterRadius": 50,
            },
        ).add_to(m)

        # Filter by selected types if provided
        display_df = fire_df
        if selected_types and "fire_type" in fire_df.columns:
            display_df = fire_df[fire_df["fire_type"].isin(selected_types)]

        self.overlay_manager.add_category_subgroups(m, parent_cluster, display_df)

        # Inject client-side marker registry script for live filtering
        reg_script = self.overlay_manager.generate_marker_registry_script()
        if reg_script:
            m.get_root().html.add_child(branca.element.Element(reg_script))

        # 3. Interactive GIS Controls (Fullscreen, Measure, Coordinates, MiniMap)
        # Prevent duplicate top-right LayerControl when single GIS dock is enabled
        self._add_gis_controls(m, enable_layer_control=not enable_controls)

        # 4. Part 4.3 Dashboard Controls & HUD
        fire_count = len(display_df) if not display_df.empty else 0
        fac_count = len(facilities_gdf) if facilities_gdf is not None and not (hasattr(facilities_gdf, "empty") and facilities_gdf.empty) else 0

        if enable_controls:
            self.dashboard_controls.add_geocoder(m, position="topleft", collapsed=True)
            self.dashboard_controls.add_interactive_filter_panel(
                m, fire_df=display_df, facilities_gdf=facilities_gdf, selected_types=selected_types
            )
            self.dashboard_controls.add_legend_panel(m, fire_count=fire_count, facility_count=fac_count)
        else:
            self._add_floating_legend(m, fire_count=fire_count, facility_count=fac_count)

        return m

    def export_html(
        self,
        output_path: Path,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        selected_types: Optional[List[str]] = None,
        enable_temporal_slider: Optional[bool] = None,
        enable_controls: Optional[bool] = None,
    ) -> Path:
        """
        Build and save the interactive map to a standalone HTML file.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        m = self.create_interactive_map(
            fire_df=fire_df,
            facilities_gdf=facilities_gdf,
            selected_types=selected_types,
            enable_temporal_slider=enable_temporal_slider,
            enable_controls=enable_controls,
        )

        m.save(str(output_path))
        logger.info(f"Interactive GIS map exported successfully to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 6. Data Loading Helpers
    # ------------------------------------------------------------------
    @classmethod
    def load_datasets(
        cls, simulate: bool = False, days: int = 2
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Load fire detections and industrial facilities from:
        1. SQLite persistent database (`data/fire_monitoring.db`)
        2. Processed CSV/GeoJSON files
        3. Demo data generator fallback
        """
        fire_df = pd.DataFrame()
        facilities_gdf = None

        if not simulate:
            # 1. Try SQLite persistent database
            try:
                from src.pipeline_automation.database import FireMonitoringDatabase

                db = FireMonitoringDatabase()
                db_fires = db.get_all_fires(limit=5000)
                if not db_fires.empty:
                    fire_df = db_fires
                    logger.info(
                        f"Loaded {len(fire_df)} fire records from SQLite persistent database."
                    )
            except Exception as e:
                logger.debug(f"Could not load from SQLite: {e}")

            # 2. Try processed CSV files if database was empty
            if fire_df.empty:
                for candidate in [
                    settings.PROCESSED_DATA_DIR / "master_enriched_fires.csv",
                    settings.PROCESSED_DATA_DIR / "classified_fires.csv",
                    settings.PROCESSED_DATA_DIR / "proximity_scored_fires.csv",
                    settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv",
                ]:
                    if candidate.exists():
                        try:
                            df = pd.read_csv(candidate)
                            if not df.empty:
                                fire_df = df
                                logger.info(f"Loaded {len(fire_df)} fire records from {candidate.name}")
                                break
                        except Exception as e:
                            logger.warning(f"Error reading {candidate}: {e}")

            # 3. Load industrial facilities
            fac_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
            if fac_path.exists():
                try:
                    if gpd is not None:
                        facilities_gdf = gpd.read_file(fac_path)
                    else:
                        facilities_gdf = pd.read_json(fac_path)
                    logger.info(f"Loaded {len(facilities_gdf)} industrial facilities from GeoJSON.")
                except Exception as e:
                    logger.warning(f"Error reading facilities GeoJSON: {e}")

        # 4. Fallback or Simulation Mode
        if simulate or fire_df.empty or facilities_gdf is None or (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
            logger.info("Initializing high-fidelity simulation dataset (DemoDataGenerator)...")
            from src.web.demo_data import DemoDataGenerator

            demo = DemoDataGenerator()
            if facilities_gdf is None or (hasattr(facilities_gdf, "empty") and facilities_gdf.empty):
                facilities_gdf = demo.generate_facilities()
            if fire_df.empty or simulate:
                fire_df = demo.generate_fire_data(n_fires=350, days_back=days)

        return fire_df, facilities_gdf
