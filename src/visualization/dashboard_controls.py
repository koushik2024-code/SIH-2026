"""
Dashboard Controls Engine (Part 4.3).

Provides interactive, production-grade GIS controls for the SIH-2026 visualization system:
1. Search by Location:
   - folium.plugins.Geocoder (Nominatim geocoding search box)
   - Facility / City Quick-Jump search widget with autocomplete & flyTo()
2. Interactive Client-Side Filter Dock:
   - Date range filter (start & end calendar pickers + 24h, 48h, 7d presets)
   - Fire type category filter (checkboxes with colors & live count badges)
   - Confidence level slider (0% to 100% threshold pruning)
   - Layer quick-action toggles (Heatmap, Buffers, Reset)
   - Live status indicator ("Showing X of Y active fire anomalies")
3. Legend Panel:
   - Glassmorphic collapsible legend detailing classifications, FRP scale, & safety buffer rings
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import os
import folium
from folium.plugins import Geocoder
import branca.element
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class DashboardControlManager:
    """
    Manages all Part 4.3 interactive controls for Folium/Leaflet maps.
    """

    FIRE_TYPE_COLORS: Dict[str, str] = {
        "Industrial Fire": "#e74c3c",       # Vibrant Red
        "Gas Flare": "#e67e22",             # High-vis Orange
        "Forest Fire": "#27ae60",           # Forest Green
        "Agricultural Burning": "#f1c40f",  # Amber Yellow
        "Mining Activity": "#7f8c8d",       # Slate Gray
        "Other/Unknown": "#95a5a6",         # Light Cool Gray
    }

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

    HAZARD_BUFFER_RINGS = [
        (500, "#e74c3c", 0.20, "500m - Immediate Danger Zone"),
        (1000, "#e67e22", 0.12, "1km - Critical Impact Zone"),
        (2000, "#f39c12", 0.07, "2km - Thermal Exposure Zone"),
        (5000, "#3498db", 0.03, "5km - Regional Surveillance Perimeter"),
    ]

    def add_geocoder(
        self,
        m: folium.Map,
        position: str = "topleft",
        collapsed: bool = True,
        add_marker: bool = True,
    ) -> None:
        """
        Add OpenStreetMap Nominatim Geocoder search box to the map.
        Allows users to search any city, region, or address in India / worldwide.
        """
        try:
            Geocoder(
                collapsed=collapsed,
                position=position,
                add_marker=add_marker,
                zoom=11,
            ).add_to(m)
            logger.info("Geocoder search control successfully mounted to map.")
        except Exception as e:
            logger.warning(f"Could not initialize folium.plugins.Geocoder: {e}")

    def add_interactive_filter_panel(
        self,
        m: folium.Map,
        fire_df: pd.DataFrame,
        facilities_gdf: Any,
        selected_types: Optional[List[str]] = None,
    ) -> None:
        """
        Inject an interactive, glassmorphic floating filter control dock
        with client-side real-time filtering for:
        - Date range
        - Fire type checkboxes
        - Confidence level slider
        - Facility / Location quick-jump search
        - Live visible anomaly counter
        """
        # Calculate date bounds
        start_date_str = ""
        end_date_str = ""
        if not fire_df.empty and "acq_date" in fire_df.columns:
            start_date_str = str(fire_df["acq_date"].min())
            end_date_str = str(fire_df["acq_date"].max())

        # Category counts
        cat_counts = {}
        if not fire_df.empty and "fire_type" in fire_df.columns:
            cat_counts = fire_df["fire_type"].value_counts().to_dict()

        # Build facilities JSON list for fast search autocomplete
        facilities_list = []
        if facilities_gdf is not None and not (
            hasattr(facilities_gdf, "empty") and facilities_gdf.empty
        ):
            for _, row in facilities_gdf.iterrows():
                name = row.get("name", "")
                fac_type = row.get("facility_type", "industrial")
                lat = row.get("latitude", None)
                lon = row.get("longitude", None)

                # Fallback to geometry
                if (lat is None or lon is None) and hasattr(row, "geometry") and row.geometry:
                    lat = row.geometry.y
                    lon = row.geometry.x

                if lat is not None and lon is not None and not pd.isna(lat) and not pd.isna(lon):
                    clean_name = str(name).strip() if pd.notna(name) and str(name).strip() else f"Industrial Site ({fac_type})"
                    type_clean = str(fac_type).replace("_", " ").title()
                    facilities_list.append({
                        "name": f"{clean_name} ({type_clean})",
                        "lat": round(float(lat), 5),
                        "lon": round(float(lon), 5),
                    })

        facilities_json = json.dumps(facilities_list[:150])

        # Prepare checkboxes HTML
        checkboxes_html = ""
        for cat, color in self.FIRE_TYPE_COLORS.items():
            count = cat_counts.get(cat, 0)
            is_checked = "checked" if (selected_types is None or cat in selected_types) else ""
            cat_id = cat.lower().replace(" ", "_").replace("/", "_")
            checkboxes_html += f"""
            <label class="ctrl-cb-label" for="cb_{cat_id}">
                <input type="checkbox" id="cb_{cat_id}" class="map-type-cb" value="{cat}" {is_checked}>
                <span class="ctrl-color-dot" style="background-color: {color};"></span>
                <span class="ctrl-cb-text">{cat}</span>
                <span class="ctrl-count-pill">{count}</span>
            </label>
            """

        total_fires = len(fire_df)

        carto_key = getattr(settings, "CARTO_API_KEY", "") or os.getenv("CARTO_API_KEY", "")

        panel_html = f"""
        <div id="gisFilterDock" class="gis-filter-dock">
            <!-- Header with Minimize & Close Controls -->
            <div class="filter-dock-header" id="dockHeader">
                <div class="dock-title">
                    <i class="fas fa-sliders-h"></i>
                    <span>GIS Dashboard Controls</span>
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <button type="button" id="btnToggleDock" class="dock-toggle-btn" title="Minimize/Maximize Content">
                        <i class="fas fa-chevron-up" id="dockToggleIcon"></i>
                    </button>
                    <button type="button" id="btnCloseDock" class="dock-toggle-btn" title="Close Controls Panel" style="font-size: 13px;">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>

            <!-- Collapsible Dock Content -->
            <div class="filter-dock-body" id="dockBody">
                <!-- 0. Basemap Layer Switcher -->
                <div class="dock-section">
                    <label class="dock-label">
                        <i class="fas fa-layer-group text-info"></i> Base Map Style:
                    </label>
                    <div class="basemap-pill-group" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 4px;">
                        <button type="button" class="preset-pill active" data-tile="osm" style="padding: 4px 2px; font-size: 10px; text-align: center; background: #0284c7; border-color: #38bdf8; color: #ffffff;">Streets</button>
                        <button type="button" class="preset-pill" data-tile="sat" style="padding: 4px 2px; font-size: 10px; text-align: center;">Satellite</button>
                        <button type="button" class="preset-pill" data-tile="dark" style="padding: 4px 2px; font-size: 10px; text-align: center;">Dark</button>
                        <button type="button" class="preset-pill" data-tile="topo" style="padding: 4px 2px; font-size: 10px; text-align: center;">Terrain</button>
                    </div>
                </div>

                <!-- 1. Search Location & Facilities -->
                <div class="dock-section">
                    <label class="dock-label">
                        <i class="fas fa-search-location text-danger"></i> Quick Jump to Facility / Site:
                    </label>
                    <div class="search-input-wrapper">
                        <input type="text" id="facSearchInput" list="facSearchDatalist" 
                               placeholder="Type plant, refinery, or city..." class="dock-input">
                        <datalist id="facSearchDatalist">
                            <!-- Populated dynamically -->
                        </datalist>
                        <button type="button" id="btnGoFac" class="dock-search-btn" title="Fly to Selected Site">
                            <i class="fas fa-crosshairs"></i>
                        </button>
                    </div>
                </div>

                <!-- 2. Date Range Filter -->
                <div class="dock-section">
                    <label class="dock-label">
                        <i class="far fa-calendar-alt text-warning"></i> Acquisition Date Range:
                    </label>
                    <div class="date-range-row">
                        <input type="date" id="dockDateStart" value="{start_date_str}" class="dock-input date-input" title="Start Date">
                        <span class="date-sep">to</span>
                        <input type="date" id="dockDateEnd" value="{end_date_str}" class="dock-input date-input" title="End Date">
                    </div>
                    <div class="date-presets">
                        <button type="button" class="preset-pill" data-preset="all">All Dates</button>
                        <button type="button" class="preset-pill" data-preset="24h">Last 24h</button>
                        <button type="button" class="preset-pill" data-preset="48h">Last 48h</button>
                        <button type="button" class="preset-pill" data-preset="7d">Last 7d</button>
                    </div>
                </div>

                <!-- 3. Confidence Level Slider -->
                <div class="dock-section">
                    <div class="slider-header-row">
                        <label class="dock-label">
                            <i class="fas fa-shield-alt text-success"></i> Min Confidence:
                        </label>
                        <span id="dockConfBadge" class="conf-badge">All (0%)</span>
                    </div>
                    <input type="range" id="dockConfSlider" min="0" max="95" step="5" value="0" class="dock-slider">
                    <div class="slider-marks">
                        <span>0%</span>
                        <span>50% (Nominal)</span>
                        <span>85% (High)</span>
                    </div>
                </div>

                <!-- 4. Fire Type Checkboxes -->
                <div class="dock-section">
                    <div class="cb-section-header">
                        <label class="dock-label">
                            <i class="fas fa-fire-alt text-danger"></i> Classification Filter:
                        </label>
                        <div class="cb-quick-actions">
                            <a href="javascript:void(0)" id="btnSelectAllTypes">All</a>
                            <span class="text-muted">|</span>
                            <a href="javascript:void(0)" id="btnClearAllTypes">None</a>
                        </div>
                    </div>
                    <div class="cb-container">
                        {checkboxes_html}
                    </div>
                    <div class="dock-share-bar" id="dockShareBar" style="display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 8px; background: #21262d;" title="Classification Proportions"></div>
                </div>

                <!-- 5. Quick Layer Action Buttons -->
                <div class="dock-actions-row">
                    <button type="button" id="btnQuickReset" class="dock-btn-reset">
                        <i class="fas fa-undo me-1"></i> Reset All Filters
                    </button>
                </div>

                <!-- 6. Real-Time Anomaly Status Counter -->
                <div class="dock-status-bar">
                    <span>Active Anomalies: </span>
                    <strong id="dockVisibleCount" class="text-warning">{total_fires}</strong>
                    <span class="text-muted"> / </span>
                    <span id="dockTotalCount">{total_fires}</span>
                </div>
            </div>
        </div>

        <style>
            /* Glassmorphic GIS Filter Dock - Positioned neatly on the right to eliminate left congestion */
            .gis-filter-dock {{
                position: fixed;
                top: 72px;
                right: 20px;
                width: 320px;
                background: rgba(22, 27, 34, 0.95);
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
                border: 1px solid rgba(48, 54, 61, 0.85);
                border-radius: 12px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.65);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
                color: #e6edf3;
                font-size: 12px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                overflow: hidden;
            }}

            .filter-dock-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px 14px;
                background: rgba(33, 38, 45, 0.95);
                border-bottom: 1px solid rgba(48, 54, 61, 0.8);
                cursor: pointer;
                user-select: none;
            }}

            .dock-title {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 700;
                font-size: 13px;
                color: #f0f6fc;
                letter-spacing: 0.3px;
            }}

            .dock-title i {{
                color: #e74c3c;
            }}

            .dock-toggle-btn {{
                background: transparent;
                border: none;
                color: #8b949e;
                cursor: pointer;
                font-size: 12px;
                padding: 4px 6px;
                border-radius: 4px;
                transition: color 0.2s, background 0.2s;
            }}

            .dock-toggle-btn:hover {{
                color: #ffffff;
                background: rgba(255, 255, 255, 0.1);
            }}

            .filter-dock-body {{
                padding: 12px 14px;
                max-height: calc(100vh - 180px);
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}

            .filter-dock-body::-webkit-scrollbar {{
                width: 5px;
            }}
            .filter-dock-body::-webkit-scrollbar-thumb {{
                background: #30363d;
                border-radius: 4px;
            }}

            .dock-section {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}

            .dock-label {{
                font-size: 11px;
                font-weight: 600;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.4px;
                display: flex;
                align-items: center;
                gap: 5px;
            }}

            /* Search Inputs */
            .search-input-wrapper {{
                display: flex;
                gap: 6px;
                align-items: center;
            }}

            .dock-input {{
                flex: 1;
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                padding: 6px 10px;
                font-size: 12px;
                outline: none;
                transition: border-color 0.2s, box-shadow 0.2s;
            }}

            .dock-input:focus {{
                border-color: #58a6ff;
                box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
            }}

            .dock-search-btn {{
                background: #21262d;
                border: 1px solid #30363d;
                color: #58a6ff;
                border-radius: 6px;
                padding: 6px 10px;
                cursor: pointer;
                transition: all 0.2s;
            }}

            .dock-search-btn:hover {{
                background: #30363d;
                color: #ffffff;
            }}

            /* Date Range */
            .date-range-row {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            .date-input {{
                width: 120px;
                font-size: 11px;
                padding: 5px 6px;
            }}

            .date-sep {{
                color: #8b949e;
                font-size: 11px;
            }}

            .date-presets {{
                display: flex;
                gap: 4px;
                margin-top: 2px;
            }}

            .preset-pill {{
                flex: 1;
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #8b949e;
                font-size: 10px;
                padding: 3px 4px;
                cursor: pointer;
                transition: all 0.2s;
                text-align: center;
            }}

            .preset-pill:hover, .preset-pill.active {{
                background: #30363d;
                color: #58a6ff;
                border-color: #58a6ff;
            }}

            /* Slider */
            .slider-header-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}

            .conf-badge {{
                background: rgba(46, 160, 67, 0.2);
                border: 1px solid #2ea043;
                color: #3fb950;
                padding: 1px 6px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: 600;
            }}

            .dock-slider {{
                width: 100%;
                accent-color: #2ea043;
                cursor: pointer;
                height: 4px;
                background: #30363d;
                border-radius: 2px;
                outline: none;
            }}

            .slider-marks {{
                display: flex;
                justify-content: space-between;
                font-size: 9px;
                color: #6e7681;
                margin-top: -2px;
            }}

            /* Checkboxes */
            .cb-section-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .cb-quick-actions a {{
                color: #58a6ff;
                text-decoration: none;
                font-size: 10px;
            }}

            .cb-quick-actions a:hover {{
                text-decoration: underline;
            }}

            .cb-container {{
                display: flex;
                flex-direction: column;
                gap: 4px;
                background: rgba(13, 17, 23, 0.5);
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 8px;
            }}

            .ctrl-cb-label {{
                display: flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
                padding: 3px 4px;
                border-radius: 4px;
                transition: background 0.15s;
                font-size: 11px;
            }}

            .ctrl-cb-label:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}

            .ctrl-color-dot {{
                width: 9px;
                height: 9px;
                border-radius: 50%;
                flex-shrink: 0;
            }}

            .ctrl-cb-text {{
                flex: 1;
                color: #c9d1d9;
            }}

            .ctrl-count-pill {{
                font-size: 9px;
                background: #21262d;
                color: #8b949e;
                padding: 1px 5px;
                border-radius: 8px;
                border: 1px solid #30363d;
            }}

            /* Reset Button */
            .dock-actions-row {{
                display: flex;
                gap: 6px;
            }}

            .dock-btn-reset {{
                width: 100%;
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #e6edf3;
                padding: 6px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .dock-btn-reset:hover {{
                background: #30363d;
                border-color: #8b949e;
                color: #ffffff;
            }}

            /* Status Bar */
            .dock-status-bar {{
                background: rgba(13, 17, 23, 0.8);
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 10px;
                text-align: center;
                font-size: 11px;
                color: #8b949e;
            }}

            /* Suppress duplicate top-right layer control completely */
            .leaflet-control-layers,
            .leaflet-control-layers-expanded {{
                display: none !important;
            }}

            /* High-visibility pulsing radar animation for target alert locator */
            @keyframes alertRadarPulse {{
                0% {{ transform: scale(0.85); opacity: 0.9; }}
                50% {{ transform: scale(1.35); opacity: 0.25; }}
                100% {{ transform: scale(0.85); opacity: 0.9; }}
            }}
            .pulsing-radar-marker {{
                animation: alertRadarPulse 1.8s infinite ease-in-out;
                transform-origin: center;
            }}

            /* Responsive */
            @media (max-width: 600px) {{
                .gis-filter-dock {{
                    width: calc(100vw - 24px);
                    left: 12px;
                    top: 60px;
                }}
            }}
        </style>

        <script>
            (function() {{
                var facilitiesData = {facilities_json};

                // Global Map accessor - Guaranteed to find main Folium map instance
                function getLeafletMap() {{
                    var foliumContainer = document.querySelector('.folium-map') || document.querySelector('.leaflet-container');
                    if (foliumContainer && foliumContainer.id && window[foliumContainer.id]) {{
                        return window[foliumContainer.id];
                    }}
                    if (foliumContainer && foliumContainer._leaflet_map) {{
                        return foliumContainer._leaflet_map;
                    }}
                    for (var k in window) {{
                        if (k.indexOf('map_') === 0 && window[k] && typeof window[k].setView === 'function') {{
                            return window[k];
                        }}
                    }}
                    for (var key in window) {{
                        if (window[key] && window[key]._layers && typeof window[key].flyTo === 'function') {{
                            if (!window[key]._isMiniMap) return window[key];
                        }}
                    }}
                    return null;
                }}
                window.getLeafletMap = getLeafletMap;

                // Basemap Switcher Logic
                function switchBasemap(key) {{
                    var map = getLeafletMap();
                    if (!map || typeof L === 'undefined') return;

                    map.eachLayer(function(l) {{
                        if (l instanceof L.TileLayer && !l._miniMap && (!l.options || !l.options.isMiniMap)) {{
                            map.removeLayer(l);
                        }}
                    }});

                    var cartoKey = '{carto_key}';
                    var darkUrl = cartoKey 
                        ? 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png?api_key=' + encodeURIComponent(cartoKey)
                        : 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}';
                    var darkOpts = cartoKey
                        ? {{ maxZoom: 20, subdomains: 'abcd', attribution: '&copy; CartoDB' }}
                        : {{ maxZoom: 18, attribution: '&copy; Esri &mdash; Esri, DeLorme, NAVTEQ' }};

                    var tileUrls = {{
                        osm: {{ url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', opts: {{ maxZoom: 19, attribution: 'Tiles &copy; Esri &mdash; Street Map (English)' }} }},
                        sat: {{ url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', opts: {{ maxZoom: 19, attribution: 'Esri' }} }},
                        dark: {{ url: darkUrl, opts: darkOpts }},
                        topo: {{ url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', opts: {{ maxZoom: 18, attribution: 'Esri' }} }}
                    }};

                    var cfg = tileUrls[key] || tileUrls.osm;
                    var newLayer = L.tileLayer(cfg.url, cfg.opts);
                    newLayer.addTo(map);
                    newLayer.bringToBack();

                    if (key === 'dark' && !cartoKey) {{
                        var refLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                            maxZoom: 18,
                            opacity: 0.85,
                            interactive: false
                        }});
                        refLayer.addTo(map);
                    }}

                    document.querySelectorAll('.basemap-pill-group .preset-pill').forEach(function(btn) {{
                        if (btn.getAttribute('data-tile') === key) {{
                            btn.style.background = '#0284c7';
                            btn.style.borderColor = '#38bdf8';
                            btn.style.color = '#ffffff';
                        }} else {{
                            btn.style.background = '#21262d';
                            btn.style.borderColor = '#30363d';
                            btn.style.color = '#c9d1d9';
                        }}
                    }});

                    if (window.parent && window.parent !== window) {{
                        try {{ window.parent.postMessage({{ action: 'basemapSwitched', tile: key }}, '*'); }} catch(e) {{}}
                    }}
                }}
                window.switchBasemap = switchBasemap;

                // Dock Visibility Toggle
                function toggleDock(force) {{
                    var d = document.getElementById('gisFilterDock');
                    var b = document.getElementById('dockBody');
                    var ic = document.getElementById('dockToggleIcon');
                    if (!d) return;

                    var isHidden = (d.style.display === 'none' || getComputedStyle(d).display === 'none');
                    var willOpen = (force === 'show' || (force !== 'hide' && isHidden));
                    if (willOpen) {{
                        d.style.display = 'block';
                        if (b) b.style.display = 'flex';
                        if (ic) ic.className = 'fas fa-chevron-up';
                    }} else {{
                        d.style.display = 'none';
                    }}
                    if (window.parent && window.parent !== window) {{
                        try {{ window.parent.postMessage({{ action: 'dockStateChanged', open: willOpen }}, '*'); }} catch(err) {{}}
                    }}
                }}
                window.toggleDock = toggleDock;

                function resetMapView() {{
                    var map = getLeafletMap();
                    if (map && typeof map.setView === 'function') {{
                        map.setView([22.5, 78.5], 5);
                    }}
                }}
                window.resetMapView = resetMapView;

                function toggleMiniMap() {{
                    var toggleBtn = document.querySelector('.leaflet-control-minimap-toggle-display') || document.querySelector('.leaflet-control-minimap-toggle');
                    if (toggleBtn) toggleBtn.click();
                }}
                window.toggleMiniMap = toggleMiniMap;

                // Exact Alert Coordinates Locator & Zoom Engine
                function locateAlert(lat, lon, name, frp, zoomLevel) {{
                    var targetLat = parseFloat(lat);
                    var targetLon = parseFloat(lon);
                    if (isNaN(targetLat) || isNaN(targetLon)) return false;

                    var z = zoomLevel || 15;
                    var attempts = 0;
                    var maxAttempts = 60; // Poll for up to 6s
                    var pollTimer = setInterval(function() {{
                        attempts++;
                        var map = getLeafletMap();
                        if (map && typeof map.setView === 'function') {{
                            clearInterval(pollTimer);

                            // Smooth fly to exact detection coordinates
                            if (typeof map.flyTo === 'function') {{
                                map.flyTo([targetLat, targetLon], z, {{ duration: 1.6 }});
                            }} else {{
                                map.setView([targetLat, targetLon], z);
                            }}

                            // Add prominent pulsing radar marker and popup
                            if (typeof L !== 'undefined') {{
                                if (window._currentAlertLayer) {{
                                    try {{ map.removeLayer(window._currentAlertLayer); }} catch(e) {{}}
                                }}
                                var alertGroup = L.layerGroup();
                                window._currentAlertLayer = alertGroup;

                                // Animated outer pulsing ring
                                var outerRing = L.circleMarker([targetLat, targetLon], {{
                                    radius: 32,
                                    color: '#ef4444',
                                    weight: 3,
                                    fillColor: '#ef4444',
                                    fillOpacity: 0.35,
                                    className: 'pulsing-radar-marker'
                                }}).addTo(alertGroup);

                                // Solid core beacon
                                var coreBeacon = L.circleMarker([targetLat, targetLon], {{
                                    radius: 9,
                                    color: '#ffffff',
                                    weight: 2.5,
                                    fillColor: '#dc2626',
                                    fillOpacity: 1.0
                                }}).addTo(alertGroup);

                                var titleStr = name || 'Active Thermal Anomaly';
                                var frpStr = frp ? ('<div style="font-size:12px; color:#d97706; font-weight:700; margin-top:3px;">Thermal Radiative Power: ' + frp + ' MW</div>') : '';
                                var popupHtml = 
                                    '<div style="font-family:Inter,sans-serif; min-width:210px; padding:2px;">' +
                                    '<div style="font-size:11px; font-weight:800; color:#ef4444; letter-spacing:0.5px; text-transform:uppercase;">🚨 INCIDENT ALERT LOCATION</div>' +
                                    '<div style="font-size:14px; font-weight:700; color:#0f172a; margin:3px 0;">' + titleStr + '</div>' +
                                    '<div style="font-size:12px; color:#475569;">Detection Coordinates:<br><b>' + targetLat.toFixed(4) + '&deg;N, ' + targetLon.toFixed(4) + '&deg;E</b></div>' +
                                    frpStr +
                                    '</div>';

                                coreBeacon.bindPopup(popupHtml, {{ autoClose: false, closeOnClick: false }}).openPopup();
                                alertGroup.addTo(map);

                                var searchBox = document.getElementById('facSearchInput');
                                if (searchBox) searchBox.value = titleStr;
                            }}
                            return true;
                        }}
                        if (attempts >= maxAttempts) {{
                            clearInterval(pollTimer);
                        }}
                    }}, 100);
                    return true;
                }}
                window.locateAlert = locateAlert;

                // Cross-window message listener active immediately
                window.addEventListener('message', function(e) {{
                    if (!e.data) return;
                    var msg = typeof e.data === 'string' ? e.data : (e.data.action || e.data.type || '');
                    if (msg === 'locateAlert') {{
                        locateAlert(e.data.lat, e.data.lon, e.data.name, e.data.frp, e.data.zoom || 15);
                    }} else if (msg === 'searchFacility' || msg === 'jumpToFacility') {{
                        if (e.data.lat && e.data.lon) {{
                            locateAlert(e.data.lat, e.data.lon, e.data.query || e.data.name || e.data.tile, e.data.frp, e.data.zoom || 15);
                        }} else {{
                            jumpToFacility(e.data.query || e.data.tile || e.data.name || '');
                        }}
                    }} else if (msg === 'toggleDock') {{
                        toggleDock();
                    }} else if (msg === 'switchBasemap' && (e.data.tile || e.data.key)) {{
                        switchBasemap(e.data.tile || e.data.key);
                    }} else if (msg === 'resetMapView') {{
                        resetMapView();
                    }} else if (msg === 'toggleMiniMap') {{
                        toggleMiniMap();
                    }}
                }});

                // Wait for Leaflet map to be initialized
                function initGISControls() {{
                    var map = getLeafletMap();

                    document.querySelectorAll('.basemap-pill-group .preset-pill').forEach(function(btn) {{
                        btn.addEventListener('click', function() {{
                            var key = this.getAttribute('data-tile');
                            switchBasemap(key);
                        }});
                    }});

                    // Populate Datalist
                    var datalist = document.getElementById('facSearchDatalist');
                    if (datalist && facilitiesData && facilitiesData.length) {{
                        facilitiesData.forEach(function(fac) {{
                            var opt = document.createElement('option');
                            opt.value = fac.name;
                            opt.setAttribute('data-lat', fac.lat);
                            opt.setAttribute('data-lon', fac.lon);
                            datalist.appendChild(opt);
                        }});
                    }}

                    // Facility Search FlyTo with Smart Token and Fire Registry Fallback
                    function jumpToFacility(overrideName) {{
                        var inputVal = (overrideName || (document.getElementById('facSearchInput') ? document.getElementById('facSearchInput').value : '')).trim();
                        if (!inputVal) return;
                        if (document.getElementById('facSearchInput')) {{
                            document.getElementById('facSearchInput').value = inputVal;
                        }}

                        var noiseWords = ['and', 'the', 'of', 'in', 'near', 'plant', 'station', 'complex', 'refinery', 'power', 'thermal', 'super', 'tps', 'sector', 'block'];
                        var tokens = inputVal.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(function(w) {{
                            return w.length >= 3 && noiseWords.indexOf(w) === -1;
                        }});

                        var target = null;

                        // 1. First priority: match in facilitiesData
                        if (facilitiesData && facilitiesData.length) {{
                            for (var i = 0; i < facilitiesData.length; i++) {{
                                var fNameLower = (facilitiesData[i].name || '').toLowerCase();
                                var searchLower = inputVal.toLowerCase();
                                if (fNameLower.indexOf(searchLower) !== -1 || searchLower.indexOf(fNameLower) !== -1) {{
                                    target = facilitiesData[i];
                                    break;
                                }}
                                for (var t = 0; t < tokens.length; t++) {{
                                    if (fNameLower.indexOf(tokens[t]) !== -1) {{
                                        target = facilitiesData[i];
                                        break;
                                    }}
                                }}
                                if (target) break;
                            }}
                        }}

                        // 2. Second priority: match in _fireMarkerRegistry
                        if (!target && window._fireMarkerRegistry && window._fireMarkerRegistry.length) {{
                            for (var j = 0; j < window._fireMarkerRegistry.length; j++) {{
                                var mData = window._fireMarkerRegistry[j].data;
                                if (!mData) continue;
                                var mFac = (mData.nearest_facility_name || mData.facility_name || '').toLowerCase();
                                for (var k = 0; k < tokens.length; k++) {{
                                    if (mFac.indexOf(tokens[k]) !== -1) {{
                                        target = {{
                                            name: mData.nearest_facility_name || mData.facility_name || inputVal,
                                            lat: parseFloat(mData.latitude),
                                            lon: parseFloat(mData.longitude),
                                            frp: mData.frp
                                        }};
                                        break;
                                    }}
                                }}
                                if (target) break;
                            }}
                        }}

                        if (target) {{
                            locateAlert(target.lat, target.lon, target.name, target.frp, 15);
                        }}
                    }}
                    window.jumpToFacility = jumpToFacility;

                    var searchInput = document.getElementById('facSearchInput');
                    if (searchInput) {{
                        searchInput.addEventListener('change', jumpToFacility);
                        searchInput.addEventListener('keydown', function(e) {{
                            if (e.key === 'Enter') jumpToFacility();
                        }});
                    }}
                    var goBtn = document.getElementById('btnGoFac');
                    if (goBtn) goBtn.addEventListener('click', jumpToFacility);

                    // Dock header collapse button
                    var dockHeader = document.getElementById('dockHeader');
                    var btnToggleDock = document.getElementById('btnToggleDock');

                    if (dockHeader) {{
                        dockHeader.addEventListener('click', function(e) {{
                            if (e.target.closest('#btnToggleDock')) return;
                            var b = document.getElementById('dockBody');
                            var ic = document.getElementById('dockToggleIcon');
                            if (b) {{
                                if (b.style.display === 'none') {{
                                    b.style.display = 'flex';
                                    if (ic) ic.className = 'fas fa-chevron-up';
                                }} else {{
                                    b.style.display = 'none';
                                    if (ic) ic.className = 'fas fa-chevron-down';
                                }}
                            }}
                        }});
                    }}
                    if (btnToggleDock) {{
                        btnToggleDock.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            var b = document.getElementById('dockBody');
                            var ic = document.getElementById('dockToggleIcon');
                            if (b) {{
                                if (b.style.display === 'none') {{
                                    b.style.display = 'flex';
                                    if (ic) ic.className = 'fas fa-chevron-up';
                                }} else {{
                                    b.style.display = 'none';
                                    if (ic) ic.className = 'fas fa-chevron-down';
                                }}
                            }}
                        }});
                    }}

                    var btnCloseDock = document.getElementById('btnCloseDock');
                    if (btnCloseDock) {{
                        btnCloseDock.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            toggleDock('hide');
                        }});
                    }}

                    // Live Filter Engine
                    function applyMapFilters(customFilters) {{
                        var startDate = (customFilters && customFilters.startDate !== undefined)
                            ? customFilters.startDate
                            : (document.getElementById('dockDateStart') ? document.getElementById('dockDateStart').value : '');
                        var endDate = (customFilters && customFilters.endDate !== undefined)
                            ? customFilters.endDate
                            : (document.getElementById('dockDateEnd') ? document.getElementById('dockDateEnd').value : '');
                        var minConf = (customFilters && customFilters.minConf !== undefined)
                            ? customFilters.minConf
                            : (document.getElementById('dockConfSlider') ? parseInt(document.getElementById('dockConfSlider').value, 10) : 0);

                        // Update confidence badge if present
                        var confBadge = document.getElementById('dockConfBadge');
                        if (confBadge) {{
                            if (minConf === 0) {{
                                confBadge.innerText = "All (0%)";
                                confBadge.style.borderColor = "#2ea043";
                                confBadge.style.color = "#3fb950";
                            }} else if (minConf >= 80) {{
                                confBadge.innerText = "High (≥" + minConf + "%)";
                                confBadge.style.borderColor = "#cf222e";
                                confBadge.style.color = "#ff7b72";
                            }} else {{
                                confBadge.innerText = "≥" + minConf + "%";
                                confBadge.style.borderColor = "#d29922";
                                confBadge.style.color = "#e3b341";
                            }}
                        }}

                        var checkedTypes = new Set();
                        if (customFilters && customFilters.types) {{
                            customFilters.types.forEach(function(t) {{ checkedTypes.add(t); }});
                        }} else {{
                            document.querySelectorAll('.map-type-cb:checked').forEach(function(cb) {{
                                checkedTypes.add(cb.value);
                            }});
                        }}

                        var startVal = startDate ? String(startDate).trim().substring(0, 10) : '';
                        var endVal = endDate ? String(endDate).trim().substring(0, 10) : '';

                        var visibleCount = 0;
                        var totalCount = 0;
                        var catCounts = {{}};

                        if (window._fireMarkerRegistry && window._fireMarkerRegistry.length) {{
                            totalCount = window._fireMarkerRegistry.length;
                            window._fireMarkerRegistry.forEach(function(item) {{
                                var m = item.marker;
                                var data = item.data;

                                var typeMatch = checkedTypes.size === 0 || checkedTypes.has(data.fire_type);
                                var confMatch = (data.confidence_num >= minConf);
                                var dateMatch = true;
                                var rowDate = String(data.acq_date || '').trim().substring(0, 10);
                                if (startVal && rowDate < startVal) dateMatch = false;
                                if (endVal && rowDate > endVal) dateMatch = false;

                                var shouldShow = typeMatch && confMatch && dateMatch;
                                if (shouldShow) {{
                                    visibleCount++;
                                    catCounts[data.fire_type] = (catCounts[data.fire_type] || 0) + 1;
                                    if (item.parent && !item.parent.hasLayer(m)) {{
                                        item.parent.addLayer(m);
                                    }}
                                }} else {{
                                    if (item.parent && item.parent.hasLayer(m)) {{
                                        item.parent.removeLayer(m);
                                    }}
                                }}
                            }});
                        }}

                        var visCountElem = document.getElementById('dockVisibleCount');
                        if (visCountElem) visCountElem.innerText = visibleCount;

                        // Render Category Share Bar
                        var shareBar = document.getElementById('dockShareBar');
                        if (shareBar && visibleCount > 0) {{
                            var colorMap = {json.dumps(self.FIRE_TYPE_COLORS)};
                            var barHtml = '';
                            for (var cName in catCounts) {{
                                var pct = ((catCounts[cName] / visibleCount) * 100).toFixed(1);
                                var cColor = colorMap[cName] || '#38bdf8';
                                barHtml += '<div style="width:' + pct + '%; background-color:' + cColor + ';" title="' + cName + ': ' + catCounts[cName] + ' (' + pct + '%)"></div>';
                            }}
                            shareBar.innerHTML = barHtml;
                        }}

                        // Synchronize with parent HUD chips if embedded
                        if (window.parent && window.parent !== window) {{
                            try {{
                                var hudThermal = window.parent.document.getElementById('hudThermalCount');
                                if (hudThermal) hudThermal.innerText = visibleCount.toLocaleString();
                                var hudInd = window.parent.document.getElementById('hudIndustrialCount');
                                if (hudInd && catCounts['Industrial Fire'] !== undefined) {{
                                    hudInd.innerText = catCounts['Industrial Fire'].toLocaleString();
                                }}
                            }} catch(e) {{}}
                        }}

                        return {{ visible: visibleCount, total: totalCount, categories: catCounts }};
                    }}

                    window.applyMapFilters = applyMapFilters;

                    // Attach Listeners
                    document.querySelectorAll('.map-type-cb').forEach(function(cb) {{
                        cb.addEventListener('change', function() {{ applyMapFilters(); }});
                    }});

                    var dateStart = document.getElementById('dockDateStart');
                    var dateEnd = document.getElementById('dockDateEnd');
                    if (dateStart) dateStart.addEventListener('change', function() {{ applyMapFilters(); }});
                    if (dateEnd) dateEnd.addEventListener('change', function() {{ applyMapFilters(); }});

                    var confSlider = document.getElementById('dockConfSlider');
                    if (confSlider) confSlider.addEventListener('input', function() {{ applyMapFilters(); }});

                    // Date Presets
                    document.querySelectorAll('.preset-pill').forEach(function(pill) {{
                        pill.addEventListener('click', function() {{
                            document.querySelectorAll('.preset-pill').forEach(function(p) {{ p.classList.remove('active'); }});
                            this.classList.add('active');
                            var preset = this.getAttribute('data-preset');
                            var now = new Date();

                            if (preset === 'all') {{
                                if (dateStart) dateStart.value = "{start_date_str}";
                                if (dateEnd) dateEnd.value = "{end_date_str}";
                            }} else {{
                                var daysBack = (preset === '24h') ? 1 : ((preset === '48h') ? 2 : 7);
                                var pastDate = new Date();
                                pastDate.setDate(now.getDate() - daysBack);
                                if (dateEnd) dateEnd.value = now.toISOString().split('T')[0];
                                if (dateStart) dateStart.value = pastDate.toISOString().split('T')[0];
                            }}
                            applyMapFilters();
                        }});
                    }});

                    // Checkbox All / None
                    var btnAll = document.getElementById('btnSelectAllTypes');
                    if (btnAll) {{
                        btnAll.addEventListener('click', function() {{
                            document.querySelectorAll('.map-type-cb').forEach(function(cb) {{ cb.checked = true; }});
                            applyMapFilters();
                        }});
                    }}
                    var btnNone = document.getElementById('btnClearAllTypes');
                    if (btnNone) {{
                        btnNone.addEventListener('click', function() {{
                            document.querySelectorAll('.map-type-cb').forEach(function(cb) {{ cb.checked = false; }});
                            applyMapFilters();
                        }});
                    }}

                    // Reset Button
                    var btnReset = document.getElementById('btnQuickReset');
                    if (btnReset) {{
                        btnReset.addEventListener('click', function() {{
                            if (dateStart) dateStart.value = "{start_date_str}";
                            if (dateEnd) dateEnd.value = "{end_date_str}";
                            if (confSlider) confSlider.value = 0;
                            document.querySelectorAll('.map-type-cb').forEach(function(cb) {{ cb.checked = true; }});
                            document.querySelectorAll('.preset-pill').forEach(function(p) {{ p.classList.remove('active'); }});
                            applyMapFilters();
                        }});
                    }}

                    // Initial filter pass once markers are populated
                    setTimeout(applyMapFilters, 600);

                    // Parse URL search/alert parameters for direct pinpoint zoom (linking from Alerts page)
                    var sp = null;
                    try {{
                        var rawSearch = window.location.search;
                        if ((!rawSearch || rawSearch.length < 2) && window.parent && window.parent !== window) {{
                            rawSearch = window.parent.location.search;
                        }}
                        if (rawSearch && rawSearch.length > 1) {{
                            sp = new URLSearchParams(rawSearch);
                        }}
                    }} catch(e) {{}}

                    if (sp) {{
                        var pLat = parseFloat(sp.get('lat'));
                        var pLon = parseFloat(sp.get('lon'));
                        var pName = sp.get('name') || sp.get('search');
                        var pFrp = sp.get('frp');
                        var pZoom = parseInt(sp.get('zoom') || '15', 10);

                        if (!isNaN(pLat) && !isNaN(pLon)) {{
                            locateAlert(pLat, pLon, pName, pFrp, pZoom);
                        }} else if (pName) {{
                            setTimeout(function() {{ jumpToFacility(pName); }}, 500);
                        }}
                    }}
                }}

                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initGISControls);
                }} else {{
                    setTimeout(initGISControls, 300);
                }}
            }})();
        </script>
        """

        m.get_root().html.add_child(branca.element.Element(panel_html))
        logger.info("Interactive GIS Filter Dock successfully injected into Folium map.")

    def add_legend_panel(
        self,
        m: folium.Map,
        fire_count: int = 0,
        facility_count: int = 0,
    ) -> None:
        """
        Add a responsive, glassmorphic collapsible legend panel to the map.
        """
        category_rows = ""
        for cat, color in self.FIRE_TYPE_COLORS.items():
            category_rows += f"""
            <div class="legend-row">
                <span class="legend-dot" style="background-color: {color};"></span>
                <span class="legend-name">{cat}</span>
            </div>
            """

        buffer_rows = ""
        for radius, color, opacity, label in self.HAZARD_BUFFER_RINGS:
            buffer_rows += f"""
            <div class="legend-row">
                <span class="legend-ring" style="border-color: {color}; background-color: {color}; opacity: {max(opacity * 3.5, 0.4):.2f};"></span>
                <span class="legend-name">{label}</span>
            </div>
            """

        legend_html = f"""
        <div id="gisLegendHUD" class="gis-legend-hud">
            <div class="legend-header" id="legendHeader">
                <div class="legend-title">
                    <i class="fas fa-layer-group text-primary"></i>
                    <span>GIS Legend & Specifications</span>
                </div>
            </div>

            <div class="legend-body" id="legendBody" style="display: flex;">
                <div class="legend-sec-title">Fire Classifications</div>
                <div class="legend-grid">
                    {category_rows}
                </div>

                <div class="legend-sec-title">Thermal Power (FRP Scale)</div>
                <div class="frp-scale-row">
                    <div class="frp-dot-item">
                        <span class="frp-bubble bubble-sm"></span>
                        <span>&lt;15 MW</span>
                    </div>
                    <div class="frp-dot-item">
                        <span class="frp-bubble bubble-md"></span>
                        <span>15-50 MW</span>
                    </div>
                    <div class="frp-dot-item">
                        <span class="frp-bubble bubble-lg"></span>
                        <span>&gt;50 MW</span>
                    </div>
                </div>

                <div class="legend-sec-title">Facility Hazard Buffer Zones</div>
                <div class="legend-grid">
                    {buffer_rows}
                </div>

                <div class="legend-metrics-row">
                    <div class="metric-box">
                        <span class="metric-val text-danger">{fire_count}</span>
                        <span class="metric-lbl">Fire Points</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val text-primary">{facility_count}</span>
                        <span class="metric-lbl">Facilities</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-val text-success">4</span>
                        <span class="metric-lbl">Base Maps</span>
                    </div>
                </div>
            </div>
        </div>

        <style>
            .gis-legend-hud {{
                position: fixed;
                bottom: 54px;
                left: 16px;
                width: 225px;
                background: rgba(14, 21, 36, 0.94);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(48, 54, 61, 0.85);
                border-radius: 10px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
                z-index: 998;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
                color: #e6edf3;
                font-size: 11px;
                overflow: hidden;
                pointer-events: auto;
            }}

            @media (max-width: 768px) {{
                .gis-legend-hud {{
                    left: 12px;
                    bottom: 54px;
                    width: 200px;
                }}
            }}

            .legend-header {{
                display: flex;
                align-items: center;
                padding: 7px 10px;
                background: rgba(22, 27, 34, 0.96);
                border-bottom: 1px solid rgba(48, 54, 61, 0.8);
                user-select: none;
            }}

            .legend-title {{
                display: flex;
                align-items: center;
                gap: 6px;
                font-weight: 700;
                font-size: 11px;
                color: #f0f6fc;
            }}

            .legend-body {{
                padding: 8px 10px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                max-height: 250px;
                overflow-y: auto;
            }}

            .legend-sec-title {{
                font-size: 10px;
                font-weight: 700;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 1px solid rgba(48, 54, 61, 0.6);
                padding-bottom: 3px;
                margin-top: 2px;
            }}

            .legend-grid {{
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}

            .legend-row {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .legend-dot {{
                width: 9px;
                height: 9px;
                border-radius: 50%;
                flex-shrink: 0;
            }}

            .legend-ring {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                border: 2px solid;
                flex-shrink: 0;
            }}

            .legend-name {{
                font-size: 11px;
                color: #c9d1d9;
            }}

            /* FRP Scale */
            .frp-scale-row {{
                display: flex;
                justify-content: space-around;
                align-items: center;
                background: rgba(13, 17, 23, 0.6);
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 5px;
            }}

            .frp-dot-item {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 3px;
                font-size: 9px;
                color: #8b949e;
            }}

            .frp-bubble {{
                border-radius: 50%;
                background: #e74c3c;
                border: 1px solid #ffffff;
            }}
            .bubble-sm {{ width: 8px; height: 8px; }}
            .bubble-md {{ width: 13px; height: 13px; }}
            .bubble-lg {{ width: 18px; height: 18px; }}

            /* Metrics Box */
            .legend-metrics-row {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 5px;
                margin-top: 2px;
            }}

            .metric-box {{
                background: rgba(13, 17, 23, 0.8);
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 4px 2px;
                text-align: center;
                display: flex;
                flex-direction: column;
            }}

            .metric-val {{
                font-size: 12px;
                font-weight: 700;
            }}

            .metric-lbl {{
                font-size: 9px;
                color: #8b949e;
                text-transform: uppercase;
            }}

            @media (max-width: 600px) {{
                .gis-legend-hud {{
                    display: none; /* Hide secondary HUD on tiny mobile screens to preserve view */
                }}
            }}
        </style>

        <script>
            (function() {{
                var legendBody = document.getElementById('legendBody');
                if (legendBody) legendBody.style.display = 'flex';
                // Always-visible legend (no toggle)
                window.toggleLegend = function() {{
                    if (legendBody) {{
                        legendBody.style.display = 'flex';
                    }}
                }};
            }})();
        </script>
        """

        m.get_root().html.add_child(branca.element.Element(legend_html))
        logger.info("Glassmorphic GIS Legend HUD successfully injected into Folium map.")
