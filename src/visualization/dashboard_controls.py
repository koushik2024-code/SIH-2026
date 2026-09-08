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

import folium
from folium.plugins import Geocoder
import branca.element
import pandas as pd

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

        panel_html = f"""
        <div id="gisFilterDock" class="gis-filter-dock">
            <!-- Header with Minimize Toggle -->
            <div class="filter-dock-header" id="dockHeader">
                <div class="dock-title">
                    <i class="fas fa-sliders-h"></i>
                    <span>GIS Dashboard Controls</span>
                </div>
                <button type="button" id="btnToggleDock" class="dock-toggle-btn" title="Minimize/Maximize Controls">
                    <i class="fas fa-chevron-up" id="dockToggleIcon"></i>
                </button>
            </div>

            <!-- Collapsible Dock Content -->
            <div class="filter-dock-body" id="dockBody">
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
            /* Glassmorphic GIS Filter Dock */
            .gis-filter-dock {{
                position: fixed;
                top: 80px;
                left: 12px;
                width: 320px;
                background: rgba(22, 27, 34, 0.94);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(48, 54, 61, 0.85);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
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

                // Wait for Leaflet map to be initialized
                function initGISControls() {{
                    var map = null;
                    for (var key in window) {{
                        if (window[key] && window[key]._layers && typeof window[key].flyTo === 'function') {{
                            map = window[key];
                            break;
                        }}
                    }}

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

                    // Facility Search FlyTo
                    function jumpToFacility() {{
                        var inputVal = document.getElementById('facSearchInput').value.trim();
                        if (!inputVal) return;

                        var target = null;
                        for (var i = 0; i < facilitiesData.length; i++) {{
                            if (facilitiesData[i].name.toLowerCase().indexOf(inputVal.toLowerCase()) !== -1) {{
                                target = facilitiesData[i];
                                break;
                            }}
                        }}

                        if (target && map) {{
                            map.flyTo([target.lat, target.lon], 14, {{ duration: 1.5 }});
                            if (typeof L !== 'undefined') {{
                                var pulse = L.circleMarker([target.lat, target.lon], {{
                                    radius: 20,
                                    color: '#58a6ff',
                                    weight: 3,
                                    fillColor: '#58a6ff',
                                    fillOpacity: 0.3
                                }}).addTo(map);
                                pulse.bindPopup("<b>" + target.name + "</b><br>Coordinates: " + target.lat + ", " + target.lon).openPopup();
                                setTimeout(function() {{ map.removeLayer(pulse); }}, 8000);
                            }}
                        }}
                    }}

                    var searchInput = document.getElementById('facSearchInput');
                    if (searchInput) {{
                        searchInput.addEventListener('change', jumpToFacility);
                        searchInput.addEventListener('keydown', function(e) {{
                            if (e.key === 'Enter') jumpToFacility();
                        }});
                    }}
                    var goBtn = document.getElementById('btnGoFac');
                    if (goBtn) goBtn.addEventListener('click', jumpToFacility);

                    // Dock Minimize / Maximize Toggle
                    var dockHeader = document.getElementById('dockHeader');
                    var dockBody = document.getElementById('dockBody');
                    var dockIcon = document.getElementById('dockToggleIcon');
                    var isDockCollapsed = false;

                    function toggleDock() {{
                        isDockCollapsed = !isDockCollapsed;
                        if (isDockCollapsed) {{
                            dockBody.style.display = 'none';
                            dockIcon.className = 'fas fa-chevron-down';
                        }} else {{
                            dockBody.style.display = 'flex';
                            dockIcon.className = 'fas fa-chevron-up';
                        }}
                    }}

                    if (dockHeader) dockHeader.addEventListener('click', toggleDock);

                    // Live Filter Engine
                    function applyMapFilters() {{
                        var startDate = document.getElementById('dockDateStart').value;
                        var endDate = document.getElementById('dockDateEnd').value;
                        var minConf = parseInt(document.getElementById('dockConfSlider').value, 10);

                        // Update confidence badge
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
                        document.querySelectorAll('.map-type-cb:checked').forEach(function(cb) {{
                            checkedTypes.add(cb.value);
                        }});

                        var visibleCount = 0;
                        var totalCount = 0;

                        if (window._fireMarkerRegistry && window._fireMarkerRegistry.length) {{
                            totalCount = window._fireMarkerRegistry.length;
                            window._fireMarkerRegistry.forEach(function(item) {{
                                var m = item.marker;
                                var data = item.data;

                                var typeMatch = checkedTypes.has(data.fire_type);
                                var confMatch = (data.confidence_num >= minConf);
                                var dateMatch = true;
                                if (startDate && data.acq_date < startDate) dateMatch = false;
                                if (endDate && data.acq_date > endDate) dateMatch = false;

                                var shouldShow = typeMatch && confMatch && dateMatch;
                                if (shouldShow) {{
                                    visibleCount++;
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
                    }}

                    // Attach Listeners
                    document.querySelectorAll('.map-type-cb').forEach(function(cb) {{
                        cb.addEventListener('change', applyMapFilters);
                    }});

                    var dateStart = document.getElementById('dockDateStart');
                    var dateEnd = document.getElementById('dockDateEnd');
                    if (dateStart) dateStart.addEventListener('change', applyMapFilters);
                    if (dateEnd) dateEnd.addEventListener('change', applyMapFilters);

                    var confSlider = document.getElementById('dockConfSlider');
                    if (confSlider) confSlider.addEventListener('input', applyMapFilters);

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
                    }}));

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
                <button type="button" id="btnToggleLegend" class="legend-toggle-btn" title="Toggle Legend View">
                    <i class="fas fa-chevron-up" id="legendToggleIcon"></i>
                </button>
            </div>

            <div class="legend-body" id="legendBody">
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
                bottom: 24px;
                right: 24px;
                width: 290px;
                background: rgba(22, 27, 34, 0.93);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(48, 54, 61, 0.85);
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
                color: #e6edf3;
                font-size: 11px;
                transition: all 0.3s ease;
                overflow: hidden;
            }}

            .legend-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 9px 12px;
                background: rgba(33, 38, 45, 0.95);
                border-bottom: 1px solid rgba(48, 54, 61, 0.8);
                cursor: pointer;
                user-select: none;
            }}

            .legend-title {{
                display: flex;
                align-items: center;
                gap: 6px;
                font-weight: 700;
                font-size: 12px;
                color: #f0f6fc;
            }}

            .legend-toggle-btn {{
                background: transparent;
                border: none;
                color: #8b949e;
                cursor: pointer;
                padding: 3px 6px;
                border-radius: 4px;
            }}
            .legend-toggle-btn:hover {{
                color: #ffffff;
                background: rgba(255, 255, 255, 0.1);
            }}

            .legend-body {{
                padding: 10px 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                max-height: 380px;
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
                padding: 6px;
            }}

            .frp-dot-item {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
                font-size: 9px;
                color: #8b949e;
            }}

            .frp-bubble {{
                border-radius: 50%;
                background: #e74c3c;
                border: 1px solid #ffffff;
            }}
            .bubble-sm {{ width: 8px; height: 8px; }}
            .bubble-md {{ width: 14px; height: 14px; }}
            .bubble-lg {{ width: 20px; height: 20px; }}

            /* Metrics Box */
            .legend-metrics-row {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 6px;
                margin-top: 4px;
            }}

            .metric-box {{
                background: rgba(13, 17, 23, 0.8);
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 4px;
                text-align: center;
                display: flex;
                flex-direction: column;
            }}

            .metric-val {{
                font-size: 13px;
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
                var legendHeader = document.getElementById('legendHeader');
                var legendBody = document.getElementById('legendBody');
                var legendIcon = document.getElementById('legendToggleIcon');
                var isCollapsed = false;

                if (legendHeader && legendBody) {{
                    legendHeader.addEventListener('click', function() {{
                        isCollapsed = !isCollapsed;
                        if (isCollapsed) {{
                            legendBody.style.display = 'none';
                            legendIcon.className = 'fas fa-chevron-down';
                        }} else {{
                            legendBody.style.display = 'flex';
                            legendIcon.className = 'fas fa-chevron-up';
                        }}
                    }});
                }}
            }})();
        </script>
        """

        m.get_root().html.add_child(branca.element.Element(legend_html))
        logger.info("Glassmorphic GIS Legend HUD successfully injected into Folium map.")
