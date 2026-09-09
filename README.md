---

# 🔥 AI-Based Detection and Classification of Industrial Fires & Persistent Thermal Sources

> Using NASA FIRMS, OpenStreetMap & Satellite Data | SIH 2026 | NTRO

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/koushik2024-code/SIH-2026)
[![Live Deployed Website](https://img.shields.io/badge/Live%20Demo-Website%20Active-success?logo=google-chrome&logoColor=white)](https://koushik2024-code.github.io/SIH-2026/)
![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Part%201%20%26%20Web%20Dashboard%20Complete-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

### 🔗 Project Links
| Resource | URL |
| :--- | :--- |
| **🌐 Live Deployed Website** | [https://koushik2024-code.github.io/SIH-2026/](https://koushik2024-code.github.io/SIH-2026/) |
| **💻 Official GitHub Repository** | [https://github.com/koushik2024-code/SIH-2026](https://github.com/koushik2024-code/SIH-2026) |

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Project Parts Overview](#-project-parts-overview)
- [Detailed Flow of All 5 Parts](#-detailed-flow-of-all-5-parts)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Team](#-team)

## 🎯 Problem Statement
Explain the SIH problem: Industrial facilities generate thermal signatures observable from space. NASA FIRMS detects thermal anomalies but cannot distinguish between industrial fires, gas flares, agricultural burning, mining activity, and wildfires. The challenge is to build an AI system that classifies and monitors these thermal sources.

## 🏗️ Solution Architecture
```text
┌─────────────────────────────────────────────────────────────────┐
│                    SOLUTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐ │
│  │ NASA     │   │ OSM      │   │ Sentinel  │   │ Land Cover │ │
│  │ FIRMS    │   │ Overpass │   │ Satellite │   │ CORINE     │ │
│  │ API      │   │ API      │   │ Imagery   │   │ Data       │ │
│  └────┬─────┘   └────┬─────┘   └─────┬─────┘   └─────┬──────┘ │
│       │              │               │               │         │
│       ▼              ▼               ▼               ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          PART 1: DATA INGESTION & PREPROCESSING         │   │
│  │  • Fetch thermal anomalies  • Fetch industrial sites    │   │
│  │  • Clean & validate data    • Engineer features         │   │
│  │  • Spatial joins            • Land cover labeling       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          PART 2: SPATIAL ANALYSIS & ENRICHMENT          │   │
│  │  • Proximity analysis       • Buffer zone creation      │   │
│  │  • Hotspot detection        • Temporal clustering       │   │
│  │  • Spatial statistics       • Industrial zone mapping   │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          PART 3: AI/ML CLASSIFICATION ENGINE             │   │
│  │  • Random Forest classifier  • Feature importance       │   │
│  │  • XGBoost ensemble          • Model evaluation         │   │
│  │  • Fire type classification  • Confidence scoring       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          PART 4: GIS VISUALIZATION & WEB DASHBOARD       │   │
│  │  • Interactive map (Folium)  • Layer controls            │   │
│  │  • Heatmap overlays          • Facility markers          │   │
│  │  • Real-time alerts          • Filter panels             │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          PART 5: MONITORING, ALERTS & DEPLOYMENT         │   │
│  │  • Automated scheduling      • Alert system (email/SMS) │   │
│  │  • Historical analysis       • API endpoint             │   │
│  │  • Reporting dashboard       • Docker deployment        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Project Parts Overview
| Part | Name | Status | Description |
|------|------|--------|-------------|
| 1 | Data Ingestion & Preprocessing | ✅ Complete | Fetches thermal data from NASA FIRMS, industrial facilities from OSM, cleans data, engineers features |
| - | Interactive Web Dashboard (Part 1 Preview) | ✅ Complete | Full-featured GIS web application (Flask + Leaflet/Folium + Chart.js) with real-time analytics & filters |
| 2 | Spatial Analysis & Enrichment | 🔲 Planned | Proximity analysis, hotspot detection, temporal clustering, spatial statistics |
| 3 | AI/ML Classification Engine | 🔲 Planned | ML models to classify fire types (industrial, forest, agricultural, etc.) |
| 4 | GIS Visualization & Advanced Dashboard | ✅ Complete (4.1, 4.2, 4.3 & 4.4 Complete) | Production Folium/Leaflet GIS map with 4 base layers, category sub-layers, multi-ring buffers, temporal HeatMapWithTime slider, interactive dashboard controls, and comprehensive visual analytics panels |
| 5 | Monitoring, Alerts & Deployment | ✅ Complete (5.1, 5.2, 5.3, 5.4 & 5.5 Complete) | Automated 6-hour ingestion pipeline, multi-channel alert dispatcher, high-performance FastAPI REST API, longitudinal analytics with PDF/HTML/CSV reporting, and multi-service Docker containerization with deep health monitoring & CI/CD |

## 🔄 Detailed Flow of All 5 Parts

### PART 1: Data Ingestion & Preprocessing Pipeline ✅ COMPLETED

#### 1.1 NASA FIRMS Data Ingestion
**Objective**: Fetch near real-time thermal anomaly data from NASA's Fire Information for Resource Management System.

**Data Sources**:
- VIIRS (Visible Infrared Imaging Radiometer Suite) on NOAA-20 satellite
- VIIRS on Suomi NPP satellite  
- MODIS (Moderate Resolution Imaging Spectroradiometer) on Aqua/Terra

**Process Flow**:
1. API Authentication: Register and obtain MAP_KEY from NASA FIRMS
2. Request Construction: Build URL with source, bounding box (India: 68°E-98°E, 6°N-38°N), day range
3. Data Fetch: GET request to `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{SOURCE}/{AREA}/{DAYS}`
4. Multi-Source Aggregation: Fetch from all 3 satellite sources and concatenate
5. Response Validation: Check HTTP status, content type, data integrity
6. Error Handling: Exponential backoff retry (3 attempts), per-source error isolation

**Output Schema** (per fire detection):
| Field | Type | Description |
|-------|------|-------------|
| latitude | float | Fire detection latitude |
| longitude | float | Fire detection longitude |
| brightness | float | Brightness temperature (Kelvin) |
| frp | float | Fire Radiative Power (MW) |
| confidence | str/int | Detection confidence level |
| acq_date | date | Acquisition date |
| acq_time | int | Acquisition time (HHMM) |
| satellite | str | Source satellite name |
| daynight | str | Day/Night flag |

#### 1.2 OSM Industrial Facilities Ingestion
**Objective**: Build a comprehensive database of industrial infrastructure from OpenStreetMap.

**Facility Types Queried**:
| Category | OSM Tags | Examples |
|----------|----------|----------|
| Thermal Power Plants | power=plant | Coal, gas, nuclear plants |
| Oil Refineries | industrial=refinery | IOCL, BPCL, HPCL refineries |
| Petrochemical Plants | industrial=petrochemical | Chemical complexes |
| Steel/Iron Works | man_made=works + product=steel | TATA Steel, SAIL plants |
| Mining Areas | industrial=mine, landuse=quarry | Coal mines, open-pit mines |
| Gas Flares | man_made=flare | Gas flaring sites |
| LNG Terminals | industrial=gas, content=lng | Dahej, Kochi LNG |
| Petroleum Wells | man_made=petroleum_well | Oil/gas extraction |

**Process Flow**:
1. Build Overpass QL query with bounding box and facility type tags
2. Execute query against Overpass API (overpass-api.de)
3. Parse nodes, ways, and relations from response
4. Extract centroid coordinates for area features (ways/relations)
5. Convert to GeoDataFrame with standardized schema
6. Cache results locally to avoid redundant API calls

#### 1.3 Data Cleaning & Validation
**Steps**:
1. Duplicate Removal: De-duplicate on (lat, lon, date, time)
2. Missing Data: Drop records with null coordinates
3. Range Validation: Ensure lat ∈ [-90, 90], lon ∈ [-180, 180]
4. Confidence Filtering: Keep only medium/high confidence detections
5. Datetime Normalization: Parse and combine date + time fields
6. Unique ID Generation: SHA-based fire_id for each detection

#### 1.4 Feature Engineering
**Spatial Features**:
- distance_to_nearest_industrial: Haversine distance (km) to closest industrial facility
- nearest_facility_type: Category of the closest facility
- nearest_facility_name: Name of the closest facility
- is_near_industrial: Boolean flag (within threshold distance)

**Temporal Features**:
- hour_of_day: Extracted from acquisition time
- day_of_week: Monday=0 to Sunday=6
- month: 1-12
- is_daytime: Boolean from daynight flag

**Thermal Features**:
- brightness_normalized: Min-max normalized brightness
- frp_normalized: Min-max normalized FRP
- brightness_frp_ratio: Ratio of brightness to FRP

**Clustering Features**:
- fire_cluster_id: Group nearby fires (within 1km) into clusters

#### 1.5 Land Cover Assignment (Rule-Based)
- Check proximity to industrial facilities (buffer zone)
- Label fires within buffer as "Industrial"
- Default to "Unknown" for enhancement in Part 3

**Output**: Enriched fire dataset saved as `data/processed/enriched_fire_data.csv`

---

### PART 2: Spatial Analysis & Enrichment 🔲 PLANNED

#### 2.1 Proximity Analysis Engine
**Objective**: Quantify spatial relationships between thermal anomalies and industrial infrastructure.

**Process Flow**:
1. Buffer Zone Creation: Generate concentric buffer zones (500m, 1km, 2km, 5km) around each industrial facility
2. Spatial Join: Intersect fire points with buffer zones to determine containment
3. Proximity Scoring: Assign proximity scores based on distance bands
4. Multi-Facility Analysis: For fires near multiple facilities, rank by distance and facility risk

#### 2.2 Hotspot Detection (Kernel Density Estimation)
**Objective**: Identify statistically significant clusters of thermal activity.

**Process Flow**:
1. Apply Getis-Ord Gi* statistic to identify statistically significant hotspots
2. Use Kernel Density Estimation (KDE) to create continuous heat surfaces
3. Classify hotspots by intensity: Low / Medium / High / Critical
4. Temporal hotspot analysis: Compare hotspot persistence over time windows

#### 2.3 Temporal Clustering
**Objective**: Detect persistent thermal sources vs. one-time fire events.

**Process Flow**:
1. Group fire detections by spatial proximity (DBSCAN clustering)
2. Track each cluster's temporal persistence (days active)
3. Classify as: Transient (<24h), Short-term (1-7 days), Persistent (>7 days)
4. Persistent sources likely = industrial activity; Transient = fire events

#### 2.4 Spatial Statistics
- Moran's I for spatial autocorrelation
- Ripley's K-function for point pattern analysis
- Nearest Neighbor analysis for clustering significance

**Output**: Spatially enriched dataset with hotspot scores, persistence metrics, and cluster assignments.

---

### PART 3: AI/ML Classification Engine 🔲 PLANNED

#### 3.1 Training Data Preparation
**Objective**: Create labeled training dataset for supervised classification.

**Process Flow**:
1. Semi-Automatic Labeling:
   - Fires within 1km of industrial facility → Label: "Industrial Fire"
   - Fires in forest land cover → Label: "Forest Fire"
   - Fires in agricultural land cover → Label: "Agricultural Burning"
   - Persistent thermal sources at known facilities → Label: "Gas Flare" or "Industrial Heat"
   - Remaining fires → Label: "Unknown/Other"
2. Manual Verification: Sample review of auto-labeled data
3. Feature Selection: Statistical analysis to identify most predictive features
4. Class Balancing: SMOTE oversampling for minority classes

#### 3.2 Model Architecture
**Primary Model**: Random Forest Classifier
- Handles mixed feature types well
- Built-in feature importance
- Robust to outliers
- 100-500 estimators, max_depth tuning

**Ensemble Model**: XGBoost Gradient Boosted Trees
- Superior performance on tabular data
- Handles imbalanced classes
- Learning rate, max_depth, n_estimators tuning

**Classification Categories**:
| Class ID | Category | Description |
|----------|----------|-------------|
| 0 | Industrial Fire | Fire at/near industrial facility |
| 1 | Gas Flare | Persistent combustion at oil/gas sites |
| 2 | Forest Fire | Wildfire in forest areas |
| 3 | Agricultural Burning | Crop residue / stubble burning |
| 4 | Mining Activity | Thermal from mining operations |
| 5 | Other/Unknown | Unclassified thermal anomaly |

#### 3.3 Feature Input Vector
All features from Part 1 + Part 2:
- Thermal: brightness, FRP, brightness_frp_ratio
- Spatial: distance_to_nearest_industrial, proximity_score, hotspot_intensity
- Temporal: hour, day_of_week, month, persistence_days
- Context: land_cover_type, facility_type, cluster_size

#### 3.4 Model Training & Evaluation
1. Train/Test Split: 80/20 stratified split
2. Cross-Validation: 5-fold stratified CV
3. Hyperparameter Tuning: GridSearchCV / RandomizedSearchCV
4. Metrics: Accuracy, Precision, Recall, F1-Score (per class), Confusion Matrix, ROC-AUC
5. Model Persistence: Save trained model with joblib

#### 3.5 Inference Pipeline
1. Load trained model
2. Accept new fire data
3. Run through feature pipeline
4. Predict fire type + confidence score
5. Return classified results

**Output**: Trained ML model (.pkl), classification results, evaluation metrics.

---
### PART 4: GIS Visualization & Web Dashboard 🔄 IN PROGRESS (4.1 Complete)

#### 4.1 Interactive Map (Folium/Leaflet.js) ✅ COMPLETED
**Objective**: Build a production-grade web-based GIS interface for real-time visualization and spatial analysis of thermal anomalies and industrial infrastructure.

**Base Map Layers**:
- 🗺️ **OpenStreetMap** (default): Standard street cartography and urban infrastructure.
- 🛰️ **ESRI World Imagery**: High-resolution optical satellite imagery for visual confirmation of facility layout, smoke plumes, and flare locations.
- 🌙 **Dark Canvas / Dark Matter**: High-contrast dark basemap (ESRI Dark Canvas default with zero watermarks; supports CARTO Dark Matter via optional `CARTO_API_KEY`) engineered specifically to accentuate glowing thermal hotspot heatmaps and fire markers.
- 🏔️ **Topographic / Terrain View**: Detailed elevation contours and relief shading to analyze terrain-driven fire behavior and dispersion.

**Interactive GIS Controls**:
- 🎛️ **Layer Control**: Quick-toggle panel to switch base maps and toggle individual data overlays.
- ⛶ **Fullscreen Mode**: Mission-critical operations center view.
- 📏 **Measurement Tool**: Interactive real-time measurement of distances (km/m) and polygon areas.
- 📍 **Mouse Position HUD**: Live cursor tracking displaying coordinates in `Lat: xx.xxxx | Lon: xx.xxxx`.
- 🗺️ **MiniMap**: Inset reference map with display toggle.
- 📊 **Floating Glassmorphic GIS Legend**: Collapsible dark-mode legend detailing fire classifications, FRP indicators, facility markers, and active stats.

**Execution**:
```bash
# Generate the interactive GIS map (outputs to output/interactive_map.html, map.html, and docs/map.html)
python main.py --part 4.1

# Run with simulated fire anomaly detections
python main.py --part 4.1 --simulate
```

#### 4.2 Data Overlay Layers ✅ COMPLETED
**Objective**: Build rich multi-layered geospatial overlays for AI fire classifications, industrial facility infrastructure, multi-ring safety buffers, and dual-mode thermal intensity surfaces.

**Fire Detection Layer (`FeatureGroupSubGroup`)**:
- Individual toggleable category sub-layers bound to a parent `MarkerCluster`:
  - 🔴 **Industrial Fire**: Combustion signatures within proximity to refineries, power plants, and chemical sites.
  - 🟠 **Gas Flare**: Persistent combustion vents at oil/gas extraction and processing complexes.
  - 🟢 **Forest Fire**: Vegetative wildfires in designated forestry tracts.
  - 🟡 **Agricultural Burning**: High-frequency seasonal crop residue / stubble burning.
  - ⚫ **Mining Activity**: Open-pit thermal anomalies and smoldering spoil heaps.
  - ⚪ **Other/Unknown**: Unclassified thermal hotspots under evaluation.
- Marker size dynamically scaled by Fire Radiative Power: $r = \min(\max(\text{FRP} / 8, 4), 22)\text{px}$.
- Interactive popup detail cards: Detection ID, coordinates, FRP (MW), brightness (K), confidence level, UTC timestamp, diurnal period (Day/Night), nearest industrial facility, and proximity distance with high-hazard alerts for fires $\le 2\text{km}$.

**Industrial Facility Layer**:
- Distinct FontAwesome 6 icons categorized by industry type (Refineries, Power plants, Steel works, Mining, Petrochemical, Gas Flares, LNG terminals, Petroleum wells).
- Interactive popup cards displaying facility category, coordinates, commercial operator, and safety buffers.

**Multi-Ring Concentric Hazard Buffer Zones**:
- Semi-transparent radial impact rings centered on industrial infrastructure:
  - ⭕ **500m**: Immediate Danger Zone (Red, high opacity)
  - ⭕ **1km**: Critical Impact Zone (Orange)
  - ⭕ **2km**: Thermal Exposure Zone (Yellow)
  - ⭕ **5km**: Regional Surveillance Perimeter (Blue)

**Dual-Mode Thermal Heatmap Layers**:
- 🌡️ **Static FRP-Weighted Heatmap**: Continuous thermal surface using `folium.plugins.HeatMap` with calibrated multi-spectral color gradient (Blue $\to$ Cyan $\to$ Green $\to$ Yellow $\to$ Red).
- ⏳ **Multi-Temporal Time-Lapse Slider**: Temporal animation using `folium.plugins.HeatMapWithTime` grouping thermal observations chronologically across satellite pass cycles with Play/Pause controls and speed scrubbing.

**Execution**:
```bash
# Generate the full Part 4.2 Data Overlay map (outputs to output/overlay_map.html, map.html, and docs/map.html)
python main.py --part 4.2

# Run Part 4.2 with simulated fire anomaly detections
python main.py --part 4.2 --simulate
```

#### 4.3 Dashboard Controls ✅ COMPLETED
**Objective**: Equip the interactive GIS environment and web dashboard with dynamic, real-time controls for precision anomaly filtering, geocoding, and temporal exploration.

**Integrated Control Capabilities**:
- 🎛️ **Layer Toggle Panel**:
  - Independent toggling of base cartography (OSM, Satellite, Dark Matter, Topo).
  - Category-specific sub-layer toggles for each fire classification.
  - Granular layer visibility controls for industrial infrastructure, concentric hazard buffers, and thermal intensity surfaces.
- 📅 **Date Range Filter**:
  - Start Date and End Date calendar pickers with rapid presets (**All Dates**, **Last 24h**, **Last 48h**, **Last 7d**).
  - Synchronized client-side Leaflet marker pruning (60 FPS, no page reload) on standalone maps and query parameter filtering (`?start_date=&end_date=`) in the web dashboard.
- ☑️ **Fire Type Filter (Checkboxes)**:
  - 6 dedicated category filter checkboxes with theme-matched color swatches and live detection count badges.
  - **Select All** / **Clear All** rapid toggle links for multi-category isolation.
- 🎯 **Confidence Level Slider**:
  - Continuous threshold slider (0% to 100%) with real-time HUD readout badge (`≥ 60%`, `High (≥ 80%)`).
  - Prunes low-confidence observations dynamically across all active clusters.
- 🔍 **Search by Location & Infrastructure**:
  - **Nominatim Geocoder**: Integrated search box (`folium.plugins.Geocoder`) for panning and zooming to any global address, city, or coordinate in India.
  - **Facility Quick-Jump**: Autocomplete search datalist of major refineries, power stations, steel works, and chemical complexes with instant `flyTo()` navigation and animated proximity pulse.
- 📊 **Glassmorphic Legend HUD**:
  - Floating, collapsible mission-control legend panel detailing classification hex colors, FRP marker sizing bubbles (<15 MW, 15–50 MW, >50 MW), multi-ring safety buffer distances (500m, 1km, 2km, 5km), and live anomaly counters.

**Execution**:
```bash
# Generate the full Part 4.3 Dashboard Controls map (outputs to output/dashboard_map.html, map.html, and docs/map.html)
python main.py --part 4.3

# Run Part 4.3 with simulated satellite anomaly detections
python main.py --part 4.3 --simulate

# Launch full web dashboard with interactive sidebar filter controls
python main.py --part web --port 5000
```

#### 4.4 Analytics Panels & Comprehensive Visual Intelligence ✅ COMPLETED
**Objective**: Deliver high-impact decision support through statistical aggregation, visual intelligence dashboards, static publication charts, and REST API endpoints.

**Core Analytics Components**:
1. **Fire Type Distribution (Pie / Donut & Category Breakdown)**:
   - Dynamic proportional breakdown across all 6 thermal anomaly classifications (*Industrial Fire*, *Gas Flare*, *Forest Fire*, *Agricultural Burning*, *Mining Activity*, *Other/Unknown*).
   - Metrics include total detection counts, percentage shares, cumulative Fire Radiative Power (MW), mean FRP per category, and average brightness temperature (K).
   - Interactive Chart.js cutout doughnut with custom tooltip hover effects and responsive legend.

2. **Time Series of Fire Detections (Temporal Frequency & Energy Evolution)**:
   - Chronological daily and hourly surveillance histograms.
   - Dual-axis visual mapping: Primary Y-axis tracks daily detection frequency (bar chart); Secondary Y-axis tracks cumulative thermal energy output (FRP in MW line overlay with spline smoothing).
   - Categorical stratification tracking industrial vs. vegetative fire trends over observation windows.

3. **Top Facilities Proximity Hazard Leaderboard**:
   - Automated spatial proximity ranking identifying critical industrial infrastructure facing active thermal exposure within 5km.
   - Granular hazard ring containment breakdown: counts within `≤ 500m` (Immediate Danger), `500m–1km` (Critical Impact), `1km–2km` (Thermal Exposure), and `2km–5km` (Regional Perimeter).
   - Dynamic Criticality Tiers (**CRITICAL**, **HIGH**, **MEDIUM**, **LOW**) assigned by proximity severity and anomaly concentration.
   - Minimum proximity distance (km), peak FRP (MW), and average radiative power.

4. **Regional Surveillance Statistics & Quadrant Risk Matrix**:
   - Executive surveillance KPIs: Total active thermal events, industrial facility exposure rate (%), mean and peak FRP (MW), and high-confidence verification rate (%).
   - Diurnal solar distribution tracking daytime (☀️) vs nighttime (🌙) thermal activity.
   - Spatial quadrant risk partitioning dividing the Indian landmass into North-East, North-West, South-East, and South-West operational surveillance sectors with dominant fire profiles and threat statuses.

**Reporting & Deployment Surfaces**:
- 🌐 **Flask Web Dashboard**: Accessible at `/analytics` with interactive Chart.js visualizations, responsive dark-mode styling, and live navbar navigation.
- 🔌 **REST API Endpoints**:
  - `GET /api/analytics`: Delivers full structured JSON payload with optional filtering (`?fire_type=&start_date=&end_date=&min_confidence=`).
  - `GET /api/analytics/export`: Triggers server-side static chart rendering and HTML report export.
- 📄 **Standalone Intelligence Report**: Self-contained HTML report (`output/analytics_dashboard.html`, `analytics.html`, `docs/analytics.html`) containing embedded Chart.js scripts and glassmorphic tables for air-gapped distribution.
- 📊 **Publication-Quality Matplotlib Figures**: High-resolution static PNG figures generated in `output/` (`fire_type_distribution.png`, `time_series_detections.png`, `top_facilities_fires.png`, `regional_quadrants.png`).

**Execution**:
```bash
# Run standalone Part 4.4 Analytics engine (generates HTML report, JSON summary, and static PNGs)
python main.py --part 4.4

# Run Part 4.4 with simulated satellite anomaly detections
python main.py --part 4.4 --simulate

# Launch web dashboard and navigate to /analytics in your browser
python main.py --part web --port 5000
```

---

### PART 5: Monitoring, Alerts & Deployment 🔄 IN PROGRESS

#### 5.1 Automated Data Pipeline ✅ COMPLETED
**Objective**: Schedule automated data collection, incremental anomaly fetching, ML auto-classification, persistent SQLite database updates, and live dashboard map refresh.

**Architecture & Process Flow**:
1. **Cron Job / Task Scheduler**:
   - Python in-process daemon: `python main.py --part 5 --interval-hours 6`
   - Windows Task Scheduler: `scripts/setup_scheduled_task.bat` or `scripts/setup_scheduled_task.ps1`
   - Linux / macOS Crontab: `scripts/setup_cron.sh` (`0 */6 * * *`)
   - Cloud CI/CD: `.github/workflows/pipeline_scheduler.yml` runs every 6 hours and auto-updates GitHub Pages!
2. **Incremental Data Fetch**:
   - Uses deterministic SHA-256 signatures (`detection_id`) per detection.
   - Compares incoming detections against `data/fire_monitoring.db` and state checkpoint `data/pipeline_state.json`.
   - Filters out previously seen detections; only processes new thermal events.
   - Includes automatic simulation fallback for offline testing or demo mode (`--simulate`).
3. **Auto-Classification Engine**:
   - Pre-trained / self-bootstrapping Random Forest classifier (`src/pipeline_automation/classifier.py`).
   - Feature vector: `[brightness, FRP, ratio, distance_to_industry, is_near_industrial, hour, day, month, is_daytime, cluster_id]`.
   - Classifies detections into: Industrial Fire (0), Gas Flare (1), Forest Fire (2), Agricultural Burning (3), Mining Activity (4), Other/Unknown (5) with calibrated confidence probabilities.
4. **Persistent Storage Update**:
   - Stores all records in SQLite database `data/fire_monitoring.db` (WAL mode enabled, indexed by coordinates, date, and fire type).
   - Records execution metrics in `pipeline_runs` table.
   - Exports synchronized `data/processed/enriched_fire_data.csv` and `data/processed/latest_detections.geojson`.
5. **Dashboard & Map Refresh**:
   - Regenerates Folium HTML maps: `map.html` (root) and `docs/map.html` (GitHub Pages).
   - Exports aggregate statistics to `data/processed/latest_stats.json`.
   - Updates live Flask dashboard cache and exposes endpoints:
     - `GET /api/pipeline/status`: Inspect pipeline health and SQLite stats.
     - `POST /api/pipeline/run`: Trigger manual incremental pass on demand.

**Execution Commands**:
```bash
# 1. Run a single incremental batch (manual / test)
python main.py --part 5 --run-once

# 2. Run with incremental simulation feed (test / demonstration)
python main.py --part 5 --run-once --simulate

# 3. Run recurring 6-hour scheduler daemon
python main.py --part 5 --interval-hours 6

# 4. View live updated web dashboard
python main.py --part web --port 5000
```

#### 5.2 Multi-Channel Alert System ✅ COMPLETED
**Objective**: Detect critical thermal anomalies in real time, evaluate deterministic & statistical hazard trigger conditions, and instantly dispatch structured incident alerts across multiple delivery channels with intelligent cooldown deduplication.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PART 5.2: MULTI-CHANNEL ALERT SYSTEM ARCHITECTURE            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Automated Pipeline (Part 5.1)] ──► Enriched Thermal & Spatial Detections │
│                                                │                            │
│                                                ▼                            │
│                               ┌──────────────────────────────────┐          │
│                               │     Trigger Condition Engine     │          │
│                               │  • New Industrial Fire (>80%)    │          │
│                               │  • Facility Thermal Spike        │          │
│                               │  • Industrial Cluster Formation  │          │
│                               │  • Persistent Fire (>48 Hours)   │          │
│                               └────────────────┬─────────────────┘          │
│                                                │                            │
│                                                ▼                            │
│                               ┌──────────────────────────────────┐          │
│                               │   Alert Dispatcher & Cooldown    │          │
│                               │  • Cooldown Check (60 min default)│         │
│                               │  • Severity Escalation Bypass    │          │
│                               │  • SQLite Persistent Logging     │          │
│                               └───────┬───────┬──────────┬───────┘          │
│                                       │       │          │                  │
│            ┌──────────────────────────┼───────┴──────────┼───────────────┐  │
│            ▼                          ▼                  ▼               ▼  │
│     ┌─────────────┐            ┌─────────────┐    ┌─────────────┐ ┌────────┐│
│     │    Email    │            │     SMS     │    │  Dashboard  │ │ Log &  ││
│     │ (SMTP/HTML) │            │  (Twilio)   │    │  (SSE Stream)│ │Webhook││
│     └─────────────┘            └─────────────┘    └─────────────┘ └────────┘│
│                                                          │                  │
│                                                          ▼                  │
│                                                 Live Toast & HUD Audio      │
│                                                 /alerts Incident Center     │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Trigger Conditions Evaluated
1. **New Industrial Fire (`NEW_INDUSTRIAL_FIRE`)**:
   - Triggers when a detection is classified as `Industrial Fire` with model confidence $> 80\%$, OR when any high-confidence ($> 80\%$) thermal anomaly is detected within $2.0\text{ km}$ of a registered industrial plant.
   - Severity: `CRITICAL` if within $1\text{ km}$, `HIGH` if within $2\text{ km}$.
2. **Facility Thermal Spike (`FACILITY_THERMAL_SPIKE`)**:
   - Detects abnormal heat surges where brightness temperature $\ge 365\text{ K}$ or Fire Radiative Power (FRP) $\ge 50\text{ MW}$, or when current FRP exceeds $2.0\times$ the historical baseline of the facility.
   - Severity: `CRITICAL` ($\ge 380\text{ K}$ or $\ge 80\text{ MW}$), `HIGH` otherwise.
3. **Industrial Cluster Formation (`INDUSTRIAL_CLUSTER_FORMATION`)**:
   - Identifies $\ge 3$ active fire detections co-located within a $3.0\text{ km}$ radius of an industrial zone, representing rapidly spreading or multi-point combustion.
   - Severity: `CRITICAL` for large clusters ($\ge 5$ points), `HIGH` for smaller clusters.
4. **Persistent Fire Burning > 48 Hours (`PERSISTENT_FIRE_48H`)**:
   - Detects long-duration thermal signatures with temporal persistence duration $> 48.0\text{ hours}$ within $5.0\text{ km}$ of a facility, indicating continuous flare anomalies, coal seam fires, or uncontrolled industrial fires.
   - Severity: `CRITICAL` if active at $< 2\text{ km}$, `HIGH` otherwise.
5. **System Diagnostic Test Alert (`DIAGNOSTIC_TEST`)**:
   - Simulates high-priority multi-spectral telemetry against critical infrastructure to test channel connectivity end-to-end.

##### 2. Multi-Channel Alert Delivery
- **📧 Email Notifications (SMTP & Responsive Dark HTML)**:
  - Custom HTML template formatted with modern dark aesthetics, severity gradient banners, key metrics (FRP, brightness, confidence, distance), geographic coordinates, and direct links to the GIS Dashboard.
  - Automatic fallback to file simulation preview (`output/latest_alert_email.html`) when SMTP credentials are not configured.
- **📱 SMS Alerts (Twilio REST API)**:
  - Standardized, concise SMS payloads under 160 characters formatted for field emergency teams:
    `[SIH ALERT] CRITICAL: Industrial Fire Detected. Near: Jamnagar Refinery (0.8km). FRP: 72MW, Conf: 95%. Coords: 22.471,70.058. ID: ALT-A1B2C3D4`
  - Automated simulation logger when Twilio credentials are not set.
- **🖥️ Dashboard Notifications (Server-Sent Events / SSE)**:
  - Non-blocking streaming endpoint (`/api/alerts/stream`) continuously broadcasting real-time alert events to all connected clients.
  - Triggers floating glassmorphic toast notifications with severity color accents and synthesized Web Audio alert frequencies.
  - Dedicated **Alert Bell HUD** on the navbar displaying active badge counts.
- **📜 Log File Alerts & JSON Sink**:
  - Detailed audit entries appended to `data/alerts.log`.
  - Structured JSON array persisted to `output/alerts.json` for machine ingestion.
- **🌐 Outbound Webhooks**:
  - Standardized JSON POST payload dispatched to external monitoring platforms, Slack, Discord, or Microsoft Teams.

##### 3. Cooldown & Deduplication Logic
- Prevents alert fatigue by maintaining an in-memory and persistent SQLite record of past alerts per `(facility_name, trigger_type)`.
- Default cooldown window: **60 minutes** (configurable via `ALERT_COOLDOWN_MINUTES`).
- **Severity Escalation Bypass**: If a new incoming anomaly escalates in severity (e.g., from `WARNING` or `HIGH` to `CRITICAL`), the cooldown is automatically bypassed to ensure safety-critical alerts are never dropped.

##### 4. Persistent Storage & Schema
Alerts are stored in the SQLite database (`data/fire_monitoring.db`) with fast indexed lookups:
```sql
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    detection_id TEXT,
    latitude REAL,
    longitude REAL,
    facility_name TEXT,
    facility_type TEXT,
    distance_km REAL,
    frp REAL,
    brightness REAL,
    confidence REAL,
    title TEXT NOT NULL,
    description TEXT,
    channels_dispatched TEXT,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    acknowledged_at TEXT,
    acknowledged_by TEXT
);
```

##### 5. Web Interface: Incident Operations Center (`/alerts`)
- Access the dedicated operations center at `http://localhost:5000/alerts`:
  - **KPI Cards**: Active alerts, Critical incidents, High-priority anomalies, Total resolved alerts.
  - **Filter Toolbar**: Instant filtering by severity (`CRITICAL`, `HIGH`, `WARNING`, `INFO`) and lifecycle status (`ACTIVE`, `ACKNOWLEDGED`).
  - **One-Click Acknowledgment**: Operators can acknowledge active alerts directly from the table.
  - **Real-Time Live Feed**: Automatically updates via SSE stream without page reload.
  - **Test Trigger Button**: Dispatches an instant diagnostic test alert across all channels.

##### 6. REST API Endpoints (Alerts)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/alerts` | Incident Operations Center Web UI |
| `GET` | `/api/alerts` | Retrieve alerts with filtering (`limit`, `severity`, `status`, `trigger_type`) |
| `GET` | `/api/alerts/stats` | Summary counts of alerts by severity and status |
| `POST` | `/api/alerts/<alert_id>/ack` | Acknowledge an active alert |
| `POST` | `/api/alerts/test` | Trigger a live diagnostic test alert across all channels |
| `GET` | `/api/alerts/stream` | Server-Sent Events (SSE) live stream for browser clients |

##### 7. Execution Commands
```bash
# 1. Run standalone Part 5.2 alert evaluation
python main.py --part 5.2

# 2. Run standalone alert evaluation with simulated multi-spectral telemetry
python main.py --part 5.2 --simulate

# 3. Fire an instant diagnostic test alert across all channels
python main.py --part 5.2 --test-alert

# 4. Run automated pipeline with integrated alert evaluation
python main.py --part 5 --run-once

# 5. Launch web application with live alert center
python main.py --part web --port 5000
```


#### 5.3 Production REST API (FastAPI) ✅ COMPLETED
**Objective**: Deliver a high-performance, asynchronous production REST API service built on **FastAPI**, **Uvicorn**, and **Pydantic v2**, providing standard JSON endpoints for active fires, industrial facility registry, thermal hotspots, analytics stats, and on-demand machine learning classification inference.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PART 5.3: FASTAPI REST ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [HTTP / Frontends / External Services]                                     │
│        │                                                                    │
│        ▼                                                                    │
│  FastAPI Gateway (Uvicorn ASGI Server)                                      │
│  • CORS Middleware (*), Interactive Swagger UI (/docs), ReDoc (/redoc)      │
│  • Pydantic v2 Strict Validation & Serialization                            │
│        │                                                                    │
│        ├──► GET  /api/fires               (Paginated detections & filters)  │
│        ├──► GET  /api/fires/{id}          (Single fire details / 404)       │
│        ├──► GET  /api/fires/type/{type}   (Filter by classification class)  │
│        ├──► GET  /api/facilities         (Industrial facility catalog)     │
│        ├──► GET  /api/hotspots           (Ranked thermal hotspot hubs)     │
│        ├──► GET  /api/stats              (System & thermal summary stats)  │
│        ├──► POST /api/classify           (On-demand ML model inference)    │
│        └──► GET  /api/health             (DB connection & engine status)   │
│                                                                             │
│        ▼                                                                    │
│  APIService Business Layer (src/api/service.py)                             │
│        ├──► FireMonitoringDatabase (SQLite persistent tables & indexes)     │
│        ├──► FireClassifier (Trained Random Forest / Heuristic engine)       │
│        └──► Fallback DemoDataGenerator & GeoJSON facility catalog           │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 1. API Endpoints Reference
| Method | Endpoint | Query Parameters | Description | Status Code |
|:---:|:---|:---|:---|:---:|
| `GET` | `/` | — | API welcome metadata and navigation links | `200` |
| `GET` | `/health` / `/api/health` | — | Health check, SQLite connection, and ML status | `200` |
| `GET` | `/docs` | — | Interactive Swagger UI API documentation | `200` |
| `GET` | `/redoc` | — | ReDoc responsive documentation interface | `200` |
| `GET` | `/api/fires` | `limit`, `offset`, `fire_type`, `min_confidence`, `is_near_industrial`, `start_date`, `end_date` | Paginated fire detections with multi-criteria filtering | `200` |
| `GET` | `/api/fires/{detection_id}` | — | Single thermal anomaly by primary detection ID | `200` / `404` |
| `GET` | `/api/fires/type/{fire_type}` | `limit`, `offset` | Direct filter by classification category | `200` |
| `GET` | `/api/facilities` | `facility_type`, `limit` | Industrial facility registry (refineries, power plants, mines) | `200` |
| `GET` | `/api/hotspots` | `min_frp`, `limit` | High-priority thermal hotspots ranked by composite intensity score | `200` |
| `GET` | `/api/stats` | — | Aggregate metrics (avg brightness, FRP, distribution, day/night) | `200` |
| `POST` | `/api/classify` | *JSON payload* | On-demand ML classification on single or batch fire observations | `200` / `422` |

##### 2. Sample Request & Response Payloads

###### On-Demand ML Classification (`POST /api/classify`)
**Request**:
```json
{
  "records": [
    {
      "latitude": 22.4707,
      "longitude": 70.0577,
      "brightness": 395.0,
      "frp": 80.0,
      "confidence": 95.0,
      "distance_to_nearest_industrial": 0.2,
      "nearest_facility_type": "petrochemical"
    }
  ]
}
```
**Response**:
```json
{
  "total_classified": 1,
  "results": [
    {
      "fire_type": "Industrial Fire",
      "fire_type_id": 0,
      "classification_confidence": 0.942,
      "is_near_industrial": 1,
      "distance_to_nearest_industrial": 0.2,
      "nearest_facility_name": "Jamnagar Refinery (Reliance)",
      "nearest_facility_type": "petrochemical"
    }
  ]
}
```

###### Thermal Hotspots (`GET /api/hotspots?min_frp=30&limit=2`)
**Response**:
```json
{
  "total": 2,
  "data": [
    {
      "detection_id": "3e186a86dac28b2d",
      "latitude": 22.471,
      "longitude": 70.058,
      "brightness": 382.4,
      "frp": 72.5,
      "confidence": 95,
      "intensity_score": 87.81,
      "fire_type": "Industrial Fire",
      "facility_nearby": "Jamnagar Petrochemical Complex",
      "distance_km": 0.45,
      "cluster_id": 1
    }
  ]
}
```

##### 3. How to Launch the API Server
```bash
# 1. Start FastAPI server on default port 8000
python main.py --part 5.3

# Or using the 'api' alias:
python main.py --part api

# 2. Specify a custom port
python main.py --part 5.3 --port 8080

# 3. Access Swagger UI documentation
# Open browser at: http://localhost:8000/docs
```


#### 5.4 Historical Analysis & Multi-Format Reporting ✅ COMPLETED
**Objective**: Perform longitudinal spatial-temporal trend modeling, compute dynamic multi-factor facility risk scores over time, and generate publication-grade executive reports in native **PDF** (via ReportLab), **CSV**, and standalone responsive **HTML** dossiers.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PART 5.4: HISTORICAL ANALYSIS & REPORTING ARCHITECTURE       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Fire Detections + Facility Registry] (SQLite Persistent Storage / DB)     │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                HistoricalAnalyzer (src/reporting/analyzer.py)         │  │
│  │  1. Monthly & Quarterly Aggregations (Counts, FRP energy, MoM/QoQ)    │  │
│  │  2. Regional & Category Trend Vectors (North, West, East, South)      │  │
│  │  3. Facility Risk Scoring Engine (Rolling 7d, 30d, 90d window)        │  │
│  │     R(f) = w1*Fires + w2*(PeakFRP/60) + w3*(5 - dist) + w4*Persistence│  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                 │
│         ▼                        ▼                        ▼                 │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐         │
│  │ PDF Generator│         │ CSV Exporter │         │HTML Dashboard│         │
│  │ (ReportLab)  │         │ (Pandas/CSV) │         │ (Executive)  │         │
│  └──────┬───────┘         └──────┬───────┘         └──────┬───────┘         │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│  output/reports/*.pdf     output/reports/*.csv     output/reports/*.html    │
│                                                                             │
│  [FastAPI Endpoints] ──► /api/reports/monthly, /quarterly, /trends, /export │
│  [Web Console]       ──► /reports (Interactive incident analytics UI)      │
│  [CLI Runner]        ──► python main.py --part 5.4                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Longitudinal & Spatial Analytics Engine
1. **Monthly & Quarterly Aggregations**:
   - Compiles total thermal anomalies, mean brightness (K), mean FRP (MW), cumulative thermal energy (MWh), day/night satellite ratios, near-industrial fire counts, and Month-over-Month (MoM) / Quarter-over-Quarter (QoQ) velocity percentages.
2. **Regional Surveillance Quadrants**:
   - **Northern Agro-Industrial Zone** (Punjab, Haryana, UP, NCR): Tracks seasonal crop residue combustion vs. thermal industrial hubs.
   - **Western Energy & Petrochemical Corridor** (Gujarat, Maharashtra, Bombay High Offshore): Monitors petrochemical complexes, oil refineries, and continuous flare stacks.
   - **Central-Eastern Mining & Steel Belt** (Odisha, Jharkhand, Chhattisgarh, MP): Detects coalfield seam fires, smoldering open-cast mines, and blast furnace emissions.
   - **Southern Coastal Infrastructure Zone** (Karnataka, Kerala, Tamil Nadu, AP): Monitors coastal LNG terminals, refineries, and power plants.
   - Assigns directional trend vectors: `Rising (▲)`, `Declining (▼)`, or `Stable (▬)`.
3. **Dynamic Facility Risk Scoring**:
   - Multi-factor risk formula ($0-100$) evaluating nearby fire frequency ($<5\text{ km}$), peak radiative energy, closest anomaly proximity, and persistence duration:
     $$R(f) = \min\left(100, 3.5 \cdot \min(N, 10) + 30 \cdot \frac{\text{FRP}_{\text{peak}}}{60} + 4 \cdot \max(0, 5 - d_{\min}) + 5 \cdot \min(D, 3)\right)$$
   - Risk Tiers:
     - 🔴 `EXTREME RISK` ($\ge 75$)
     - 🟠 `HIGH RISK` ($50 - 74$)
     - 🟡 `MODERATE RISK` ($25 - 49$)
     - 🟢 `LOW RISK` ($< 25$)

##### 2. Multi-Format Report Generation
- **📄 Publication-Grade PDF Dossier (`output/reports/fire_historical_report.pdf`)**:
  - Built with **ReportLab**, featuring official NTRO / SIH 2026 title header, executive summary metrics table, color-coded facility risk leaderboard, monthly longitudinal trend tables, regional surveillance matrix, and security classification footer.
- **🌐 Standalone Executive HTML Report (`output/reports/fire_historical_report.html`)**:
  - Responsive dark-themed dossier with glassmorphic KPI tiles, risk score progress bars, dynamic tables, and print-ready CSS (`@media print`) for 1-click browser PDF printing.
- **📊 Standardized CSV Datasets (`output/reports/*.csv`)**:
  - `monthly_fire_summary.csv`: Monthly counts, energy, and MoM velocity.
  - `quarterly_fire_summary.csv`: Quarterly aggregations and QoQ growth rates.
  - `facility_risk_scores.csv`: Longitudinal risk ranking per industrial asset.
  - `regional_trends.csv`: Surveillance statistics per geographic corridor.

##### 3. Web Console & REST API Endpoints
- **Web UI**: Access the Executive Reporting Console at `http://localhost:5000/reports` with interactive tabs, KPI cards, and one-click export downloads.
- **REST Endpoints**:
  | Method | Endpoint | Description |
  |:---:|:---|:---|
  | `GET` | `/api/reports/monthly` | Monthly aggregated thermal metrics and MoM velocity |
  | `GET` | `/api/reports/quarterly` | Quarterly aggregated metrics and QoQ growth rates |
  | `GET` | `/api/reports/trends` | Regional surveillance trend vectors |
  | `GET` | `/api/reports/facilities/risk` | Facility longitudinal risk scores and tier ratings |
  | `GET` | `/api/reports/summary` | Macro executive statistics and generated artifact metadata |
  | `GET` | `/api/reports/export/pdf` | Stream download of publication-grade PDF report |
  | `GET` | `/api/reports/export/html` | Stream download of standalone executive HTML report |

##### 4. Execution Commands
```bash
# 1. Run Part 5.4 Historical Analysis & generate all report artifacts
python main.py --part 5.4

# Or using the alias:
python main.py --part reporting

# 2. View generated PDF report
# Generated at: output/reports/fire_historical_report.pdf

# 3. View Web Reports Console
python main.py --part web --port 5000
# Open browser at: http://localhost:5000/reports
```


#### 5.5 Deployment & Operations ✅ COMPLETED
**Objective**: Deliver a production-grade containerized deployment architecture featuring multi-service Docker Compose orchestration, unprivileged non-root container security, zero-dependency health monitoring probes, deep diagnostic telemetry, centralized configuration validation, and automated GitHub Actions CI/CD workflows.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PART 5.5: PRODUCTION DEPLOYMENT TOPOLOGY                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     DOCKER COMPOSE ORCHESTRATION                      │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐  │  │
│  │  │  Service: api   │   │  Service: web   │   │   Service: worker   │  │  │
│  │  │ (FastAPI/Uvicorn│   │ (Flask/Folium/UI│   │ (Telemetry Pipeline)│  │  │
│  │  │   Port: 8000)   │   │   Port: 5000)   │   │ (6h Scheduled Cycle)│  │  │
│  │  └────────┬────────┘   └────────┬────────┘   └──────────┬──────────┘  │  │
│  │           │                     │                       │             │  │
│  │           ▼                     ▼                       ▼             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Shared Persistent Volumes: fire_data, fire_output, fire_models  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│         ┌────────────────────────────┼────────────────────────────┐         │
│         ▼                            ▼                            ▼         │
│  ┌──────────────┐             ┌──────────────┐             ┌──────────────┐ │
│  │Health Probes │             │Central Config│             │CI/CD Pipeline│ │
│  │Liveness/Deep │             │src/core/     │             │GitHub Actions│ │
│  │Diagnostics   │             │config.py     │             │ci_cd.yml     │ │
│  └──────────────┘             └──────────────┘             └──────────────┘ │
│                                                                             │
│  [CLI Preflight Runner] ──► python main.py --part 5.5                       │
│  [Docker Healthcheck]   ──► python scripts/healthcheck.py --service api     │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Production Containerization (`Dockerfile` & `docker-compose.yml`)
- **Hardened Multi-Purpose `Dockerfile`**:
  - Based on `python:3.10-slim` with compiled C-extensions (`gcc`, `g++`, `libsqlite3-dev`).
  - Unprivileged non-root execution (`appuser`, UID `10001`) adhering to CIS Docker Benchmarks.
  - Pre-created volume directories (`/app/data`, `/app/output/reports`, `/app/models`) with guaranteed write permissions.
  - Built-in `HEALTHCHECK` directive invoking `python scripts/healthcheck.py --service ${SERVICE_TYPE:-api}`.
  - Dynamic service startup supporting `SERVICE_TYPE=api`, `SERVICE_TYPE=web`, or `SERVICE_TYPE=worker`.
- **Multi-Container Topology (`docker-compose.yml`)**:
  - `api`: High-performance FastAPI REST server running on `http://localhost:8000`.
  - `web`: Flask GIS operations center and executive reporting dashboard on `http://localhost:5000`.
  - `worker`: Automated background daemon executing satellite ingestion every 6 hours.
  - Isolated bridge network `fire_monitoring_net` with named local volumes for zero data loss upon container recreation.
- **Developer Experience (`docker-compose.dev.yml`)**:
  - Hot-reloading live mount overrides for `./src` and `./config`.

##### 2. Centralized Configuration Management (`src/core/config.py`)
- Strongly-typed `AppConfig` manager with automatic `.env` ingestion.
- Pre-flight diagnostic engine (`validate_environment()`) verifying directory writability, SQLite database accessibility, ML model presence, and port collision avoidance.

##### 3. System Health & Deep Diagnostics (`src/monitoring/health.py`)
- **Probes**:
  - `get_liveness()`: Ultra-low overhead ping for orchestrator liveness checks.
  - `get_readiness()`: Confirms database responsiveness and file storage availability.
  - `get_deep_diagnostics()`: Comprehensive system audit analyzing:
    - **Database**: SQLite schema quick-check integrity, total detections, registered facilities, and alert table sizes.
    - **Storage**: Free disk space (`shutil.disk_usage`) and runtime directory write tests.
    - **ML Classifier**: Machine learning model file presence and memory allocation status.
    - **Telemetry Recency**: Timestamp and row count of latest satellite ingestion cycle.
    - **Alert Channels**: Active notification channel status (console, SSE stream, email, SMS, webhook).
- **Standalone Probe Script (`scripts/healthcheck.py`)**: Zero external dependencies (uses standard library `urllib`), suitable for container and Kubernetes probes.
- **REST Endpoints**:
  | Method | Endpoint | Description |
  |:---:|:---|:---|
  | `GET` | `/health` | Standard readiness probe (HTTP 200 / 503) |
  | `GET` | `/health/deep` | Deep diagnostic system report with database, storage & ML metrics |
  | `GET` | `/api/health/deep` | API-prefixed deep diagnostic endpoint |

##### 4. Automated CI/CD Pipeline (`.github/workflows/ci_cd.yml`)
- Multi-stage GitHub Actions workflow triggered on push and pull requests:
  1. `lint-and-validate`: Syntax checks, Python code compilation, and Docker Compose YAML validation.
  2. `test-suite`: Executes complete test suite (64 tests across all modules).
  3. `docker-build-verify`: Builds the production Docker image and executes an in-container direct health check.
  4. `deployment-readiness`: Executes pre-flight verification to confirm production readiness.

##### 5. Execution Commands
```bash
# 1. Run Part 5.5 Deployment Pre-Flight Check
python main.py --part 5.5
# Or using the alias:
python main.py --part deployment

# 2. Run standalone system healthcheck probe
python scripts/healthcheck.py --service system

# 3. Launch with Docker Compose (Production)
docker compose up -d

# 4. Launch with Docker Compose (Development Hot-Reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# 5. Query deep diagnostic health via API
curl -s http://localhost:8000/api/health/deep
curl -s http://localhost:5000/health
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy, GeoPandas |
| Geospatial | Shapely, Folium, Overpy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Folium, Matplotlib, Plotly |
| API Framework | FastAPI |
| Database | GeoJSON, SQLite/PostGIS |
| Deployment | Docker, GitHub Actions |
| Data Sources | NASA FIRMS, OpenStreetMap, CORINE |

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/SIH-2026.git
cd SIH-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your NASA FIRMS API key

# Run Part 1: Data Ingestion & Preprocessing Pipeline
python main.py --part 1

# Launch the Interactive Web Dashboard (Flask + Folium + Chart.js)
python main.py --part web --port 5000
```

### 🌐 Web Dashboard Features (Live Preview)
Navigate to `http://localhost:5000` to view:
- **Interactive Geospatial Map**: Clustered NASA FIRMS hotspots with satellite/dark-mode basemaps and 2km industrial buffers
- **Categorized Thermal Markers**: Distinct color-coding for Industrial Fires (Red), Gas Flares (Orange), Forest Fires (Green), Agricultural Stubble (Yellow), Mining Operations (Gray)
- **Live Analytical Counters**: Real-time stats on average brightness (Kelvin), Fire Radiative Power (MW), day vs. night ratios, and high-confidence detections
- **Dynamic Chart.js Visualizations**: Breakdown of thermal sources across industrial clusters
- **Filtering & Refresh**: Instant filter by fire category and instant data re-synthesis

#### REST API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive Web Dashboard UI |
| `/api/stats` | GET | Thermal detection summary statistics (JSON) |
| `/api/fires` | GET | Active fire detection records with engineered features |
| `/api/facilities` | GET | Industrial facility registry with coordinates and types |
| `/api/refresh` | POST/GET | Re-query / refresh active thermal sources |

## 📂 Project Structure
```text
SIH-2026/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                          # CLI & Server entry point
├── config/
│   ├── __init__.py
│   └── settings.py                  # Configuration & constants
├── data/
│   ├── raw/                         # Raw fetched data
│   ├── processed/                   # Cleaned & enriched data
│   └── cache/                       # Cached API responses
├── src/
│   ├── __init__.py
│   ├── pipeline.py                  # Pipeline orchestrator
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── firms_client.py          # NASA FIRMS API client (VIIRS/MODIS)
│   │   ├── osm_client.py            # OSM Overpass API client (8 facility types)
│   │   └── land_cover.py            # Land cover classification
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py          # Data cleaning & validation
│   │   └── feature_engineer.py      # Spatial & temporal feature engineering
│   ├── web/                         # Interactive GIS Web Dashboard
│   │   ├── __init__.py
│   │   ├── app.py                   # Flask server application & API routes
│   │   ├── demo_data.py             # Realistic thermal & facility synthesizer
│   │   ├── map_generator.py         # Folium/Leaflet map rendering engine
│   │   └── templates/
│   │       ├── dashboard.html       # Responsive dark-theme dashboard UI
│   │       └── alerts.html          # Incident Operations Center UI
│   ├── spatial_analysis/            # Part 2
│   ├── ml_classifier/               # Part 3
│   ├── visualization/               # Part 4
│   ├── pipeline_automation/         # Part 5.1 Automated Pipeline & SQLite DB
│   │   ├── __init__.py
│   │   ├── incremental_fetcher.py   # State tracking & FIRMS delta fetcher
│   │   ├── automated_pipeline.py    # Multi-step pipeline execution engine
│   │   ├── database.py              # Persistent SQLite storage & audit logging
│   │   └── scheduler.py             # Daemon & OS-level task scheduler
│   ├── core/                        # Part 5.5 Centralized Settings & Validation
│   │   ├── __init__.py
│   │   └── config.py                # AppConfig and pre-flight validation
│   ├── monitoring/                  # Part 5.2 Multi-Channel Alerts & Part 5.5 Health
│   │   ├── __init__.py
│   │   ├── models.py                # AlertEvent, AlertSeverity, TriggerType
│   │   ├── trigger_engine.py        # 4 Trigger conditions + diagnostic test
│   │   ├── dispatcher.py            # Cooldown suppression & dispatch orchestrator
│   │   ├── alert_engine.py          # Unified AlertEngine coordinator
│   │   ├── health.py                # SystemHealthManager (liveness, readiness, deep diagnostics)
│   │   └── channels/                # Email, SMS, Dashboard (SSE), Log, Webhook
│   ├── api/                         # Part 5.3 Production REST API (FastAPI)
│   │   ├── __init__.py              # Package exports
│   │   ├── app.py                   # FastAPI app factory & CORS
│   │   ├── routes.py                # APIRouter with all 10+ endpoints
│   │   ├── models.py                # Pydantic v2 schemas
│   │   └── service.py               # APIService business logic layer
│   └── reporting/                   # Part 5.4 Historical Analysis & Multi-Format Reporting
│       ├── __init__.py              # Package exports
│       ├── analyzer.py              # Monthly/quarterly aggregation & facility risk scoring
│       ├── csv_exporter.py          # Multi-dataset CSV exporter
│       ├── pdf_generator.py         # Publication-grade PDF generator (ReportLab)
│       ├── html_generator.py        # Executive standalone HTML dossier
│       └── report_engine.py         # Unified analytical coordinator
├── Dockerfile                       # Multi-service non-root production Docker image
├── docker-compose.yml               # Production multi-container orchestration
├── docker-compose.dev.yml           # Development live hot-reload override
├── .dockerignore                    # Build context exclusions
├── models/                          # Trained ML models
├── output/                          # Generated maps, emails & reports
└── tests/                           # Unit test suites (64 tests)
```

## 📊 Data Sources
| Source | URL | Data Type |
|--------|-----|----------|
| NASA FIRMS | https://firms.modaps.eosdis.nasa.gov | Active fire/thermal anomaly data |
| OpenStreetMap | https://www.openstreetmap.org | Industrial infrastructure locations |
| CORINE Land Cover | https://land.copernicus.eu | Land use/land cover classification |
| Sentinel-2 | https://dataspace.copernicus.eu | Multispectral satellite imagery |

## 🗺️ Next Steps & Execution Roadmap

Following the successful completion of **Part 1 (Data Ingestion & Preprocessing)** and the **Web Dashboard Preview**, here is the actionable phase-by-phase execution roadmap for subsequent parts:

### 📍 Phase 2: Spatial Analysis & Thermal Hotspot Analytics (Part 2)
1. **Multi-Ring Buffer Analysis (`src/spatial_analysis/buffer_analysis.py`)**:
   - Construct radial impact rings (500m, 1km, 2km, 5km) around high-risk assets (refineries, power plants, chemical reactors).
   - Evaluate intersecting thermal points to calculate facility exposure scores.
2. **Spatial Autocorrelation & Density Estimation (`src/spatial_analysis/hotspot_detection.py`)**:
   - Implement **Getis-Ord Gi\*** statistics to separate statistically significant hot clusters from isolated anomalies.
   - Deploy Kernel Density Estimation (KDE) surfaces across regional industrial hubs (Jamnagar, Singrauli, Korba, Dahej).
3. **Temporal DBSCAN Persistence Clustering (`src/spatial_analysis/temporal_clustering.py`)**:
   - Spatio-temporal DBSCAN ($eps_{spatial} \le 1\text{km}$, $eps_{temporal} \le 48\text{h}$) to identify long-term combustion (gas flares, smoldering mines) vs. episodic wildfires.

### 📍 Phase 3: Supervised AI/ML Classification Engine (Part 3)
1. **Dataset Synthesis & Ground-Truth Annotation (`src/ml_classifier/dataset_builder.py`)**:
   - Cross-reference FIRMS observations with known OSM facility polygon intersections.
   - Aggregate labelled training instances across 6 target classes: *Industrial Fire, Gas Flare, Forest Fire, Agricultural Burning, Mining Activity, Other/Unknown*.
2. **Model Training & Ensembling (`src/ml_classifier/train.py`)**:
   - Train baseline **Random Forest** and high-performance **XGBoost / LightGBM** models.
   - Perform feature importance attribution (SHAP values) on thermal radiative power, spatial distances, and diurnal ratios.
   - Save serialized inference pipelines (`models/fire_classifier_xgb.joblib`).
3. **Real-time Model Inference (`src/ml_classifier/inference.py`)**:
   - Integrate inference directly into the ingestion pipeline, assigning classification labels and confidence percentages (0-100%).

### 📍 Phase 4: Full Production GIS & Multi-Temporal Dashboard (Part 4)
1. **Copernicus Sentinel-2 Surface Reflectance Integration**:
   - Fetch before/after SWIR (Short-Wave Infrared) bands for high-confidence industrial fires to confirm structural damage.
2. **Temporal Time-Lapse Slider**:
   - Enable interactive temporal playback showing thermal plume evolution and dispersion over 7-30 days.
3. **Analytical Drilldown & PDF Risk Reports**:
   - Export automated forensic incident reports for NTRO/disaster management teams.

### 📍 Phase 5: Automated Alerting, Monitoring & Docker Deployment (Part 5)
1. **Scheduled Ingestion Worker**:
   - Background daemon/cron pipeline polling FIRMS API every 3 hours as satellite passes are published.
2. **Multi-Channel Alert Dispatcher**:
   - Immediate notification dispatch (Email/SMTP, Webhooks, Telegram) whenever an anomaly triggers high industrial confidence near critical infrastructure.
3. **Production Containerization**:
   - Multi-stage `Dockerfile` and `docker-compose.yml` for zero-configuration multi-platform deployment.

---

## 📜 License
MIT License

## 👥 Team
SIH 2026 Hackathon Team

