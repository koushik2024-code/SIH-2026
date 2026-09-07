---

# 🔥 AI-Based Detection and Classification of Industrial Fires & Persistent Thermal Sources

> Using NASA FIRMS, OpenStreetMap & Satellite Data | SIH 2026 | NTRO

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Part%201%20Complete-brightgreen) ![License](https://img.shields.io/badge/License-MIT-yellow)

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
| 2 | Spatial Analysis & Enrichment | 🔲 Planned | Proximity analysis, hotspot detection, temporal clustering, spatial statistics |
| 3 | AI/ML Classification Engine | 🔲 Planned | ML models to classify fire types (industrial, forest, agricultural, etc.) |
| 4 | GIS Visualization & Dashboard | 🔲 Planned | Interactive web map with Folium/Leaflet, heatmaps, overlays, filter panels |
| 5 | Monitoring, Alerts & Deployment | 🔲 Planned | Automated pipeline, alert system, REST API, Docker deployment |

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

### PART 4: GIS Visualization & Web Dashboard 🔲 PLANNED

#### 4.1 Interactive Map (Folium/Leaflet.js)
**Objective**: Build a web-based GIS interface for visualization and analysis.

**Base Map Layers**:
- OpenStreetMap (default)
- ESRI Satellite Imagery
- CartoDB Dark Matter (for heatmap contrast)
- Terrain/Topographic view

#### 4.2 Data Overlay Layers
**Fire Detection Layer**:
- Color-coded markers by fire type classification
  - 🔴 Red: Industrial Fire
  - 🟠 Orange: Gas Flare
  - 🟢 Green: Forest Fire
  - 🟡 Yellow: Agricultural Burning
  - ⚫ Gray: Mining Activity
  - ⚪ White: Unknown
- Marker size proportional to FRP
- Popup with detailed info: coordinates, datetime, confidence, classification

**Industrial Facility Layer**:
- Custom icons by facility type
- Buffer zones shown as semi-transparent circles
- Click for facility details

**Heatmap Layer**:
- FRP-weighted heatmap using folium.plugins.HeatMap
- Temporal heatmap with time slider (HeatMapWithTime)
- Adjustable radius and blur parameters

#### 4.3 Dashboard Controls
- Layer toggle panel (show/hide layers)
- Date range filter
- Fire type filter (checkboxes)
- Confidence level slider
- Search by location
- Legend panel

#### 4.4 Analytics Panels
- Fire type distribution pie chart
- Time series of fire detections
- Top facilities by nearby fire count
- Regional summary statistics

**Output**: Interactive HTML map files, embeddable web dashboard.

---

### PART 5: Monitoring, Alerts & Deployment 🔲 PLANNED

#### 5.1 Automated Data Pipeline
**Objective**: Schedule automated data collection and classification.

**Process Flow**:
1. Cron Job / Task Scheduler: Run pipeline every 6 hours
2. Incremental Data Fetch: Only fetch new detections since last run
3. Auto-Classification: Run new data through trained ML model
4. Database Update: Append results to persistent storage
5. Dashboard Refresh: Update map with latest data

#### 5.2 Alert System
**Trigger Conditions**:
- New industrial fire detected (confidence > 80%)
- Unusual thermal spike at known facility
- New fire cluster formation near industrial zone
- Persistent fire burning for >48 hours

**Alert Channels**:
- Email notifications (SMTP)
- SMS alerts (Twilio API)
- Dashboard notifications (WebSocket)
- Log file alerts

#### 5.3 REST API (FastAPI)
**Endpoints**:
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/fires | Get all classified fires |
| GET | /api/fires/{id} | Get specific fire details |
| GET | /api/fires/type/{type} | Filter by fire type |
| GET | /api/facilities | Get all industrial facilities |
| GET | /api/hotspots | Get current hotspots |
| GET | /api/stats | Get summary statistics |
| POST | /api/classify | Classify new fire data |

#### 5.4 Historical Analysis & Reporting
- Monthly/quarterly fire reports
- Trend analysis by region/type
- Facility risk scoring over time
- Export to PDF/CSV

#### 5.5 Deployment
- Docker containerization (Dockerfile + docker-compose)
- Environment variable configuration
- Health checks and monitoring
- CI/CD pipeline (GitHub Actions)

**Output**: Production-ready system with automated monitoring, alerts, API, and Docker deployment.

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

# Run Part 1: Data Ingestion & Preprocessing
python main.py --part 1
```

## 📂 Project Structure
```text
SIH-2026/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                          # Entry point
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
│   │   ├── firms_client.py          # NASA FIRMS API client
│   │   ├── osm_client.py            # OSM Overpass API client
│   │   └── land_cover.py            # Land cover classification
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py          # Data cleaning & validation
│   │   └── feature_engineer.py      # Feature engineering
│   ├── spatial_analysis/            # Part 2 (planned)
│   ├── ml_classifier/               # Part 3 (planned)
│   ├── visualization/               # Part 4 (planned)
│   └── monitoring/                  # Part 5 (planned)
├── models/                          # Trained ML models
├── output/                          # Generated maps & reports
└── tests/                           # Unit tests
```

## 📊 Data Sources
| Source | URL | Data Type |
|--------|-----|----------|
| NASA FIRMS | https://firms.modaps.eosdis.nasa.gov | Active fire/thermal anomaly data |
| OpenStreetMap | https://www.openstreetmap.org | Industrial infrastructure locations |
| CORINE Land Cover | https://land.copernicus.eu | Land use/land cover classification |
| Sentinel-2 | https://dataspace.copernicus.eu | Multispectral satellite imagery |

## 📜 License
MIT License

## 👥 Team
SIH 2026 Hackathon Team

---
