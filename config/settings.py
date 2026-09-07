import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DATA_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

# NASA FIRMS API Settings
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]
DEFAULT_DAY_RANGE = 2

# Geographic boundaries (India)
INDIA_BBOX = {
    "west": 68.0,
    "south": 6.0,
    "east": 98.0,
    "north": 38.0
}

# OpenStreetMap / Overpass API Settings
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Industrial Facility Types Mapping
INDUSTRIAL_CATEGORIES = {
    "thermal_power_plant": ["node[\"power\"=\"plant\"];", "way[\"power\"=\"plant\"];", "relation[\"power\"=\"plant\"];"],
    "oil_refinery": ["node[\"industrial\"=\"refinery\"];", "way[\"industrial\"=\"refinery\"];", "relation[\"industrial\"=\"refinery\"];"],
    "petrochemical": ["node[\"industrial\"=\"petrochemical\"];", "way[\"industrial\"=\"petrochemical\"];", "relation[\"industrial\"=\"petrochemical\"];"],
    "petroleum_well": ["node[\"man_made\"=\"petroleum_well\"];", "way[\"man_made\"=\"petroleum_well\"];"],
    "mining": ["node[\"industrial\"=\"mine\"];", "way[\"industrial\"=\"mine\"];", "node[\"landuse\"=\"quarry\"];", "way[\"landuse\"=\"quarry\"];"],
    "steel_plant": ["node[\"man_made\"=\"works\"][\"product\"~\"steel|iron\"];", "way[\"man_made\"=\"works\"][\"product\"~\"steel|iron\"];"],
    "gas_flare": ["node[\"man_made\"=\"flare\"];", "way[\"man_made\"=\"flare\"];"],
    "lng_terminal": ["node[\"industrial\"=\"gas\"];", "way[\"industrial\"=\"gas\"];", "node[\"man_made\"=\"storage_tank\"][\"content\"=\"lng\"];", "way[\"man_made\"=\"storage_tank\"][\"content\"=\"lng\"];"]
}

# Land Cover Classifications
LAND_COVER_TYPES = {
    1: 'Urban/Built-up',
    2: 'Industrial',
    3: 'Agricultural',
    4: 'Forest',
    5: 'Grassland/Shrubland',
    6: 'Wetland',
    7: 'Water',
    8: 'Barren',
    9: 'Unknown'
}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IndustrialFireDetection")
