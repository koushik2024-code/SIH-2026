import logging
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import overpy

from config import settings

logger = logging.getLogger(__name__)

class OSMIndustrialClient:
    """Client for fetching industrial facility data from OpenStreetMap via Overpass API."""

    def __init__(self, bbox: dict = None):
        self.bbox = bbox or settings.INDIA_BBOX
        self.api = overpy.Overpass(url=settings.OVERPASS_URL)
        
    def fetch_industrial_facilities(self) -> gpd.GeoDataFrame:
        """Fetch industrial facilities and return as GeoDataFrame."""
        cache_path = settings.CACHE_DATA_DIR / "osm_industrial_facilities.geojson"
        
        if cache_path.exists():
            logger.info(f"Loading industrial facilities from cache: {cache_path}")
            return gpd.read_file(cache_path)
            
        logger.info("Fetching industrial facilities from Overpass API...")
        
        bbox_str = f"{self.bbox['south']},{self.bbox['west']},{self.bbox['north']},{self.bbox['east']}"
        
        query = f"[out:json][timeout:90];\n(\n"
        for category, tags in settings.INDUSTRIAL_CATEGORIES.items():
            for tag in tags:
                query += f"  {tag.replace(';', f'( {bbox_str} );')}\n"
        query += ");\nout center;"
        
        try:
            result = self.api.query(query)
            parsed_data = self._parse_overpass_results(result)
            
            if parsed_data:
                df = gpd.pd.DataFrame(parsed_data)
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
                gdf.set_crs(epsg=4326, inplace=True)
                
                # Cache the results
                settings.CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
                gdf.to_file(cache_path, driver="GeoJSON")
                logger.info(f"Saved {len(gdf)} facilities to cache.")
                
                return gdf
            else:
                logger.warning("No industrial facilities found.")
                return gpd.GeoDataFrame()
                
        except overpy.exception.OverpassTooManyRequests:
            logger.error("Overpass API rate limit exceeded.")
            return gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Error fetching OSM data: {e}")
            return gpd.GeoDataFrame()

    def _parse_overpass_results(self, result: overpy.Result) -> list[dict]:
        """Parse nodes, ways, and relations into a structured list of dicts."""
        data = []
        
        # Process Nodes
        for node in result.nodes:
            facility_type = self._determine_facility_type(node.tags)
            data.append({
                "osm_id": f"node/{node.id}",
                "name": node.tags.get("name", "Unknown"),
                "facility_type": facility_type,
                "latitude": float(node.lat),
                "longitude": float(node.lon),
                "tags": json.dumps(node.tags)
            })
            
        # Process Ways
        for way in result.ways:
            facility_type = self._determine_facility_type(way.tags)
            lat, lon = self._get_center_of_way(way)
            if lat is not None and lon is not None:
                data.append({
                    "osm_id": f"way/{way.id}",
                    "name": way.tags.get("name", "Unknown"),
                    "facility_type": facility_type,
                    "latitude": lat,
                    "longitude": lon,
                    "tags": json.dumps(way.tags)
                })
                
        # Process Relations (center not reliably provided by all overpy relation objects, so simplified)
        for rel in result.relations:
            facility_type = self._determine_facility_type(rel.tags)
            if hasattr(rel, 'center_lat') and hasattr(rel, 'center_lon'):
                lat, lon = float(rel.center_lat), float(rel.center_lon)
            else:
                continue # Skip relations without center coordinates in this basic implementation
            
            data.append({
                "osm_id": f"relation/{rel.id}",
                "name": rel.tags.get("name", "Unknown"),
                "facility_type": facility_type,
                "latitude": lat,
                "longitude": lon,
                "tags": json.dumps(rel.tags)
            })
            
        return data

    def _determine_facility_type(self, tags: dict) -> str:
        """Determine facility category based on tags."""
        for cat, queries in settings.INDUSTRIAL_CATEGORIES.items():
            for query in queries:
                # Basic matching based on key-value presence
                query_str = query.split('[')[1:]
                match = True
                for q in query_str:
                    q = q.replace(']', '').replace(';', '').replace('"', '')
                    if '=' in q:
                        k, v = q.split('=', 1)
                        if tags.get(k) != v:
                            match = False
                            break
                    elif '~' in q:
                        k, v = q.split('~', 1)
                        if k not in tags or v not in str(tags.get(k)): # Very simplified regex match
                            match = False
                            break
                if match:
                    return cat
        return "other_industrial"

    def _get_center_of_way(self, way: overpy.Way) -> tuple:
        """Extract centroid coordinates for a way, assuming 'out center' was used."""
        if hasattr(way, 'center_lat') and hasattr(way, 'center_lon'):
             return float(way.center_lat), float(way.center_lon)
        return None, None
