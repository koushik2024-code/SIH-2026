import geopandas as gpd
from shapely.geometry import Point
import logging

from config.settings import LAND_COVER_TYPES

logger = logging.getLogger(__name__)

class LandCoverClassifier:
    """Classifies land cover for fire points, primarily identifying industrial fires."""
    
    LAND_COVER_TYPES = LAND_COVER_TYPES

    def assign_land_cover_from_osm(self, fire_gdf: gpd.GeoDataFrame, facilities_gdf: gpd.GeoDataFrame, buffer_km: float = 2.0) -> gpd.GeoDataFrame:
        """Assign land cover classification based on proximity to industrial facilities."""
        if fire_gdf.empty or facilities_gdf.empty:
            logger.warning("Empty fire or facilities GeoDataFrame. Returning original data.")
            if 'land_cover' not in fire_gdf.columns:
                fire_gdf['land_cover'] = 9 # Unknown
            return fire_gdf
            
        logger.info(f"Assigning land cover based on OSM facilities with {buffer_km}km buffer...")
        
        # Ensure CRS is geographic for distance calculation later if needed, but buffer is better in projected CRS
        # For simplicity, rough degree conversion (1 deg ~ 111km)
        buffer_deg = buffer_km / 111.0 
        
        facilities_buffered = facilities_gdf.copy()
        facilities_buffered['geometry'] = facilities_buffered.geometry.buffer(buffer_deg)
        
        # Initialize column
        if 'land_cover' not in fire_gdf.columns:
            fire_gdf['land_cover'] = 9 # Unknown default
            
        # Spatial join to find fires within facility buffers
        joined = gpd.sjoin(fire_gdf, facilities_buffered, how="left", predicate="within")
        
        # Where joined['index_right'] is not null, it's an industrial fire
        industrial_indices = joined[joined['index_right'].notna()].index
        
        # Assign land cover 2 (Industrial)
        fire_gdf.loc[industrial_indices, 'land_cover'] = 2
        
        counts = fire_gdf['land_cover'].value_counts()
        logger.info(f"Land cover assignment complete. Counts:\n{counts}")
        
        return fire_gdf
