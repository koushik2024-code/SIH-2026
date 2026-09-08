import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None
import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)

class MapGenerator:
    """Generates interactive Folium maps for fire visualization."""
    
    FIRE_TYPE_COLORS = {
        "Industrial Fire": "#e74c3c",    # Red
        "Gas Flare": "#e67e22",          # Orange
        "Forest Fire": "#27ae60",        # Green
        "Agricultural Burning": "#f1c40f", # Yellow
        "Mining Activity": "#7f8c8d",    # Gray
        "Other/Unknown": "#95a5a6",      # Light gray
    }
    
    FACILITY_ICONS = {
        "oil_refinery": ("tint", "red"),
        "thermal_power_plant": ("bolt", "orange"),
        "steel_plant": ("industry", "darkred"),
        "mining": ("gem", "gray"),
        "petrochemical": ("flask", "purple"),
        "gas_flare": ("fire", "orange"),
        "lng_terminal": ("ship", "blue"),
        "petroleum_well": ("oil-can", "black"),
    }
    
    def create_dashboard_map(self, fire_df: pd.DataFrame, facilities_gdf: Any, 
                              selected_types: list = None,
                              enable_controls: bool = True) -> str:
        """Create the main dashboard map using Part 4.1-4.3 InteractiveGISMap engine and return as HTML string."""
        from src.visualization.interactive_map import InteractiveGISMap
        gis_engine = InteractiveGISMap(enable_controls=enable_controls)
        m = gis_engine.create_interactive_map(
            fire_df=fire_df,
            facilities_gdf=facilities_gdf,
            selected_types=selected_types,
            enable_controls=enable_controls,
        )
        return m._repr_html_()
    
    def get_fire_statistics(self, fire_df: pd.DataFrame) -> dict:
        """Calculate summary statistics from fire data."""
        stats = {
            "total_fires": len(fire_df),
            "avg_brightness": round(fire_df['brightness'].mean(), 1) if 'brightness' in fire_df.columns else 0,
            "avg_frp": round(fire_df['frp'].mean(), 1) if 'frp' in fire_df.columns else 0,
            "high_confidence": len(fire_df[fire_df['confidence'] == 'high']) if 'confidence' in fire_df.columns else 0,
            "industrial_count": len(fire_df[fire_df.get('is_near_industrial', pd.Series(dtype=int)) == 1]) if 'is_near_industrial' in fire_df.columns else 0,
            "daytime_fires": len(fire_df[fire_df['is_daytime'] == 1]) if 'is_daytime' in fire_df.columns else 0,
            "nighttime_fires": len(fire_df[fire_df['is_daytime'] == 0]) if 'is_daytime' in fire_df.columns else 0,
        }
        
        # Fire type distribution
        if 'fire_type' in fire_df.columns:
            type_counts = fire_df['fire_type'].value_counts().to_dict()
            stats['fire_type_distribution'] = type_counts
        else:
            stats['fire_type_distribution'] = {}
        
        # Facility type distribution of nearest facilities
        if 'nearest_facility_type' in fire_df.columns:
            stats['nearest_facility_distribution'] = fire_df['nearest_facility_type'].value_counts().to_dict()
        
        # Date range
        if 'acq_date' in fire_df.columns:
            stats['date_range'] = {
                'start': str(fire_df['acq_date'].min()),
                'end': str(fire_df['acq_date'].max())
            }
        
        return stats
