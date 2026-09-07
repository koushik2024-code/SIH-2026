import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineers spatial and temporal features for fire data."""
    
    def engineer_features(self, fire_df: pd.DataFrame, facilities_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create new features based on fire data and industrial facilities."""
        if fire_df.empty:
             return fire_df
             
        logger.info("Engineering temporal and thermal features...")
        
        # Temporal features
        if 'datetime' in fire_df.columns:
            fire_df['hour_of_day'] = fire_df['datetime'].dt.hour
            fire_df['day_of_week'] = fire_df['datetime'].dt.dayofweek
            fire_df['month'] = fire_df['datetime'].dt.month
            
        if 'daynight' in fire_df.columns:
            fire_df['is_daytime'] = fire_df['daynight'].map({'D': 1, 'N': 0})
        else:
            # Fallback based on hour
            if 'hour_of_day' in fire_df.columns:
                 fire_df['is_daytime'] = ((fire_df['hour_of_day'] >= 6) & (fire_df['hour_of_day'] <= 18)).astype(int)
                 
        # Thermal features
        if 'brightness' in fire_df.columns and 'frp' in fire_df.columns:
            # Prevent division by zero
            fire_df['brightness_frp_ratio'] = fire_df['brightness'] / (fire_df['frp'] + 1e-6)
            
        # Spatial features (Proximity to industry)
        if not facilities_gdf.empty:
            logger.info("Engineering spatial features (distance to industry)...")
            
            # Prepare BallTree for fast nearest neighbor search
            # Convert degrees to radians for haversine
            fire_coords = np.radians(fire_df[['latitude', 'longitude']].values)
            facility_coords = np.radians(facilities_gdf[['latitude', 'longitude']].values)
            
            tree = BallTree(facility_coords, metric='haversine')
            
            # Query nearest facility
            distances, indices = tree.query(fire_coords, k=1)
            
            # Earth radius in km
            R = 6371.0
            
            fire_df['distance_to_nearest_industrial'] = distances.flatten() * R
            
            nearest_facilities = facilities_gdf.iloc[indices.flatten()]
            
            fire_df['nearest_facility_type'] = nearest_facilities['facility_type'].values
            fire_df['nearest_facility_name'] = nearest_facilities['name'].values
            
            # Is near threshold (e.g., 2km)
            fire_df['is_near_industrial'] = (fire_df['distance_to_nearest_industrial'] <= 2.0).astype(int)
        else:
            logger.warning("No facility data provided. Skipping spatial feature engineering.")
            fire_df['distance_to_nearest_industrial'] = -1.0
            fire_df['nearest_facility_type'] = 'none'
            fire_df['nearest_facility_name'] = 'none'
            fire_df['is_near_industrial'] = 0
            
        # Simple Clustering (Persistence features)
        logger.info("Clustering fires...")
        fire_coords = np.radians(fire_df[['latitude', 'longitude']].values)
        if len(fire_coords) > 0:
            tree = BallTree(fire_coords, metric='haversine')
            # Find neighbors within ~1km (1 / 6371 radians)
            radius = 1.0 / 6371.0
            indices_list = tree.query_radius(fire_coords, r=radius)
            
            # Basic connected components for clusters
            cluster_id = np.zeros(len(fire_df), dtype=int)
            current_id = 1
            visited = set()
            
            for i, neighbors in enumerate(indices_list):
                if i not in visited:
                    for n in neighbors:
                        cluster_id[n] = current_id
                        visited.add(n)
                    current_id += 1
                    
            fire_df['fire_cluster_id'] = cluster_id
            
        return fire_df
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate haversine distance in km."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
