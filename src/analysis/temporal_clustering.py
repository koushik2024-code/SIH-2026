import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class TemporalClusteringEngine:
    """
    Engine to detect persistent thermal sources vs. one-time fire events
    by grouping fires spatially and tracking their temporal persistence.
    """
    def __init__(self, eps_m=1500, min_samples=2):
        """
        Initializes the TemporalClusteringEngine.
        
        :param eps_m: Maximum distance between two samples for one to be considered as in the neighborhood of the other. (in meters)
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps_m = eps_m
        self.min_samples = min_samples
        self.crs_metric = "EPSG:3857"
        self.crs_wgs84 = "EPSG:4326"

    def _ensure_metric(self, gdf):
        """Ensures the GeoDataFrame is in the metric CRS."""
        gdf = gdf.copy()
        if gdf.crs != self.crs_metric:
            gdf = gdf.to_crs(self.crs_metric)
        return gdf

    def extract_datetime(self, df, date_col="acq_date", time_col="acq_time"):
        """
        Combines date and time columns into a single datetime column if possible.
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in data.")
            
        datetime_series = pd.to_datetime(df[date_col])
        
        # If time is available, try to add it (assuming it's a string like 'HHMM' or 'HH:MM')
        if time_col in df.columns:
            try:
                # Convert numeric time like 1205 to string '1205'
                time_str = df[time_col].astype(str).str.zfill(4)
                # Ensure it's in HHMM format before converting
                # For some FIRMS data it might just be HHMM
                hours = time_str.str[:2]
                mins = time_str.str[2:4]
                
                # Create a timedelta
                time_td = pd.to_timedelta(hours + ':' + mins + ':00', errors='coerce')
                
                datetime_series = datetime_series + time_td.fillna(pd.Timedelta(seconds=0))
            except Exception as e:
                logger.warning(f"Could not parse time column: {e}. Falling back to date only.")
                
        return datetime_series

    def cluster_spatially(self, fires_gdf):
        """
        Groups fire detections by spatial proximity using DBSCAN.
        """
        logger.info(f"Running DBSCAN clustering (eps={self.eps_m}m, min_samples={self.min_samples})...")
        if fires_gdf.empty:
            return fires_gdf
            
        fires_metric = self._ensure_metric(fires_gdf)
        coords = np.vstack([fires_metric.geometry.x, fires_metric.geometry.y]).T
        
        dbscan = DBSCAN(eps=self.eps_m, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(coords)
        
        fires_out = fires_gdf.copy()
        fires_out["spatial_cluster_id"] = clusters
        
        return fires_out

    def analyze_persistence(self, clustered_gdf, date_col="acq_date", time_col="acq_time"):
        """
        Tracks each cluster's temporal persistence (days active) and classifies it.
        """
        logger.info("Analyzing temporal persistence of clusters...")
        if clustered_gdf.empty or "spatial_cluster_id" not in clustered_gdf.columns:
            return clustered_gdf, gpd.GeoDataFrame()
            
        gdf = clustered_gdf.copy()
        gdf["datetime"] = self.extract_datetime(gdf, date_col, time_col)
        
        # We only care about valid clusters (DBSCAN assigns -1 to noise/outliers)
        valid_clusters = gdf[gdf["spatial_cluster_id"] != -1]
        noise_points = gdf[gdf["spatial_cluster_id"] == -1].copy()
        
        # Calculate cluster statistics
        cluster_stats = valid_clusters.groupby("spatial_cluster_id").agg(
            first_detection=("datetime", "min"),
            last_detection=("datetime", "max"),
            fire_count=("datetime", "count")
        ).reset_index()
        
        # Calculate persistence in days
        cluster_stats["duration_days"] = (cluster_stats["last_detection"] - cluster_stats["first_detection"]).dt.total_seconds() / (24 * 3600)
        
        # Classify clusters
        # Transient: <24h
        # Short-term: 1-7 days
        # Persistent: >7 days
        def classify_duration(days):
            if days < 1.0:
                return "Transient"
            elif days <= 7.0:
                return "Short-term"
            else:
                return "Persistent"
                
        cluster_stats["cluster_class"] = cluster_stats["duration_days"].apply(classify_duration)
        
        # Merge back to the original points
        gdf = gdf.merge(cluster_stats[["spatial_cluster_id", "duration_days", "cluster_class"]], 
                        on="spatial_cluster_id", 
                        how="left")
        
        # Assign noise points as Transient (since they are isolated single events)
        gdf["duration_days"] = gdf["duration_days"].fillna(0.0)
        gdf["cluster_class"] = gdf["cluster_class"].astype(object).fillna("Transient (Isolated)")
        
        # Create a summary GeoDataFrame of the clusters (Convex Hulls)
        cluster_hulls = []
        valid_clusters_metric = self._ensure_metric(valid_clusters)
        
        for cluster_id, group in valid_clusters_metric.groupby("spatial_cluster_id"):
            if len(group) >= 3:
                hull = group.geometry.unary_union.convex_hull
            elif len(group) == 2:
                # Buffer a line connecting two points
                hull = group.geometry.unary_union.buffer(50) 
            else:
                hull = group.geometry.iloc[0].buffer(50)
                
            stats_row = cluster_stats[cluster_stats["spatial_cluster_id"] == cluster_id].iloc[0]
            
            cluster_hulls.append({
                "spatial_cluster_id": cluster_id,
                "first_detection": stats_row["first_detection"],
                "last_detection": stats_row["last_detection"],
                "fire_count": stats_row["fire_count"],
                "duration_days": stats_row["duration_days"],
                "cluster_class": stats_row["cluster_class"],
                "geometry": hull
            })
            
        hulls_gdf = gpd.GeoDataFrame(cluster_hulls, crs=self.crs_metric).to_crs(self.crs_wgs84) if cluster_hulls else gpd.GeoDataFrame()
        
        return gdf, hulls_gdf

    def run_pipeline(self, fires_gdf):
        """
        Runs the full temporal clustering pipeline.
        """
        logger.info("--- Starting Temporal Clustering Engine ---")
        clustered = self.cluster_spatially(fires_gdf)
        analyzed_fires, cluster_hulls = self.analyze_persistence(clustered)
        logger.info("--- Temporal Clustering Complete ---")
        return analyzed_fires, cluster_hulls
