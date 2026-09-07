import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import json

try:
    import libpysal
    from esda.moran import Moran
    from pointpats import PointPattern
    HAS_SPATIAL_LIBS = True
except ImportError:
    HAS_SPATIAL_LIBS = False

from .hotspot_engine import HotspotEngine
from .temporal_clustering import TemporalClusteringEngine

logger = logging.getLogger(__name__)

class SpatialStatisticsEngine:
    """
    Engine to calculate global spatial statistics (Moran's I, Ripley's K, Nearest Neighbor)
    and orchestrate the full analysis pipeline.
    """
    def __init__(self, crs_metric="EPSG:3857"):
        self.crs_metric = crs_metric
        self.crs_wgs84 = "EPSG:4326"

    def _ensure_metric(self, gdf):
        gdf = gdf.copy()
        if gdf.crs != self.crs_metric:
            gdf = gdf.to_crs(self.crs_metric)
        return gdf

    def calculate_morans_i(self, fires_gdf, value_col="frp"):
        """Calculates global Moran's I for spatial autocorrelation."""
        if not HAS_SPATIAL_LIBS:
            logger.warning("Spatial libraries missing. Cannot calculate Moran's I.")
            return {"I": np.nan, "p_value": np.nan}
            
        fires_metric = self._ensure_metric(fires_gdf)
        if len(fires_metric) < 10:
            return {"I": np.nan, "p_value": np.nan}
            
        # Handle NaNs and missing columns
        if value_col not in fires_metric.columns:
            logger.warning(f"Column '{value_col}' missing for Moran's I. Using constant.")
            y = np.ones(len(fires_metric))
        else:
            col_mean = fires_metric[value_col].mean()
            fill_val = col_mean if pd.notna(col_mean) else 0.0
            y = fires_metric[value_col].fillna(fill_val).values
            
        if np.nanstd(y) == 0:
            logger.warning("Zero variance in value column. Moran's I undefined.")
            return {"I": np.nan, "p_value": np.nan}
            
        k = min(8, len(fires_metric)-1)
        w = libpysal.weights.KNN.from_dataframe(fires_metric, k=k)
        w.transform = 'R'
        
        mi = Moran(y, w)
        return {"I": mi.I, "p_value": mi.p_sim}

    def calculate_nearest_neighbor(self, fires_gdf):
        """Calculates Nearest Neighbor index (Clark-Evans R)."""
        from sklearn.neighbors import NearestNeighbors
        fires_metric = self._ensure_metric(fires_gdf)
        if len(fires_metric) < 5:
            return {"r_ratio": np.nan, "mean_nn_dist": np.nan, "expected_nn_dist": np.nan}
            
        coords = np.vstack([fires_metric.geometry.x, fires_metric.geometry.y]).T
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        
        nn_dists = distances[:, 1]
        mean_nn_dist = np.mean(nn_dists)
        
        minx, miny, maxx, maxy = fires_metric.total_bounds
        area = (maxx - minx) * (maxy - miny)
        if area == 0:
            return {"r_ratio": np.nan, "mean_nn_dist": mean_nn_dist, "expected_nn_dist": np.nan}
            
        density = len(fires_metric) / area
        expected_nn_dist = 1 / (2 * np.sqrt(density))
        
        r_ratio = mean_nn_dist / expected_nn_dist
        
        return {
            "r_ratio": r_ratio, 
            "mean_nn_dist": mean_nn_dist, 
            "expected_nn_dist": expected_nn_dist,
            "interpretation": "Clustered" if r_ratio < 0.9 else ("Dispersed" if r_ratio > 1.1 else "Random")
        }

    def calculate_ripleys_k(self, fires_gdf):
        """Calculates Ripley's K-function across several distance bands."""
        if not HAS_SPATIAL_LIBS or len(fires_gdf) < 10:
            return {}
            
        fires_metric = self._ensure_metric(fires_gdf)
        coords = np.vstack([fires_metric.geometry.x, fires_metric.geometry.y]).T
        
        try:
            pp = PointPattern(coords)
            # Evaluate from 1km to 10km
            support = np.linspace(1000, 10000, 10)
            k_stat = pp.k(support)
            
            # CSR (Complete Spatial Randomness) expected K is pi * r^2
            k_csr = np.pi * (support ** 2)
            
            return {
                "distances_m": support.tolist(), 
                "k_observed": k_stat.tolist(),
                "k_expected_csr": k_csr.tolist()
            }
        except Exception as e:
            logger.warning(f"Ripley's K calculation failed: {e}")
            return {}

    def run_master_pipeline(self, fires_gdf):
        """
        Runs the full enrichment pipeline (Hotspots -> Temporal -> Spatial Stats)
        and outputs the globally enriched dataset and statistics summary.
        """
        logger.info("--- Starting Master Analysis Pipeline (Problem 2.4) ---")
        
        # 1. Hotspot Detection
        logger.info("[1/3] Running Hotspot Detection...")
        hotspot_engine = HotspotEngine()
        hotspots_gdf, _ = hotspot_engine.run_pipeline(fires_gdf)
        
        # 2. Temporal Clustering
        logger.info("[2/3] Running Temporal Clustering...")
        temporal_engine = TemporalClusteringEngine()
        enriched_gdf, _ = temporal_engine.run_pipeline(hotspots_gdf)
        
        # 3. Global Spatial Statistics
        logger.info("[3/3] Calculating Global Spatial Statistics...")
        val_col = "frp" if "frp" in enriched_gdf.columns else "confidence"
        
        morans_i = self.calculate_morans_i(enriched_gdf, value_col=val_col)
        nn_stats = self.calculate_nearest_neighbor(enriched_gdf)
        ripleys_k = self.calculate_ripleys_k(enriched_gdf)
        
        stats_summary = {
            "total_fires_analyzed": len(enriched_gdf),
            "morans_i": morans_i,
            "nearest_neighbor": nn_stats,
            "ripleys_k": ripleys_k
        }
        
        logger.info("--- Master Analysis Pipeline Complete ---")
        return enriched_gdf, stats_summary
