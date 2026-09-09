import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point
import logging

try:
    import libpysal
    from esda.getisord import G_Local
    HAS_PYSAL = True
except ImportError:
    HAS_PYSAL = False

logger = logging.getLogger(__name__)

class HotspotEngine:
    """
    Engine to identify statistically significant clusters of thermal activity.
    Implements Kernel Density Estimation (KDE) and Getis-Ord Gi* statistics.
    """
    def __init__(self, kde_bandwidth_m=5000, gi_k_neighbors=8):
        """
        Initializes the HotspotEngine.
        :param kde_bandwidth_m: Bandwidth for KDE in meters.
        :param gi_k_neighbors: Number of nearest neighbors for spatial weights in Gi*.
        """
        self.kde_bandwidth_m = kde_bandwidth_m
        self.gi_k_neighbors = gi_k_neighbors
        self.crs_metric = "EPSG:3857"
        self.crs_wgs84 = "EPSG:4326"

    def _ensure_metric(self, gdf):
        if gdf.crs != self.crs_metric:
            return gdf.to_crs(self.crs_metric)
        return gdf

    def generate_kde_surface(self, fires_gdf, grid_size=100):
        """
        Generates a continuous KDE heat surface over the bounding box of the fire points.
        
        :param fires_gdf: GeoDataFrame of fire points.
        :param grid_size: Resolution of the grid.
        :return: GeoDataFrame of grid points with density and intensity classification.
        """
        logger.info("Generating KDE heat surface...")
        if fires_gdf.empty:
            return gpd.GeoDataFrame()
            
        fires_metric = self._ensure_metric(fires_gdf)
        coords = np.vstack([fires_metric.geometry.x, fires_metric.geometry.y]).T
        
        # Grid bounds with a small buffer (e.g. 10km)
        minx, miny, maxx, maxy = fires_metric.total_bounds
        buffer = 10000
        x_grid = np.linspace(minx - buffer, maxx + buffer, grid_size)
        y_grid = np.linspace(miny - buffer, maxy + buffer, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # KDE
        kde = KernelDensity(bandwidth=self.kde_bandwidth_m, kernel='gaussian')
        kde.fit(coords)
        
        log_dens = kde.score_samples(grid_coords)
        dens = np.exp(log_dens)
        
        # Normalize density to 0-100 score
        max_dens = dens.max() if dens.max() > 0 else 1
        norm_dens = (dens / max_dens) * 100
        
        # Classify Intensity
        intensity = []
        for d in norm_dens:
            if d < 10:
                intensity.append("Minimal")
            elif d < 40:
                intensity.append("Low")
            elif d < 70:
                intensity.append("Medium")
            elif d < 90:
                intensity.append("High")
            else:
                intensity.append("Critical")
                
        # Create GeoDataFrame
        points = [Point(x, y) for x, y in grid_coords]
        kde_gdf = gpd.GeoDataFrame({
            "density_score": norm_dens,
            "intensity": intensity,
            "geometry": points
        }, crs=self.crs_metric)
        
        # Keep only meaningful areas to reduce file size
        kde_gdf = kde_gdf[kde_gdf["density_score"] > 1.0].copy()
        
        return kde_gdf.to_crs(self.crs_wgs84)

    def calculate_gi_star(self, fires_gdf, value_col="frp"):
        """
        Identifies statistically significant hotspots using Getis-Ord Gi*.
        
        :param fires_gdf: GeoDataFrame of fire points.
        :param value_col: Numerical column to analyze (e.g., Fire Radiative Power).
        :return: GeoDataFrame with Gi* Z-scores, p-values, and hotspot flags.
        """
        logger.info("Calculating Getis-Ord Gi* statistics...")
        if not HAS_PYSAL:
            logger.warning("libpysal/esda not installed. Skipping Gi* calculation.")
            fires_gdf_out = fires_gdf.copy()
            fires_gdf_out["is_hotspot"] = False
            return fires_gdf_out
            
        if len(fires_gdf) <= self.gi_k_neighbors:
            logger.warning("Not enough points for Gi* spatial weights.")
            fires_gdf_out = fires_gdf.copy()
            fires_gdf_out["is_hotspot"] = False
            return fires_gdf_out
            
        fires_metric = self._ensure_metric(fires_gdf)
        
        # Handle missing value column by falling back to a constant (analyzing point density essentially)
        if value_col not in fires_metric.columns:
            logger.warning(f"Column '{value_col}' not found. Using constant value 1.0.")
            fires_metric["_weight_val"] = 1.0
            val_col = "_weight_val"
        else:
            # Handle NaNs in the value column
            fires_metric["_weight_val"] = fires_metric[value_col].fillna(fires_metric[value_col].mean())
            val_col = "_weight_val"
            
        y = fires_metric[val_col].values
        
        # Use KNN weights to guarantee everyone has neighbors and avoid island issues
        k = min(self.gi_k_neighbors, len(fires_metric) - 1)
        w = libpysal.weights.KNN.from_dataframe(fires_metric, k=k)
        w.transform = 'R'
        
        gi_star = G_Local(y, w, transform='R', star=True)
        
        fires_out = fires_gdf.copy()
        fires_out["gi_zscore"] = gi_star.Zs
        fires_out["gi_pvalue"] = gi_star.p_sim
        
        # Identify hotspots: Z-score > 1.96 (95% confidence) and p-value < 0.05
        fires_out["is_hotspot"] = (fires_out["gi_zscore"] > 1.96) & (fires_out["gi_pvalue"] < 0.05)
        
        return fires_out

    def analyze_temporal_persistence(self, fires_gdf, time_col="acq_date"):
        """
        Compares hotspot persistence over time windows.
        Splits data into 'Historical' and 'Recent' and identifies emerging vs persistent hotspots.
        """
        logger.info("Performing temporal hotspot analysis...")
        if time_col not in fires_gdf.columns or len(fires_gdf) < 10:
            logger.warning("Insufficient data or missing time column for temporal analysis.")
            return fires_gdf
            
        fires_temp = fires_gdf.copy()
        fires_temp[time_col] = pd.to_datetime(fires_temp[time_col])
        median_time = fires_temp[time_col].median()
        
        historical = fires_temp[fires_temp[time_col] <= median_time]
        recent = fires_temp[fires_temp[time_col] > median_time]
        
        if len(historical) < 5 or len(recent) < 5:
            return fires_gdf
            
        hist_scored = self.calculate_gi_star(historical)
        rec_scored = self.calculate_gi_star(recent)
        
        combined = pd.concat([hist_scored, rec_scored])
        
        rec_hotspots = rec_scored[rec_scored["is_hotspot"]]
        hist_hotspots = hist_scored[hist_scored["is_hotspot"]]
        
        if not rec_hotspots.empty and not hist_hotspots.empty:
            # Buffer historical hotspots by 5km to define a "persistent area"
            hist_buffers = hist_hotspots.to_crs(self.crs_metric).copy()
            hist_buffers.geometry = hist_buffers.buffer(5000)
            
            rec_hotspots_metric = rec_hotspots.to_crs(self.crs_metric)
            joined = gpd.sjoin(rec_hotspots_metric, hist_buffers, how='inner', predicate='intersects')
            persistent_ids = joined.index.unique()
            
            combined["temporal_status"] = "Emerging"
            combined.loc[combined.index.isin(persistent_ids), "temporal_status"] = "Persistent"
            combined.loc[combined[time_col] <= median_time, "temporal_status"] = "Historical"
            combined.loc[~combined["is_hotspot"], "temporal_status"] = "Not a Hotspot"
        else:
            combined["temporal_status"] = "Unknown"
            
        return combined

    def run_pipeline(self, fires_gdf):
        """
        Runs the full hotspot detection pipeline.
        """
        logger.info("--- Starting Hotspot Detection Engine ---")
        kde_surface = self.generate_kde_surface(fires_gdf)
        
        # Temporal analysis includes Gi* internally for both time windows
        if "acq_date" in fires_gdf.columns:
            analyzed_fires = self.analyze_temporal_persistence(fires_gdf, "acq_date")
        else:
            analyzed_fires = self.calculate_gi_star(fires_gdf)
            analyzed_fires["temporal_status"] = "N/A"
            
        logger.info("--- Hotspot Detection Complete ---")
        return analyzed_fires, kde_surface
