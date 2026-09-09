import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

class ProximityAnalysisEngine:
    """
    Engine to quantify spatial relationships between thermal anomalies (fires) 
    and industrial infrastructure.
    """
    def __init__(self, crs_metric="EPSG:3857"):
        """
        Initializes the engine.
        :param crs_metric: A projected coordinate reference system (in meters) for accurate buffering.
                           Default is Web Mercator (EPSG:3857), but for specific regions UTM is better.
        """
        self.crs_metric = crs_metric
        self.buffer_distances = [500, 1000, 2000, 5000]
        self.buffer_scores = {500: 10, 1000: 7, 2000: 4, 5000: 1}

    def _ensure_crs(self, gdf):
        """Ensures the GeoDataFrame is in the target metric CRS for accurate distance calculations."""
        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame must have a defined CRS.")
        if gdf.crs != self.crs_metric:
            return gdf.to_crs(self.crs_metric)
        return gdf

    def create_buffers(self, facilities_gdf, facility_id_col="facility_id"):
        """
        Generates concentric buffer zones around each industrial facility.
        
        :param facilities_gdf: GeoDataFrame containing facility locations (points or polygons).
        :param facility_id_col: Column name identifying each facility.
        :return: A new GeoDataFrame with concentric buffer polygons.
        """
        facilities_metric = self._ensure_crs(facilities_gdf)
        
        buffer_records = []
        for _, facility in facilities_metric.iterrows():
            geom = facility.geometry
            fac_id = facility[facility_id_col]
            risk = facility.get("risk_score", 1.0) # default risk is 1.0
            
            for dist in self.buffer_distances:
                buffer_geom = geom.buffer(dist)
                buffer_records.append({
                    facility_id_col: fac_id,
                    "buffer_distance": dist,
                    "risk_score": risk,
                    "geometry": buffer_geom
                })
                
        buffers_gdf = gpd.GeoDataFrame(buffer_records, crs=self.crs_metric)
        return buffers_gdf

    def intersect_fires(self, fires_gdf, buffers_gdf):
        """
        Intersects fire points with facility buffer zones.
        
        :param fires_gdf: GeoDataFrame of thermal anomalies.
        :param buffers_gdf: GeoDataFrame of facility buffers.
        :return: GeoDataFrame of fires that fall within any facility buffer.
        """
        fires_metric = self._ensure_crs(fires_gdf)
        
        # Spatial join: fires that are within the buffers
        # How='inner' keeps only fires that intersect at least one buffer
        joined_gdf = gpd.sjoin(fires_metric, buffers_gdf, how="inner", predicate="intersects")
        
        return joined_gdf

    def calculate_scores(self, joined_gdf, fire_id_col="fire_id", facility_id_col="facility_id"):
        """
        Calculates proximity score based on the closest intersected buffer ring.
        Since a fire within 500m is also within 1000m, 2000m, etc.,
        we group by fire and facility, and take the MINIMUM distance buffer it intersected.
        
        :param joined_gdf: Result from intersect_fires.
        :param fire_id_col: Column identifying unique fires.
        :param facility_id_col: Column identifying unique facilities.
        :return: DataFrame with scored relationships.
        """
        if joined_gdf.empty:
            return joined_gdf
            
        fac_col = facility_id_col if facility_id_col in joined_gdf.columns else "facility_id"
        # A single fire might intersect multiple concentric rings for the same facility.
        # We want to find the smallest ring it intersected to give the highest score.
        scored = joined_gdf.sort_values(by="buffer_distance").drop_duplicates(subset=[fire_id_col, fac_col], keep="first").copy()
        
        # Assign score based on the smallest buffer distance
        scored["proximity_score"] = scored["buffer_distance"].map(self.buffer_scores)
        
        return scored

    def rank_multi_facility_fires(self, scored_gdf, fire_id_col="fire_id"):
        """
        For fires near multiple facilities, rank the involved facilities by distance and facility risk.
        Ranking formula: (proximity_score * risk_score)
        
        :param scored_gdf: Result from calculate_scores.
        :param fire_id_col: Column identifying unique fires.
        :return: DataFrame sorted and ranked by the combined threat level.
        """
        if scored_gdf.empty:
            return scored_gdf
            
        # Calculate combined threat score
        scored_gdf["threat_score"] = scored_gdf["proximity_score"] * scored_gdf["risk_score"]
        
        # Sort by fire_id and then by threat_score descending
        ranked_gdf = scored_gdf.sort_values(by=[fire_id_col, "threat_score"], ascending=[True, False])
        
        # Add a rank column per fire
        ranked_gdf["facility_rank_for_fire"] = ranked_gdf.groupby(fire_id_col).cumcount() + 1
        
        return ranked_gdf

    def run_analysis(self, fires_gdf, facilities_gdf, fire_id_col="fire_id", facility_id_col="facility_id"):
        """
        Executes the full proximity analysis pipeline.
        
        :param fires_gdf: GeoDataFrame of thermal anomalies.
        :param facilities_gdf: GeoDataFrame of industrial facilities.
        :return: Final ranked and scored GeoDataFrame of critical fires, and the generated buffers GeoDataFrame.
        """
        print("1. Generating facility buffer zones...")
        buffers_gdf = self.create_buffers(facilities_gdf, facility_id_col)
        
        print("2. Performing spatial intersection...")
        joined_gdf = self.intersect_fires(fires_gdf, buffers_gdf)
        
        if joined_gdf.empty:
            print("No fires detected within facility buffer zones.")
            return joined_gdf, buffers_gdf
            
        print("3. Calculating proximity scores...")
        scored_gdf = self.calculate_scores(joined_gdf, fire_id_col, facility_id_col)
        
        print("4. Ranking multi-facility conflicts...")
        final_ranked_gdf = self.rank_multi_facility_fires(scored_gdf, fire_id_col)
        
        # Optionally reproject back to WGS84 (EPSG:4326) for common mapping tools
        final_ranked_wgs84 = final_ranked_gdf.to_crs("EPSG:4326")
        buffers_wgs84 = buffers_gdf.to_crs("EPSG:4326")
        
        print(f"Analysis complete! Found {len(final_ranked_wgs84[fire_id_col].unique())} fires posing a threat.")
        
        return final_ranked_wgs84, buffers_wgs84

if __name__ == "__main__":
    pass
