import logging
import pandas as pd
import geopandas as gpd

from config import settings
from src.data_ingestion import FIRMSClient, OSMIndustrialClient, LandCoverClassifier
from src.preprocessing import FireDataCleaner, FeatureEngineer

logger = logging.getLogger(__name__)

class DataPipeline:
    """Orchestrates the data ingestion and preprocessing pipeline (Part 1)."""
    
    def __init__(self):
        self.firms_client = FIRMSClient()
        self.osm_client = OSMIndustrialClient()
        self.cleaner = FireDataCleaner()
        self.engineer = FeatureEngineer()
        self.lc_classifier = LandCoverClassifier()
        
    def run_ingestion(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """Fetch raw data from sources."""
        logger.info("--- Starting Data Ingestion ---")
        
        # 1. Fetch FIRMS Data
        firms_df = self.firms_client.fetch_all_sources()
        
        # 2. Fetch OSM Facilities
        facilities_gdf = self.osm_client.fetch_industrial_facilities()
        
        return firms_df, facilities_gdf
        
    def run_preprocessing(self, firms_df: pd.DataFrame, facilities_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Clean and enrich the raw data."""
        logger.info("--- Starting Data Preprocessing ---")
        
        # 1. Clean Data
        clean_df = self.cleaner.clean_firms_data(firms_df)
        
        # 2. Engineer Features
        enriched_df = self.engineer.engineer_features(clean_df, facilities_gdf)
        
        # 3. Assign Land Cover
        # Convert to GeoDataFrame for spatial join
        if not enriched_df.empty:
            fire_gdf = gpd.GeoDataFrame(
                enriched_df, 
                geometry=gpd.points_from_xy(enriched_df.longitude, enriched_df.latitude),
                crs="EPSG:4326"
            )
            
            final_gdf = self.lc_classifier.assign_land_cover_from_osm(fire_gdf, facilities_gdf)
            # Convert back to regular DF, optionally drop geometry to save space
            final_df = pd.DataFrame(final_gdf.drop(columns=['geometry']))
            return final_df
            
        return enriched_df

    def run_full_pipeline(self) -> pd.DataFrame:
        """Run the complete ingestion and preprocessing pipeline."""
        try:
            settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            firms_df, facilities_gdf = self.run_ingestion()
            
            if not facilities_gdf.empty:
                facilities_path = settings.PROCESSED_DATA_DIR / "industrial_facilities.geojson"
                facilities_gdf.to_file(facilities_path, driver="GeoJSON")
                logger.info(f"Saved facilities to {facilities_path}")
            
            if firms_df.empty:
                logger.error("Pipeline stopped: No fire data ingested.")
                return pd.DataFrame()
                
            processed_df = self.run_preprocessing(firms_df, facilities_gdf)
            
            if not processed_df.empty:
                output_path = settings.PROCESSED_DATA_DIR / "enriched_fire_data.csv"
                processed_df.to_csv(output_path, index=False)
                logger.info(f"Saved enriched data to {output_path}")
                
                logger.info("--- Pipeline Summary ---")
                logger.info(f"Total processed fire records: {len(processed_df)}")
                if 'land_cover' in processed_df.columns:
                    ind_fires = len(processed_df[processed_df['land_cover'] == 2])
                    logger.info(f"Identified potential industrial fires: {ind_fires}")
            
            return processed_df
            
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_full_pipeline()
