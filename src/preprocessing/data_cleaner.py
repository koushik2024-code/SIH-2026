import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FireDataCleaner:
    """Cleans and validates FIRMS fire data."""
    
    def clean_firms_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive cleaning of FIRMS data."""
        if df.empty:
            logger.warning("Empty DataFrame passed for cleaning.")
            return df
            
        initial_len = len(df)
        logger.info(f"Starting cleaning on {initial_len} records.")
        
        # Drop missing coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['latitude', 'longitude', 'acq_date', 'acq_time'])
        
        # Validate coordinate ranges
        df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
        
        # Process dates and times
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        
        # Handle acq_time format (HHMM integer to string, then pad)
        df['acq_time_str'] = df['acq_time'].astype(str).str.zfill(4)
        df['datetime'] = pd.to_datetime(df['acq_date'].dt.strftime('%Y-%m-%d') + ' ' + df['acq_time_str'], format='%Y-%m-%d %H%M')
        df = df.drop(columns=['acq_time_str'])
        
        # Filter confidence
        # VIIRS uses 'low', 'nominal', 'high'. MODIS uses 0-100%
        if 'confidence' in df.columns:
            viirs_mask = df['confidence'].isin(['nominal', 'high', 'n', 'h'])
            modis_mask = pd.to_numeric(df['confidence'], errors='coerce').ge(50)
            df = df[viirs_mask | modis_mask]
            
        # Normalize brightness (handle different column names for different instruments)
        if 'brightness' in df.columns:
            df['brightness_normalized'] = (df['brightness'] - df['brightness'].min()) / (df['brightness'].max() - df['brightness'].min() + 1e-6)
        
        if 'frp' in df.columns:
             df['frp_normalized'] = (df['frp'] - df['frp'].min()) / (df['frp'].max() - df['frp'].min() + 1e-6)
             
        # Add unique ID
        df['fire_id'] = [f"FIRE_{i}" for i in range(len(df))]
        
        final_len = len(df)
        logger.info(f"Cleaning complete. Total records: {initial_len}, Removed: {initial_len - final_len}, Remaining: {final_len}")
        
        return df
