import io
import time
import logging
import requests
import pandas as pd
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

class FIRMSClient:
    """Client for fetching active fire data from NASA FIRMS API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.FIRMS_MAP_KEY
        if not self.api_key:
            logger.warning("FIRMS API key not provided. Fetching might fail.")
        self.base_url = settings.FIRMS_BASE_URL
        
    def fetch_active_fires(self, source: str, area_coords: Optional[str] = None, 
                           day_range: Optional[int] = None, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch active fires for a specific source."""
        if not area_coords:
            bbox = settings.INDIA_BBOX
            area_coords = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        
        day_range = day_range or settings.DEFAULT_DAY_RANGE
        
        url = f"{self.base_url}/{self.api_key}/{source}/{area_coords}/{day_range}"
        if date:
            url += f"/{date}"
            
        logger.info(f"Fetching FIRMS data from source {source}...")
        
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                if self._validate_response(response):
                    df = pd.read_csv(io.StringIO(response.text))
                    logger.info(f"Fetched {len(df)} records from {source}.")
                    return df
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt+1}/3 failed for {source}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to fetch data for {source} after 3 attempts.")
        return pd.DataFrame()

    def fetch_all_sources(self, area_coords: Optional[str] = None, day_range: Optional[int] = None) -> pd.DataFrame:
        """Fetch active fires from all configured sources and combine them."""
        all_dfs = []
        for source in settings.FIRMS_SOURCES:
            df = self.fetch_active_fires(source, area_coords, day_range)
            if not df.empty:
                df['source'] = source
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Combined fetched records: {len(combined_df)} total fires.")
            return combined_df
        else:
            logger.warning("No fire data fetched from any source.")
            return pd.DataFrame()

    def _validate_response(self, response: requests.Response) -> bool:
        """Validate API response."""
        if response.status_code == 200:
            if "text/csv" in response.headers.get("Content-Type", "") or "text/plain" in response.headers.get("Content-Type", ""):
                return True
            else:
                logger.error(f"Unexpected content type: {response.headers.get('Content-Type')}")
                return False
        else:
            logger.error(f"API Error {response.status_code}: {response.text}")
            return False
