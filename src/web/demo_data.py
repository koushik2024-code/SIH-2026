import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import random
import hashlib

class DemoDataGenerator:
    """Generates realistic demo data for testing without API keys."""
    
    # Major Indian industrial clusters with real coordinates
    INDUSTRIAL_FACILITIES = [
        # Oil Refineries
        {"name": "Jamnagar Refinery (Reliance)", "lat": 22.4707, "lon": 70.0577, "type": "oil_refinery"},
        {"name": "Mathura Refinery (IOCL)", "lat": 27.4924, "lon": 77.6737, "type": "oil_refinery"},
        {"name": "Mangalore Refinery (MRPL)", "lat": 12.9141, "lon": 74.8560, "type": "oil_refinery"},
        {"name": "Kochi Refinery (BPCL)", "lat": 9.9312, "lon": 76.2673, "type": "oil_refinery"},
        {"name": "Haldia Refinery (IOCL)", "lat": 22.0667, "lon": 88.0698, "type": "oil_refinery"},
        # Thermal Power Plants
        {"name": "Vindhyachal STPP", "lat": 24.0880, "lon": 82.6629, "type": "thermal_power_plant"},
        {"name": "Mundra Thermal Power", "lat": 22.8394, "lon": 69.7225, "type": "thermal_power_plant"},
        {"name": "Talcher Super TPS", "lat": 20.9517, "lon": 85.2133, "type": "thermal_power_plant"},
        {"name": "Sipat STPP", "lat": 22.1219, "lon": 82.2715, "type": "thermal_power_plant"},
        {"name": "Korba STPP", "lat": 22.3595, "lon": 82.7501, "type": "thermal_power_plant"},
        # Steel Plants
        {"name": "Tata Steel Jamshedpur", "lat": 22.7876, "lon": 86.2025, "type": "steel_plant"},
        {"name": "SAIL Bhilai Steel Plant", "lat": 21.2093, "lon": 81.4285, "type": "steel_plant"},
        {"name": "JSW Steel Vijayanagar", "lat": 15.1394, "lon": 76.6339, "type": "steel_plant"},
        {"name": "SAIL Rourkela Steel", "lat": 22.2604, "lon": 84.8536, "type": "steel_plant"},
        # Mining Areas
        {"name": "Jharia Coalfield", "lat": 23.7466, "lon": 86.4132, "type": "mining"},
        {"name": "Singrauli Coalfield", "lat": 24.0996, "lon": 82.6373, "type": "mining"},
        {"name": "Talcher Coalfield", "lat": 20.9500, "lon": 85.2300, "type": "mining"},
        {"name": "Korba Coalfield", "lat": 22.3500, "lon": 82.7000, "type": "mining"},
        # Petrochemical
        {"name": "GAIL Pata Petrochemical", "lat": 26.2320, "lon": 80.5360, "type": "petrochemical"},
        {"name": "RIL Dahej Petrochemical", "lat": 21.7063, "lon": 72.5834, "type": "petrochemical"},
        # LNG Terminals
        {"name": "Dahej LNG Terminal", "lat": 21.7100, "lon": 72.5800, "type": "lng_terminal"},
        {"name": "Kochi LNG Terminal", "lat": 9.9400, "lon": 76.2700, "type": "lng_terminal"},
        # Gas Flares
        {"name": "Hazira Gas Processing", "lat": 21.1000, "lon": 72.6500, "type": "gas_flare"},
        {"name": "Bombay High Offshore", "lat": 19.3700, "lon": 71.3700, "type": "gas_flare"},
    ]
    
    # Known forest fire prone regions in India
    FOREST_FIRE_ZONES = [
        {"name": "Uttarakhand Forests", "lat_center": 30.0, "lon_center": 79.0},
        {"name": "Odisha Simlipal", "lat_center": 21.8, "lon_center": 86.4},
        {"name": "Madhya Pradesh Forests", "lat_center": 23.5, "lon_center": 78.0},
        {"name": "Mizoram Forests", "lat_center": 23.2, "lon_center": 92.8},
        {"name": "Chhattisgarh Forests", "lat_center": 21.5, "lon_center": 82.0},
        {"name": "Karnataka Western Ghats", "lat_center": 14.5, "lon_center": 75.5},
    ]
    
    # Agricultural burning zones (Punjab, Haryana - stubble burning)
    AGRI_BURNING_ZONES = [
        {"name": "Punjab Stubble", "lat_center": 30.7, "lon_center": 75.5},
        {"name": "Haryana Stubble", "lat_center": 29.5, "lon_center": 76.5},
        {"name": "UP Western", "lat_center": 29.0, "lon_center": 78.0},
    ]
    
    def generate_facilities(self) -> gpd.GeoDataFrame:
        """Generate industrial facilities GeoDataFrame."""
        records = []
        for i, f in enumerate(self.INDUSTRIAL_FACILITIES):
            records.append({
                "osm_id": f"demo/{i+1}",
                "name": f["name"],
                "facility_type": f["type"],
                "latitude": f["lat"],
                "longitude": f["lon"],
                "tags": "{}"
            })
        df = pd.DataFrame(records)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return gdf
    
    def generate_fire_data(self, n_fires: int = 500, days_back: int = 3) -> pd.DataFrame:
        """Generate realistic fire detection data."""
        np.random.seed(42)
        random.seed(42)
        
        fires = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # 1. Industrial fires (~15% near facilities)
        n_industrial = int(n_fires * 0.15)
        for _ in range(n_industrial):
            facility = random.choice(self.INDUSTRIAL_FACILITIES)
            fires.append(self._create_fire(
                lat=facility["lat"] + np.random.normal(0, 0.005),
                lon=facility["lon"] + np.random.normal(0, 0.005),
                brightness_range=(330, 500),
                frp_range=(10, 150),
                confidence="high",
                base_date=base_date,
                days_back=days_back,
                fire_type="Industrial Fire"
            ))
        
        # 2. Forest fires (~30%)
        n_forest = int(n_fires * 0.30)
        for _ in range(n_forest):
            zone = random.choice(self.FOREST_FIRE_ZONES)
            fires.append(self._create_fire(
                lat=zone["lat_center"] + np.random.normal(0, 0.3),
                lon=zone["lon_center"] + np.random.normal(0, 0.3),
                brightness_range=(305, 370),
                frp_range=(2, 50),
                confidence=random.choice(["nominal", "high"]),
                base_date=base_date,
                days_back=days_back,
                fire_type="Forest Fire"
            ))
        
        # 3. Agricultural burning (~35%)
        n_agri = int(n_fires * 0.35)
        for _ in range(n_agri):
            zone = random.choice(self.AGRI_BURNING_ZONES)
            fires.append(self._create_fire(
                lat=zone["lat_center"] + np.random.normal(0, 0.5),
                lon=zone["lon_center"] + np.random.normal(0, 0.5),
                brightness_range=(300, 340),
                frp_range=(1, 20),
                confidence=random.choice(["nominal", "low"]),
                base_date=base_date,
                days_back=days_back,
                fire_type="Agricultural Burning"
            ))
        
        # 4. Gas flares (~10%)
        n_flares = int(n_fires * 0.10)
        for _ in range(n_flares):
            facility = random.choice([f for f in self.INDUSTRIAL_FACILITIES if f["type"] in ["gas_flare", "oil_refinery", "lng_terminal"]])
            fires.append(self._create_fire(
                lat=facility["lat"] + np.random.normal(0, 0.003),
                lon=facility["lon"] + np.random.normal(0, 0.003),
                brightness_range=(350, 600),
                frp_range=(5, 80),
                confidence="high",
                base_date=base_date,
                days_back=days_back,
                fire_type="Gas Flare"
            ))
        
        # 5. Other/Unknown (~10%)
        n_other = n_fires - n_industrial - n_forest - n_agri - n_flares
        for _ in range(n_other):
            fires.append(self._create_fire(
                lat=np.random.uniform(8, 35),
                lon=np.random.uniform(70, 95),
                brightness_range=(300, 380),
                frp_range=(1, 30),
                confidence=random.choice(["nominal", "low", "high"]),
                base_date=base_date,
                days_back=days_back,
                fire_type="Other/Unknown"
            ))
        
        df = pd.DataFrame(fires)
        
        # Add engineered features
        facilities_gdf = self.generate_facilities()
        df = self._add_spatial_features(df, facilities_gdf)
        df = self._add_thermal_features(df)
        
        return df
    
    def _create_fire(self, lat, lon, brightness_range, frp_range, confidence, base_date, days_back, fire_type):
        """Create a single fire record."""
        acq_date = base_date + timedelta(hours=np.random.randint(0, days_back * 24))
        acq_time_int = int(acq_date.strftime("%H%M"))
        
        fire_id = hashlib.sha256(f"{lat:.5f}_{lon:.5f}_{acq_date.isoformat()}".encode()).hexdigest()[:12]
        
        return {
            "fire_id": fire_id,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "brightness": round(np.random.uniform(*brightness_range), 1),
            "frp": round(np.random.uniform(*frp_range), 1),
            "scan": round(np.random.uniform(0.4, 1.5), 1),
            "track": round(np.random.uniform(0.4, 1.2), 1),
            "acq_date": acq_date.strftime("%Y-%m-%d"),
            "acq_time": acq_time_int,
            "satellite": random.choice(["N", "1"]),
            "instrument": "VIIRS",
            "confidence": confidence,
            "version": "2.0NRT",
            "daynight": "D" if 6 <= acq_date.hour <= 18 else "N",
            "source": random.choice(["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"]),
            "datetime": acq_date,
            "hour_of_day": acq_date.hour,
            "day_of_week": acq_date.weekday(),
            "month": acq_date.month,
            "is_daytime": 1 if 6 <= acq_date.hour <= 18 else 0,
            "fire_type": fire_type,
            "land_cover": self._fire_type_to_land_cover(fire_type),
        }
    
    def _fire_type_to_land_cover(self, fire_type):
        mapping = {
            "Industrial Fire": 2, "Gas Flare": 2,
            "Forest Fire": 4, "Agricultural Burning": 3,
            "Mining Activity": 2, "Other/Unknown": 9
        }
        return mapping.get(fire_type, 9)
    
    def _add_spatial_features(self, df, facilities_gdf):
        from sklearn.neighbors import BallTree
        fire_coords = np.radians(df[['latitude', 'longitude']].values)
        facility_coords = np.radians(facilities_gdf[['latitude', 'longitude']].values)
        tree = BallTree(facility_coords, metric='haversine')
        distances, indices = tree.query(fire_coords, k=1)
        df['distance_to_nearest_industrial'] = np.round(distances.flatten() * 6371.0, 2)
        nearest = facilities_gdf.iloc[indices.flatten()]
        df['nearest_facility_type'] = nearest['facility_type'].values
        df['nearest_facility_name'] = nearest['name'].values
        df['is_near_industrial'] = (df['distance_to_nearest_industrial'] <= 2.0).astype(int)
        return df
    
    def _add_thermal_features(self, df):
        df['brightness_normalized'] = (df['brightness'] - df['brightness'].min()) / (df['brightness'].max() - df['brightness'].min() + 1e-6)
        df['frp_normalized'] = (df['frp'] - df['frp'].min()) / (df['frp'].max() - df['frp'].min() + 1e-6)
        df['brightness_frp_ratio'] = df['brightness'] / (df['frp'] + 1e-6)
        return df
