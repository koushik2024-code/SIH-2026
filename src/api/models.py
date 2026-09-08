from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FireRecord(BaseModel):
    """Schema representing a classified active fire detection."""
    detection_id: str
    latitude: float
    longitude: float
    brightness: Optional[float] = None
    frp: Optional[float] = None
    brightness_frp_ratio: Optional[float] = None
    acq_date: Optional[str] = None
    acq_time: Optional[str] = None
    datetime: Optional[str] = None
    satellite: Optional[str] = "UNKNOWN"
    confidence: Optional[Any] = "nominal"
    is_daytime: Optional[int] = 1
    distance_to_nearest_industrial: Optional[float] = -1.0
    nearest_facility_name: Optional[str] = "none"
    nearest_facility_type: Optional[str] = "none"
    is_near_industrial: Optional[int] = 0
    fire_type: Optional[str] = "Other/Unknown"
    fire_type_id: Optional[int] = 5
    classification_confidence: Optional[float] = 0.80
    land_cover: Optional[int] = 9
    fire_cluster_id: Optional[int] = 0
    ingested_at: Optional[str] = None

class FireListResponse(BaseModel):
    """Paginated list of fire detections."""
    total: int
    count: int
    limit: int
    offset: int
    data: List[FireRecord]

class FacilityRecord(BaseModel):
    """Schema for a registered industrial facility."""
    facility_id: str
    name: str
    facility_type: str
    latitude: float
    longitude: float
    source: Optional[str] = "OSM"

class FacilityListResponse(BaseModel):
    """List of registered industrial facilities."""
    total: int
    data: List[FacilityRecord]

class HotspotRecord(BaseModel):
    """Schema for a high-intensity thermal anomaly hotspot."""
    detection_id: str
    latitude: float
    longitude: float
    brightness: float
    frp: float
    confidence: Any
    intensity_score: float
    fire_type: str
    facility_nearby: Optional[str] = "none"
    distance_km: Optional[float] = -1.0
    cluster_id: Optional[int] = 0

class HotspotListResponse(BaseModel):
    """List of high-priority thermal hotspots."""
    total: int
    data: List[HotspotRecord]

class StatsResponse(BaseModel):
    """Aggregated system and thermal analytics statistics."""
    total_fires: int
    avg_brightness: float
    avg_frp: float
    fire_type_distribution: Dict[str, int]
    day_fires: int
    night_fires: int
    near_industrial_count: int
    critical_hotspots: int
    last_pipeline_run: Optional[Dict[str, Any]] = None

class ClassifyFeatureInput(BaseModel):
    """Single fire observation for on-demand classification."""
    latitude: float = Field(..., description="Latitude in decimal degrees")
    longitude: float = Field(..., description="Longitude in decimal degrees")
    brightness: float = Field(320.0, description="Brightness temperature in Kelvin")
    frp: float = Field(15.0, description="Fire Radiative Power in MW")
    confidence: Optional[Any] = Field(80.0, description="Detection confidence percentage or level")
    distance_to_nearest_industrial: Optional[float] = Field(None, description="Distance in km to nearest facility")
    nearest_facility_type: Optional[str] = Field("unknown", description="Type of nearest industrial plant")
    acq_date: Optional[str] = Field(None, description="Acquisition date (YYYY-MM-DD)")
    acq_time: Optional[str] = Field("1200", description="Acquisition time (HHMM)")
    satellite: Optional[str] = Field("VIIRS", description="Observing satellite source")

class ClassifyBatchRequest(BaseModel):
    """Batch payload of fire observations for classification."""
    records: List[ClassifyFeatureInput]

class ClassifyResult(BaseModel):
    """Classification inference result."""
    fire_type: str
    fire_type_id: int
    classification_confidence: float
    is_near_industrial: int
    distance_to_nearest_industrial: float
    nearest_facility_name: Optional[str] = "none"
    nearest_facility_type: Optional[str] = "none"

class ClassifyBatchResponse(BaseModel):
    """Response containing classification results for all submitted records."""
    total_classified: int
    results: List[ClassifyResult]

class HealthResponse(BaseModel):
    """Service health and backend connectivity status."""
    status: str
    version: str
    database_connected: bool
    total_fires_in_db: int
    total_facilities_in_db: int
    ml_classifier_loaded: bool
