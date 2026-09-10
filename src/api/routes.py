from typing import Optional, List, Union
from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.responses import FileResponse

from src.api.models import (
    FireRecord,
    FireListResponse,
    FacilityListResponse,
    HotspotListResponse,
    StatsResponse,
    ClassifyFeatureInput,
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    HealthResponse,
)
from src.api.service import APIService

router = APIRouter(prefix="/api", tags=["Thermal Fire Monitoring API"])

_service_instance: Optional[APIService] = None

def get_service() -> APIService:
    """Dependency provider returning singleton APIService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = APIService()
    return _service_instance

@router.get(
    "/fires",
    response_model=FireListResponse,
    summary="Get all classified fires",
    description="Retrieve active fire detections with multi-criteria filtering and pagination."
)
def get_fires(
    limit: int = Query(50, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    fire_type: Optional[str] = Query(None, description="Filter by fire category (e.g. 'Industrial Fire', 'Gas Flare')"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100, description="Minimum detection confidence percentage"),
    is_near_industrial: Optional[int] = Query(None, ge=0, le=1, description="1 for near-industrial, 0 for outside"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    service: APIService = Depends(get_service)
):
    return service.get_fires(
        limit=limit,
        offset=offset,
        fire_type=fire_type,
        min_confidence=min_confidence,
        is_near_industrial=is_near_industrial,
        start_date=start_date,
        end_date=end_date
    )

@router.get(
    "/fires/{detection_id}",
    response_model=FireRecord,
    summary="Get specific fire details",
    description="Look up detailed telemetry for a single thermal anomaly by primary detection ID."
)
def get_fire_by_id(
    detection_id: str,
    service: APIService = Depends(get_service)
):
    fire = service.get_fire_by_id(detection_id)
    if not fire:
        raise HTTPException(
            status_code=404,
            detail=f"Fire detection record '{detection_id}' not found."
        )
    return fire

@router.get(
    "/fires/type/{fire_type}",
    response_model=FireListResponse,
    summary="Filter fires by fire type",
    description="Convenience endpoint to query thermal detections for a specific classification class."
)
def get_fires_by_type(
    fire_type: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    service: APIService = Depends(get_service)
):
    return service.get_fires(
        limit=limit,
        offset=offset,
        fire_type=fire_type
    )

@router.get(
    "/facilities",
    response_model=FacilityListResponse,
    summary="Get all industrial facilities",
    description="Retrieve registered industrial facilities from OpenStreetMap / persistent storage."
)
def get_facilities(
    facility_type: Optional[str] = Query(None, description="Filter by facility type (e.g. 'oil_refinery', 'steel_plant')"),
    limit: int = Query(100, ge=1, le=1000, description="Max facilities to return"),
    service: APIService = Depends(get_service)
):
    return service.get_facilities(facility_type=facility_type, limit=limit)

@router.get(
    "/hotspots",
    response_model=HotspotListResponse,
    summary="Get current hotspots",
    description="Retrieve high-priority thermal anomaly hotspots ranked by composite intensity score."
)
def get_hotspots(
    min_frp: float = Query(20.0, ge=0.0, description="Minimum Fire Radiative Power (MW)"),
    limit: int = Query(50, ge=1, le=500, description="Max hotspots to return"),
    service: APIService = Depends(get_service)
):
    return service.get_hotspots(min_frp=min_frp, limit=limit)

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get summary statistics",
    description="Retrieve system-wide analytics, average brightness, FRP metrics, and fire distribution."
)
@router.get(
    "/metrics",
    response_model=StatsResponse,
    summary="Get operational metrics (alias for /stats)",
    description="Retrieve system-wide analytics, average brightness, FRP metrics, and fire distribution."
)
def get_stats(
    service: APIService = Depends(get_service)
):
    return service.get_statistics()

@router.get(
    "/analytics",
    summary="Get spatial intelligence analytics",
    description="Retrieve fire type distribution, time series, top impacted facilities, and regional surveillance summary."
)
def get_analytics(
    service: APIService = Depends(get_service)
):
    return service.get_analytics()

@router.post(
    "/classify",
    response_model=ClassifyBatchResponse,
    summary="Classify new fire data",
    description="Run machine learning classification inference on single or batch fire observations."
)
def classify_fire_data(
    payload: Union[ClassifyBatchRequest, ClassifyFeatureInput, List[ClassifyFeatureInput]] = Body(...),
    service: APIService = Depends(get_service)
):
    # Normalize input into a list of ClassifyFeatureInput
    if isinstance(payload, ClassifyBatchRequest):
        records = payload.records
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]

    return service.classify_records(records)

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Check API status, SQLite database connectivity, and ML classifier model status."
)
def health_check(
    service: APIService = Depends(get_service)
):
    return service.get_health()

@router.get(
    "/health/deep",
    summary="Deep system diagnostics",
    description="Comprehensive inspection of database integrity, table counts, disk space, ML model state, and satellite recency.",
    tags=["General"]
)
def deep_health_check():
    from src.monitoring.health import SystemHealthManager
    manager = SystemHealthManager()
    return manager.get_deep_diagnostics()

# ======================================================================
# Part 5.4: Historical Analysis & Reporting Endpoints
# ======================================================================

@router.get(
    "/reports/monthly",
    summary="Monthly Fire Intelligence Report",
    description="Retrieve monthly aggregated metrics, detection frequencies, thermal energy, and MoM velocity.",
    tags=["Historical Analysis & Reporting"]
)
def get_monthly_report(service: APIService = Depends(get_service)):
    return {"status": "success", "data": service.get_monthly_report()}

@router.get(
    "/reports/quarterly",
    summary="Quarterly Fire Intelligence Report",
    description="Retrieve quarterly aggregations and QoQ velocity trends.",
    tags=["Historical Analysis & Reporting"]
)
def get_quarterly_report(service: APIService = Depends(get_service)):
    return {"status": "success", "data": service.get_quarterly_report()}

@router.get(
    "/reports/trends",
    summary="Regional Surveillance Trends",
    description="Retrieve spatial trend vectors across Northern, Western, Central-Eastern, and Southern India.",
    tags=["Historical Analysis & Reporting"]
)
def get_regional_trends(service: APIService = Depends(get_service)):
    return {"status": "success", "data": service.get_regional_trends()}

@router.get(
    "/reports/facilities/risk",
    summary="Facility Longitudinal Risk Scores",
    description="Retrieve multi-factor dynamic risk ratings (0-100) and risk tiers for monitored industrial facilities.",
    tags=["Historical Analysis & Reporting"]
)
def get_facility_risks(service: APIService = Depends(get_service)):
    return {"status": "success", "data": service.get_facility_risks()}

@router.get(
    "/reports/summary",
    summary="Executive Surveillance Summary",
    description="Retrieve high-level macro metrics, total energy, and priority facility counts.",
    tags=["Historical Analysis & Reporting"]
)
def get_reports_summary(service: APIService = Depends(get_service)):
    bundle = service.get_full_report_bundle()
    return {"status": "success", "summary": bundle.get("summary"), "files": bundle.get("files")}

@router.get(
    "/reports/export/pdf",
    summary="Export Executive PDF Report",
    description="Download the publication-grade executive incident and risk assessment PDF.",
    tags=["Historical Analysis & Reporting"]
)
def export_pdf_report(service: APIService = Depends(get_service)):
    bundle = service.get_full_report_bundle()
    pdf_path = bundle.get("files", {}).get("pdf")
    if not pdf_path:
        raise HTTPException(status_code=500, detail="Failed to generate PDF report.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="fire_historical_report.pdf"
    )

@router.get(
    "/reports/export/html",
    summary="Export Standalone Executive HTML Dossier",
    description="Download or view the standalone printable executive HTML report.",
    tags=["Historical Analysis & Reporting"]
)
def export_html_report(service: APIService = Depends(get_service)):
    bundle = service.get_full_report_bundle()
    html_path = bundle.get("files", {}).get("html")
    if not html_path:
        raise HTTPException(status_code=500, detail="Failed to generate HTML report.")
    return FileResponse(
        html_path,
        media_type="text/html",
        filename="fire_historical_report.html"
    )

