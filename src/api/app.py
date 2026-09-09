import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.api.service import APIService

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown routines."""
    logger.info("Initializing Thermal Fire Monitoring REST API...")
    try:
        # Pre-warm service and database check
        service = APIService()
        health = service.get_health()
        logger.info(f"API Startup Check: DB Connected={health.database_connected}, Fires={health.total_fires_in_db}")
    except Exception as e:
        logger.warning(f"Startup pre-warming warning: {e}")
    yield
    logger.info("Shutting down Thermal Fire Monitoring REST API...")

def create_app() -> FastAPI:
    """Application factory for the FastAPI service."""
    application = FastAPI(
        title="PyroVision - AI Industrial Fire & Thermal Anomaly Detection API",
        description=(
            "Production REST API for satellite thermal monitoring, industrial asset proximity analysis, "
            "hotspot intensity scoring, and machine learning fire classification (SIH 2026 / NTRO)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable Cross-Origin Resource Sharing (CORS)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    application.include_router(api_router)

    # Top-level routes
    @application.get("/", tags=["General"])
    def root():
        return {
            "title": "PyroVision - Industrial Fire & Thermal Source Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "endpoints": {
                "fires": "/api/fires",
                "facilities": "/api/facilities",
                "hotspots": "/api/hotspots",
                "stats": "/api/stats",
                "classify": "/api/classify",
                "health": "/api/health",
            }
        }

    @application.get("/health", tags=["General"])
    def root_health():
        service = APIService()
        return service.get_health()

    @application.get("/health/deep", tags=["General"])
    def root_deep_health():
        from src.monitoring.health import SystemHealthManager
        manager = SystemHealthManager()
        return manager.get_deep_diagnostics()

    return application

app = create_app()
