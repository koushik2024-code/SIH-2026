"""
Pipeline Automation Package (Part 5.1)
=====================================
Automated data collection, incremental anomaly fetching, ML auto-classification,
persistent SQLite storage, dashboard refreshing, and multi-platform task scheduling.
"""

from src.pipeline_automation.database import FireMonitoringDatabase
from src.pipeline_automation.classifier import FireClassifier
from src.pipeline_automation.incremental_fetcher import IncrementalDataFetcher
from src.pipeline_automation.automated_pipeline import AutomatedPipeline
from src.pipeline_automation.scheduler import PipelineScheduler

__all__ = [
    "FireMonitoringDatabase",
    "FireClassifier",
    "IncrementalDataFetcher",
    "AutomatedPipeline",
    "PipelineScheduler",
]
