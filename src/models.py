import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    WARNING = "WARNING"
    INFO = "INFO"

class TriggerType(str, Enum):
    NEW_INDUSTRIAL_FIRE = "NEW_INDUSTRIAL_FIRE"
    FACILITY_THERMAL_SPIKE = "FACILITY_THERMAL_SPIKE"
    INDUSTRIAL_CLUSTER_FORMATION = "INDUSTRIAL_CLUSTER_FORMATION"
    PERSISTENT_FIRE_48H = "PERSISTENT_FIRE_48H"
    DIAGNOSTIC_TEST = "DIAGNOSTIC_TEST"

@dataclass
class AlertEvent:
    """Represents an alert triggered by the thermal anomaly monitoring system."""
    alert_id: str = field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trigger_type: str = TriggerType.NEW_INDUSTRIAL_FIRE.value
    severity: str = AlertSeverity.HIGH.value
    detection_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    facility_name: str = "Unknown Facility"
    facility_type: str = "Industrial"
    distance_km: float = 0.0
    frp: float = 0.0
    brightness: float = 0.0
    confidence: float = 0.0
    title: str = "Thermal Anomaly Alert"
    description: str = ""
    channels_dispatched: List[str] = field(default_factory=list)
    status: str = "ACTIVE"
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert event to serializable dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertEvent":
        """Reconstruct AlertEvent from dictionary."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)
