"""
TriSense AI - Pydantic Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class VitalSigns(BaseModel):
    """Single vital signs reading"""
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in BPM")
    systolic_bp: float = Field(..., ge=0, le=300, description="Systolic blood pressure mmHg")
    diastolic_bp: float = Field(..., ge=0, le=200, description="Diastolic blood pressure mmHg")
    respiratory_rate: float = Field(..., ge=0, le=80, description="Respiratory rate per min")
    spo2: float = Field(..., ge=0, le=100, description="Blood oxygen saturation %")
    temperature: float = Field(..., ge=30, le=45, description="Temperature in Celsius")


class VitalReading(BaseModel):
    """Vital reading with timestamp"""
    patient_id: str
    timestamp: datetime
    vitals: VitalSigns
    metadata: Optional[Dict[str, Any]] = None


class PatientInfo(BaseModel):
    """Patient demographic information"""
    patient_id: str
    name: str
    age: Optional[int] = None
    gestational_age: Optional[str] = None  # For neonatal patients
    birth_weight: Optional[str] = None
    ward: Optional[str] = None
    admission_time: Optional[datetime] = None


class PatternMatch(BaseModel):
    """Detected clinical pattern"""
    pattern_name: str
    confidence: float = Field(..., ge=0, le=1)
    description: str
    matching_criteria: List[str]


class AgentOutput(BaseModel):
    """Output from a single agent"""
    agent_name: str
    risk_contribution: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    findings: Dict[str, Any]
    reasoning: str


class ClinicalReasoning(BaseModel):
    """Clinical reasoning explanation"""
    severity: str
    primary_concern: str
    physiological_interpretation: str
    timeline_estimate: Optional[str] = None
    contributing_factors: List[str] = []


class SuggestedAction(BaseModel):
    """Recommended clinical action"""
    action: str
    priority: int = Field(..., ge=1, le=5)
    rationale: str
    protocol_reference: Optional[str] = None


class Alert(BaseModel):
    """Clinical alert"""
    level: AlertLevel
    message: str
    risk_score: float
    reasoning: ClinicalReasoning
    actions: List[SuggestedAction]
    escalate_to: List[str]
    response_time: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    """Complete risk assessment output"""
    patient_id: str
    timestamp: datetime
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    vitals: VitalSigns
    features: Dict[str, float]
    patterns: List[PatternMatch]
    agent_outputs: List[AgentOutput]
    reasoning: ClinicalReasoning
    suggestions: List[SuggestedAction]
    alert: Optional[Alert] = None


class DashboardUpdate(BaseModel):
    """WebSocket message for dashboard"""
    type: str = "vitals_update"
    patient_id: str
    timestamp: datetime
    risk_score: float
    risk_category: RiskCategory
    vitals: VitalSigns
    risk_history: List[Dict[str, Any]] = []
    vital_trends: Dict[str, List[Any]] = {}
    patterns: List[PatternMatch] = []
    reasoning: Optional[ClinicalReasoning] = None
    alert: Optional[Alert] = None
    feature_importance: Dict[str, float] = {}
    ml_output: Optional[Dict[str, Any]] = None
