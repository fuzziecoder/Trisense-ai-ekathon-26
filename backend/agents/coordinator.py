"""
TriSense AI - Agent Coordinator
Orchestrates all agents and builds consensus.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.patient_buffer import PatientBuffer
from ..models.schemas import (RiskAssessment, RiskCategory, DashboardUpdate,
                               PatternMatch, AgentOutput, Alert)
from ..ml.feature_engineering import get_engineer
from ..ml.risk_scorer import get_scorer
from .pattern_agent import PatternRecognitionAgent
from .drift_agent import BaselineDriftAgent
from .trend_agent import TrendPredictorAgent
from .reasoning_agent import ClinicalReasoningAgent
from .alert_agent import AlertEscalationAgent
from .suggestion_agent import SuggestionAgent


class AgentCoordinator:
    """
    Central orchestrator that coordinates all agents.
    Builds consensus, resolves conflicts, and produces unified output.
    """
    
    def __init__(self):
        # Initialize all agents
        self.pattern_agent = PatternRecognitionAgent()
        self.drift_agent = BaselineDriftAgent()
        self.trend_agent = TrendPredictorAgent()
        self.reasoning_agent = ClinicalReasoningAgent()
        self.alert_agent = AlertEscalationAgent()
        self.suggestion_agent = SuggestionAgent()
        
        # Feature engineering and risk scoring
        self.feature_engineer = get_engineer()
        self.risk_scorer = get_scorer()
        
        # Agent weights for consensus
        self.agent_weights = {
            "PatternRecognitionAgent": 0.25,
            "BaselineDriftAgent": 0.20,
            "TrendPredictorAgent": 0.15,
            "MLRiskScorer": 0.40
        }
        
        # Shared memory for agent communication
        self.shared_memory: Dict[str, Any] = {}
    
    def process_vitals(self, patient_id: str, buffer: PatientBuffer) -> DashboardUpdate:
        """
        Process vital signs through all agents and produce unified output.
        """
        timestamp = datetime.utcnow()
        latest_vitals = buffer.get_latest()
        
        if not latest_vitals:
            return self._empty_update(patient_id, timestamp)
        
        # 1. Feature Engineering
        features = self.feature_engineer.engineer_features(buffer)
        
        # 2. ML Risk Scoring (The source of truth)
        ml_output = self.risk_scorer.get_ml_output(features)
        ml_risk = ml_output["risk_score"]
        
        # 3. Run all agents for monitoring
        pattern_output = self.pattern_agent.get_agent_output(buffer)
        patterns = self.pattern_agent.detect_patterns(buffer)
        
        drift_output = self.drift_agent.get_agent_output(patient_id, buffer)
        trend_output = self.trend_agent.get_agent_output(buffer)
        
        # 4. Build consensus risk score (Used for overall alerting/dashboard)
        # We still use consensus for the main display, but the reasoning is strictly on ML
        consensus_risk = self._build_consensus(ml_risk, pattern_output, 
                                                drift_output, trend_output)
        
        # 5. Get clinical reasoning (STRICT: Based only on ML output)
        reasoning = self.reasoning_agent.generate_reasoning(ml_output)
        
        # 6. Generate suggestions
        suggestions = self.suggestion_agent.generate_suggestions(patterns, consensus_risk)
        
        # 7. Generate alerts
        alert = self.alert_agent.generate_alert(
            patient_id, consensus_risk, reasoning, suggestions
        )
        
        # 8. Update risk history
        buffer.add_risk_score(consensus_risk, timestamp)
        
        # 9. Get feature importance
        feature_importance = self.risk_scorer.get_top_features(features, top_n=6)
        
        # 10. Build dashboard update
        from ..models.schemas import VitalSigns
        vitals_obj = VitalSigns(**latest_vitals)
        
        return DashboardUpdate(
            type="vitals_update",
            patient_id=patient_id,
            timestamp=timestamp,
            risk_score=consensus_risk,
            risk_category=self._get_category(consensus_risk),
            vitals=vitals_obj,
            risk_history=buffer.get_risk_history(limit=50),
            vital_trends=buffer.get_vital_trends(limit=50),
            patterns=[PatternMatch(**p.model_dump()) for p in patterns[:3]],
            reasoning=reasoning,
            alert=alert,
            feature_importance=feature_importance,
            ml_output=ml_output
        )
    
    def _build_consensus(self, ml_risk: float, pattern_output: Dict, 
                         drift_output: Dict, trend_output: Dict) -> float:
        """Build consensus risk score from all agents."""
        scores = {
            "MLRiskScorer": ml_risk,
            "PatternRecognitionAgent": pattern_output.get("risk_contribution", 0),
            "BaselineDriftAgent": drift_output.get("risk_contribution", 0),
            "TrendPredictorAgent": trend_output.get("risk_contribution", 0)
        }
        
        weighted_sum = sum(
            scores[agent] * self.agent_weights.get(agent, 0.1)
            for agent in scores
        )
        
        # Take max if any agent sees high risk (safety margin)
        max_risk = max(scores.values())
        if max_risk > 0.7:
            weighted_sum = max(weighted_sum, max_risk * 0.9)
        
        return min(weighted_sum, 1.0)
    
    def _get_category(self, risk_score: float) -> RiskCategory:
        if risk_score >= 0.80:
            return RiskCategory.CRITICAL
        elif risk_score >= 0.60:
            return RiskCategory.HIGH
        elif risk_score >= 0.35:
            return RiskCategory.MODERATE
        return RiskCategory.LOW
    
    def _empty_update(self, patient_id: str, timestamp: datetime) -> DashboardUpdate:
        from ..models.schemas import VitalSigns
        return DashboardUpdate(
            type="vitals_update", patient_id=patient_id, timestamp=timestamp,
            risk_score=0.0, risk_category=RiskCategory.LOW,
            vitals=VitalSigns(heart_rate=0, systolic_bp=0, diastolic_bp=0,
                             respiratory_rate=0, spo2=0, temperature=0),
            risk_history=[], vital_trends={}, patterns=[], reasoning=None, alert=None
        )


# Singleton
_coordinator: Optional[AgentCoordinator] = None

def get_coordinator() -> AgentCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = AgentCoordinator()
    return _coordinator
