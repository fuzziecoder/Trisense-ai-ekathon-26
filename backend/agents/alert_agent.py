"""
TriSense AI - Alert Escalation Agent
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
from ..config import settings
from ..models.schemas import Alert, AlertLevel, ClinicalReasoning, SuggestedAction


class AlertEscalationAgent:
    """Tiered alert generation with smart deduplication."""
    
    def __init__(self):
        self.alert_history: Dict[str, List[Dict]] = defaultdict(list)
        self.cooldown_periods = {
            AlertLevel.INFO: 600, AlertLevel.WARNING: 300,
            AlertLevel.CRITICAL: 120, AlertLevel.EMERGENCY: 60
        }
        self.escalation_targets = {
            AlertLevel.INFO: ["Charge Nurse"],
            AlertLevel.WARNING: ["Primary Nurse", "Charge Nurse"],
            AlertLevel.CRITICAL: ["Attending Physician", "Rapid Response"],
            AlertLevel.EMERGENCY: ["ICU Team", "Code Team"]
        }
        self.response_times = {
            AlertLevel.INFO: "Within shift", AlertLevel.WARNING: "Within 1 hour",
            AlertLevel.CRITICAL: "Within 15 min", AlertLevel.EMERGENCY: "Immediate"
        }
    
    def generate_alert(self, patient_id: str, risk_score: float,
                       reasoning: ClinicalReasoning, 
                       suggestions: List[SuggestedAction]) -> Optional[Alert]:
        level = self._determine_level(risk_score)
        if level is None or self._is_on_cooldown(patient_id, level):
            return None
        
        alert = Alert(
            level=level,
            message=self._format_message(level, reasoning),
            risk_score=risk_score, reasoning=reasoning,
            actions=suggestions[:5],
            escalate_to=self.escalation_targets.get(level, []),
            response_time=self.response_times.get(level, "As needed")
        )
        self._record_alert(patient_id, alert)
        return alert
    
    def _determine_level(self, risk_score: float) -> Optional[AlertLevel]:
        if risk_score >= 0.90: return AlertLevel.EMERGENCY
        elif risk_score >= 0.75: return AlertLevel.CRITICAL
        elif risk_score >= 0.50: return AlertLevel.WARNING
        elif risk_score >= 0.35: return AlertLevel.INFO
        return None
    
    def _is_on_cooldown(self, patient_id: str, level: AlertLevel) -> bool:
        cooldown = self.cooldown_periods.get(level, 300)
        now = datetime.utcnow()
        for alert in reversed(self.alert_history.get(patient_id, [])):
            if alert['level'] == level:
                if (now - alert['timestamp']).total_seconds() < cooldown:
                    return True
                break
        return False
    
    def _format_message(self, level: AlertLevel, reasoning: ClinicalReasoning) -> str:
        emojis = {AlertLevel.INFO: "ℹ️", AlertLevel.WARNING: "⚠️",
                  AlertLevel.CRITICAL: "🚨", AlertLevel.EMERGENCY: "🆘"}
        return f"{emojis.get(level, '')} {level.value}: {reasoning.primary_concern}"
    
    def _record_alert(self, patient_id: str, alert: Alert):
        self.alert_history[patient_id].append({
            'level': alert.level, 'timestamp': datetime.utcnow(),
            'risk_score': alert.risk_score
        })
        if len(self.alert_history[patient_id]) > 20:
            self.alert_history[patient_id] = self.alert_history[patient_id][-20:]
    
    def get_agent_output(self, patient_id: str, risk_score: float,
                         reasoning: ClinicalReasoning,
                         suggestions: List[SuggestedAction]) -> Dict[str, Any]:
        alert = self.generate_alert(patient_id, risk_score, reasoning, suggestions)
        return {
            "agent_name": "AlertEscalationAgent",
            "risk_contribution": risk_score if alert else 0.0,
            "confidence": 0.95,
            "findings": {"alert_generated": alert is not None,
                        "alert_level": alert.level.value if alert else None},
            "reasoning": f"Alert {'triggered' if alert else 'suppressed/not needed'}."
        }
