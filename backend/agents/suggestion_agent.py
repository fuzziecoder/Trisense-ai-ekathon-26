"""
TriSense AI - Suggestion Agent
Recommends clinical actions based on patterns and risk.
"""
from typing import Dict, Any, List
from ..models.schemas import SuggestedAction, PatternMatch


class SuggestionAgent:
    """Agent that generates protocol-based clinical action recommendations."""
    
    def __init__(self):
        self.protocols = self._define_protocols()
    
    def _define_protocols(self) -> Dict[str, Dict]:
        return {
            "sepsis_bundle": {
                "trigger_patterns": ["early_sepsis", "septic_shock"],
                "actions": [
                    ("Obtain blood cultures before antibiotics", 1, "Sepsis Bundle"),
                    ("Administer broad-spectrum antibiotics within 1 hour", 1, "Hour-1 Bundle"),
                    ("Measure serum lactate", 2, "Sepsis Workup"),
                    ("Begin fluid resuscitation (30mL/kg crystalloid)", 2, "Fluid Therapy"),
                    ("Re-assess volume status and tissue perfusion", 3, "Monitoring")
                ]
            },
            "respiratory_distress": {
                "trigger_patterns": ["respiratory_distress", "hypoxemia", "ards_precursor"],
                "actions": [
                    ("Increase supplemental oxygen, target SpO2 >94%", 1, "Oxygenation"),
                    ("Obtain chest X-ray", 2, "Diagnostic"),
                    ("Consider non-invasive ventilation if worsening", 2, "Ventilation"),
                    ("Arterial blood gas analysis", 3, "Lab Work")
                ]
            },
            "shock_management": {
                "trigger_patterns": ["compensated_shock", "decompensated_shock"],
                "actions": [
                    ("Establish large-bore IV access", 1, "Access"),
                    ("Initiate fluid bolus", 1, "Resuscitation"),
                    ("Continuous cardiac monitoring", 2, "Monitoring"),
                    ("Consider vasopressor support if fluid-refractory", 2, "Hemodynamics")
                ]
            },
            "cardiac_monitoring": {
                "trigger_patterns": ["bradycardia_alert", "tachycardia_persistent", "cardiac_arrest_precursor"],
                "actions": [
                    ("12-lead ECG immediately", 1, "Cardiac Workup"),
                    ("Continuous telemetry monitoring", 1, "Monitoring"),
                    ("Prepare defibrillator at bedside", 2, "Emergency Prep"),
                    ("Review medications affecting heart rate", 3, "Med Review")
                ]
            }
        }
    
    def generate_suggestions(self, patterns: List[PatternMatch], 
                             risk_score: float) -> List[SuggestedAction]:
        suggestions = []
        pattern_names = {p.pattern_name for p in patterns}
        
        for protocol_name, protocol in self.protocols.items():
            triggers = set(protocol["trigger_patterns"])
            if triggers & pattern_names:
                for action, priority, reference in protocol["actions"]:
                    suggestions.append(SuggestedAction(
                        action=action, priority=priority,
                        rationale=f"Based on detected pattern",
                        protocol_reference=reference
                    ))
        
        if risk_score >= 0.7 and not suggestions:
            suggestions.extend([
                SuggestedAction(action="Increase monitoring frequency to q15min",
                               priority=2, rationale="High risk score"),
                SuggestedAction(action="Notify senior clinician",
                               priority=1, rationale="Elevated risk"),
                SuggestedAction(action="Review recent labs and imaging",
                               priority=3, rationale="Clinical assessment")
            ])
        
        suggestions.sort(key=lambda x: x.priority)
        return suggestions[:8]
    
    def get_agent_output(self, patterns: List[PatternMatch], 
                         risk_score: float) -> Dict[str, Any]:
        suggestions = self.generate_suggestions(patterns, risk_score)
        return {
            "agent_name": "SuggestionAgent",
            "risk_contribution": 0.0,
            "confidence": 0.85,
            "findings": {
                "suggestions_count": len(suggestions),
                "top_actions": [s.action for s in suggestions[:3]]
            },
            "reasoning": f"Generated {len(suggestions)} protocol-based recommendations."
        }
