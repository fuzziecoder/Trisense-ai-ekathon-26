"""
TriSense AI - Pattern Recognition Agent
Detects clinical deterioration patterns like early sepsis, septic shock, etc.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..models.patient_buffer import PatientBuffer
from ..models.schemas import PatternMatch


@dataclass
class PatternCriteria:
    """Criteria for a clinical pattern"""
    name: str
    description: str
    severity: float  # 0-1
    criteria: Dict[str, str]  # vital -> condition


class PatternRecognitionAgent:
    """
    Agent that detects clinical deterioration patterns.
    Uses 15+ predefined sepsis and deterioration patterns.
    """
    
    def __init__(self):
        self.patterns = self._define_patterns()
    
    def _define_patterns(self) -> List[PatternCriteria]:
        """Define clinical deterioration patterns"""
        return [
            # Sepsis patterns
            PatternCriteria(
                name="early_sepsis",
                description="Early signs of sepsis with inflammatory response",
                severity=0.6,
                criteria={
                    "heart_rate": "increasing",
                    "temperature": "elevated",
                    "respiratory_rate": "increasing",
                    "systolic_bp": "normal_or_decreasing"
                }
            ),
            PatternCriteria(
                name="septic_shock",
                description="Septic shock with hemodynamic instability",
                severity=0.95,
                criteria={
                    "heart_rate": "very_high",
                    "systolic_bp": "very_low",
                    "spo2": "decreasing",
                    "temperature": "abnormal"
                }
            ),
            PatternCriteria(
                name="compensated_shock",
                description="Compensatory tachycardia with declining perfusion",
                severity=0.7,
                criteria={
                    "heart_rate": "high",
                    "systolic_bp": "decreasing",
                    "spo2": "normal"
                }
            ),
            PatternCriteria(
                name="decompensated_shock",
                description="Decompensated shock with multi-organ stress",
                severity=0.9,
                criteria={
                    "heart_rate": "very_high",
                    "systolic_bp": "low",
                    "respiratory_rate": "high",
                    "spo2": "decreasing"
                }
            ),
            
            # Respiratory patterns
            PatternCriteria(
                name="respiratory_distress",
                description="Acute respiratory distress",
                severity=0.75,
                criteria={
                    "respiratory_rate": "very_high",
                    "spo2": "decreasing",
                    "heart_rate": "high"
                }
            ),
            PatternCriteria(
                name="hypoxemia",
                description="Progressive hypoxemia",
                severity=0.8,
                criteria={
                    "spo2": "very_low",
                    "respiratory_rate": "high",
                    "heart_rate": "increasing"
                }
            ),
            PatternCriteria(
                name="ards_precursor",
                description="Early ARDS pattern",
                severity=0.85,
                criteria={
                    "spo2": "decreasing",
                    "respiratory_rate": "very_high",
                    "heart_rate": "high"
                }
            ),
            
            # Cardiac patterns
            PatternCriteria(
                name="bradycardia_alert",
                description="Significant bradycardia with potential hemodynamic impact",
                severity=0.65,
                criteria={
                    "heart_rate": "very_low",
                    "systolic_bp": "decreasing"
                }
            ),
            PatternCriteria(
                name="tachycardia_persistent",
                description="Persistent tachycardia indicating stress",
                severity=0.5,
                criteria={
                    "heart_rate": "high",
                    "temperature": "elevated"
                }
            ),
            PatternCriteria(
                name="cardiac_arrest_precursor",
                description="Pre-arrest vital pattern",
                severity=1.0,
                criteria={
                    "heart_rate": "extreme",
                    "systolic_bp": "very_low",
                    "spo2": "very_low"
                }
            ),
            
            # Temperature patterns
            PatternCriteria(
                name="febrile_response",
                description="Febrile response with systemic impact",
                severity=0.45,
                criteria={
                    "temperature": "high",
                    "heart_rate": "elevated",
                    "respiratory_rate": "elevated"
                }
            ),
            PatternCriteria(
                name="hypothermia_warning",
                description="Hypothermia indicating poor perfusion",
                severity=0.7,
                criteria={
                    "temperature": "very_low",
                    "heart_rate": "low"
                }
            ),
            
            # Hemodynamic patterns
            PatternCriteria(
                name="hypotension_progressive",
                description="Progressive hypotension",
                severity=0.75,
                criteria={
                    "systolic_bp": "decreasing",
                    "diastolic_bp": "decreasing",
                    "heart_rate": "increasing"
                }
            ),
            PatternCriteria(
                name="hypertensive_crisis",
                description="Hypertensive emergency",
                severity=0.8,
                criteria={
                    "systolic_bp": "very_high",
                    "diastolic_bp": "very_high",
                    "heart_rate": "variable"
                }
            ),
            
            # Compound patterns
            PatternCriteria(
                name="multi_organ_stress",
                description="Multiple organ systems showing stress",
                severity=0.85,
                criteria={
                    "heart_rate": "abnormal",
                    "systolic_bp": "abnormal",
                    "respiratory_rate": "abnormal",
                    "spo2": "abnormal",
                    "temperature": "abnormal"
                }
            ),
        ]
    
    def detect_patterns(self, buffer: PatientBuffer) -> List[PatternMatch]:
        """
        Detect all matching clinical patterns.
        
        Args:
            buffer: Patient vital signs buffer
        
        Returns:
            List of matched patterns with confidence
        """
        if buffer.size() < 3:
            return []
        
        matches = []
        vital_analysis = self._analyze_vitals(buffer)
        
        for pattern in self.patterns:
            match_result = self._check_pattern(pattern, vital_analysis)
            if match_result:
                confidence, matching_criteria = match_result
                matches.append(PatternMatch(
                    pattern_name=pattern.name,
                    confidence=confidence * pattern.severity,
                    description=pattern.description,
                    matching_criteria=matching_criteria
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return matches
    
    def _analyze_vitals(self, buffer: PatientBuffer) -> Dict[str, str]:
        """Analyze vital trends and states"""
        analysis = {}
        window = buffer.get_window()
        latest = buffer.get_latest()
        
        # Heart rate analysis
        hr = latest.get('heart_rate', 80)
        hr_arr = window.get('heart_rate', [])
        if hr > 140:
            analysis['heart_rate'] = 'extreme'
        elif hr > 120:
            analysis['heart_rate'] = 'very_high'
        elif hr > 100:
            analysis['heart_rate'] = 'high'
        elif hr < 50:
            analysis['heart_rate'] = 'very_low'
        elif hr < 60:
            analysis['heart_rate'] = 'low'
        else:
            analysis['heart_rate'] = 'normal'
        
        # Add trend
        if len(hr_arr) >= 3:
            if hr_arr[-1] > hr_arr[0] * 1.1:
                analysis['heart_rate'] = 'increasing' if analysis['heart_rate'] == 'normal' else analysis['heart_rate']
        
        # Systolic BP analysis
        sbp = latest.get('systolic_bp', 120)
        sbp_arr = window.get('systolic_bp', [])
        if sbp > 180:
            analysis['systolic_bp'] = 'very_high'
        elif sbp > 140:
            analysis['systolic_bp'] = 'high'
        elif sbp < 80:
            analysis['systolic_bp'] = 'very_low'
        elif sbp < 90:
            analysis['systolic_bp'] = 'low'
        else:
            analysis['systolic_bp'] = 'normal'
        
        if len(sbp_arr) >= 3:
            if sbp_arr[-1] < sbp_arr[0] * 0.9:
                analysis['systolic_bp'] = 'decreasing'
        
        # Diastolic BP
        dbp = latest.get('diastolic_bp', 80)
        if dbp > 120:
            analysis['diastolic_bp'] = 'very_high'
        elif dbp < 60:
            analysis['diastolic_bp'] = 'low'
        else:
            analysis['diastolic_bp'] = 'normal'
        
        # Respiratory rate
        rr = latest.get('respiratory_rate', 16)
        if rr > 30:
            analysis['respiratory_rate'] = 'very_high'
        elif rr > 24:
            analysis['respiratory_rate'] = 'high'
        elif rr > 20:
            analysis['respiratory_rate'] = 'elevated'
        elif rr < 10:
            analysis['respiratory_rate'] = 'very_low'
        else:
            analysis['respiratory_rate'] = 'normal'
        
        rr_arr = window.get('respiratory_rate', [])
        if len(rr_arr) >= 3:
            if rr_arr[-1] > rr_arr[0] * 1.15:
                analysis['respiratory_rate'] = 'increasing'
        
        # SpO2
        spo2 = latest.get('spo2', 98)
        spo2_arr = window.get('spo2', [])
        if spo2 < 85:
            analysis['spo2'] = 'very_low'
        elif spo2 < 90:
            analysis['spo2'] = 'low'
        elif spo2 < 94:
            analysis['spo2'] = 'borderline'
        else:
            analysis['spo2'] = 'normal'
        
        if len(spo2_arr) >= 3:
            if spo2_arr[-1] < spo2_arr[0] - 2:
                analysis['spo2'] = 'decreasing'
        
        # Temperature
        temp = latest.get('temperature', 37.0)
        if temp > 39.5:
            analysis['temperature'] = 'very_high'
        elif temp > 38.3:
            analysis['temperature'] = 'high'
        elif temp > 37.5:
            analysis['temperature'] = 'elevated'
        elif temp < 35.5:
            analysis['temperature'] = 'very_low'
        elif temp < 36.0:
            analysis['temperature'] = 'low'
        else:
            analysis['temperature'] = 'normal'
        
        if temp > 38 or temp < 36:
            analysis['temperature'] = 'abnormal'
        
        return analysis
    
    def _check_pattern(self, pattern: PatternCriteria, 
                       vital_analysis: Dict[str, str]) -> Optional[tuple]:
        """Check if a pattern matches the vital analysis"""
        matching_criteria = []
        matches = 0
        total_criteria = len(pattern.criteria)
        
        for vital, expected in pattern.criteria.items():
            actual = vital_analysis.get(vital, 'normal')
            
            # Check for match
            if self._condition_matches(actual, expected):
                matches += 1
                matching_criteria.append(f"{vital}: {actual}")
        
        # Require at least 60% match
        if matches / total_criteria >= 0.6:
            confidence = matches / total_criteria
            return confidence, matching_criteria
        
        return None
    
    def _condition_matches(self, actual: str, expected: str) -> bool:
        """Check if actual condition matches expected"""
        # Direct match
        if actual == expected:
            return True
        
        # Semantic matches
        match_groups = {
            'increasing': ['increasing', 'high', 'elevated', 'very_high'],
            'decreasing': ['decreasing', 'low', 'very_low'],
            'high': ['high', 'elevated', 'very_high', 'extreme'],
            'low': ['low', 'very_low'],
            'elevated': ['elevated', 'high', 'increasing'],
            'abnormal': ['high', 'low', 'very_high', 'very_low', 'elevated', 'abnormal', 'extreme'],
            'normal_or_decreasing': ['normal', 'decreasing', 'low'],
            'variable': ['normal', 'high', 'low', 'increasing', 'decreasing']
        }
        
        if expected in match_groups:
            return actual in match_groups[expected]
        
        return False
    
    def get_agent_output(self, buffer: PatientBuffer) -> Dict[str, Any]:
        """Get formatted agent output"""
        patterns = self.detect_patterns(buffer)
        
        max_risk = max([p.confidence for p in patterns], default=0.0)
        
        return {
            "agent_name": "PatternRecognitionAgent",
            "risk_contribution": max_risk,
            "confidence": max_risk,
            "findings": {
                "patterns_detected": len(patterns),
                "top_patterns": [p.pattern_name for p in patterns[:3]],
                "descriptions": [p.description for p in patterns[:3]]
            },
            "reasoning": self._generate_reasoning(patterns)
        }
    
    def _generate_reasoning(self, patterns: List[PatternMatch]) -> str:
        """Generate reasoning text"""
        if not patterns:
            return "No concerning clinical patterns detected."
        
        top = patterns[0]
        if len(patterns) == 1:
            return f"Detected {top.pattern_name}: {top.description}. Confidence: {top.confidence:.0%}."
        else:
            return f"Detected {len(patterns)} patterns. Primary: {top.pattern_name} ({top.confidence:.0%}). {top.description}."
