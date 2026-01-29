"""
TriSense AI - Baseline Drift Detection Agent
Monitors patient-specific baselines and detects deviations.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from ..models.patient_buffer import PatientBuffer


class BaselineDriftAgent:
    """
    Agent that learns patient-specific baselines and detects drift.
    Like detecting regime change in stock markets.
    """
    
    def __init__(self):
        # Store patient baselines
        self.patient_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Drift thresholds (in standard deviations)
        self.drift_thresholds = {
            'mild': 1.5,
            'moderate': 2.0,
            'severe': 2.5,
            'critical': 3.0
        }
        
        # Vital weights for risk calculation
        self.vital_weights = {
            'heart_rate': 0.20,
            'systolic_bp': 0.25,
            'diastolic_bp': 0.10,
            'respiratory_rate': 0.15,
            'spo2': 0.20,
            'temperature': 0.10
        }
    
    def learn_baseline(self, patient_id: str, buffer: PatientBuffer):
        """
        Learn patient baseline from initial readings.
        Should be called after 4-6 hours of data collection.
        """
        if buffer.size() < 10:
            return
        
        window = buffer.get_window(window_size=min(20, buffer.size()))
        
        baseline = {}
        baseline_std = {}
        
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 
                      'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) > 0:
                baseline[vital] = float(np.mean(arr))
                baseline_std[vital] = float(np.std(arr)) if np.std(arr) > 0 else 1.0
            else:
                baseline[vital] = self._get_default_baseline(vital)
                baseline_std[vital] = 1.0
        
        self.patient_baselines[patient_id] = {
            'baseline': baseline,
            'std': baseline_std,
            'established_at': datetime.utcnow(),
            'sample_size': buffer.size()
        }
    
    def _get_default_baseline(self, vital: str) -> float:
        """Get default baseline values (population averages)"""
        defaults = {
            'heart_rate': 80.0,
            'systolic_bp': 120.0,
            'diastolic_bp': 80.0,
            'respiratory_rate': 16.0,
            'spo2': 98.0,
            'temperature': 37.0
        }
        return defaults.get(vital, 0.0)
    
    def detect_drift(self, patient_id: str, buffer: PatientBuffer) -> Dict[str, float]:
        """
        Detect drift from baseline for each vital.
        
        Returns:
            Dict mapping vital to z-score deviation
        """
        # Auto-learn baseline if not exists
        if patient_id not in self.patient_baselines:
            if buffer.has_baseline():
                self.learn_baseline(patient_id, buffer)
            else:
                return {}
        
        # Use buffer's baseline if available, otherwise use stored
        if buffer.has_baseline():
            deviations = buffer.get_deviation_from_baseline()
        else:
            baseline_data = self.patient_baselines.get(patient_id, {})
            baseline = baseline_data.get('baseline', {})
            std = baseline_data.get('std', {})
            
            latest = buffer.get_latest()
            deviations = {}
            
            for vital, value in latest.items():
                base_val = baseline.get(vital, self._get_default_baseline(vital))
                std_val = std.get(vital, 1.0)
                deviations[vital] = (value - base_val) / std_val
        
        return deviations
    
    def calculate_drift_score(self, drift_scores: Dict[str, float]) -> float:
        """
        Calculate overall drift risk score.
        
        Returns:
            Risk score 0-1
        """
        if not drift_scores:
            return 0.0
        
        weighted_score = 0.0
        
        for vital, zscore in drift_scores.items():
            weight = self.vital_weights.get(vital, 0.1)
            # Convert z-score to 0-1 risk
            vital_risk = min(abs(zscore) / 3.0, 1.0)
            weighted_score += weight * vital_risk
        
        return min(weighted_score, 1.0)
    
    def get_drift_severity(self, zscore: float) -> str:
        """Get severity level from z-score"""
        abs_z = abs(zscore)
        
        if abs_z >= self.drift_thresholds['critical']:
            return 'CRITICAL'
        elif abs_z >= self.drift_thresholds['severe']:
            return 'SEVERE'
        elif abs_z >= self.drift_thresholds['moderate']:
            return 'MODERATE'
        elif abs_z >= self.drift_thresholds['mild']:
            return 'MILD'
        else:
            return 'NORMAL'
    
    def get_direction(self, zscore: float) -> str:
        """Get drift direction"""
        if zscore > 0.5:
            return 'ELEVATED'
        elif zscore < -0.5:
            return 'DECREASED'
        else:
            return 'STABLE'
    
    def get_agent_output(self, patient_id: str, buffer: PatientBuffer) -> Dict[str, Any]:
        """Get formatted agent output"""
        drift_scores = self.detect_drift(patient_id, buffer)
        overall_risk = self.calculate_drift_score(drift_scores)
        
        # Find most concerning drifts
        concerning = []
        for vital, zscore in drift_scores.items():
            severity = self.get_drift_severity(zscore)
            if severity in ['MODERATE', 'SEVERE', 'CRITICAL']:
                direction = self.get_direction(zscore)
                concerning.append({
                    'vital': vital,
                    'zscore': zscore,
                    'severity': severity,
                    'direction': direction
                })
        
        # Sort by absolute z-score
        concerning.sort(key=lambda x: abs(x['zscore']), reverse=True)
        
        return {
            "agent_name": "BaselineDriftAgent",
            "risk_contribution": overall_risk,
            "confidence": min(0.9, 0.5 + overall_risk * 0.5),
            "findings": {
                "drift_scores": {k: round(v, 2) for k, v in drift_scores.items()},
                "concerning_vitals": concerning[:3],
                "overall_drift": 'HIGH' if overall_risk > 0.6 else 'MODERATE' if overall_risk > 0.3 else 'LOW'
            },
            "reasoning": self._generate_reasoning(concerning, overall_risk)
        }
    
    def _generate_reasoning(self, concerning: list, overall_risk: float) -> str:
        """Generate reasoning text"""
        if not concerning:
            return "Patient vitals are within their established baseline parameters."
        
        if overall_risk > 0.6:
            top = concerning[0]
            return f"Significant drift from baseline detected. {top['vital'].replace('_', ' ').title()} is {top['direction'].lower()} ({top['zscore']:.1f}σ from baseline). Multiple vitals showing concerning deviation."
        elif overall_risk > 0.3:
            top = concerning[0]
            return f"Moderate baseline drift noted. {top['vital'].replace('_', ' ').title()} trending {top['direction'].lower()}. Close monitoring advised."
        else:
            return "Minor fluctuations from baseline, within acceptable variance."
