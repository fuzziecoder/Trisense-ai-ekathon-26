"""
TriSense AI - Trend Prediction Agent
Predicts future vital values using time-series forecasting.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta

from ..models.patient_buffer import PatientBuffer


class TrendPredictorAgent:
    """
    Agent that predicts future vital values.
    Uses trend analysis and PatchTST embeddings for forecasting.
    """
    
    def __init__(self):
        # Prediction horizons (in readings)
        self.horizons = [2, 4, 8]  # 30min, 1h, 2h assuming 15min intervals
        
        # Normal ranges for vitals
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'respiratory_rate': (12, 20),
            'spo2': (95, 100),
            'temperature': (36.1, 37.5)
        }
        
        # Critical thresholds
        self.critical_thresholds = {
            'heart_rate': {'low': 50, 'high': 130},
            'systolic_bp': {'low': 85, 'high': 180},
            'diastolic_bp': {'low': 50, 'high': 110},
            'respiratory_rate': {'low': 8, 'high': 28},
            'spo2': {'low': 88, 'high': 101},
            'temperature': {'low': 35.0, 'high': 39.0}
        }
    
    def predict_trends(self, buffer: PatientBuffer) -> Dict[str, Dict[str, Any]]:
        """
        Predict future values for each vital.
        
        Returns:
            Dict mapping vital to prediction info
        """
        predictions = {}
        window = buffer.get_window()
        
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp',
                      'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) >= 3:
                predictions[vital] = self._predict_vital(vital, arr)
            else:
                predictions[vital] = self._no_prediction(vital)
        
        return predictions
    
    def _predict_vital(self, vital: str, values: np.ndarray) -> Dict[str, Any]:
        """Predict future values for a single vital"""
        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate volatility
        std = np.std(values)
        
        # Make predictions
        future_predictions = []
        current_val = values[-1]
        
        for horizon in self.horizons:
            pred_val = intercept + slope * (len(values) + horizon - 1)
            
            # Add confidence interval based on volatility
            confidence_interval = 1.96 * std * np.sqrt(horizon / len(values))
            
            future_predictions.append({
                'horizon': horizon,
                'predicted_value': float(pred_val),
                'lower_bound': float(pred_val - confidence_interval),
                'upper_bound': float(pred_val + confidence_interval),
                'confidence': self._calculate_confidence(len(values), horizon, std)
            })
        
        # Determine trend direction
        if slope > 0.5:
            trend = 'INCREASING'
        elif slope < -0.5:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'
        
        # Calculate time to threshold breach
        breach_time = self._estimate_breach_time(vital, current_val, slope)
        
        # Risk assessment
        risk_level = self._assess_prediction_risk(vital, future_predictions)
        
        return {
            'current_value': float(current_val),
            'trend': trend,
            'slope': float(slope),
            'volatility': float(std),
            'predictions': future_predictions,
            'breach_estimate': breach_time,
            'risk_level': risk_level
        }
    
    def _no_prediction(self, vital: str) -> Dict[str, Any]:
        """Return empty prediction when insufficient data"""
        return {
            'current_value': 0.0,
            'trend': 'UNKNOWN',
            'slope': 0.0,
            'volatility': 0.0,
            'predictions': [],
            'breach_estimate': None,
            'risk_level': 'UNKNOWN'
        }
    
    def _calculate_confidence(self, data_points: int, horizon: int, std: float) -> float:
        """Calculate prediction confidence (0-1)"""
        # More data = higher confidence
        data_factor = min(data_points / 10, 1.0)
        
        # Shorter horizon = higher confidence
        horizon_factor = 1.0 - (horizon / 20)
        
        # Lower volatility = higher confidence
        vol_factor = max(0, 1.0 - std / 10)
        
        return data_factor * horizon_factor * vol_factor
    
    def _estimate_breach_time(self, vital: str, current: float, slope: float) -> Optional[Dict[str, Any]]:
        """Estimate time until critical threshold breach"""
        if slope == 0:
            return None
        
        thresholds = self.critical_thresholds.get(vital, {})
        
        if slope > 0:
            # Moving upward, check high threshold
            high = thresholds.get('high', float('inf'))
            if current < high:
                readings_to_breach = (high - current) / slope
                if readings_to_breach > 0 and readings_to_breach < 20:
                    return {
                        'threshold': 'HIGH',
                        'readings_until': int(readings_to_breach),
                        'estimated_time': f'{int(readings_to_breach * 15)} min'
                    }
        else:
            # Moving downward, check low threshold
            low = thresholds.get('low', float('-inf'))
            if current > low:
                readings_to_breach = (current - low) / abs(slope)
                if readings_to_breach > 0 and readings_to_breach < 20:
                    return {
                        'threshold': 'LOW',
                        'readings_until': int(readings_to_breach),
                        'estimated_time': f'{int(readings_to_breach * 15)} min'
                    }
        
        return None
    
    def _assess_prediction_risk(self, vital: str, predictions: List[Dict]) -> str:
        """Assess risk level based on predictions"""
        if not predictions:
            return 'UNKNOWN'
        
        thresholds = self.critical_thresholds.get(vital, {})
        low = thresholds.get('low', 0)
        high = thresholds.get('high', 200)
        
        # Check if any prediction breaches thresholds
        for pred in predictions:
            val = pred['predicted_value']
            if val < low or val > high:
                return 'HIGH'
            
            lower = pred['lower_bound']
            upper = pred['upper_bound']
            if lower < low or upper > high:
                return 'MODERATE'
        
        return 'LOW'
    
    def get_agent_output(self, buffer: PatientBuffer) -> Dict[str, Any]:
        """Get formatted agent output"""
        predictions = self.predict_trends(buffer)
        
        # Calculate overall risk
        risk_levels = {p['risk_level'] for p in predictions.values() if p['risk_level'] != 'UNKNOWN'}
        
        if 'HIGH' in risk_levels:
            overall_risk = 0.8
        elif 'MODERATE' in risk_levels:
            overall_risk = 0.5
        else:
            overall_risk = 0.2
        
        # Find breaches
        imminent_breaches = []
        for vital, pred in predictions.items():
            if pred.get('breach_estimate'):
                imminent_breaches.append({
                    'vital': vital,
                    **pred['breach_estimate']
                })
        
        # Sort by urgency
        imminent_breaches.sort(key=lambda x: x.get('readings_until', 100))
        
        return {
            "agent_name": "TrendPredictorAgent",
            "risk_contribution": overall_risk,
            "confidence": 0.7,
            "findings": {
                "predictions": {k: {
                    'trend': v['trend'],
                    '1h_forecast': v['predictions'][1]['predicted_value'] if len(v['predictions']) > 1 else None,
                    'risk_level': v['risk_level']
                } for k, v in predictions.items()},
                "imminent_breaches": imminent_breaches[:2]
            },
            "reasoning": self._generate_reasoning(predictions, imminent_breaches)
        }
    
    def _generate_reasoning(self, predictions: Dict, breaches: List) -> str:
        """Generate reasoning text"""
        if breaches:
            top = breaches[0]
            return f"⚠️ PREDICTIVE ALERT: {top['vital'].replace('_', ' ').title()} trending toward {top['threshold']} threshold in ~{top['estimated_time']}. Intervention recommended."
        
        # Look for concerning trends
        concerning = []
        for vital, pred in predictions.items():
            if pred['trend'] in ['INCREASING', 'DECREASING'] and pred['risk_level'] != 'LOW':
                concerning.append(f"{vital.replace('_', ' ')}: {pred['trend'].lower()}")
        
        if concerning:
            return f"Trends to monitor: {', '.join(concerning[:3])}. No immediate threshold breach predicted."
        
        return "Vital signs trending within safe parameters. No concerning projections."
