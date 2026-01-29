"""
TriSense AI - Feature Engineering Pipeline
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from ..models.patient_buffer import PatientBuffer
from .patchtst_encoder import get_encoder


class FeatureEngineer:
    """
    Feature engineering pipeline for vital signs.
    Generates comprehensive features including:
    - PatchTST embeddings
    - Clinical scores (qSOFA, SIRS, shock index)
    - Statistical features (slope, variance, z-score)
    - Rate of change features
    """
    
    def __init__(self):
        self.encoder = get_encoder()
    
    def engineer_features(self, buffer: PatientBuffer) -> Dict[str, float]:
        """
        Generate all features from patient buffer.
        
        Args:
            buffer: PatientBuffer with time-series vitals
        
        Returns:
            Dict of feature name to value
        """
        features = {}
        
        # Get vitals window
        window = buffer.get_window()
        latest = buffer.get_latest()
        
        if not latest:
            return self._empty_features()
        
        # 1. PatchTST Embeddings (32 dimensions)
        embeddings = self.encoder.encode(window)
        for i, val in enumerate(embeddings):
            features[f'embed_{i}'] = float(val)
        
        # 2. Clinical Scores
        features['qsofa_score'] = self._calculate_qsofa(latest)
        features['sirs_score'] = self._calculate_sirs(latest)
        features['shock_index'] = self._calculate_shock_index(latest)
        
        # 3. Percentage Change Features
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) >= 2:
                pct_change = (arr[-1] - arr[0]) / arr[0] * 100 if arr[0] != 0 else 0
                features[f'{vital}_pct_change'] = float(pct_change)
            else:
                features[f'{vital}_pct_change'] = 0.0
        
        # 4. Slope Features (trend direction)
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) >= 2:
                x = np.arange(len(arr))
                slope = np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0
                features[f'{vital}_slope'] = float(slope)
            else:
                features[f'{vital}_slope'] = 0.0
        
        # 5. Variance Features
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) >= 2:
                features[f'{vital}_variance'] = float(np.var(arr))
            else:
                features[f'{vital}_variance'] = 0.0
        
        # 6. Z-Score Features (deviation from baseline)
        deviations = buffer.get_deviation_from_baseline()
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            features[f'{vital}_zscore'] = float(deviations.get(vital, 0.0))
        
        # 7. Min/Max Features
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            arr = window.get(vital, np.array([]))
            if len(arr) > 0:
                features[f'{vital}_min'] = float(np.min(arr))
                features[f'{vital}_max'] = float(np.max(arr))
            else:
                features[f'{vital}_min'] = 0.0
                features[f'{vital}_max'] = 0.0
        
        # 8. Latest Values
        for vital, value in latest.items():
            features[f'{vital}_latest'] = float(value)
        
        return features
    
    def _calculate_qsofa(self, vitals: Dict[str, float]) -> int:
        """Calculate quick SOFA score (0-3)"""
        score = 0
        
        # Respiratory rate >= 22
        if vitals.get('respiratory_rate', 0) >= 22:
            score += 1
        
        # Systolic BP <= 100
        if vitals.get('systolic_bp', 120) <= 100:
            score += 1
        
        # Altered mental status (assumed from low GCS - using HR as proxy)
        # In real system, GCS would come from separate data
        hr = vitals.get('heart_rate', 80)
        if hr > 120 or hr < 50:  # Proxy for altered status
            score += 1
        
        return score
    
    def _calculate_sirs(self, vitals: Dict[str, float]) -> int:
        """Calculate SIRS criteria score (0-4)"""
        score = 0
        
        # Heart rate > 90
        if vitals.get('heart_rate', 0) > 90:
            score += 1
        
        # Respiratory rate > 20
        if vitals.get('respiratory_rate', 0) > 20:
            score += 1
        
        # Temperature < 36 or > 38
        temp = vitals.get('temperature', 37.0)
        if temp < 36 or temp > 38:
            score += 1
        
        # WBC abnormal (assumed criterion - would come from labs)
        # Using SpO2 as proxy indicator
        if vitals.get('spo2', 98) < 92:
            score += 1
        
        return score
    
    def _calculate_shock_index(self, vitals: Dict[str, float]) -> float:
        """Calculate shock index (HR / Systolic BP)"""
        hr = vitals.get('heart_rate', 80)
        sys_bp = vitals.get('systolic_bp', 120)
        
        if sys_bp > 0:
            return hr / sys_bp
        return 0.0
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict with correct keys"""
        features = {}
        
        # Embeddings
        for i in range(32):
            features[f'embed_{i}'] = 0.0
        
        # Clinical scores
        features['qsofa_score'] = 0.0
        features['sirs_score'] = 0.0
        features['shock_index'] = 0.0
        
        # Per-vital features
        for vital in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate', 'spo2', 'temperature']:
            features[f'{vital}_pct_change'] = 0.0
            features[f'{vital}_slope'] = 0.0
            features[f'{vital}_variance'] = 0.0
            features[f'{vital}_zscore'] = 0.0
            features[f'{vital}_min'] = 0.0
            features[f'{vital}_max'] = 0.0
            features[f'{vital}_latest'] = 0.0
        
        return features


# Singleton instance
_engineer_instance = None


def get_engineer() -> FeatureEngineer:
    """Get or create feature engineer instance"""
    global _engineer_instance
    if _engineer_instance is None:
        _engineer_instance = FeatureEngineer()
    return _engineer_instance
