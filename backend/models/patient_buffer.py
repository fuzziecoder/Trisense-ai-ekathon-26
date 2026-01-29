"""
TriSense AI - Patient Time-Series Buffer
"""
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..config import settings


class PatientBuffer:
    """
    Rolling window buffer for patient vital signs.
    Stores time-series data for real-time analysis.
    """
    
    def __init__(self, patient_id: str, max_size: int = None):
        self.patient_id = patient_id
        self.max_size = max_size or settings.BUFFER_MAX_SIZE
        
        # Vital sign buffers
        self.timestamps: deque = deque(maxlen=self.max_size)
        self.heart_rate: deque = deque(maxlen=self.max_size)
        self.systolic_bp: deque = deque(maxlen=self.max_size)
        self.diastolic_bp: deque = deque(maxlen=self.max_size)
        self.respiratory_rate: deque = deque(maxlen=self.max_size)
        self.spo2: deque = deque(maxlen=self.max_size)
        self.temperature: deque = deque(maxlen=self.max_size)
        
        # Risk score history
        self.risk_scores: deque = deque(maxlen=self.max_size)
        self.risk_timestamps: deque = deque(maxlen=self.max_size)
        
        # Baseline (learned from first 4-6 hours)
        self.baseline: Optional[Dict[str, float]] = None
        self.baseline_std: Optional[Dict[str, float]] = None
        self.baseline_established: bool = False
        
        # Metadata
        self.last_update: Optional[datetime] = None
        self.created_at: datetime = datetime.utcnow()
    
    def add_vitals(self, vitals: Dict[str, float], timestamp: datetime = None):
        """Add new vital signs reading to buffer"""
        timestamp = timestamp or datetime.utcnow()
        
        self.timestamps.append(timestamp)
        self.heart_rate.append(vitals.get('heart_rate', 0))
        self.systolic_bp.append(vitals.get('systolic_bp', 0))
        self.diastolic_bp.append(vitals.get('diastolic_bp', 0))
        self.respiratory_rate.append(vitals.get('respiratory_rate', 0))
        self.spo2.append(vitals.get('spo2', 0))
        self.temperature.append(vitals.get('temperature', 0))
        
        self.last_update = timestamp
        
        # Establish baseline after collecting enough data
        if not self.baseline_established and self.size() >= 20:
            self._establish_baseline()
    
    def add_risk_score(self, score: float, timestamp: datetime = None):
        """Add risk score to history"""
        timestamp = timestamp or datetime.utcnow()
        self.risk_scores.append(score)
        self.risk_timestamps.append(timestamp)
    
    def size(self) -> int:
        """Number of readings in buffer"""
        return len(self.timestamps)
    
    def get_series(self, vital_name: str) -> pd.Series:
        """Get time series for a specific vital"""
        data_map = {
            'heart_rate': self.heart_rate,
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'respiratory_rate': self.respiratory_rate,
            'spo2': self.spo2,
            'temperature': self.temperature
        }
        
        data = list(data_map.get(vital_name, []))
        timestamps = list(self.timestamps)
        
        if not data:
            return pd.Series(dtype=float)
        
        return pd.Series(data, index=pd.to_datetime(timestamps))
    
    def get_window(self, window_size: int = None) -> Dict[str, np.ndarray]:
        """Get recent window of vitals as numpy arrays"""
        window_size = window_size or settings.VITAL_WINDOW_SIZE
        
        return {
            'heart_rate': np.array(list(self.heart_rate)[-window_size:]),
            'systolic_bp': np.array(list(self.systolic_bp)[-window_size:]),
            'diastolic_bp': np.array(list(self.diastolic_bp)[-window_size:]),
            'respiratory_rate': np.array(list(self.respiratory_rate)[-window_size:]),
            'spo2': np.array(list(self.spo2)[-window_size:]),
            'temperature': np.array(list(self.temperature)[-window_size:])
        }
    
    def get_latest(self) -> Dict[str, float]:
        """Get most recent vital readings"""
        if self.size() == 0:
            return {}
        
        return {
            'heart_rate': self.heart_rate[-1],
            'systolic_bp': self.systolic_bp[-1],
            'diastolic_bp': self.diastolic_bp[-1],
            'respiratory_rate': self.respiratory_rate[-1],
            'spo2': self.spo2[-1],
            'temperature': self.temperature[-1]
        }
    
    def get_risk_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get risk score history"""
        scores = list(self.risk_scores)[-limit:]
        timestamps = list(self.risk_timestamps)[-limit:]
        
        return [
            {'timestamp': ts.isoformat(), 'risk_score': score}
            for ts, score in zip(timestamps, scores)
        ]
    
    def get_vital_trends(self, limit: int = 50) -> Dict[str, List[float]]:
        """Get vital trends for charting"""
        return {
            'heart_rate': list(self.heart_rate)[-limit:],
            'systolic_bp': list(self.systolic_bp)[-limit:],
            'diastolic_bp': list(self.diastolic_bp)[-limit:],
            'respiratory_rate': list(self.respiratory_rate)[-limit:],
            'spo2': list(self.spo2)[-limit:],
            'temperature': list(self.temperature)[-limit:],
            'timestamps': [ts.isoformat() for ts in list(self.timestamps)[-limit:]]
        }
    
    def _establish_baseline(self):
        """Establish patient baseline from initial readings"""
        if self.size() < 10:
            return
        
        # Use first portion of data as baseline
        baseline_data = {
            'heart_rate': list(self.heart_rate)[:20],
            'systolic_bp': list(self.systolic_bp)[:20],
            'diastolic_bp': list(self.diastolic_bp)[:20],
            'respiratory_rate': list(self.respiratory_rate)[:20],
            'spo2': list(self.spo2)[:20],
            'temperature': list(self.temperature)[:20]
        }
        
        self.baseline = {k: np.mean(v) for k, v in baseline_data.items()}
        self.baseline_std = {k: np.std(v) if np.std(v) > 0 else 1.0 for k, v in baseline_data.items()}
        self.baseline_established = True
    
    def has_baseline(self) -> bool:
        """Check if baseline has been established"""
        return self.baseline_established
    
    def get_baseline(self) -> Optional[Dict[str, float]]:
        """Get established baseline values"""
        return self.baseline
    
    def get_deviation_from_baseline(self) -> Dict[str, float]:
        """Get Z-score deviation from baseline for each vital"""
        if not self.has_baseline() or self.size() == 0:
            return {}
        
        latest = self.get_latest()
        deviations = {}
        
        for vital, value in latest.items():
            baseline_val = self.baseline.get(vital, value)
            baseline_std = self.baseline_std.get(vital, 1.0)
            deviations[vital] = (value - baseline_val) / baseline_std
        
        return deviations
    
    def is_stale(self, threshold_minutes: int = 30) -> bool:
        """Check if data is stale (no recent updates)"""
        if self.last_update is None:
            return True
        
        age = datetime.utcnow() - self.last_update
        return age > timedelta(minutes=threshold_minutes)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to pandas DataFrame"""
        return pd.DataFrame({
            'timestamp': list(self.timestamps),
            'heart_rate': list(self.heart_rate),
            'systolic_bp': list(self.systolic_bp),
            'diastolic_bp': list(self.diastolic_bp),
            'respiratory_rate': list(self.respiratory_rate),
            'spo2': list(self.spo2),
            'temperature': list(self.temperature)
        })
