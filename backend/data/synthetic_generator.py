"""
TriSense AI - Synthetic Patient Data Generator
Generates realistic time-series vital data for demo.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random


class SyntheticDataGenerator:
    """Generates realistic synthetic patient vital sign data."""
    
    def __init__(self):
        self.patients = self._create_demo_patients()
        self.current_indices = {p['patient_id']: 0 for p in self.patients}
        self.patient_data = {p['patient_id']: self._generate_patient_stream(p) 
                            for p in self.patients}
    
    def _create_demo_patients(self) -> List[Dict]:
        return [
            {"patient_id": "PAT-001", "name": "Marisse Meeus", 
             "age_days": 95, "gestational_age": "34 Weeks, 6 Days",
             "birth_weight": "750g", "ward": "NICU-3", "pattern": "stable"},
            {"patient_id": "PAT-002", "name": "James Chen",
             "age_days": 42, "gestational_age": "32 Weeks, 2 Days",
             "birth_weight": "1200g", "ward": "NICU-2", "pattern": "deteriorating"},
            {"patient_id": "PAT-003", "name": "Sarah Johnson",
             "age_days": 28, "gestational_age": "36 Weeks, 0 Days",
             "birth_weight": "2100g", "ward": "NICU-1", "pattern": "sepsis"},
        ]
    
    def _generate_patient_stream(self, patient: Dict) -> List[Dict]:
        pattern = patient.get("pattern", "stable")
        readings = []
        
        if pattern == "stable":
            readings = self._generate_stable_stream(200)
        elif pattern == "deteriorating":
            readings = self._generate_deteriorating_stream(200)
        elif pattern == "sepsis":
            readings = self._generate_sepsis_stream(200)
        
        return readings
    
    def _generate_stable_stream(self, count: int) -> List[Dict]:
        baselines = {"hr": 75, "sys": 120, "dia": 75, "rr": 16, "spo2": 98, "temp": 36.8}
        readings = []
        for i in range(count):
            readings.append({
                "heart_rate": baselines["hr"] + np.random.normal(0, 3),
                "systolic_bp": baselines["sys"] + np.random.normal(0, 5),
                "diastolic_bp": baselines["dia"] + np.random.normal(0, 3),
                "respiratory_rate": baselines["rr"] + np.random.normal(0, 1.5),
                "spo2": min(100, baselines["spo2"] + np.random.normal(0, 1)),
                "temperature": baselines["temp"] + np.random.normal(0, 0.2)
            })
        return readings
    
    def _generate_deteriorating_stream(self, count: int) -> List[Dict]:
        readings = []
        for i in range(count):
            progress = i / count
            hr = 75 + progress * 35 + np.random.normal(0, 4)
            sbp = 120 - progress * 25 + np.random.normal(0, 5)
            readings.append({
                "heart_rate": hr,
                "systolic_bp": max(70, sbp),
                "diastolic_bp": max(50, 75 - progress * 15 + np.random.normal(0, 3)),
                "respiratory_rate": 16 + progress * 10 + np.random.normal(0, 2),
                "spo2": min(100, max(85, 98 - progress * 8 + np.random.normal(0, 1.5))),
                "temperature": 36.8 + progress * 1.5 + np.random.normal(0, 0.3)
            })
        return readings
    
    def _generate_sepsis_stream(self, count: int) -> List[Dict]:
        readings = []
        sepsis_start = count // 3
        for i in range(count):
            if i < sepsis_start:
                readings.append(self._generate_stable_stream(1)[0])
            else:
                progress = (i - sepsis_start) / (count - sepsis_start)
                hr = 80 + progress * 50 + np.random.normal(0, 5)
                sbp = 115 - progress * 35 + np.random.normal(0, 6)
                readings.append({
                    "heart_rate": min(150, hr),
                    "systolic_bp": max(75, sbp),
                    "diastolic_bp": max(45, 70 - progress * 20 + np.random.normal(0, 4)),
                    "respiratory_rate": min(35, 18 + progress * 15 + np.random.normal(0, 2)),
                    "spo2": min(100, max(82, 97 - progress * 12 + np.random.normal(0, 2))),
                    "temperature": 37.0 + progress * 2.5 + np.random.normal(0, 0.3)
                })
        return readings
    
    def get_next_reading(self, patient_id: str) -> Optional[Dict]:
        if patient_id not in self.patient_data:
            return None
        
        data = self.patient_data[patient_id]
        idx = self.current_indices[patient_id]
        
        if idx >= len(data):
            idx = 0  # Loop
        
        reading = data[idx].copy()
        self.current_indices[patient_id] = idx + 1
        
        return {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow(),
            "vitals": reading
        }
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict]:
        for p in self.patients:
            if p["patient_id"] == patient_id:
                return p
        return None
    
    def get_all_patients(self) -> List[Dict]:
        return self.patients


# Singleton
_generator = None

def get_generator() -> SyntheticDataGenerator:
    global _generator
    if _generator is None:
        _generator = SyntheticDataGenerator()
    return _generator
