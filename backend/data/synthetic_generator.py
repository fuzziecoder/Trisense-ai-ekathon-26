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
             "patient_age": "45 Years", "admitted_ward": "General Ward A",
             "nurse_id": "NUR-75", "ward": "General-A", "pattern": "stable"},
            {"patient_id": "PAT-002", "name": "James Chen",
             "patient_age": "28 Years", "admitted_ward": "ICU-3",
             "nurse_id": "NUR-42", "ward": "ICU", "pattern": "deteriorating"},
            {"patient_id": "PAT-003", "name": "Sarah Johnson",
             "patient_age": "62 Years", "admitted_ward": "Emergency B",
             "nurse_id": "NUR-12", "ward": "Emergency", "pattern": "sepsis"},
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
        # Force immediate critical sepsis state
        # High HR (>180), Low BP (<90), Fever (>38.5), Low SpO2 (<90)
        for i in range(count):
            readings.append({
                "heart_rate": 185 + np.random.normal(0, 5),
                "systolic_bp": 85 + np.random.normal(0, 4),
                "diastolic_bp": 45 + np.random.normal(0, 3),
                "respiratory_rate": 35 + np.random.normal(0, 2),
                "spo2": 88 + np.random.normal(0, 2),
                "temperature": 39.5 + np.random.normal(0, 0.2)
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

if __name__ == "__main__":
    print("Test run of SyntheticDataGenerator...")
    gen = SyntheticDataGenerator()
    patients = gen.get_all_patients()
    print(f"Loaded {len(patients)} patients:")
    for p in patients:
        print(f" - {p['name']} ({p['pattern']})")
    
    print("\nGenerating sample stream for PAT-002 (Deteriorating)...")
    for _ in range(5):
        data = gen.get_next_reading("PAT-002")
        print(f"Time: {data['timestamp'].strftime('%H:%M:%S')} | HR: {data['vitals']['heart_rate']:.1f} | SpO2: {data['vitals']['spo2']:.1f}")
