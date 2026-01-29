"""
TriSense AI - Supabase Database Client
"""
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from .config import settings


class SupabaseClient:
    """Supabase database operations"""
    
    _client: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Optional[Client]:
        """Get or create Supabase client"""
        if cls._client is None:
            if settings.SUPABASE_URL.startswith("YOUR_"):
                print("⚠️ Supabase not configured - running in demo mode")
                return None
            try:
                cls._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            except Exception as e:
                print(f"⚠️ Failed to connect to Supabase: {e}")
                return None
        return cls._client
    
    @classmethod
    async def save_patient(cls, patient_data: Dict[str, Any]) -> Optional[Dict]:
        """Save or update patient record"""
        client = cls.get_client()
        if not client:
            return patient_data  # Demo mode
        
        try:
            result = client.table("patients").upsert(patient_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving patient: {e}")
            return None
    
    @classmethod
    async def save_vitals(cls, vitals_data: Dict[str, Any]) -> Optional[Dict]:
        """Save vital signs reading"""
        client = cls.get_client()
        if not client:
            return vitals_data  # Demo mode
        
        try:
            result = client.table("vitals").insert(vitals_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving vitals: {e}")
            return None
    
    @classmethod
    async def save_alert(cls, alert_data: Dict[str, Any]) -> Optional[Dict]:
        """Save alert record"""
        client = cls.get_client()
        if not client:
            return alert_data  # Demo mode
        
        try:
            result = client.table("alerts").insert(alert_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving alert: {e}")
            return None
    
    @classmethod
    async def get_patient(cls, patient_id: str) -> Optional[Dict]:
        """Get patient by ID"""
        client = cls.get_client()
        if not client:
            return None
        
        try:
            result = client.table("patients").select("*").eq("id", patient_id).single().execute()
            return result.data
        except Exception as e:
            print(f"Error getting patient: {e}")
            return None
    
    @classmethod
    async def get_recent_vitals(cls, patient_id: str, limit: int = 50) -> List[Dict]:
        """Get recent vital readings for patient"""
        client = cls.get_client()
        if not client:
            return []
        
        try:
            result = client.table("vitals")\
                .select("*")\
                .eq("patient_id", patient_id)\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting vitals: {e}")
            return []
    
    @classmethod
    async def get_active_alerts(cls, patient_id: str) -> List[Dict]:
        """Get unacknowledged alerts for patient"""
        client = cls.get_client()
        if not client:
            return []
        
        try:
            result = client.table("alerts")\
                .select("*")\
                .eq("patient_id", patient_id)\
                .eq("acknowledged", False)\
                .order("created_at", desc=True)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []


# SQL for creating tables in Supabase (run in SQL Editor)
SUPABASE_SCHEMA = """
-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    age INTEGER,
    gestational_age TEXT,
    birth_weight TEXT,
    ward TEXT,
    admission_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vitals table
CREATE TABLE IF NOT EXISTS vitals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT REFERENCES patients(patient_id),
    timestamp TIMESTAMPTZ NOT NULL,
    heart_rate REAL,
    systolic_bp REAL,
    diastolic_bp REAL,
    respiratory_rate REAL,
    spo2 REAL,
    temperature REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id TEXT REFERENCES patients(patient_id),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    risk_score REAL,
    reasoning TEXT,
    actions JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_vitals_patient_time ON vitals(patient_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id, acknowledged);

-- Enable realtime
ALTER PUBLICATION supabase_realtime ADD TABLE vitals;
ALTER PUBLICATION supabase_realtime ADD TABLE alerts;
"""
