"""
TriSense AI - Configuration
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # App settings
    APP_NAME: str = "TriSense AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Supabase settings (UPDATE THESE WITH YOUR CREDENTIALS)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_ANON_KEY")
    
    # NVIDIA API settings (for Qwen model)
    NVIDIA_API_KEY: Optional[str] = "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"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "qwen/qwen3-next-80b-a3b-thinking"
    
    # OpenAI settings (Fallback)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    OPENAI_MODEL: str = "gpt-4o"
    
    # ML Model paths
    MODEL_DIR: str = r"c:\Users\Tharun\OneDrive\Desktop\fa\eka_care_models"
    PATCHTST_MODEL: str = "patchtst_encoder.pt"
    XGBOOST_CLASSIFIER: str = "xgboost_model.json"
    XGBOOST_REGRESSOR: str = "xgboost_regressor.json"
    FEATURE_SCHEMA: str = "feature_schema.json"
    
    # Monitoring settings
    VITAL_WINDOW_SIZE: int = 6  # Number of readings for time window
    BUFFER_MAX_SIZE: int = 100  # Max readings to keep per patient
    ALERT_COOLDOWN_SECONDS: int = 300  # 5 min cooldown between same alerts
    
    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 0.30
    RISK_MODERATE_THRESHOLD: float = 0.60
    RISK_HIGH_THRESHOLD: float = 0.80
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
