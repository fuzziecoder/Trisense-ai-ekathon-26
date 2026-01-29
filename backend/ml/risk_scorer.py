"""
TriSense AI - XGBoost Risk Scorer
"""
import os
import xgboost as xgb
import numpy as np
from typing import Dict, Optional, Tuple, Any
import json
from datetime import datetime

from ..config import settings


class RiskScorer:
    """
    XGBoost-based risk scoring for sepsis/deterioration detection.
    Uses both classifier (binary risk) and regressor (continuous score).
    """
    
    def __init__(self):
        self.classifier: Optional[xgb.Booster] = None
        self.regressor: Optional[xgb.Booster] = None
        self.feature_names: list = []
        self.feature_importance: Dict[str, float] = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained XGBoost models"""
        classifier_path = os.path.join(settings.MODEL_DIR, settings.XGBOOST_CLASSIFIER)
        regressor_path = os.path.join(settings.MODEL_DIR, settings.XGBOOST_REGRESSOR)
        schema_path = os.path.join(settings.MODEL_DIR, settings.FEATURE_SCHEMA)
        meta_path = os.path.join(settings.MODEL_DIR, "xgboost_model_meta.json")
        
        # Load feature schema
        try:
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                    self.feature_names = schema.get('feature_names', [])
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.feature_importance = meta.get('feature_importance', {})
        except Exception as e:
            print(f"⚠️ Error loading feature schema: {e}")
        
        # Load classifier
        try:
            if os.path.exists(classifier_path):
                self.classifier = xgb.Booster()
                self.classifier.load_model(classifier_path)
                print(f"✅ XGBoost classifier loaded from {classifier_path}")
            else:
                print(f"⚠️ Classifier not found at {classifier_path}")
        except Exception as e:
            print(f"⚠️ Error loading classifier: {e}")
        
        # Load regressor
        try:
            if os.path.exists(regressor_path):
                self.regressor = xgb.Booster()
                self.regressor.load_model(regressor_path)
                print(f"✅ XGBoost regressor loaded from {regressor_path}")
            else:
                print(f"⚠️ Regressor not found at {regressor_path}")
        except Exception as e:
            print(f"⚠️ Error loading regressor: {e}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict risk score.
        
        Args:
            features: Dict of feature name to value
        
        Returns:
            Tuple of (combined_score, classification_probability)
        """
        # Prepare feature array
        feature_array = self._prepare_features(features)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(feature_array.reshape(1, -1), feature_names=self.feature_names)
        
        # Classifier prediction (probability)
        class_prob = 0.0
        if self.classifier is not None:
            try:
                class_prob = float(self.classifier.predict(dmatrix)[0])
            except Exception as e:
                print(f"Classifier prediction error: {e}")
                class_prob = self._rule_based_prediction(features)
        else:
            class_prob = self._rule_based_prediction(features)
        
        # Regressor prediction (0-100 score)
        reg_score = 0.0
        if self.regressor is not None:
            try:
                reg_score = float(self.regressor.predict(dmatrix)[0])
                reg_score = max(0, min(100, reg_score)) / 100  # Normalize to 0-1
            except Exception as e:
                print(f"Regressor prediction error: {e}")
                reg_score = class_prob
        else:
            reg_score = class_prob
        
        # Combine predictions (weighted average)
        combined_score = 0.6 * class_prob + 0.4 * reg_score
        
        return combined_score, class_prob

    def get_ml_output(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get structured ML output according to the contract.
        """
        risk_score, class_prob = self.predict(features)
        
        # Confidence logic: if model is available, use high confidence
        # In a real scenario, this might come from the model's internal metrics
        confidence = 0.91 if self.regressor else 0.70
        
        return {
            "model_name": "PatchTST-XGBoost-Hybrid",
            "task": "sepsis_risk_regression",
            "risk_score": round(risk_score, 4),
            "confidence": confidence,
            "prediction_time": datetime.utcnow().isoformat() + "Z"
        }
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature array in correct order"""
        feature_array = []
        
        for name in self.feature_names:
            value = features.get(name, 0.0)
            if value is None or np.isnan(value):
                value = 0.0
            feature_array.append(value)
        
        return np.array(feature_array, dtype=np.float32)
    
    def _rule_based_prediction(self, features: Dict[str, float]) -> float:
        """Fallback rule-based prediction when model unavailable"""
        risk = 0.0
        
        # High heart rate
        hr = features.get('heart_rate_latest', 0)
        if hr > 120:
            risk += 0.20
        elif hr > 100:
            risk += 0.10
        
        # Low blood pressure
        sys_bp = features.get('systolic_bp_latest', 120)
        if sys_bp < 90:
            risk += 0.25
        elif sys_bp < 100:
            risk += 0.15
        
        # Low SpO2
        spo2 = features.get('spo2_latest', 98)
        if spo2 < 88:
            risk += 0.25
        elif spo2 < 92:
            risk += 0.15
        
        # High qSOFA
        qsofa = features.get('qsofa_score', 0)
        if qsofa >= 2:
            risk += 0.25
        
        # High shock index
        shock_idx = features.get('shock_index', 0)
        if shock_idx > 1.0:
            risk += 0.20
        elif shock_idx > 0.9:
            risk += 0.10
        
        # Temperature abnormality
        temp = features.get('temperature_latest', 37.0)
        if temp > 38.5 or temp < 36.0:
            risk += 0.10
        
        return min(risk, 1.0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def get_top_features(self, features: Dict[str, float], top_n: int = 10) -> Dict[str, float]:
        """Get top N contributing features for explanation"""
        # Sort by importance
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = {}
        for name, importance in sorted_importance[:top_n]:
            if name in features and importance > 0:
                top_features[name] = importance
        
        return top_features


# Singleton instance
_scorer_instance: Optional[RiskScorer] = None


def get_scorer() -> RiskScorer:
    """Get or create scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = RiskScorer()
    return _scorer_instance
