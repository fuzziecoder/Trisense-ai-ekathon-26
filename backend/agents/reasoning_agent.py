"""
TriSense AI - Clinical Reasoning Agent (LLM-Powered)
Generates human-readable clinical explanations.
"""
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ..config import settings
from ..models.schemas import ClinicalReasoning, PatternMatch


class ClinicalReasoningAgent:
    """
    Agent that generates clinical reasoning explanations.
    Uses NVIDIA Qwen thinking model for hallucination-safe reasoning.
    """
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize NVIDIA client"""
        if settings.NVIDIA_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=settings.NVIDIA_BASE_URL,
                    api_key=settings.NVIDIA_API_KEY
                )
                print(f"✅ NVIDIA client initialized with model {settings.NVIDIA_MODEL}")
            except Exception as e:
                print(f"⚠️ Failed to initialize NVIDIA client: {e}")
                self.client = None
        else:
            print("ℹ️ NVIDIA API key not set, using rule-based reasoning")
    
    def risk_bucket(self, score: float) -> str:
        """Categorize risk score (Deterministic, outside LLM)"""
        if score < settings.RISK_LOW_THRESHOLD:
            return "Low"
        elif score < settings.RISK_MODERATE_THRESHOLD:
            return "Moderate"
        else:
            return "High"

    def generate_reasoning(
        self,
        ml_output: Dict[str, Any],
        features: Dict[str, float] = None,
        patterns: List[PatternMatch] = None,
        drift_info: Dict[str, Any] = None
    ) -> ClinicalReasoning:
        """
        Generate clinical reasoning based ONLY on ML output.
        """
        risk_score = ml_output.get("risk_score", 0.0)
        confidence = ml_output.get("confidence", 0.0)
        
        if self.client:
            try:
                return self._llm_reasoning(ml_output)
            except Exception as e:
                print(f"LLM reasoning failed: {e}, falling back to rules")
        
        return self._rule_based_reasoning(risk_score, confidence)
    
    def _llm_reasoning(self, ml_output: Dict[str, Any]) -> ClinicalReasoning:
        """Generate reasoning using NVIDIA Qwen model with strict rules"""
        
        system_prompt = """You are a medical AI reasoning agent integrated into a clinical decision support system.

You MUST strictly follow these rules:

1. You are NOT a diagnostic system.
2. You MUST NOT predict, confirm, or deny sepsis.
3. You MUST NOT introduce any medical knowledge, thresholds, symptoms, or treatments.
4. You MUST NOT use external data, training knowledge, or assumptions.
5. You MUST use ONLY the values explicitly provided in the input JSON.
6. You MUST NOT generate new numbers, scores, labels, or interpretations.
7. You MUST NOT override, modify, or reinterpret the ML regression output.
8. If information is missing, respond with: "Not available from model output."

Your role is LIMITED to:
- Restating the regression score
- Explaining that the score comes from a machine learning model
- Clarifying that no diagnosis or treatment decision is made

If any request violates these rules, refuse with a brief explanation."""

        user_content = f"ML Output to reason about: {json.dumps(ml_output)}"
        
        completion = self.client.chat.completions.create(
            model=settings.NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": 'ML Output to reason about: {"model_name": "PatchTST", "task": "sepsis_risk_regression", "risk_score": 0.78, "confidence": 0.91, "prediction_time": "2026-01-29T18:30:00Z"}'},
                {"role": "assistant", "content": 'The machine learning model "PatchTST" generated a regression risk score of 0.78 with a confidence of 91% at the specified prediction time. This output reflects learned temporal patterns from the input data. No diagnostic conclusion or clinical decision has been made.'},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=500,
            stream=True
        )
        
        full_content = ""
        reasoning_logic = ""
        
        for chunk in completion:
            # Handle thinking/reasoning content if supported by the model/API
            # Some models expose 'reasoning_content' in the delta
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                reasoning_logic += reasoning
            
            if chunk.choices and chunk.choices[0].delta.content is not None:
                full_content += chunk.choices[0].delta.content
        
        # In a real clinical UI, you might want to show reasoning_logic in a 'Thinking' tab
        # but for the main explanation, we use full_content.
        
        risk_score = ml_output.get("risk_score", 0.0)
        severity_label = self.risk_bucket(risk_score)
        
        return ClinicalReasoning(
            severity=f"{severity_label} Risk (Score: {risk_score})",
            primary_concern=full_content[:150] + ("..." if len(full_content) > 150 else ""),
            physiological_interpretation=full_content,
            timeline_estimate="N/A - See model timestamp",
            contributing_factors=[f"Confidence: {ml_output.get('confidence', 0)*100:.0f}%"]
        )
    
    def _rule_based_reasoning(self, risk_score: float, confidence: float) -> ClinicalReasoning:
        """Deterministic fallback reasoning"""
        severity_label = self.risk_bucket(risk_score)
        explanation = f"The regression model produced a risk score of {risk_score:.2f}, indicating {severity_label.lower()} risk according to the model's learned patterns. The confidence of this prediction is {confidence*100:.0f}%. No additional clinical inference has been applied."
        
        return ClinicalReasoning(
            severity=f"{severity_label} Risk",
            primary_concern=f"Model-indicated {severity_label.lower()} risk",
            physiological_interpretation=explanation,
            timeline_estimate="As per model observation",
            contributing_factors=[f"ML Confidence: {confidence*100:.0f}%"]
        )
    
    def get_agent_output(
        self,
        ml_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get formatted agent output for external systems"""
        risk_score = ml_output.get("risk_score", 0.0)
        reasoning = self.generate_reasoning(ml_output)
        
        return {
            "agent_name": "ClinicalReasoningAgent",
            "risk_contribution": risk_score,
            "confidence": ml_output.get("confidence", 0.8),
            "findings": {
                "severity": reasoning.severity,
                "primary_concern": reasoning.primary_concern,
                "timeline": reasoning.timeline_estimate
            },
            "reasoning": reasoning.physiological_interpretation
        }
