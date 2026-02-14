# 🩺 TriSense AI — Early Clinical Deterioration & Sepsis Prediction System

TriSense AI is an **AI-powered early warning system** that detects clinical deterioration before visible collapse occurs. It analyzes time-series vital sign data to generate **explainable risk alerts**, enabling clinicians to intervene early and improve patient outcomes.

Built for real-world healthcare environments, TriSense focuses on **trend-based detection**, not static threshold monitoring.

---

## 🚀 Overview

In hospital settings, patients often show subtle physiological changes hours before critical events such as:

- Sepsis
- Cardiac arrest
- Respiratory failure
- Septic shock

Traditional monitoring systems rely on fixed thresholds and snapshot readings, which may miss gradual deterioration.

### TriSense solves this by:

- Learning patient baseline patterns
- Monitoring vital trends continuously
- Detecting abnormal deviations early
- Providing explainable risk alerts
- Supporting clinical decision-making

---

## 🎯 Key Features

- 📈 **Time-Series Vital Analysis** — detects trends instead of static values
- ⚠️ **Early Deterioration Detection**
- 🧠 **Machine Learning Risk Prediction**
- 🔍 **Explainable Alerts**
- 📊 **Risk Scoring (0–100%)**
  - 🟢 Low (0–39%)
  - 🟡 Moderate (40–69%)
  - 🔴 High (70–100%)
- 🏥 Deployment-ready for resource-limited settings
- 🔒 Privacy-first healthcare AI

---

## 🧠 System Architecture

Patient Vitals → Feature Engineering → Time-Series Encoder → Risk Classifier → Explanation Engine → Alert System

yaml
Copy code

### Pipeline Flow

1. Vital signs collected
2. Features extracted (trend, change rate, clinical scores)
3. Time-series model learns patterns
4. Risk prediction generated
5. Explanation provided
6. Alert triggered if risk high

---

## ⚙️ Core Components

### 1. Data Generator (`data_generator.py`)
- Generates labeled clinical data
- Simulates patient vital patterns
- Uses clinical rules (qSOFA, SIRS, Shock Index)

### 2. Feature Engineering (`feature_engineering.py`)
Creates clinical features such as:
- Rate of change
- Vital slopes
- Percentage change
- Z-score normalization
- Shock index
- Trend deviations

### 3. Time-Series Encoder (`patchtst_encoder.py`)
- Transformer-based model
- Learns temporal patterns in vital signs
- Generates time-series embeddings

### 4. Risk Classifier (`xgboost_classifier.py`)
- Predicts deterioration probability
- Optimized for high recall
- Outputs risk percentage

### 5. Training Pipeline (`train_pipeline.py`)
- Data preprocessing
- Model training
- Performance evaluation
- Model saving

### 6. Inference Engine (`inference.py`)
Main prediction interface:

```python
{
  "risk_percentage": 78,
  "risk_category": "HIGH",
  "explanation": "Rising heart rate + falling blood pressure"
}
7. AI Agents (Decision Layer)
Alert Agent — triggers alerts

Pattern Agent — detects physiological patterns

Trend Agent — analyzes time-series changes

Reasoning Agent — explains predictions

Suggestion Agent — provides recommendations

8. Backend API (main.py)
Serves prediction endpoints

Connects ML pipeline

9. Frontend Dashboard
Displays patient risk

Shows alerts

Visualizes trends

📊 Input Data
TriSense uses physiological parameters:

Heart Rate

Blood Pressure

Oxygen Saturation (SpO2)

Respiratory Rate

Temperature

Possible data sources:

ICU monitors

Wearables

Electronic Health Records

Simulated clinical data

📦 Installation
bash
Copy code
git clone https://github.com/your-repo/trisense-ai
cd trisense-ai
pip install -r requirements.txt
▶️ Run Training
bash
Copy code
python train_pipeline.py
🔮 Run Prediction
python
Copy code
from inference import predict_deterioration_risk

result = predict_deterioration_risk(vitals_6hr)
print(result)
🧪 Run Demo
bash
Copy code
python demo.py
📈 Model Performance
The system prioritizes:

High recall for critical cases

Early detection capability

Clinical interpretability

Evaluation metrics include:

Precision

Recall

F1 Score

ROC-AUC

🏥 Use Cases
ICU monitoring

Emergency departments

Rural healthcare centers

Early sepsis detection

Continuous patient monitoring

Clinical decision support

🔐 Ethics & Safety
Clinical decision-support only

Does not replace medical professionals

Uses anonymized data

Explainable predictions

Privacy-first design

🌍 Impact
TriSense enables:

Earlier intervention

Reduced ICU admissions

Lower mortality risk

Reduced clinician workload

Proactive healthcare delivery

👨‍💻 Team
Flexiroasters — Healthcare AI Innovation Team
Built during Ekathon 2026 — Health AI on India’s Digital Rails.

📜 License
MIT License

yaml
Copy code

---

If you want next, I can also help you add:

✅ GitHub badges (build status, license, Python version)  
✅ architecture diagram image  
✅ demo screenshots section  
✅ API docs section  
✅ Docker deployment guide  
✅ “How it works” visual flow  
✅ research references  

Just tell me 👍
