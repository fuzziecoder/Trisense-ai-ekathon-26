"""
TriSense AI - FastAPI Backend
Real-time healthcare monitoring with WebSocket support.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import asyncio
import json
from datetime import datetime

from .config import settings
from .models.patient_buffer import PatientBuffer
from .models.schemas import VitalReading, PatientInfo
from .agents.coordinator import get_coordinator
from .data.synthetic_generator import get_generator
from .database import SupabaseClient

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 TriSense AI Starting (Lifespan)...")
    try:
        patients = get_generator().get_all_patients()
        print(f"📊 Demo patients: {len(patients)}")
        asyncio.create_task(continuous_data_stream())
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    global is_streaming
    is_streaming = False

app = FastAPI(
    title="TriSense AI",
    description="Real-time clinical deterioration detection system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
patient_buffers: Dict[str, PatientBuffer] = {}
active_connections: List[WebSocket] = []
is_streaming = False


@app.websocket("/ws/vitals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"📱 Client connected. Total: {len(active_connections)}")
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") == "subscribe":
                await websocket.send_json({"type": "subscribed", "status": "ok"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"📱 Client disconnected. Total: {len(active_connections)}")


async def broadcast_update(data: dict):
    # print(f"Broadcasting update for {data.get('patient_id')} to {len(active_connections)} clients")
    message = json.dumps(data, default=str)
    for conn in active_connections:
        try:
            await conn.send_text(message)
        except Exception as e:
            print(f"⚠️ Broadcast error: {e}")
            pass


async def continuous_data_stream():
    global is_streaming
    is_streaming = True
    generator = get_generator()
    coordinator = get_coordinator()
    
    print("🔄 Starting continuous data stream...")
    
    while is_streaming:
        try:
            for patient in generator.get_all_patients():
                patient_id = patient["patient_id"]
                reading = generator.get_next_reading(patient_id)
                
                if reading:
                    if patient_id not in patient_buffers:
                        patient_buffers[patient_id] = PatientBuffer(patient_id)
                    
                    buffer = patient_buffers[patient_id]
                    buffer.add_vitals(reading["vitals"], datetime.utcnow())
                    
                    if buffer.size() >= 4:
                        update = coordinator.process_vitals(patient_id, buffer)
                        update_dict = update.model_dump()
                        update_dict["patient_info"] = patient
                        await broadcast_update(update_dict)
            
            await asyncio.sleep(2)  # Update every 2 seconds
        except Exception as e:
            print(f"❌ Error in data stream: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(2)


@app.get("/")
async def root():
    return {"message": "TriSense AI API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "patients": len(patient_buffers)}


@app.get("/api/patients")
async def get_patients():
    return get_generator().get_all_patients()


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    info = get_generator().get_patient_info(patient_id)
    if not info:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    buffer = patient_buffers.get(patient_id)
    return {
        "info": info,
        "vital_trends": buffer.get_vital_trends() if buffer else {},
        "risk_history": buffer.get_risk_history() if buffer else []
    }


@app.post("/api/vitals")
async def post_vitals(reading: VitalReading):
    patient_id = reading.patient_id
    
    if patient_id not in patient_buffers:
        patient_buffers[patient_id] = PatientBuffer(patient_id)
    
    buffer = patient_buffers[patient_id]
    buffer.add_vitals(reading.vitals.model_dump(), reading.timestamp)
    
    if buffer.size() >= 4:
        coordinator = get_coordinator()
        update = coordinator.process_vitals(patient_id, buffer)
        await broadcast_update(update.model_dump())
        return update.model_dump()
    
    return {"status": "learning_baseline", "readings": buffer.size()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
