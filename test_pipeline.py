
import sys
import os
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from backend.data.synthetic_generator import get_generator
from backend.models.patient_buffer import PatientBuffer
from backend.agents.coordinator import get_coordinator

async def test():
    print("Testing pipeline...")
    generator = get_generator()
    coordinator = get_coordinator()
    
    patients = generator.get_all_patients()
    p = patients[0]
    pid = p['patient_id']
    buffer = PatientBuffer(pid)
    
    print(f"Feeding data for {pid}...")
    
    for i in range(10):
        reading = generator.get_next_reading(pid)
        print(f"Reading {i}: {reading['vitals']['heart_rate']:.1f} BPM")
        buffer.add_vitals(reading['vitals'], datetime.utcnow())
        
        if buffer.size() >= 4:
            print("Processing vitals...")
            try:
                update = coordinator.process_vitals(pid, buffer)
                print("Success!")
                print("Risk Score:", update.risk_score)
                print("ML Output:", update.ml_output)
            except Exception as e:
                with open("error.log", "w") as f:
                    f.write(str(e) + "\n")
                    import traceback
                    traceback.print_exc(file=f)
                print("CRASHED. See error.log")
                break

if __name__ == "__main__":
    asyncio.run(test())
