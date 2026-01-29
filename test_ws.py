
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/vitals"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Send subscribe (as frontend does)
            await websocket.send(json.dumps({"type": "subscribe"}))
            print("Sent subscribe")
            
            count = 0
            while count < 3:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                    data = json.loads(message)
                    print(f"Msg {count+1}: Type={data.get('type')}")
                    if data.get('type') == 'vitals_update':
                        print(f"Received data for patient: {data.get('patient_id')}")
                        print(f"Risk Score: {data.get('risk_score')}")
                        count += 1
                except asyncio.TimeoutError:
                    print("Timeout waiting for data (5s)")
                    break
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
