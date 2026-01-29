"""
Test WebSocket connection to backend
"""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/vitals"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Send subscribe message
            await websocket.send(json.dumps({"type": "subscribe"}))
            print("📤 Sent subscribe message")
            
            # Wait for messages
            print("⏳ Waiting for data...")
            for i in range(5):
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(message)
                print(f"\n📨 Message {i+1}:")
                if 'patient_id' in data:
                    print(f"  Patient: {data.get('patient_id')}")
                    print(f"  Risk Score: {data.get('risk_score', 'N/A')}")
                    print(f"  Risk Category: {data.get('risk_category', 'N/A')}")
                else:
                    print(f"  {data}")
                    
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for data")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
