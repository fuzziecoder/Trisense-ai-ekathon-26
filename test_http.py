
import urllib.request
import json

try:
    print("Testing HTTP health check...")
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as response:
        print(f"Status: {response.getcode()}")
        print(f"Response: {response.read().decode()}")
except Exception as e:
    print(f"Failed: {e}")
