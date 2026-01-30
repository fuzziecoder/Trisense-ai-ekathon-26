
try:
    import uvicorn
    from backend.main import app
    print("App imported successfully")
    uvicorn.run(app, host="127.0.0.1", port=8001)
except Exception as e:
    import traceback
    traceback.print_exc()
