"""
Simple script to start the FastAPI server for testing.
"""

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("Starting Ceramic Armor ML API server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )