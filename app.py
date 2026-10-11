"""
Model Mind - Decision Tree Studio Launcher
Starts the FastAPI Backend & Serves the Interactive Decision Tree Frontend.
Usage: python3 app.py
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    print("\n" + "="*60)
    print("🌲 Model Mind — Decision Tree Visualization Studio")
    print("🚀 Starting FastAPI Backend & Frontend Server...")
    print("📍 URL: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
