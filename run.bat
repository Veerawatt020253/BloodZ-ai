@echo off
echo 🚀 Starting Blood Analyzer API on port 5050...
set MODEL_PATH=C:\BloodZ\model\best.pt
uvicorn main:app --host 0.0.0.0 --port 5050 --reload
