@echo off
REM Start HydroAI Backend
cd /d "D:\SIH2025\groundwater-backend\backend"
echo Starting HydroAI Backend...
echo.
python -m uvicorn app:app --reload --port 8000
pause
