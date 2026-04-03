from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import joblib
import numpy as np
import json
import os
import sys
import logging
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
from fastapi import Request
from starlette.responses import JSONResponse
from time import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hydroai")

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth_routes import router as auth_router
from forecast_routes import router as forecast_router
from policy_routes import router as policy_router
from optimizer_routes import router as optimizer_router
from validation_routes import router as validation_router
from rainfall_routes import router as rainfall_router
from alerts_routes import router as alerts_router
from drivers_routes import router as drivers_router
from location_routes import router as location_router
from database import Database, create_indexes

load_dotenv()

app = FastAPI(title="Groundwater Prototype API", version="2.0")

# Simple in-memory rate limiter (per-IP sliding window)
# Increase RATE_LIMIT_MAX_PER_WINDOW in .env if you hit 429 during development
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX_PER_WINDOW", "120"))
_rate_store = defaultdict(list)


REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
# Set to "true" in production (HTTPS only); leave unset or "false" in local dev
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"


@app.middleware("http")
async def cookie_to_bearer(request: Request, call_next):
    """
    If the request has no Authorization header but carries an access_token cookie,
    inject it as a Bearer token so all existing route handlers work unchanged.
    """
    has_auth_header = any(k == b"authorization" for k, _ in request.scope.get("headers", []))
    if not has_auth_header and "access_token" in request.cookies:
        token = request.cookies["access_token"]
        new_header = (b"authorization", f"Bearer {token}".encode())
        request.scope["headers"] = list(request.scope.get("headers", [])) + [new_header]
    return await call_next(request)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Request timed out: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=504, content={"detail": "Request timed out"})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time()
    response = await call_next(request)
    duration_ms = (time() - start) * 1000
    logger.info(
        "%s %s %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.middleware("http")
async def simple_rate_limiter(request: Request, call_next):
    # Allow internal health checks without limiting
    ip = request.client.host if request.client else "unknown"
    now = time()
    window = RATE_LIMIT_WINDOW
    arr = _rate_store[ip]
    # drop old timestamps
    while arr and arr[0] <= now - window:
        arr.pop(0)
    if len(arr) >= RATE_LIMIT_MAX:
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})
    arr.append(now)
    _rate_store[ip] = arr
    response = await call_next(request)
    return response

# --- Enable CORS for frontend ---
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize database connection"""
    try:
        Database.connect_db()
        create_indexes()
        print("[OK] Database initialized successfully")
    except Exception as e:
        print(f"[ERROR] Database initialization error: {e}")

# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection"""
    Database.close_db()

# --- Include Auth Routes ---
app.include_router(auth_router)
app.include_router(forecast_router)
app.include_router(policy_router)
app.include_router(optimizer_router)
app.include_router(validation_router)
app.include_router(rainfall_router)
app.include_router(alerts_router)
app.include_router(drivers_router)
app.include_router(location_router)

# --- Helper function ---
def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# --- Load Data (using correct paths from backend directory) ---
backend_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(backend_dir, "..", "data")

rainfall = pd.read_csv(os.path.join(data_dir, "rainfall.csv"))
rainfall["state_name"] = _clean(rainfall["state_name"])
rainfall["year_month"] = rainfall["year_month"].astype(str)

groundwater = pd.read_csv(os.path.join(data_dir, "groundwater.csv"))
groundwater["state_name"] = _clean(groundwater["state_name"])
groundwater["district_name"] = _clean(groundwater["district_name"])
groundwater["year_month"] = groundwater["year_month"].astype(str)

# --- Load trained model ---
model = joblib.load(os.path.join(backend_dir, "..", "models", "groundwater_predictor.pkl"))


# --- API Routes ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Groundwater Backend API is running"}

# --- Authentication ---
# Legacy fuzzy-login removed. Use the secure `/api/auth` endpoints for registration and login.

# --- Data Endpoints ---
@app.get("/api/months")
def get_months():
    return sorted(rainfall["year_month"].unique().tolist())

@app.get("/api/states")
def get_states():
    return sorted(rainfall["state_name"].unique().tolist())

@app.get("/api/districts")
def get_districts(state: str = Query(...)):
    df = groundwater[groundwater["state_name"] == state]
    return sorted(df["district_name"].dropna().unique().tolist())

@app.get("/api/timeseries/state")
def state_timeseries(state: str = Query(...)):
    r = rainfall[rainfall["state_name"] == state]
    g = groundwater[groundwater["state_name"] == state].groupby(
        ["state_name", "year_month"], as_index=False
    )["gw_level_m_bgl"].mean()
    merged = pd.merge(r, g, on=["state_name", "year_month"], how="outer").sort_values("year_month")
    return merged.fillna(0).to_dict(orient="records")

# --- Prediction Endpoint ---
@app.get("/api/predict")
def predict(
    state: str = Query(...),
    year_month: str = Query(...),
    rainfall_value: float = Query(..., description="Rainfall in mm"),
    lag_gw: float = Query(..., description="Lag groundwater level (m bgl)")
):
    X = np.array([[rainfall_value, lag_gw]])
    pred = model.predict(X)[0]

    return {
        "state": state,
        "year_month": year_month,
        "rainfall_mm": rainfall_value,
        "lag_gw": lag_gw,
        "predicted_groundwater_level": round(float(pred), 2)
    }