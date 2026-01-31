"""Rainfall forecast API - IMD/NOAA/ECMWF integration with fallback."""
from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional
from auth_utils import verify_token
from database import get_users_collection
from rainfall_service import fetch_rainfall_forecast

router = APIRouter(prefix="/api/rainfall", tags=["rainfall"])


def _require_auth(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = authorization.split(" ")[1]
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    users = get_users_collection()
    if not users.find_one({"email": email}):
        raise HTTPException(status_code=401, detail="User not found")


@router.get("/forecast")
async def get_rainfall_forecast(
    state: str = Query(...),
    district: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    authorization: str = Header(None),
):
    """Fetch rainfall forecast from IMD/NOAA with automatic fallback to local data."""
    _require_auth(authorization)
    result = fetch_rainfall_forecast(state, district, lat, lon)
    return result
