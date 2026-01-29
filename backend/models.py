from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from bson.objectid import ObjectId


class ForecastParams(BaseModel):
    """Parameters used for a forecast"""
    state: str
    district: Optional[str] = None
    forecast_horizon: int
    rainfall_value: Optional[float] = None
    lag_gw: Optional[float] = None


class ForecastResult(BaseModel):
    """Result of a forecast prediction"""
    predicted_level: float
    confidence: float
    uncertainty: float
    physics_compliance: float


class UserForecast(BaseModel):
    """User forecast history entry"""
    user_id: str
    params: ForecastParams
    result: ForecastResult
    created_at: datetime
    timestamp: str  # Human readable


class UserStats(BaseModel):
    """User statistics"""
    user_id: str
    total_forecasts: int
    total_optimizations: int
    total_policies: int
    last_forecast: Optional[datetime] = None
    created_at: datetime
