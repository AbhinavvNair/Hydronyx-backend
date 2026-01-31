"""Alerts for critical groundwater stress."""
from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional, List
from pydantic import BaseModel
from auth_utils import verify_token
from database import get_users_collection
import pandas as pd
import os
import numpy as np

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class AlertItem(BaseModel):
    state: str
    district: Optional[str]
    severity: str  # critical, high, medium
    message: str
    gw_level: float
    trend: str
    threshold_exceeded: bool


class AlertsResponse(BaseModel):
    alerts: List[AlertItem]
    count: int
    timestamp: str


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


def _load_groundwater() -> pd.DataFrame:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(backend_dir, "..", "data", "groundwater.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["state_name"] = df["state_name"].astype(str).str.strip().str.lower()
    if "district_name" in df.columns:
        df["district_name"] = df["district_name"].astype(str).str.strip().str.lower()
    if "gw_level_m_bgl" in df.columns:
        df["gw_level_m_bgl"] = pd.to_numeric(df["gw_level_m_bgl"], errors="coerce")
    return df.dropna(subset=["gw_level_m_bgl"])


@router.get("", response_model=AlertsResponse)
async def get_alerts(
    state: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    authorization: str = Header(None),
):
    """Get alerts for regions with critical groundwater stress."""
    from datetime import datetime
    _require_auth(authorization)
    df = _load_groundwater()
    alerts: List[AlertItem] = []
    if df.empty:
        return AlertsResponse(alerts=[], count=0, timestamp=datetime.utcnow().isoformat())
    # Critical: GW > 25m bgl (deep); High: > 15m; Medium: > 10m
    CRITICAL_THRESHOLD = 25.0
    HIGH_THRESHOLD = 15.0
    MEDIUM_THRESHOLD = 10.0
    group_cols = ["state_name", "district_name"] if "district_name" in df.columns else ["state_name"]
    for key, grp in df.groupby(group_cols):
        state_name = key[0] if isinstance(key, tuple) else key
        district_name = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        dist = str(district_name) if district_name is not None else None
        recent = grp.sort_values("year_month", ascending=False).head(12)
        mean_gw = float(recent["gw_level_m_bgl"].mean())
        if len(recent) >= 2:
            trend_val = np.polyfit(range(len(recent)), recent["gw_level_m_bgl"].values, 1)[0]
            trend = "declining" if trend_val > 0.1 else ("improving" if trend_val < -0.1 else "stable")
        else:
            trend = "unknown"
        if mean_gw >= CRITICAL_THRESHOLD:
            sev = "critical"
            msg = f"Critical: GW level {mean_gw:.1f}m bgl (deep stress)"
        elif mean_gw >= HIGH_THRESHOLD:
            sev = "high"
            msg = f"High stress: GW level {mean_gw:.1f}m bgl"
        elif mean_gw >= MEDIUM_THRESHOLD:
            sev = "medium"
            msg = f"Moderate stress: GW level {mean_gw:.1f}m bgl"
        else:
            continue
        if state and state.strip().lower() != state_name:
            continue
        if severity and severity != sev:
            continue
        alerts.append(AlertItem(
            state=state_name.title(),
            district=dist.title() if dist else None,
            severity=sev,
            message=msg,
            gw_level=round(mean_gw, 2),
            trend=trend,
            threshold_exceeded=True,
        ))
    alerts.sort(key=lambda a: (0 if a.severity == "critical" else 1 if a.severity == "high" else 2, -a.gw_level))
    return AlertsResponse(alerts=alerts[:50], count=len(alerts), timestamp=datetime.utcnow().isoformat())
