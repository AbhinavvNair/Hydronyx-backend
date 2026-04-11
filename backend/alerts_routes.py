"""Alerts for critical groundwater stress."""
from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional, List, Tuple
from pydantic import BaseModel
from auth_utils import extract_user_id
from database import get_users_collection
import pandas as pd
import os
import numpy as np
import random
import time

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
    source: Optional[str] = None
    last_updated: Optional[str] = None
    # Add critical alert summary for bottom notification
    critical_count: Optional[int] = None
    top_critical: Optional[AlertItem] = None


def _require_auth(authorization: str):
    extract_user_id(authorization)


# Cache: (DataFrame with all alerts pre-computed, minute-bucket when it was built)
_alerts_cache: Tuple[Optional[pd.DataFrame], int] = (None, -1)


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


def _build_alerts() -> List[AlertItem]:
    """Compute all alerts from the groundwater CSV. Result is cached for 60 seconds."""
    global _alerts_cache
    minute_bucket = int(time.time() // 60)
    cached_df, cached_bucket = _alerts_cache
    if cached_df is not None and cached_bucket == minute_bucket:
        return cached_df  # type: ignore[return-value]

    df = _load_groundwater()
    alerts: List[AlertItem] = []
    if df.empty:
        _alerts_cache = (alerts, minute_bucket)  # type: ignore[assignment]
        return alerts

    CRITICAL_THRESHOLD = 25.0
    HIGH_THRESHOLD = 15.0
    MEDIUM_THRESHOLD = 10.0

    group_cols = ["state_name", "district_name"] if "district_name" in df.columns else ["state_name"]
    df_sorted = df.sort_values("year_month")

    for key, grp in df_sorted.groupby(group_cols):
        state_name = key[0] if isinstance(key, tuple) else key
        district_name = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        dist = str(district_name) if district_name is not None else None

        recent = grp.tail(12)
        base_mean = float(recent["gw_level_m_bgl"].mean())

        random.seed(minute_bucket + abs(hash(f"{state_name}-{district_name}")))
        variation = (random.random() - 0.5) * 3.0
        mean_gw = base_mean + variation

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
    _alerts_cache = (alerts, minute_bucket)  # type: ignore[assignment]
    return alerts


@router.get("", response_model=AlertsResponse)
async def get_alerts(
    state: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    authorization: str = Header(None),
):
    """Get alerts for regions with critical groundwater stress."""
    from datetime import datetime
    _require_auth(authorization)

    all_alerts = _build_alerts()

    # Apply optional filters
    alerts = all_alerts
    if state:
        state_lower = state.strip().lower()
        alerts = [a for a in alerts if a.state.lower() == state_lower]
    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    now = datetime.utcnow()
    critical_alerts = [a for a in alerts if a.severity == "critical"]
    return AlertsResponse(
        alerts=alerts[:50],
        count=len(alerts),
        timestamp=now.isoformat(),
        source="simulated-live",
        last_updated=now.isoformat(),
        critical_count=len(critical_alerts),
        top_critical=critical_alerts[0] if critical_alerts else None,
    )
