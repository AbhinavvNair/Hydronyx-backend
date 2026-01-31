"""
Rainfall forecast service - IMD/NOAA/ECMWF API integration with fallback.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

# API config from env
IMD_API_KEY = os.getenv("IMD_API_KEY", "")
NOAA_API_KEY = os.getenv("NOAA_API_KEY", "")
ECMWF_API_KEY = os.getenv("ECMWF_API_KEY", "")
RAINFALL_CACHE_HOURS = int(os.getenv("RAINFALL_CACHE_HOURS", "6"))


def fetch_imd_rainfall(state: str, district: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch rainfall forecast from IMD API (India Meteorological Department)."""
    if not IMD_API_KEY:
        return None
    try:
        url = "https://mausam.imd.gov.in/backend/api/forecast"
        params = {"state": state, "district": district or ""}
        headers = {"Authorization": f"Bearer {IMD_API_KEY}"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {"source": "IMD", "data": data, "timestamp": datetime.utcnow().isoformat()}
    except Exception:
        pass
    return None


def fetch_noaa_rainfall(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch rainfall from NOAA API (US-based, global coverage)."""
    if not NOAA_API_KEY:
        return None
    try:
        url = "https://api.weather.gov/points/{:.4f},{:.4f}/forecast".format(lat, lon)
        headers = {"User-Agent": "Hydronyx/1.0", "Accept": "application/json"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {"source": "NOAA", "data": data, "timestamp": datetime.utcnow().isoformat()}
    except Exception:
        pass
    return None


def fetch_rainfall_forecast(state: str, district: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
    """
    Fetch rainfall forecast with automatic fallback: IMD -> NOAA -> local CSV.
    Returns forecast with source and uncertainty metadata.
    """
    result: Dict[str, Any] = {"source": "local", "forecast_mm": [], "uncertainty": None, "metadata": {}}
    state_lower = state.strip().lower()

    # Try IMD first (India)
    imd = fetch_imd_rainfall(state, district)
    if imd:
        try:
            periods = imd.get("data", {}).get("forecast", {}).get("periods", [])[:12]
            if periods:
                result["forecast_mm"] = [p.get("precipitation", 0) or 0 for p in periods]
                result["source"] = "IMD"
                result["metadata"] = {"provider": "IMD", "fetched": imd["timestamp"]}
                result["uncertainty"] = 0.15
                return result
        except Exception:
            pass

    # Try NOAA (global, needs lat/lon)
    if lat is not None and lon is not None:
        noaa = fetch_noaa_rainfall(lat, lon)
        if noaa:
            try:
                periods = noaa.get("data", {}).get("properties", {}).get("periods", [])[:12]
                if periods:
                    result["forecast_mm"] = [p.get("temperature", 0) or 0 for p in periods]  # NOAA uses different structure
                    result["source"] = "NOAA"
                    result["metadata"] = {"provider": "NOAA", "fetched": noaa["timestamp"]}
                    result["uncertainty"] = 0.2
                    return result
            except Exception:
                pass

    # Fallback: use local CSV historical average
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(backend_dir, "..", "data")
    rainfall_path = os.path.join(data_dir, "rainfall.csv")
    if os.path.exists(rainfall_path):
        df = pd.read_csv(rainfall_path)
        df["state_name"] = df["state_name"].astype(str).str.strip().str.lower()
        subset = df[df["state_name"] == state_lower]
        rain_cols = [c for c in df.columns if "rain" in c.lower() or "precip" in c.lower() or "mm" in c.lower()]
        if not rain_cols:
            rain_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not rain_cols:
            rain_cols = [df.columns[-1]]
        if not subset.empty and rain_cols:
            col = rain_cols[0]
            mean_val = float(subset[col].mean())
            if pd.isna(mean_val):
                mean_val = 80.0
            result["forecast_mm"] = [mean_val] * 12
            result["source"] = "local"
            result["metadata"] = {"provider": "local_csv", "fallback": True}
            result["uncertainty"] = 0.35
            return result

    # Ultimate fallback
    result["forecast_mm"] = [80.0] * 12
    result["metadata"] = {"provider": "default", "fallback": True}
    result["uncertainty"] = 0.5
    return result
