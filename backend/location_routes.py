"""
Location-based Groundwater Forecasting using IDW Interpolation
================================================================
Provides endpoints for querying groundwater at arbitrary geographic coordinates
using Inverse Distance Weighting (IDW) from nearby monitoring stations.
"""

from fastapi import APIRouter, HTTPException, status, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import os
import io
import math
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from scipy.spatial.distance import cdist
from data_loader import (
    estimate_gwl, confidence_score, field_water_map, 
    calculate_trend, get_nearest_wells, latest_gwl,
    get_trend_with_confidence, forecast_from_trend
)
from shapely.geometry import shape, Point

from auth_utils import verify_token, extract_user_id
from database import get_forecast_collection, get_users_collection
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/location", tags=["location"])

# Cached data
_groundwater_df: Optional[pd.DataFrame] = None
_rainfall_df: Optional[pd.DataFrame] = None


def _get_groundwater_df() -> pd.DataFrame:
    """Load and cache groundwater CSV, generating missing spatial data."""
    global _groundwater_df
    if _groundwater_df is not None:
        return _groundwater_df
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(backend_dir, "..", "data")
    path = os.path.join(data_dir, "groundwater.csv")
    
    if not os.path.exists(path):
        _groundwater_df = pd.DataFrame()
        return _groundwater_df
    
    df = pd.read_csv(path)
    
    # Check which required columns are missing
    required_cols = ["state_name", "district_name", "gw_level_m_bgl", "year_month"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Add synthetic station codes and names if missing
    if "station_code" not in df.columns:
        df["station_code"] = df.apply(
            lambda row: f"STN_{row['state_name'][:3].upper()}_{row['district_name'][:3].upper()}_{len(df) % 1000:03d}",
            axis=1
        )
    
    if "station_name" not in df.columns:
        df["station_name"] = df.apply(
            lambda row: f"{row['district_name']} Station {row['state_name']}",
            axis=1
        )
    
    # Add synthetic coordinates if missing
    if "latitude" not in df.columns or "longitude" not in df.columns:
        # Define approximate center coordinates for major Indian states
        state_coords = {
            "Maharashtra": (19.7515, 75.7139),
            "Haryana": (29.0588, 76.0856),
            "Punjab": (31.1471, 75.3412),
            "Uttar Pradesh": (26.8467, 80.9462),
            "Rajasthan": (27.0238, 74.2179),
            "Gujarat": (22.2587, 71.1924),
        }
        
        def get_coords(state):
            base_lat, base_lon = state_coords.get(state, (25.0, 75.0))
            # Add small random offset for each district
            np.random.seed(hash(state) % 2**32)
            return (
                base_lat + np.random.uniform(-1, 1),
                base_lon + np.random.uniform(-1, 1)
            )
        
        if "latitude" not in df.columns:
            df["latitude"] = df["state_name"].apply(lambda s: get_coords(s)[0])
        if "longitude" not in df.columns:
            df["longitude"] = df["state_name"].apply(lambda s: get_coords(s)[1])
    
    # Convert to numeric types
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["gw_level_m_bgl"] = pd.to_numeric(df["gw_level_m_bgl"], errors="coerce")
    
    # Drop rows with missing critical values
    df = df.dropna(subset=["latitude", "longitude", "gw_level_m_bgl"])
    
    _groundwater_df = df
    return _groundwater_df


def _get_rainfall_df() -> pd.DataFrame:
    """Load and cache rainfall CSV."""
    global _rainfall_df
    if _rainfall_df is not None:
        return _rainfall_df
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(backend_dir, "..", "data")
    path = os.path.join(data_dir, "rainfall.csv")
    
    if not os.path.exists(path):
        _rainfall_df = pd.DataFrame()
        return _rainfall_df
    
    df = pd.read_csv(path)
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    _rainfall_df = df.dropna(subset=["rainfall_mm"])
    return _rainfall_df


def _idw_interpolation(
    target_lat: float,
    target_lon: float,
    stations_df: pd.DataFrame,
    power: float = 2.0
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Perform Inverse Distance Weighting interpolation.
    
    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        stations_df: DataFrame with station data (must have latitude, longitude, gw_level_m_bgl)
        power: IDW power parameter (default 2.0)
    
    Returns:
        (interpolated_value, list_of_contributing_stations)
    """
    if stations_df.empty:
        raise ValueError("No station data available")
    
    # Calculate distances
    stations_coords = stations_df[["latitude", "longitude"]].values
    target_coord = np.array([[target_lat, target_lon]])
    distances = cdist(target_coord, stations_coords, metric="euclidean")[0]
    
    # Handle case where target is exactly at a station
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] < 0.001:  # ~100m
        exact_station = stations_df.iloc[min_dist_idx]
        return float(exact_station["gw_level_m_bgl"]), [{
            "station_code": exact_station["station_code"],
            "station_name": exact_station["station_name"],
            "state": exact_station["state_name"],
            "district": exact_station["district_name"],
            "latitude": float(exact_station["latitude"]),
            "longitude": float(exact_station["longitude"]),
            "distance_km": 0.0,
            "gw_latest": float(exact_station["gw_level_m_bgl"]),
            "weight": 1.0,
        }]
    
    # Calculate weights (inverse distance with power)
    weights = 1.0 / (distances ** power)
    weights = weights / np.sum(weights)  # Normalize
    
    # Interpolated value
    gw_values = stations_df["gw_level_m_bgl"].values
    interpolated = np.sum(weights * gw_values)
    
    # Build station list with weights
    contributing_stations = []
    for idx, (dist, weight) in enumerate(zip(distances, weights)):
        station = stations_df.iloc[idx]
        contributing_stations.append({
            "station_code": station["station_code"],
            "station_name": station["station_name"],
            "state": station["state_name"],
            "district": station["district_name"],
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
            "distance_km": float(dist * 111.0),  # Approximate km conversion
            "gw_latest": float(station["gw_level_m_bgl"]),
            "weight": float(weight),
        })
    
    return float(interpolated), contributing_stations


def _get_nearest_stations(
    target_lat: float,
    target_lon: float,
    k: int = 8
) -> pd.DataFrame:
    """Get k nearest stations to target location."""
    df = _get_groundwater_df()
    if df.empty:
        raise ValueError("No groundwater data available")
    
    # For each station_code, keep only the latest record
    df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    latest_df = df.sort_values("year_month").groupby("station_code").tail(1).reset_index(drop=True)
    
    # Calculate distances
    coords = latest_df[["latitude", "longitude"]].values
    target = np.array([[target_lat, target_lon]])
    distances = cdist(target, coords, metric="euclidean")[0]
    
    # Get k nearest
    nearest_indices = np.argsort(distances)[:k]
    return latest_df.iloc[nearest_indices].reset_index(drop=True)


def _estimate_trend(station_code: str) -> Tuple[float, str]:
    """
    Estimate groundwater trend (m/month) and trend type.
    
    Args:
        station_code: Station code
    
    Returns:
        (trend_m_per_month, trend_type)
    """
    df = _get_groundwater_df()
    if df.empty:
        return 0.0, "stable"
    
    station_data = df[df["station_code"] == station_code].copy()
    if station_data.empty or len(station_data) < 3:
        return 0.0, "stable"
    
    station_data["year_month"] = pd.to_datetime(station_data["year_month"], errors="coerce")
    station_data = station_data.sort_values("year_month")
    
    # Simple linear regression
    x = np.arange(len(station_data))
    y = station_data["gw_level_m_bgl"].values
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        trend_m_per_month = float(z[0])
    else:
        trend_m_per_month = 0.0
    
    # Classify trend
    if trend_m_per_month < -0.01:
        trend_type = "declining"
    elif trend_m_per_month > 0.01:
        trend_type = "improving"
    else:
        trend_type = "stable"
    
    return trend_m_per_month, trend_type


def _estimate_uncertainty(k: int, power: float) -> float:
    """
    Estimate uncertainty based on number of stations and IDW power.
    Higher k and lower power → lower uncertainty.
    """
    # Simple heuristic: base uncertainty decreases with k
    base_uncertainty = 5.0  # m
    k_factor = max(1.0, 1.0 - (k / 25.0))  # Decreases with k
    power_factor = max(0.5, 3.0 - power)  # Higher power → slightly lower uncertainty
    
    uncertainty = base_uncertainty * k_factor * (power_factor / 2.0)
    return max(0.5, uncertainty)


def _estimate_confidence(k: int) -> str:
    """Estimate confidence level based on number of stations used."""
    if k >= 15:
        return "High"
    elif k >= 8:
        return "Medium"
    else:
        return "Low"


class LocationInsightRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    months_ahead: int = Field(6, ge=1, le=12)
    k: int = Field(8, ge=3, le=25)
    power: float = Field(2.0, ge=1.0, le=5.0)


class LocationInsightResponse(BaseModel):
    current_level_m_bgl: float
    trend_m_per_month: float
    trend: str
    uncertainty_m: float
    confidence: str
    forecast: List[Dict[str, Any]]
    nearest_stations: List[Dict[str, Any]]
    meta: Dict[str, Any]


@router.post("/groundwater", response_model=LocationInsightResponse)
async def get_location_groundwater_insight(
    req: LocationInsightRequest,
    authorization: str = Header(None)
):
    """
    Get groundwater insight at arbitrary lat/lon using real stations data.
    
    Request body:
    {
        "latitude": 26.9124,
        "longitude": 75.7873,
        "months_ahead": 12,
        "k": 8,
        "power": 2.0
    }
    """
    # Verify authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization[7:]  # safe: startswith("Bearer ") already checked above
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        # Use real dataset for estimation
        estimated_gwl, used_wells = estimate_gwl(req.latitude, req.longitude, k=req.k)
        
        if estimated_gwl is None:
            raise ValueError("No nearby stations with valid data found")
        
        # Calculate trend from nearby wells using improved method
        trends = []
        trend_confidences = []
        for well in used_wells[:5]:  # Use top 5 wells for trend
            trend, trend_type, trend_conf = get_trend_with_confidence(well["station_code"])
            if trend != 0.0:  # Only include non-zero trends
                trends.append(trend)
                trend_confidences.append(trend_conf)
        
        avg_trend = float(np.mean(trends)) if trends else 0.0
        
        # Determine overall trend confidence
        if trend_confidences:
            confidence_counts = {"High": trend_confidences.count("High"), 
                              "Medium": trend_confidences.count("Medium"), 
                              "Low": trend_confidences.count("Low")}
            overall_trend_confidence = max(confidence_counts, key=confidence_counts.get)
        else:
            overall_trend_confidence = "Low"
        
        # Classify trend
        if avg_trend < -0.01:
            trend_type = "declining"
        elif avg_trend > 0.01:
            trend_type = "improving"
        else:
            trend_type = "stable"
        
        # Generate forecast using improved method
        forecast_data = forecast_from_trend(estimated_gwl, avg_trend, req.months_ahead)
        
        # Calculate confidence
        confidence = confidence_score(used_wells)
        
        # Calculate uncertainty
        avg_distance = np.mean([w["distance_km"] for w in used_wells])
        uncertainty_m = 2.0 + (avg_distance * 0.5)
        
        response = LocationInsightResponse(
            current_level_m_bgl=float(estimated_gwl),
            trend_m_per_month=float(avg_trend),
            trend=trend_type,
            uncertainty_m=float(uncertainty_m),
            confidence=confidence,
            forecast=forecast_data,
            nearest_stations=used_wells,
            meta={
                "method": "IDW",
                "k": req.k,
                "power": req.power,
                "stations_used": len(used_wells),
                "avg_distance_km": round(avg_distance, 2),
                "generated_at": datetime.utcnow().isoformat(),
            }
        )
        
        # Log to database
        try:
            forecast_collection = get_forecast_collection()
            forecast_collection.insert_one({
                "user_id": user_id,
                "type": "location_insight",
                "latitude": req.latitude,
                "longitude": req.longitude,
                "result": response.dict(),
                "created_at": datetime.utcnow(),
            })
        except Exception as db_error:
            logger.warning("Failed to log location query to database: %s", db_error)

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Location insight error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


class PlotInsightRequest(BaseModel):
    polygon: Dict[str, Any]  # Changed from polygon_geojson to polygon
    months_ahead: int = 12
    k: int = 8
    power: float = 2.0
    grid_size: int = 14


class PlotPointInsight(BaseModel):
    latitude: float
    longitude: float
    current_level_m_bgl: float
    trend_m_per_month: float
    uncertainty_m: float
    confidence: str
    zone: str
    risk_score: float


class PlotInsightResponse(BaseModel):
    plot_stats: Dict[str, Any]
    grid: List[PlotPointInsight]
    recommended_point: Dict[str, Any]
    generated_at: str


def _extract_user_id_from_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


def _compute_location_metrics(lat: float, lon: float, months_ahead: int, k: int, power: float) -> Dict[str, Any]:
    """Compute location metrics without DB logging; reuses IDW + trend logic."""
    # Get nearest stations
    nearest = _get_nearest_stations(lat, lon, k=k)
    if nearest.empty:
        raise ValueError("No nearby stations found")

    current_level, contributing_stations = _idw_interpolation(lat, lon, nearest, power=power)

    # Weighted-ish average trend: here simple mean of station trends (consistent with existing endpoint)
    trends = []
    for _, station in nearest.iterrows():
        t_m_month, _ = _estimate_trend(station["station_code"])
        trends.append(t_m_month)
    avg_trend = float(np.mean(trends)) if trends else 0.0

    if avg_trend < -0.01:
        trend_type = "declining"
    elif avg_trend > 0.01:
        trend_type = "improving"
    else:
        trend_type = "stable"

    uncertainty = _estimate_uncertainty(k, power)
    confidence = _estimate_confidence(k)

    # Build forecast (kept simple; plot-insight uses stats primarily)
    forecast_data: List[Dict[str, Any]] = []
    for month in range(1, months_ahead + 1):
        pred_level = current_level + (avg_trend * month)
        lower = pred_level - uncertainty
        upper = pred_level + uncertainty
        forecast_data.append({
            "month": month,
            "predicted_level": float(pred_level),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        })

    contributing_stations.sort(key=lambda x: x["weight"], reverse=True)

    return {
        "current_level_m_bgl": float(current_level),
        "trend_m_per_month": float(avg_trend),
        "trend": trend_type,
        "uncertainty_m": float(uncertainty),
        "confidence": confidence,
        "forecast": forecast_data,
        "nearest_stations": contributing_stations,
        "meta": {
            "method": "IDW",
            "k": k,
            "power": power,
            "stations_used": len(contributing_stations),
        },
    }


def _risk_score(current_level_m_bgl: float, trend_m_per_month: float, uncertainty_m: float, confidence: str) -> float:
    """Heuristic risk score for borewell viability; higher is worse."""
    score = 0.0
    # Deep water table is riskier
    if current_level_m_bgl > 60:
        score += 3.0
    elif current_level_m_bgl > 40:
        score += 2.0
    elif current_level_m_bgl > 25:
        score += 1.0

    # Declining (in this dataset negative is labeled declining in _estimate_trend)
    if trend_m_per_month < -0.03:
        score += 3.0
    elif trend_m_per_month < -0.01:
        score += 2.0
    elif trend_m_per_month < 0.0:
        score += 1.0

    # Uncertainty penalty
    if uncertainty_m > 5:
        score += 2.0
    elif uncertainty_m > 3:
        score += 1.0

    if confidence == "Low":
        score += 1.5
    elif confidence == "Medium":
        score += 0.75

    return float(score)


def _zone_from_score(score: float) -> str:
    if score >= 6.0:
        return "Red"
    if score >= 3.5:
        return "Yellow"
    return "Green"


@router.post("/plot-insight", response_model=PlotInsightResponse)
async def get_plot_groundwater_insight(req: PlotInsightRequest, authorization: str = Header(None)):
    """Plot-level insight using real stations data for a farm boundary."""
    extract_user_id(authorization)

    try:
        if req.k < 3 or req.k > 25:
            raise ValueError("k must be between 3 and 25")
        if req.months_ahead < 1 or req.months_ahead > 12:
            raise ValueError("months_ahead must be between 1 and 12")
        if req.grid_size < 6 or req.grid_size > 40:
            raise ValueError("grid_size must be between 6 and 40")

        geom = shape(req.polygon)
        if geom.is_empty:
            raise ValueError("Empty polygon")
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("polygon must be a Polygon or MultiPolygon")

        minx, miny, maxx, maxy = geom.bounds
        # bounds sanity
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
            raise ValueError("Polygon bounds look invalid")

        # Generate grid points inside polygon using real dataset
        step_deg = 0.001  # ~100m steps
        grid_points = []
        
        lat = miny
        while lat <= maxy:
            lon = minx
            while lon <= maxx:
                point = Point(lon, lat)
                if geom.contains(point):
                    grid_points.append((lat, lon))
                lon += step_deg
            lat += step_deg

        if not grid_points:
            raise ValueError("No grid points fell inside the polygon. Try a larger plot or smaller grid_size.")

        # Sample points if too many
        max_points = req.grid_size * req.grid_size
        if len(grid_points) > max_points:
            indices = np.random.choice(len(grid_points), max_points, replace=False)
            grid_points = [grid_points[i] for i in indices]

        # Process each grid point with real data
        sampled: List[Dict[str, Any]] = []
        for lat, lon in grid_points:
            try:
                # Use real dataset estimation
                estimated_gwl, used_wells = estimate_gwl(lat, lon, k=req.k)
                
                if estimated_gwl is None:
                    continue
                
                # Calculate trend from nearby wells using improved method
                trends = []
                for well in used_wells[:3]:
                    trend, trend_type, trend_conf = get_trend_with_confidence(well["station_code"])
                    if trend != 0.0:  # Only include non-zero trends
                        trends.append(trend)
                
                avg_trend = float(np.mean(trends)) if trends else 0.0
                
                # Calculate risk score
                rs = _risk_score(
                    current_level_m_bgl=estimated_gwl,
                    trend_m_per_month=avg_trend,
                    uncertainty_m=2.0 + (np.mean([w["distance_km"] for w in used_wells]) * 0.5),
                    confidence=confidence_score(used_wells)
                )
                
                sampled.append({
                    "latitude": lat,
                    "longitude": lon,
                    "current_level_m_bgl": estimated_gwl,
                    "trend_m_per_month": avg_trend,
                    "uncertainty_m": 2.0 + (np.mean([w["distance_km"] for w in used_wells]) * 0.5),
                    "confidence": confidence_score(used_wells),
                    "risk_score": rs,
                    "zone": _zone_from_score(rs),
                    "_nearest_stations": used_wells,
                })
                
            except Exception as e:
                print(f"Error processing grid point {lat}, {lon}: {e}")
                continue

        if not sampled:
            raise ValueError("No valid grid points could be processed. Try a different area.")

        # Calculate statistics
        levels = np.array([s["current_level_m_bgl"] for s in sampled], dtype=float)
        trends = np.array([s["trend_m_per_month"] for s in sampled], dtype=float)
        scores = np.array([s["risk_score"] for s in sampled], dtype=float)

        best_idx = int(np.argmin(scores))
        best = sampled[best_idx]

        plot_stats = {
            "n_samples": int(len(sampled)),
            "current_level_mean_m_bgl": float(np.mean(levels)),
            "current_level_min_m_bgl": float(np.min(levels)),
            "current_level_max_m_bgl": float(np.max(levels)),
            "trend_mean_m_per_month": float(np.mean(trends)),
            "risk_score_mean": float(np.mean(scores)),
            "zone_counts": {
                "Green": int(sum(1 for s in sampled if s["zone"] == "Green")),
                "Yellow": int(sum(1 for s in sampled if s["zone"] == "Yellow")),
                "Red": int(sum(1 for s in sampled if s["zone"] == "Red")),
            },
        }

        reasons: List[str] = []
        if best["current_level_m_bgl"] > plot_stats["current_level_mean_m_bgl"]:
            reasons.append("Is point par paani thoda gehra hai, lekin plot ke hisaab se yeh sabse stable zone hai.")
        else:
            reasons.append("Is point par paani plot ke average se upar (better) hai.")
        if best["trend_m_per_month"] < -0.01:
            reasons.append("Trend declining hai — pumping ko control/recharge consider karein.")
        if best["confidence"] == "Low":
            reasons.append("Nearby monitoring wells kam hain — isliye confidence low hai.")

        recommended_point = {
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "zone": best["zone"],
            "risk_score": best["risk_score"],
            "current_level_m_bgl": best["current_level_m_bgl"],
            "trend_m_per_month": best["trend_m_per_month"],
            "uncertainty_m": best["uncertainty_m"],
            "confidence": best["confidence"],
            "reasons": reasons,
            "nearest_stations": best.get("_nearest_stations", [])[:8],
        }

        grid = [
            PlotPointInsight(
                latitude=s["latitude"],
                longitude=s["longitude"],
                current_level_m_bgl=s["current_level_m_bgl"],
                trend_m_per_month=s["trend_m_per_month"],
                uncertainty_m=s["uncertainty_m"],
                confidence=s["confidence"],
                zone=s["zone"],
                risk_score=s["risk_score"],
            )
            for s in sampled
        ]

        return PlotInsightResponse(
            plot_stats=plot_stats,
            grid=grid,
            recommended_point=recommended_point,
            generated_at=datetime.utcnow().isoformat(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plot insight error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


def _generate_location_pdf(result: LocationInsightResponse, req: LocationInsightRequest) -> bytes:
    """Generate PDF report for location insight."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    y = height - margin
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#06b6d4"))
    c.drawString(margin, y, "Hydronyx Location Groundwater Report")
    y -= 1.5 * cm
    
    # Generated date
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#666666"))
    c.drawString(margin, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 0.6 * cm
    
    # Location Details
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#000000"))
    c.drawString(margin, y, "Location Details")
    y -= 0.4 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#333333"))
    details_list = [
        f"Latitude: {req.latitude}",
        f"Longitude: {req.longitude}",
        f"Forecast Horizon: {req.months_ahead} months",
        f"Nearest Stations Used (k): {req.k}",
    ]
    for detail in details_list:
        c.drawString(margin + 0.3 * cm, y, detail)
        y -= 0.35 * cm
    y -= 0.4 * cm
    
    # Current Status
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#000000"))
    c.drawString(margin, y, "Current Groundwater Status")
    y -= 0.4 * cm
    
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#333333"))
    status_list = [
        f"Current Level: {result.current_level_m_bgl:.2f} m bgl",
        f"Trend: {result.trend} ({result.trend_m_per_month:.3f} m/month)",
        f"Uncertainty: ±{result.uncertainty_m:.2f} m",
        f"Confidence: {result.confidence}",
    ]
    for status in status_list:
        c.drawString(margin + 0.3 * cm, y, status)
        y -= 0.35 * cm
    y -= 0.4 * cm
    
    # Forecast Summary
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#000000"))
    c.drawString(margin, y, "Forecast Summary")
    y -= 0.4 * cm
    
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#333333"))
    if result.forecast:
        first = result.forecast[0]
        last = result.forecast[-1]
        c.drawString(margin + 0.3 * cm, y, f"Month 1 Level: {first['predicted_level']:.2f} m bgl")
        y -= 0.35 * cm
        c.drawString(margin + 0.3 * cm, y, f"Month {last['month']} Level: {last['predicted_level']:.2f} m bgl")
        y -= 0.35 * cm
        change = last['predicted_level'] - first['predicted_level']
        c.drawString(margin + 0.3 * cm, y, f"Expected Change: {change:.2f} m")
        y -= 0.35 * cm
    y -= 0.4 * cm
    
    # Contributing Stations
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#000000"))
    c.drawString(margin, y, "Top Contributing Stations")
    y -= 0.4 * cm
    
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#333333"))
    for i, station in enumerate(result.nearest_stations[:5], 1):
        c.drawString(
            margin + 0.3 * cm, 
            y, 
            f"{i}. {station['station_name']} ({station['station_code']}) - {station['distance_km']:.1f} km, weight: {station['weight']:.3f}"
        )
        y -= 0.3 * cm
    y -= 0.3 * cm
    
    # Method info
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#666666"))
    method_text = "This report uses Inverse Distance Weighting (IDW) interpolation to estimate groundwater levels at the specified location based on nearby monitoring stations."
    text_object = c.beginText(margin, y)
    text_object.setTextOrigin(margin, y)
    text_object.setFont("Helvetica", 8)
    for line in method_text.split('\n'):
        text_object.textLine(line)
    
    c.drawString(margin, y - 1 * cm, "For more information, visit: www.hydronyx.io")
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


@router.post("/report.pdf")
async def download_location_report(
    req: LocationInsightRequest,
    authorization: str = Header(None)
):
    """
    Download PDF report for location groundwater insight.
    """
    # Verify authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization[7:]  # safe: startswith("Bearer ") already checked above
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        # Get insight data
        insight_response = await get_location_groundwater_insight(req, authorization)
        
        # Generate PDF
        pdf_bytes = _generate_location_pdf(insight_response, req)
        
        # Return as file
        return FileResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=location_groundwater_report.pdf"}
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] PDF generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF")
