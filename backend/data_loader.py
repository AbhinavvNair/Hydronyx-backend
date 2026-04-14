"""
Data loader for groundwater stations dataset.
Loads and processes stations.csv once at startup.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os
from shapely.geometry import Point, Polygon

# Global variables to store loaded data
_meta_df = None
_ts_df = None
_time_cols = None

def load_stations_data():
    """Load and clean stations.csv dataset once at startup."""
    global _meta_df, _ts_df, _time_cols
    
    if _meta_df is not None:
        return  # Already loaded
    
    try:
        # Load the stations.csv file
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "stations.csv")
        df = pd.read_csv(csv_path)
        
        print(f"Loaded stations dataset: {len(df)} rows")
        
        # Keep only usable rows with valid coordinates
        df = df.dropna(subset=["Latitude", "Longitude"])
        df = df[df["Latitude"].astype(str).str.replace('.', '', 1).str.isdigit() & 
                df["Longitude"].astype(str).str.replace('.', '', 1).str.isdigit()]
        
        # Convert to float
        df["Latitude"] = df["Latitude"].astype(float)
        df["Longitude"] = df["Longitude"].astype(float)
        
        # Filter out invalid coordinates
        df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))]
        
        print(f"After cleaning: {len(df)} valid stations")
        
        # Define metadata columns
        META_COLS = [
            "Station Code", "Station Name", "State", "District",
            "Block", "Village", "Latitude", "Longitude",
            "Aquifer Type", "Well Depth", "Latest  Data Available"
        ]
        
        # Get time columns (those with '-')
        TIME_COLS = [c for c in df.columns if "-" in c and c not in META_COLS]
        
        # Split metadata and time series
        _meta_df = df[META_COLS].copy()
        _ts_df = df[["Station Code"] + TIME_COLS].copy()
        _time_cols = TIME_COLS
        
        print(f"Metadata columns: {len(_meta_df.columns)}")
        print(f"Time columns: {len(_time_cols)} (from {_time_cols[0]} to {_time_cols[-1]})")
        
    except Exception as e:
        print(f"Error loading stations data: {e}")
        # Create empty dataframes as fallback
        _meta_df = pd.DataFrame(columns=META_COLS)
        _ts_df = pd.DataFrame(columns=["Station Code"])
        _time_cols = []

def get_meta_df() -> pd.DataFrame:
    """Get metadata dataframe."""
    global _meta_df
    if _meta_df is None:
        load_stations_data()
    return _meta_df

def get_ts_df() -> pd.DataFrame:
    """Get time series dataframe."""
    global _ts_df
    if _ts_df is None:
        load_stations_data()
    return _ts_df

def get_time_cols() -> List[str]:
    """Get time column names."""
    global _time_cols
    if _time_cols is None:
        load_stations_data()
    return _time_cols

def get_nearest_wells(lat: float, lon: float, k: int = 8) -> List[Tuple[float, Dict]]:
    """
    Find k nearest wells to given lat/lon.
    Uses a ±3° bounding box pre-filter + vectorized numpy distances to avoid
    iterating all 32k+ stations with geodesic() (was the 56s bottleneck).

    Returns:
        List of (distance_km, well_info) tuples
    """
    meta_df = get_meta_df()

    if len(meta_df) == 0:
        return []

    # --- Bounding box pre-filter (vectorised, no Python loop) ---
    bbox_deg = 3.0
    mask = (
        (meta_df["Latitude"] >= lat - bbox_deg) &
        (meta_df["Latitude"] <= lat + bbox_deg) &
        (meta_df["Longitude"] >= lon - bbox_deg) &
        (meta_df["Longitude"] <= lon + bbox_deg)
    )
    nearby = meta_df[mask]

    # Fall back to full dataset if bounding box yields fewer than k stations
    if len(nearby) < k:
        nearby = meta_df

    # --- Vectorised approximate distance (Haversine via numpy) ---
    R = 6371.0  # Earth radius in km
    lat_r = np.radians(nearby["Latitude"].values)
    lon_r = np.radians(nearby["Longitude"].values)
    dlat = lat_r - np.radians(lat)
    dlon = lon_r - np.radians(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
    distances_km = 2 * R * np.arcsin(np.sqrt(a))

    # Take only the k nearest
    nearest_idx = np.argsort(distances_km)[:k]
    rows = nearby.iloc[nearest_idx]
    dists = distances_km[nearest_idx]

    wells = []
    for dist, (_, row) in zip(dists, rows.iterrows()):
        wells.append((float(dist), {
            "Station Code": row["Station Code"],
            "Station Name": row["Station Name"],
            "State": row["State"],
            "District": row["District"],
            "Block": row["Block"],
            "Village": row["Village"],
            "Latitude": row["Latitude"],
            "Longitude": row["Longitude"],
            "Aquifer Type": row["Aquifer Type"],
            "Well Depth": row["Well Depth"],
            "Latest Data": row.get("Latest  Data Available", ""),
        }))

    return wells

def latest_gwl(station_code: str) -> Optional[float]:
    """
    Get latest groundwater level for a station.
    
    Args:
        station_code: Station code identifier
        
    Returns:
        Latest groundwater level in meters bgl or None if not found
    """
    ts_df = get_ts_df()
    time_cols = get_time_cols()
    
    if len(time_cols) == 0:
        return None
    
    try:
        station_row = ts_df[ts_df["Station Code"] == station_code]
        if station_row.empty:
            return None
        
        row = station_row.iloc[0]
        values = row[time_cols].dropna()
        
        if len(values) == 0:
            return None
        
        # Get the latest (last non-null) value
        latest_value = values.iloc[-1]
        return float(latest_value) if pd.notna(latest_value) else None
        
    except Exception as e:
        print(f"Error getting GWL for station {station_code}: {e}")
        return None

def estimate_gwl(lat: float, lon: float, k: int = 8) -> Tuple[Optional[float], List[Dict]]:
    """
    Estimate groundwater level at lat/lon using IDW interpolation.
    
    Args:
        lat: Latitude
        lon: Longitude  
        k: Number of nearest wells to use
        
    Returns:
        Tuple of (estimated_gwl, used_wells_info)
    """
    nearest_wells = get_nearest_wells(lat, lon, k)
    
    if not nearest_wells:
        return None, []
    
    num, den = 0.0, 0.0
    used_wells = []
    
    for dist, well in nearest_wells:
        gw = latest_gwl(well["Station Code"])
        
        if gw is None or dist == 0:
            continue
        
        # IDW weight: 1/(distance^2)
        weight = 1.0 / (dist ** 2)
        num += gw * weight
        den += weight
        
        used_wells.append({
            "station_code": well["Station Code"],
            "station_name": well["Station Name"],
            "state": well["State"],
            "district": well["District"],
            "latitude": well["Latitude"],
            "longitude": well["Longitude"],
            "distance_km": round(dist, 2),
            "gw_latest": gw,
            "weight": weight
        })
    
    estimated_gwl = num / den if den > 0 else None
    
    # Sort used wells by weight (descending)
    used_wells.sort(key=lambda x: x["weight"], reverse=True)
    
    return estimated_gwl, used_wells

def confidence_score(used_wells: List[Dict]) -> str:
    """
    Calculate confidence score based on distances to used wells.
    
    Args:
        used_wells: List of wells used in estimation
        
    Returns:
        Confidence level: "High", "Medium", or "Low"
    """
    if not used_wells:
        return "Low"
    
    distances = [w["distance_km"] for w in used_wells]
    avg_distance = sum(distances) / len(distances)
    
    if avg_distance < 2:
        return "High"
    elif avg_distance < 5:
        return "Medium"
    return "Low"

def generate_grid_in_polygon(polygon_coords: List[Tuple[float, float]], step_deg: float = 0.001) -> List[Tuple[float, float]]:
    """
    Generate grid points inside a polygon.
    
    Args:
        polygon_coords: List of (lat, lon) coordinates for polygon
        step_deg: Step size in degrees (~100m)
        
    Returns:
        List of (lat, lon) grid points inside polygon
    """
    try:
        # Convert to (lon, lat) for shapely
        coords = [(lon, lat) for lat, lon in polygon_coords]
        poly = Polygon(coords)
        
        if not poly.is_valid:
            return []
        
        minx, miny, maxx, maxy = poly.bounds
        
        points = []
        lat = miny
        while lat <= maxy:
            lon = minx
            while lon <= maxx:
                point = Point(lon, lat)
                if poly.contains(point):
                    points.append((lat, lon))
                lon += step_deg
            lat += step_deg
        
        return points
        
    except Exception as e:
        print(f"Error generating grid: {e}")
        return []

def field_water_map(polygon_coords: List[Tuple[float, float]], k: int = 8) -> List[Dict]:
    """
    Create water level map for a field/polygon.
    
    Args:
        polygon_coords: List of (lat, lon) coordinates
        k: Number of nearest wells for IDW
        
    Returns:
        List of grid points with estimated water levels
    """
    grid_points = generate_grid_in_polygon(polygon_coords)
    results = []
    
    for lat, lon in grid_points:
        gwl, used_wells = estimate_gwl(lat, lon, k)
        
        if gwl is not None:
            results.append({
                "latitude": lat,
                "longitude": lon,
                "estimated_gwl": gwl,
                "used_wells_count": len(used_wells),
                "confidence": confidence_score(used_wells)
            })
    
    return results

def extract_numeric_series(row, time_cols):
    """
    Extract and clean time series data for trend calculation.
    
    Args:
        row: DataFrame row containing time series data
        time_cols: List of time column names
        
    Returns:
        Cleaned numeric pandas Series
    """
    values = row[time_cols]
    
    # Convert everything to numeric, errors become NaN
    values = pd.to_numeric(values, errors="coerce")
    
    # Drop NaN values
    values = values.dropna()
    
    return values

def calculate_trend_safe(row, time_cols):
    """
    Calculate trend using safe linear regression.
    
    Args:
        row: DataFrame row containing time series data
        time_cols: List of time column names
        
    Returns:
        Trend (meters per time-step) or None if insufficient data
    """
    series = extract_numeric_series(row, time_cols)
    
    if len(series) < 4:
        return None  # Not enough data for reliable trend
    
    y = series.values.astype(float)
    x = np.arange(len(y)).astype(float)
    
    # Linear trend: y = mx + c using polyfit (safer than lstsq)
    try:
        m, c = np.polyfit(x, y, 1)
        return float(m)  # meters per time-step
    except (np.linalg.LinAlgError, ValueError):
        return None

def trend_confidence(series_len):
    """
    Calculate confidence based on series length.
    
    Args:
        series_len: Length of valid data points
        
    Returns:
        Confidence level: "High", "Medium", or "Low"
    """
    if series_len > 60:
        return "High"
    elif series_len > 30:
        return "Medium"
    return "Low"

def calculate_trend(station_code: str) -> Tuple[float, str]:
    """
    Calculate groundwater trend for a station using safe methods.
    
    Args:
        station_code: Station code
        
    Returns:
        Tuple of (trend_m_per_month, trend_type)
    """
    ts_df = get_ts_df()
    time_cols = get_time_cols()
    
    if len(time_cols) < 2:
        return 0.0, "stable"
    
    try:
        station_row = ts_df[ts_df["Station Code"] == station_code]
        if station_row.empty:
            return 0.0, "stable"
        
        row = station_row.iloc[0]
        
        # Use safe trend calculation
        trend_per_timestep = calculate_trend_safe(row, time_cols)
        
        if trend_per_timestep is None:
            return 0.0, "stable"
        
        # Convert from per timestep to per month
        # CGWB data is roughly quarterly (4 measurements per year)
        trend_m_per_month = float(trend_per_timestep * 3.0)  # Approximate monthly rate
        
        # Classify trend
        if trend_m_per_month < -0.01:
            trend_type = "declining"
        elif trend_m_per_month > 0.01:
            trend_type = "improving"
        else:
            trend_type = "stable"
        
        return trend_m_per_month, trend_type
        
    except Exception as e:
        print(f"Error calculating trend for {station_code}: {e}")
        return 0.0, "stable"

def get_trend_with_confidence(station_code: str) -> Tuple[float, str, str]:
    """
    Calculate trend with confidence level.
    
    Args:
        station_code: Station code
        
    Returns:
        Tuple of (trend_m_per_month, trend_type, confidence)
    """
    ts_df = get_ts_df()
    time_cols = get_time_cols()
    
    try:
        station_row = ts_df[ts_df["Station Code"] == station_code]
        if station_row.empty:
            return 0.0, "stable", "Low"
        
        row = station_row.iloc[0]
        
        # Get the cleaned series
        series = extract_numeric_series(row, time_cols)
        series_len = len(series)
        
        if series_len < 4:
            return 0.0, "stable", "Low"
        
        # Calculate trend
        trend_per_timestep = calculate_trend_safe(row, time_cols)
        
        if trend_per_timestep is None:
            return 0.0, "stable", "Low"
        
        # Convert to monthly
        trend_m_per_month = float(trend_per_timestep * 3.0)
        
        # Classify trend
        if trend_m_per_month < -0.01:
            trend_type = "declining"
        elif trend_m_per_month > 0.01:
            trend_type = "improving"
        else:
            trend_type = "stable"
        
        # Get confidence based on data length
        confidence = trend_confidence(series_len)
        
        return trend_m_per_month, trend_type, confidence
        
    except Exception as e:
        print(f"Error calculating trend for {station_code}: {e}")
        return 0.0, "stable", "Low"

def forecast_from_trend(current_gwl: float, trend_m_per_month: float, months_ahead: int = 12) -> List[Dict[str, Any]]:
    """
    Generate forecast from current level and trend.
    
    Args:
        current_gwl: Current groundwater level
        trend_m_per_month: Trend in meters per month
        months_ahead: Number of months to forecast
        
    Returns:
        List of forecast points
    """
    forecast_data = []
    
    for month in range(1, months_ahead + 1):
        pred_level = current_gwl + (trend_m_per_month * month)
        
        # Add uncertainty based on trend magnitude and time
        base_uncertainty = 2.0  # Base 2m uncertainty
        trend_uncertainty = abs(trend_m_per_month) * month * 0.5  # Increases with trend and time
        total_uncertainty = base_uncertainty + trend_uncertainty
        
        forecast_data.append({
            "month": month,
            "predicted_level": float(pred_level),
            "lower_bound": float(pred_level - total_uncertainty),
            "upper_bound": float(pred_level + total_uncertainty),
        })
    
    return forecast_data

# Load data at import
load_stations_data()
