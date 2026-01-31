"""
GIS constraints for recharge site optimization.
Extensible structure for LULC, protected areas, soil permeability, slope/DEM.
When data layers are available, integrate here.
"""
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Paths for constraint layers (set via env or config when available)
LULC_PATH = os.getenv("LULC_LAYER_PATH", "")
PROTECTED_AREAS_PATH = os.getenv("PROTECTED_AREAS_PATH", "")
SOIL_PERMEABILITY_PATH = os.getenv("SOIL_PERMEABILITY_PATH", "")
DEM_SLOPE_PATH = os.getenv("DEM_SLOPE_PATH", "")


@dataclass
class ConstraintResult:
    """Result of constraint check for a candidate site."""
    name: str
    passed: bool
    score: float  # 0-1, higher = better
    details: str


def check_lulc_constraint(lat: float, lon: float) -> ConstraintResult:
    """
    Check if site is in suitable land-use/land-cover.
    Returns ConstraintResult. When LULC layer not loaded, returns pass.
    """
    if not LULC_PATH or not os.path.exists(LULC_PATH):
        return ConstraintResult("lulc", True, 1.0, "LULC layer not configured; constraint skipped")
    # TODO: Load LULC raster/vector, query at (lon, lat), exclude water/forest/urban
    return ConstraintResult("lulc", True, 0.8, "LULC check placeholder")


def check_protected_area_constraint(lat: float, lon: float) -> ConstraintResult:
    """
    Check if site overlaps protected areas (wildlife, forests, water bodies).
    Returns ConstraintResult. When layer not loaded, returns pass.
    """
    if not PROTECTED_AREAS_PATH or not os.path.exists(PROTECTED_AREAS_PATH):
        return ConstraintResult("protected", True, 1.0, "Protected areas layer not configured; constraint skipped")
    # TODO: Load protected areas GeoJSON/shapefile, point-in-polygon check
    return ConstraintResult("protected", True, 1.0, "Protected areas check placeholder")


def check_soil_permeability_constraint(lat: float, lon: float) -> ConstraintResult:
    """
    Check soil permeability suitability for recharge.
    Returns ConstraintResult. When layer not loaded, returns pass.
    """
    if not SOIL_PERMEABILITY_PATH or not os.path.exists(SOIL_PERMEABILITY_PATH):
        return ConstraintResult("soil", True, 1.0, "Soil permeability layer not configured; constraint skipped")
    # TODO: Load soil raster, query at (lon, lat), score by permeability
    return ConstraintResult("soil", True, 0.85, "Soil permeability check placeholder")


def check_slope_constraint(lat: float, lon: float, max_slope_deg: float = 15.0) -> ConstraintResult:
    """
    Check slope/DEM constraint (flat terrain preferred for recharge).
    Returns ConstraintResult. When DEM not loaded, returns pass.
    """
    if not DEM_SLOPE_PATH or not os.path.exists(DEM_SLOPE_PATH):
        return ConstraintResult("slope", True, 1.0, "DEM/slope layer not configured; constraint skipped")
    # TODO: Load DEM, compute slope at (lon, lat), compare to max_slope_deg
    return ConstraintResult("slope", True, 0.9, "Slope check placeholder")


def apply_gis_constraints(lat: float, lon: float, exclude_protected: bool = True, max_slope_deg: float = 15.0) -> Tuple[bool, List[ConstraintResult]]:
    """
    Apply all GIS constraints to a candidate site.
    Returns (passed, list of ConstraintResult).
    """
    results: List[ConstraintResult] = []
    results.append(check_lulc_constraint(lat, lon))
    if exclude_protected:
        results.append(check_protected_area_constraint(lat, lon))
    results.append(check_soil_permeability_constraint(lat, lon))
    results.append(check_slope_constraint(lat, lon, max_slope_deg))
    passed = all(r.passed for r in results)
    return passed, results


def get_constraint_feasibility_score(lat: float, lon: float) -> float:
    """
    Compute overall feasibility score 0-1 from GIS constraints.
    """
    passed, results = apply_gis_constraints(lat, lon)
    if not passed:
        return 0.0
    return float(np.mean([r.score for r in results]))
