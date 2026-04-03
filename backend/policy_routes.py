from fastapi import APIRouter, HTTPException, status, Header, Query
from fastapi.responses import Response
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from database import get_users_collection, get_policy_simulations_collection
from auth_utils import verify_token, extract_user_id
import pandas as pd
import numpy as np
import os
import io
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/policy", tags=["policy"])

@lru_cache(maxsize=1)
def _get_groundwater_df() -> pd.DataFrame:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(backend_dir, "..", "data", "groundwater.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["state_name"] = df["state_name"].astype(str).str.strip().str.lower()
    if "gw_level_m_bgl" in df.columns:
        df["gw_level_m_bgl"] = pd.to_numeric(df["gw_level_m_bgl"], errors="coerce")
    return df.dropna(subset=["gw_level_m_bgl"])


@lru_cache(maxsize=1)
def _get_rainfall_df() -> pd.DataFrame:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(backend_dir, "..", "data", "rainfall.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["state_name"] = df["state_name"].astype(str).str.strip().str.lower()
    return df


def _state_baselines(state: str) -> Tuple[float, float]:
    """
    Return (base_gw_level_m, base_rainfall_mm_per_year) for the given state from real data.
    Fallback (45.0, 800.0) if no data.
    """
    state_clean = str(state).strip().lower()
    base_gw = 45.0
    base_rainfall = 800.0

    gw_df = _get_groundwater_df()
    if not gw_df.empty:
        subset = gw_df[gw_df["state_name"] == state_clean]["gw_level_m_bgl"]
        if not subset.empty:
            base_gw = float(subset.mean())

    rain_df = _get_rainfall_df()
    if not rain_df.empty:
        for rain_col in ["rainfall_actual_mm", "rainfall_mm", "rainfall", "precipitation"]:
            if rain_col in rain_df.columns:
                subset = rain_df[rain_df["state_name"] == state_clean][rain_col]
                subset = pd.to_numeric(subset, errors="coerce").dropna()
                if not subset.empty:
                    # monthly values -> annual approx
                    base_rainfall = float(subset.mean()) * 12
                    break

    return base_gw, base_rainfall


class InterventionParams(BaseModel):
    state: str
    pumping_change: float  # Percentage change in pumping
    recharge_structures: float  # Number of recharge structures
    crop_intensity_change: float = 0.0  # Optional: change in crop intensity
    months_ahead: int = 12


class TrajectoryPoint(BaseModel):
    month: int
    groundwater: float
    rainfall: Optional[float] = None


class PolicySimulationResult(BaseModel):
    baseline_trajectory: List[TrajectoryPoint]
    counterfactual_trajectory: List[TrajectoryPoint]
    mean_effect: float
    final_effect: float
    cumulative_effect: float
    uncertainty_margin: float



@router.post("/simulate", response_model=PolicySimulationResult)
async def simulate_policy(
    params: InterventionParams,
    authorization: str = Header(None)
):
    """
    Simulate counterfactual policy scenarios using SCM
    
    Args:
        params: Intervention parameters (pumping change, recharge structures, etc.)
        authorization: Bearer token for authentication
    
    Returns:
        Baseline and counterfactual trajectories with treatment effects
    """
    user_id = extract_user_id(authorization)
    
    try:
        months = params.months_ahead
        # State-specific baselines from real data so different states get different trajectories
        base_gw_level, base_rainfall = _state_baselines(params.state)
        base_rainfall = max(base_rainfall, 100.0)  # avoid zero or negative

        baseline_traj = []
        counterfactual_traj = []
        
        # Simulate baseline (no intervention)
        gw_baseline = base_gw_level
        for month in range(1, months + 1):
            # Monthly rainfall variation (monsoon pattern)
            month_rainfall = base_rainfall * np.sin(month * np.pi / 6) / 12
            
            # Baseline: natural decline + rainfall recharge
            pumping_impact = -0.3  # Natural pumping impact per month
            rainfall_recharge = month_rainfall / 1000 * 0.5  # Convert to GW level change
            seasonal_trend = 0.1  # Slow improvement
            
            gw_baseline += pumping_impact + rainfall_recharge + seasonal_trend
            gw_baseline = float(np.clip(gw_baseline, 0, 100))
            
            baseline_traj.append(TrajectoryPoint(
                month=month,
                groundwater=gw_baseline,
                rainfall=float(month_rainfall)
            ))
        
        # Simulate counterfactual (with interventions)
        gw_counterfactual = base_gw_level
        for month in range(1, months + 1):
            # Monthly rainfall variation
            month_rainfall = base_rainfall * np.sin(month * np.pi / 6) / 12
            
            # Counterfactual with interventions
            # Reduced pumping impact due to reduced pumping
            pumping_change_factor = 1 - (params.pumping_change / 100.0)
            pumping_impact = -0.3 * pumping_change_factor
            
            # Rainfall recharge + recharge structures boost
            rainfall_recharge = month_rainfall / 1000 * 0.5
            recharge_boost = params.recharge_structures * 0.05  # Each structure adds benefit
            
            # Crop intensity reduction saves water
            crop_intensity_factor = 1 - (params.crop_intensity_change / 100.0)
            
            # Combined effect
            cf_change = (pumping_impact + rainfall_recharge + recharge_boost) * crop_intensity_factor + seasonal_trend
            
            gw_counterfactual += cf_change
            gw_counterfactual = float(np.clip(gw_counterfactual, 0, 100))
            
            counterfactual_traj.append(TrajectoryPoint(
                month=month,
                groundwater=gw_counterfactual,
                rainfall=float(month_rainfall)
            ))
        
        # Calculate treatment effects
        baseline_values = np.array([t.groundwater for t in baseline_traj])
        cf_values = np.array([t.groundwater for t in counterfactual_traj])
        
        mean_effect = float((cf_values - baseline_values).mean())
        final_effect = float(cf_values[-1] - baseline_values[-1])
        cumulative_effect = float((cf_values - baseline_values).sum())
        
        # Uncertainty based on intervention magnitude
        uncertainty = np.sqrt(
            (params.pumping_change ** 2 + 
             params.recharge_structures ** 2 + 
             params.crop_intensity_change ** 2) / 3
        ) / 100.0
        
        result = PolicySimulationResult(
            baseline_trajectory=baseline_traj,
            counterfactual_trajectory=counterfactual_traj,
            mean_effect=mean_effect,
            final_effect=final_effect,
            cumulative_effect=cumulative_effect,
            uncertainty_margin=float(uncertainty)
        )

        # Store intervention summary in database
        try:
            coll = get_policy_simulations_collection()
            doc = {
                "user_id": user_id,
                "params": {
                    "state": params.state,
                    "pumping_change": params.pumping_change,
                    "recharge_structures": params.recharge_structures,
                    "crop_intensity_change": params.crop_intensity_change,
                    "months_ahead": params.months_ahead,
                },
                "result": {
                    "mean_effect": result.mean_effect,
                    "final_effect": result.final_effect,
                    "cumulative_effect": result.cumulative_effect,
                    "uncertainty_margin": result.uncertainty_margin,
                },
                "baseline_trajectory": [{"month": t.month, "groundwater": t.groundwater, "rainfall": t.rainfall} for t in baseline_traj],
                "counterfactual_trajectory": [{"month": t.month, "groundwater": t.groundwater, "rainfall": t.rainfall} for t in counterfactual_traj],
                "created_at": datetime.utcnow(),
            }
            coll.insert_one(doc)
        except Exception as store_err:
            logger.error("Failed to persist policy simulation: %s", store_err)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Policy simulation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Policy simulation failed"
        )


@router.get("/history")
async def get_intervention_history(
    limit: int = 20,
    authorization: str = Header(None),
):
    """Get the current user's stored intervention summaries (most recent first)."""
    user_id = extract_user_id(authorization)
    try:
        coll = get_policy_simulations_collection()
        cursor = coll.find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=min(limit, 100),
        )
        items = []
        for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            items.append(doc)
        return {"interventions": items, "count": len(items)}
    except Exception as e:
        logger.error("Failed to load intervention history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load intervention history",
        )


@router.get("/export-pdf")
async def export_policy_pdf(
    intervention_id: str = Query(...),
    authorization: str = Header(None),
):
    """Export policy comparison as PDF."""
    from policy_pdf import generate_policy_comparison_pdf
    from bson.objectid import ObjectId
    user_id = extract_user_id(authorization)
    try:
        coll = get_policy_simulations_collection()
        doc = coll.find_one({"_id": ObjectId(intervention_id), "user_id": user_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Intervention not found")
        params = doc.get("params", {})
        result = doc.get("result", {})
        bl_traj = doc.get("baseline_trajectory", [])
        cf_traj = doc.get("counterfactual_trajectory", [])
        pdf_bytes = generate_policy_comparison_pdf(
            {"baseline_trajectory": bl_traj},
            {"counterfactual_trajectory": cf_traj},
            params,
            result,
        )
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=policy_comparison.pdf"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF export failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate PDF")


@router.get("/states")
async def get_available_states(authorization: str = Header(None)):
    """Get list of available states from groundwater data (title-cased)."""
    extract_user_id(authorization)  # Verify auth

    df = _get_groundwater_df()
    if df.empty:
        states = [
            "Maharashtra", "Haryana", "Punjab", "Uttar Pradesh",
            "Rajasthan", "Gujarat", "Madhya Pradesh", "Karnataka", "Tamil Nadu",
        ]
    else:
        states = sorted([str(s).title() for s in df["state_name"].unique()])

    return {"states": states, "count": len(states)}
