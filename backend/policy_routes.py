from fastapi import APIRouter, HTTPException, status, Header
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from database import get_users_collection
from auth_utils import verify_token
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api/policy", tags=["policy"])


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


def extract_user_id(authorization: str) -> str:
    """Extract user_id from Bearer token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization scheme"
            )
        
        user_email = verify_token(token)
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        users_collection = get_users_collection()
        user = users_collection.find_one({"email": user_email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return str(user["_id"])
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}"
        )


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
        # Physics-informed simulation
        months = params.months_ahead
        
        # Base parameters (typical for Indian groundwater systems)
        base_gw_level = 45.0  # meters below ground level
        base_rainfall = 800.0  # mm/year
        
        # Initial values
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
        
        return PolicySimulationResult(
            baseline_trajectory=baseline_traj,
            counterfactual_trajectory=counterfactual_traj,
            mean_effect=mean_effect,
            final_effect=final_effect,
            cumulative_effect=cumulative_effect,
            uncertainty_margin=float(uncertainty)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy simulation failed: {str(e)}"
        )


@router.get("/states")
async def get_available_states(authorization: str = Header(None)):
    """Get list of available states for policy simulation"""
    extract_user_id(authorization)  # Verify auth
    
    # Available states in the system
    states = [
        "Maharashtra",
        "Haryana",
        "Punjab",
        "Uttar Pradesh",
        "Rajasthan",
        "Gujarat",
        "Madhya Pradesh",
        "Karnataka",
        "Tamil Nadu"
    ]
    
    return {
        "states": states,
        "count": len(states)
    }
