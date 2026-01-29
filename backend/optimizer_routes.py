from fastapi import APIRouter, HTTPException, status, Header
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from database import get_users_collection
from auth_utils import verify_token
import numpy as np
import pandas as pd
import random

router = APIRouter(prefix="/api/optimizer", tags=["optimizer"])


class OptimizationRequest(BaseModel):
    state: str
    objectives: Optional[List[str]] = None  # ['impact', 'cost', 'equity', 'accessibility']
    max_budget: Optional[float] = None  # lakhs
    nl_query: Optional[str] = None
    n_sites: int = 10
    search_radius: float = 100  # km


class SiteInfo(BaseModel):
    id: str
    latitude: float
    longitude: float
    state: str
    district: Optional[str] = None
    impact_score: float
    cost_score: float
    accessibility_score: float
    equity_score: float
    total_score: float
    estimated_cost: float  # lakhs
    explanation: str


class OptimizationResponse(BaseModel):
    state: str
    selected_sites: List[SiteInfo]
    total_impact: float
    average_cost: float
    map_center: Dict[str, float]
    search_area_bounds: Dict[str, float]
    metadata: Dict[str, Any]


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


# State coordinates for map centering (latitude, longitude)
STATE_COORDINATES = {
    'maharashtra': {'lat': 19.7515, 'lon': 75.7139, 'bounds': {'north': 22.0, 'south': 16.5, 'east': 80.9, 'west': 72.6}},
    'haryana': {'lat': 29.0588, 'lon': 77.0745, 'bounds': {'north': 30.9, 'south': 27.7, 'east': 77.6, 'west': 74.4}},
    'punjab': {'lat': 31.1471, 'lon': 75.3412, 'bounds': {'north': 32.5, 'south': 29.6, 'east': 76.8, 'west': 73.5}},
    'uttar pradesh': {'lat': 26.8467, 'lon': 80.9462, 'bounds': {'north': 30.4, 'south': 23.8, 'east': 84.8, 'west': 77.0}},
    'rajasthan': {'lat': 27.0238, 'lon': 74.2179, 'bounds': {'north': 30.2, 'south': 23.8, 'east': 78.9, 'west': 68.8}},
    'gujarat': {'lat': 22.2587, 'lon': 71.1924, 'bounds': {'north': 24.7, 'south': 20.1, 'east': 74.4, 'west': 68.1}},
}


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_recharge_sites(
    request: OptimizationRequest,
    authorization: str = Header(None)
):
    """
    Find optimal recharge siting locations using multi-objective optimization
    """
    user_id = extract_user_id(authorization)
    
    try:
        state = request.state.lower().strip()
        
        if state not in STATE_COORDINATES:
            raise HTTPException(
                status_code=404,
                detail=f"State '{request.state}' not found. Available states: {list(STATE_COORDINATES.keys())}"
            )
        
        state_info = STATE_COORDINATES[state]
        
        # Generate candidate sites within state bounds
        n_candidates = min(100, request.n_sites * 15)
        candidates = []
        
        bounds = state_info['bounds']
        for i in range(n_candidates):
            lat = np.random.uniform(bounds['south'], bounds['north'])
            lon = np.random.uniform(bounds['west'], bounds['east'])
            candidates.append({
                'id': f'site_{i:03d}',
                'lat': float(lat),
                'lon': float(lon),
                'state': request.state
            })
        
        # Score each candidate
        scored_sites = []
        for site in candidates:
            # Impact score: higher in declining GW areas (simulated)
            impact_score = float(np.random.uniform(0.6, 0.99))
            
            # Cost score: varies by location (0-1, lower is better)
            cost_score = float(np.random.uniform(0.3, 0.9))
            cost_in_lakhs = 15 + cost_score * 30  # 15-45 lakhs
            
            # Accessibility score: higher near settlements (simulated)
            accessibility_score = float(np.random.uniform(0.5, 0.95))
            
            # Equity score: higher in underserved areas (simulated)
            equity_score = float(np.random.uniform(0.5, 0.95))
            
            # Apply objectives weights
            weights = {
                'impact': 2.0,
                'cost': 1.5,
                'accessibility': 1.0,
                'equity': 1.0
            }
            
            if request.objectives:
                weights = {k: (2.0 if k in request.objectives else 0.5) for k in weights.keys()}
            
            # Normalize cost score (invert so higher is better)
            cost_score_normalized = 1.0 - cost_score
            
            # Compute weighted total score
            total_score = (
                weights['impact'] * impact_score +
                weights['cost'] * cost_score_normalized +
                weights['accessibility'] * accessibility_score +
                weights['equity'] * equity_score
            ) / sum(weights.values())
            
            scored_sites.append({
                'site': site,
                'impact': impact_score,
                'cost': cost_score,
                'accessibility': accessibility_score,
                'equity': equity_score,
                'total': float(total_score),
                'cost_lakhs': cost_in_lakhs
            })
        
        # Apply budget constraint if specified
        if request.max_budget:
            scored_sites = [s for s in scored_sites if s['cost_lakhs'] <= request.max_budget]
        
        # Sort by total score and select top sites
        scored_sites.sort(key=lambda x: x['total'], reverse=True)
        selected = scored_sites[:request.n_sites]
        
        # Format response
        selected_sites = [
            SiteInfo(
                id=s['site']['id'],
                latitude=s['site']['lat'],
                longitude=s['site']['lon'],
                state=s['site']['state'],
                impact_score=s['impact'],
                cost_score=1.0 - s['cost'],  # Invert back for display (higher is better)
                accessibility_score=s['accessibility'],
                equity_score=s['equity'],
                total_score=s['total'],
                estimated_cost=s['cost_lakhs'],
                explanation=f"High-impact recharge site with {s['impact']*100:.1f}% potential. Cost: ₹{s['cost_lakhs']:.1f}L"
            )
            for s in selected
        ]
        
        # Calculate statistics
        total_impact = float(np.mean([s.impact_score for s in selected_sites]))
        average_cost = float(np.mean([s.estimated_cost for s in selected_sites]))
        
        return OptimizationResponse(
            state=request.state,
            selected_sites=selected_sites,
            total_impact=total_impact,
            average_cost=average_cost,
            map_center={'latitude': state_info['lat'], 'longitude': state_info['lon']},
            search_area_bounds={
                'north': bounds['north'],
                'south': bounds['south'],
                'east': bounds['east'],
                'west': bounds['west']
            },
            metadata={
                'n_candidates_evaluated': len(candidates),
                'n_sites_selected': len(selected_sites),
                'optimization_method': 'multi-objective weighted sum',
                'objectives': request.objectives or ['impact', 'cost', 'accessibility', 'equity']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.get("/states")
async def get_available_states(authorization: str = Header(None)):
    """Get list of available states for optimization"""
    extract_user_id(authorization)  # Verify auth
    
    states = list(STATE_COORDINATES.keys())
    return {
        "states": [s.title() for s in states],
        "count": len(states)
    }
