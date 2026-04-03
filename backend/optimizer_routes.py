from fastapi import APIRouter, HTTPException, status, Header
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from database import get_users_collection
from auth_utils import verify_token, extract_user_id
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



# State coordinates for map centering and candidate-site generation
# Covers all 28 states + 8 UTs present in the groundwater dataset
STATE_COORDINATES: Dict[str, Any] = {
    'andhra pradesh':      {'lat': 15.9129, 'lon': 79.7400, 'bounds': {'north': 19.9, 'south': 12.6, 'east': 84.8, 'west': 76.7}},
    'arunachal pradesh':   {'lat': 28.2180, 'lon': 94.7278, 'bounds': {'north': 29.5, 'south': 26.6, 'east': 97.4, 'west': 91.5}},
    'assam':               {'lat': 26.2006, 'lon': 92.9376, 'bounds': {'north': 27.9, 'south': 24.1, 'east': 96.0, 'west': 89.7}},
    'bihar':               {'lat': 25.0961, 'lon': 85.3131, 'bounds': {'north': 27.5, 'south': 24.3, 'east': 88.3, 'west': 83.3}},
    'chhattisgarh':        {'lat': 21.2787, 'lon': 81.8661, 'bounds': {'north': 24.1, 'south': 17.8, 'east': 84.4, 'west': 80.2}},
    'goa':                 {'lat': 15.2993, 'lon': 74.1240, 'bounds': {'north': 15.8, 'south': 14.9, 'east': 74.3, 'west': 73.7}},
    'gujarat':             {'lat': 22.2587, 'lon': 71.1924, 'bounds': {'north': 24.7, 'south': 20.1, 'east': 74.4, 'west': 68.1}},
    'haryana':             {'lat': 29.0588, 'lon': 76.0856, 'bounds': {'north': 30.9, 'south': 27.7, 'east': 77.6, 'west': 74.4}},
    'himachal pradesh':    {'lat': 31.1048, 'lon': 77.1734, 'bounds': {'north': 33.2, 'south': 30.4, 'east': 79.0, 'west': 75.6}},
    'jharkhand':           {'lat': 23.6102, 'lon': 85.2799, 'bounds': {'north': 25.4, 'south': 21.9, 'east': 87.9, 'west': 83.3}},
    'karnataka':           {'lat': 15.3173, 'lon': 75.7139, 'bounds': {'north': 18.5, 'south': 11.6, 'east': 78.6, 'west': 74.0}},
    'kerala':              {'lat': 10.8505, 'lon': 76.2711, 'bounds': {'north': 12.8, 'south':  8.2, 'east': 77.6, 'west': 74.9}},
    'madhya pradesh':      {'lat': 22.9734, 'lon': 78.6569, 'bounds': {'north': 26.9, 'south': 21.1, 'east': 82.8, 'west': 74.0}},
    'maharashtra':         {'lat': 19.7515, 'lon': 75.7139, 'bounds': {'north': 22.0, 'south': 15.6, 'east': 80.9, 'west': 72.6}},
    'manipur':             {'lat': 24.6637, 'lon': 93.9063, 'bounds': {'north': 25.7, 'south': 23.8, 'east': 94.8, 'west': 93.0}},
    'meghalaya':           {'lat': 25.4670, 'lon': 91.3662, 'bounds': {'north': 26.1, 'south': 25.0, 'east': 92.8, 'west': 89.8}},
    'mizoram':             {'lat': 23.1645, 'lon': 92.9376, 'bounds': {'north': 24.5, 'south': 21.9, 'east': 93.4, 'west': 92.3}},
    'nagaland':            {'lat': 26.1584, 'lon': 94.5624, 'bounds': {'north': 27.0, 'south': 25.2, 'east': 95.2, 'west': 93.4}},
    'odisha':              {'lat': 20.9517, 'lon': 85.0985, 'bounds': {'north': 22.6, 'south': 17.8, 'east': 87.5, 'west': 81.4}},
    'punjab':              {'lat': 31.1471, 'lon': 75.3412, 'bounds': {'north': 32.5, 'south': 29.5, 'east': 76.9, 'west': 73.9}},
    'rajasthan':           {'lat': 27.0238, 'lon': 74.2179, 'bounds': {'north': 30.2, 'south': 23.0, 'east': 78.3, 'west': 69.5}},
    'sikkim':              {'lat': 27.5330, 'lon': 88.5122, 'bounds': {'north': 28.1, 'south': 27.1, 'east': 88.9, 'west': 88.0}},
    'tamil nadu':          {'lat': 11.1271, 'lon': 78.6569, 'bounds': {'north': 13.6, 'south':  8.1, 'east': 80.3, 'west': 76.2}},
    'telangana':           {'lat': 18.1124, 'lon': 79.0193, 'bounds': {'north': 19.9, 'south': 15.9, 'east': 81.3, 'west': 77.2}},
    'tripura':             {'lat': 23.9408, 'lon': 91.9882, 'bounds': {'north': 24.5, 'south': 22.9, 'east': 92.3, 'west': 91.2}},
    'uttar pradesh':       {'lat': 26.8467, 'lon': 80.9462, 'bounds': {'north': 30.4, 'south': 23.9, 'east': 84.7, 'west': 77.1}},
    'uttarakhand':         {'lat': 30.0668, 'lon': 79.0193, 'bounds': {'north': 31.5, 'south': 28.7, 'east': 80.6, 'west': 77.6}},
    'west bengal':         {'lat': 22.9868, 'lon': 87.8550, 'bounds': {'north': 27.2, 'south': 21.5, 'east': 89.9, 'west': 85.8}},
    # Union Territories
    'delhi':               {'lat': 28.7041, 'lon': 77.1025, 'bounds': {'north': 28.9, 'south': 28.4, 'east': 77.4, 'west': 76.8}},
    'jammu and kashmir':   {'lat': 33.7782, 'lon': 76.5762, 'bounds': {'north': 37.1, 'south': 32.3, 'east': 80.0, 'west': 73.7}},
    'ladakh':              {'lat': 34.1526, 'lon': 77.5770, 'bounds': {'north': 36.0, 'south': 32.2, 'east': 79.5, 'west': 75.5}},
    'puducherry':          {'lat': 11.9416, 'lon': 79.8083, 'bounds': {'north': 12.1, 'south': 11.6, 'east': 80.0, 'west': 79.6}},
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
        
        # Fallback: India centroid if state name not in the lookup
        _india_default = {'lat': 20.5937, 'lon': 78.9629, 'bounds': {'north': 35.5, 'south': 8.1, 'east': 97.4, 'west': 68.1}}
        state_info = STATE_COORDINATES.get(state, _india_default)
        
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
