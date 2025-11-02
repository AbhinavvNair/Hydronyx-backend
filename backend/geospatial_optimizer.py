"""
Conversational Geospatial Optimization for Recharge Siting
===========================================================
Converts natural language objectives to multi-objective optimization.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from scipy.spatial import distance_matrix
from shapely.geometry import Point
import json


@dataclass
class OptimizationObjective:
    """Represents an optimization objective"""
    name: str
    weight: float
    maximize: bool
    description: str


@dataclass
class OptimizationConstraint:
    """Represents an optimization constraint"""
    name: str
    constraint_type: str  # 'min', 'max', 'range', 'categorical'
    value: any
    description: str


@dataclass
class CandidateSite:
    """Represents a candidate recharge site"""
    id: str
    lat: float
    lon: float
    state: str
    district: Optional[str]
    scores: Dict[str, float]
    total_score: float
    explanation: str


class NLObjectiveParser:
    """Parse natural language to structured objectives"""
    
    def __init__(self):
        self.objective_keywords = {
            'impact': ['impact', 'effectiveness', 'benefit', 'recharge potential'],
            'cost': ['cost', 'budget', 'expense', 'cheap', 'affordable'],
            'equity': ['equity', 'fairness', 'distribution', 'equal', 'underserved'],
            'accessibility': ['access', 'accessibility', 'reachable', 'proximity', 'distance']
        }
        
        self.constraint_keywords = {
            'budget': ['budget', 'cost limit', 'maximum cost'],
            'distance': ['distance', 'proximity', 'near', 'far from'],
            'slope': ['slope', 'gradient', 'flat', 'terrain'],
            'protected': ['protected', 'conservation', 'restricted']
        }
        
    def parse(self, nl_query: str) -> Tuple[List[OptimizationObjective], List[OptimizationConstraint]]:
        """
        Parse natural language query to objectives and constraints
        
        Args:
            nl_query: Natural language description
            
        Returns:
            (objectives, constraints)
        """
        nl_lower = nl_query.lower()
        
        objectives = []
        constraints = []
        
        # Parse objectives
        for obj_name, keywords in self.objective_keywords.items():
            if any(kw in nl_lower for kw in keywords):
                # Determine weight from context
                weight = 1.0
                
                # Look for weight indicators
                if 'prioritize' in nl_lower or 'focus on' in nl_lower or 'maximize' in nl_lower:
                    if any(kw in nl_lower for kw in keywords):
                        weight = 2.0
                
                # Determine if maximize or minimize
                maximize = True
                if obj_name == 'cost':
                    maximize = False  # Minimize cost
                
                objectives.append(OptimizationObjective(
                    name=obj_name,
                    weight=weight,
                    maximize=maximize,
                    description=f"{'Maximize' if maximize else 'Minimize'} {obj_name}"
                ))
        
        # Parse constraints
        # Budget constraint
        budget_match = re.search(r'budget.*?(\d+(?:\.\d+)?)\s*(lakh|crore|million)?', nl_lower)
        if budget_match:
            amount = float(budget_match.group(1))
            unit = budget_match.group(2) or 'lakh'
            
            # Convert to lakhs
            if unit == 'crore':
                amount *= 100
            elif unit == 'million':
                amount *= 10
            
            constraints.append(OptimizationConstraint(
                name='budget',
                constraint_type='max',
                value=amount,
                description=f"Maximum budget: {amount} lakhs"
            ))
        
        # Distance constraint
        distance_match = re.search(r'(?:within|less than|at most)\s+(\d+(?:\.\d+)?)\s*(km|meter|m)', nl_lower)
        if distance_match:
            dist = float(distance_match.group(1))
            unit = distance_match.group(2)
            
            if unit in ['meter', 'm']:
                dist /= 1000  # Convert to km
            
            constraints.append(OptimizationConstraint(
                name='max_distance_to_settlement',
                constraint_type='max',
                value=dist,
                description=f"Maximum distance to settlement: {dist} km"
            ))
        
        # Default objectives if none found
        if not objectives:
            objectives = [
                OptimizationObjective('impact', 1.0, True, "Maximize recharge impact"),
                OptimizationObjective('cost', 1.0, False, "Minimize cost"),
                OptimizationObjective('equity', 0.5, True, "Maximize equity"),
                OptimizationObjective('accessibility', 0.5, True, "Maximize accessibility")
            ]
        
        return objectives, constraints


class GeospatialOptimizer:
    """
    Multi-objective optimization for recharge site selection
    """
    
    def __init__(self, regions_gdf: gpd.GeoDataFrame):
        """
        Initialize optimizer
        
        Args:
            regions_gdf: GeoDataFrame with district/state boundaries
        """
        self.regions_gdf = regions_gdf
        self.parser = NLObjectiveParser()
        
    def generate_candidate_sites(
        self,
        state: str,
        n_candidates: int = 100,
        method: str = 'grid'
    ) -> List[Dict]:
        """
        Generate candidate sites within a state
        
        Args:
            state: State name
            n_candidates: Number of candidate sites
            method: 'grid' or 'random'
            
        Returns:
            List of candidate site dictionaries
        """
        # Get state geometry
        state_rows = self.regions_gdf[
            self.regions_gdf['state_name'] == state.lower()
        ]
        
        if state_rows.empty:
            # Fallback: use first available state or default bounds
            if not self.regions_gdf.empty:
                state_geom = self.regions_gdf.geometry.iloc[0]
            else:
                # Use default India bounds if no geometry available
                from shapely.geometry import box
                state_geom = box(68, 8, 97, 35)  # Approximate India bounds
        else:
            state_geom = state_rows.geometry.iloc[0]
        
        bounds = state_geom.bounds  # (minx, miny, maxx, maxy)
        
        candidates = []
        
        if method == 'grid':
            # Generate grid
            n_grid = int(np.sqrt(n_candidates))
            lons = np.linspace(bounds[0], bounds[2], n_grid)
            lats = np.linspace(bounds[1], bounds[3], n_grid)
            
            for lon in lons:
                for lat in lats:
                    point = Point(lon, lat)
                    if state_geom.contains(point):
                        candidates.append({
                            'id': f"site_{len(candidates)}",
                            'lon': lon,
                            'lat': lat,
                            'state': state,
                            'geometry': point
                        })
        
        elif method == 'random':
            # Random sampling within bounds
            attempts = 0
            while len(candidates) < n_candidates and attempts < n_candidates * 10:
                lon = np.random.uniform(bounds[0], bounds[2])
                lat = np.random.uniform(bounds[1], bounds[3])
                point = Point(lon, lat)
                
                if state_geom.contains(point):
                    candidates.append({
                        'id': f"site_{len(candidates)}",
                        'lon': lon,
                        'lat': lat,
                        'state': state,
                        'geometry': point
                    })
                
                attempts += 1
        
        return candidates
    
    def compute_impact_score(
        self,
        site: Dict,
        groundwater_data: pd.DataFrame
    ) -> float:
        """
        Compute recharge impact score for a site
        
        Higher score = more impact (e.g., areas with declining GW)
        
        Args:
            site: Candidate site
            groundwater_data: Historical groundwater data
            
        Returns:
            Impact score (0-1)
        """
        # Simple heuristic: impact is higher where GW is declining
        # In practice, use hydrogeological models
        
        # Get nearby GW data (within some radius)
        # For now, use state-level average
        state_data = groundwater_data[
            groundwater_data['state_name'] == site['state'].lower()
        ]
        
        if state_data.empty:
            return 0.5  # Default
        
        # Compute trend (negative trend = declining GW = higher impact)
        recent_data = state_data.sort_values('year_month').tail(24)
        if len(recent_data) > 1:
            gw_levels = recent_data['gw_level_m_bgl'].values
            trend = np.polyfit(range(len(gw_levels)), gw_levels, 1)[0]
            
            # Normalize: positive trend (declining GW) -> higher score
            impact = 1.0 / (1.0 + np.exp(-trend * 10))  # Sigmoid
        else:
            impact = 0.5
        
        return impact
    
    def compute_cost_score(self, site: Dict) -> float:
        """
        Estimate cost score (lower is better)
        
        Args:
            site: Candidate site
            
        Returns:
            Cost score (normalized 0-1, lower is better)
        """
        # Simple heuristic based on terrain, accessibility
        # In practice, use detailed cost models
        
        # Random for now (would use terrain, land cost, etc.)
        base_cost = np.random.uniform(10, 50)  # Lakhs
        
        # Normalize to 0-1
        cost_score = base_cost / 100.0
        
        return cost_score
    
    def compute_equity_score(
        self,
        site: Dict,
        existing_sites: List[Dict]
    ) -> float:
        """
        Compute equity score (higher = better distribution)
        
        Args:
            site: Candidate site
            existing_sites: List of existing recharge sites
            
        Returns:
            Equity score (0-1)
        """
        if not existing_sites:
            return 1.0  # No existing sites, all equal
        
        # Compute distance to nearest existing site
        site_point = np.array([site['lon'], site['lat']])
        existing_points = np.array([[s['lon'], s['lat']] for s in existing_sites])
        
        distances = np.linalg.norm(existing_points - site_point, axis=1)
        min_distance = distances.min()
        
        # Higher distance = better equity
        equity = min(min_distance / 1.0, 1.0)  # Normalize by 1 degree (~111km)
        
        return equity
    
    def compute_accessibility_score(self, site: Dict) -> float:
        """
        Compute accessibility score
        
        Args:
            site: Candidate site
            
        Returns:
            Accessibility score (0-1)
        """
        # Simple heuristic
        # In practice, use road network, settlement proximity
        
        # Random for now
        accessibility = np.random.uniform(0.3, 1.0)
        
        return accessibility
    
    def optimize(
        self,
        state: str,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        groundwater_data: pd.DataFrame,
        n_candidates: int = 100,
        n_select: int = 10,
        existing_sites: Optional[List[Dict]] = None
    ) -> List[CandidateSite]:
        """
        Run multi-objective optimization
        
        Args:
            state: State name
            objectives: List of objectives
            constraints: List of constraints
            groundwater_data: Historical GW data
            n_candidates: Number of candidate sites to evaluate
            n_select: Number of sites to select
            existing_sites: Existing recharge sites
            
        Returns:
            List of selected sites with scores
        """
        if existing_sites is None:
            existing_sites = []
        
        # Generate candidates
        candidates = self.generate_candidate_sites(state, n_candidates)
        
        # Score each candidate
        scored_sites = []
        
        for site in candidates:
            scores = {}
            
            # Compute objective scores
            scores['impact'] = self.compute_impact_score(site, groundwater_data)
            scores['cost'] = self.compute_cost_score(site)
            scores['equity'] = self.compute_equity_score(site, existing_sites)
            scores['accessibility'] = self.compute_accessibility_score(site)
            
            # Check constraints
            satisfies_constraints = True
            
            for constraint in constraints:
                if constraint.name == 'budget':
                    # Estimate site cost
                    site_cost = scores['cost'] * 100  # Convert to lakhs
                    if site_cost > constraint.value:
                        satisfies_constraints = False
                        break
            
            if not satisfies_constraints:
                continue
            
            # Compute weighted total score
            total_score = 0.0
            for obj in objectives:
                if obj.name in scores:
                    score = scores[obj.name]
                    
                    # Invert if minimizing
                    if not obj.maximize:
                        score = 1.0 - score
                    
                    total_score += obj.weight * score
            
            # Normalize by total weight
            total_weight = sum(obj.weight for obj in objectives)
            total_score /= total_weight
            
            # Generate explanation
            explanation = self._generate_explanation(site, scores, objectives)
            
            scored_sites.append(CandidateSite(
                id=site['id'],
                lat=site['lat'],
                lon=site['lon'],
                state=site['state'],
                district=None,
                scores=scores,
                total_score=total_score,
                explanation=explanation
            ))
        
        # Select top N sites
        scored_sites.sort(key=lambda x: x.total_score, reverse=True)
        selected = scored_sites[:n_select]
        
        return selected
    
    def _generate_explanation(
        self,
        site: Dict,
        scores: Dict[str, float],
        objectives: List[OptimizationObjective]
    ) -> str:
        """Generate human-readable explanation for site selection"""
        
        explanations = []
        
        for obj in objectives:
            if obj.name in scores:
                score = scores[obj.name]
                
                if obj.name == 'impact':
                    if score > 0.7:
                        explanations.append("High recharge impact potential")
                    elif score > 0.4:
                        explanations.append("Moderate recharge impact")
                    else:
                        explanations.append("Lower impact area")
                
                elif obj.name == 'cost':
                    if score < 0.3:
                        explanations.append("Low cost site")
                    elif score < 0.6:
                        explanations.append("Moderate cost")
                    else:
                        explanations.append("Higher cost site")
                
                elif obj.name == 'equity':
                    if score > 0.7:
                        explanations.append("Good geographic distribution")
                    else:
                        explanations.append("Near existing sites")
                
                elif obj.name == 'accessibility':
                    if score > 0.7:
                        explanations.append("Easily accessible")
                    else:
                        explanations.append("Limited accessibility")
        
        return "; ".join(explanations)
    
    def optimize_from_nl(
        self,
        nl_query: str,
        state: str,
        groundwater_data: pd.DataFrame,
        n_select: int = 10
    ) -> Tuple[List[CandidateSite], List[OptimizationObjective], List[OptimizationConstraint]]:
        """
        Run optimization from natural language query
        
        Args:
            nl_query: Natural language description
            state: State name
            groundwater_data: GW data
            n_select: Number of sites to select
            
        Returns:
            (selected_sites, objectives, constraints)
        """
        # Parse NL query
        objectives, constraints = self.parser.parse(nl_query)
        
        # Run optimization
        selected = self.optimize(
            state=state,
            objectives=objectives,
            constraints=constraints,
            groundwater_data=groundwater_data,
            n_select=n_select
        )
        
        return selected, objectives, constraints


if __name__ == "__main__":
    # Test optimizer
    print("Testing Geospatial Optimizer...")
    
    # Create dummy GeoDataFrame
    from shapely.geometry import box
    
    regions = gpd.GeoDataFrame({
        'state_name': ['maharashtra', 'karnataka'],
        'geometry': [
            box(73, 15, 80, 22),  # Maharashtra approx
            box(74, 11, 78, 18)   # Karnataka approx
        ]
    })
    
    # Create dummy GW data
    gw_data = pd.DataFrame({
        'state_name': ['maharashtra'] * 24,
        'year_month': pd.date_range('2020-01', periods=24, freq='M').astype(str),
        'gw_level_m_bgl': np.linspace(5, 7, 24) + np.random.normal(0, 0.5, 24)
    })
    
    # Create optimizer
    optimizer = GeospatialOptimizer(regions)
    
    # Test NL parsing
    nl_query = "Find sites with maximum impact and minimum cost, budget 50 lakh, within 10 km of settlements"
    
    selected, objectives, constraints = optimizer.optimize_from_nl(
        nl_query=nl_query,
        state='maharashtra',
        groundwater_data=gw_data,
        n_select=5
    )
    
    print(f"\nParsed {len(objectives)} objectives and {len(constraints)} constraints")
    print("\nObjectives:")
    for obj in objectives:
        print(f"  - {obj.name}: weight={obj.weight}, maximize={obj.maximize}")
    
    print("\nConstraints:")
    for const in constraints:
        print(f"  - {const.name}: {const.description}")
    
    print(f"\nSelected {len(selected)} sites:")
    for i, site in enumerate(selected[:3]):
        print(f"\n  Site {i+1}:")
        print(f"    Location: ({site.lat:.4f}, {site.lon:.4f})")
        print(f"    Total Score: {site.total_score:.3f}")
        print(f"    Explanation: {site.explanation}")
    
    print("\nOptimizer test successful!")
