"""
Structural Causal Model for Groundwater Policy Counterfactuals
================================================================
Implements SCM for estimating causal effects of interventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings


@dataclass
class CausalVariable:
    """Represents a variable in the causal model"""
    name: str
    var_type: str  # 'exogenous', 'endogenous', 'intervention'
    parents: List[str]
    structural_eq: Optional[Callable] = None
    noise_dist: str = 'normal'  # 'normal', 'uniform', etc.
    noise_params: Dict = None


class StructuralCausalModel:
    """
    Structural Causal Model for Groundwater System
    
    Variables:
    - Rainfall (exogenous)
    - Pumping intensity (endogenous, can be intervened)
    - Recharge efforts (endogenous, can be intervened)
    - Crop mix (endogenous, can be intervened)
    - Groundwater level (endogenous, outcome)
    """
    
    def __init__(self):
        self.variables: Dict[str, CausalVariable] = {}
        self.data: Optional[pd.DataFrame] = None
        self.fitted_equations: Dict[str, any] = {}
        self.topological_order: List[str] = []
        
    def add_variable(self, variable: CausalVariable):
        """Add a variable to the causal model"""
        self.variables[variable.name] = variable
        
    def define_default_groundwater_scm(self):
        """Define default SCM structure for groundwater"""
        
        # Exogenous: Rainfall
        self.add_variable(CausalVariable(
            name='rainfall',
            var_type='exogenous',
            parents=[],
            noise_dist='normal',
            noise_params={'mean': 0, 'std': 10}
        ))
        
        # Endogenous: Pumping (depends on rainfall, previous GW)
        self.add_variable(CausalVariable(
            name='pumping',
            var_type='endogenous',
            parents=['rainfall', 'gw_lag'],
            noise_dist='normal',
            noise_params={'mean': 0, 'std': 5}
        ))
        
        # Endogenous: Recharge efforts (policy variable)
        self.add_variable(CausalVariable(
            name='recharge',
            var_type='intervention',
            parents=['rainfall'],
            noise_dist='normal',
            noise_params={'mean': 0, 'std': 2}
        ))
        
        # Endogenous: Crop mix (irrigation intensity proxy)
        self.add_variable(CausalVariable(
            name='crop_intensity',
            var_type='intervention',
            parents=['rainfall', 'gw_lag'],
            noise_dist='normal',
            noise_params={'mean': 0, 'std': 3}
        ))
        
        # Outcome: Groundwater level
        self.add_variable(CausalVariable(
            name='groundwater',
            var_type='endogenous',
            parents=['rainfall', 'pumping', 'recharge', 'crop_intensity', 'gw_lag'],
            noise_dist='normal',
            noise_params={'mean': 0, 'std': 1}
        ))
        
        # Compute topological order
        self._compute_topological_order()
        
    def _compute_topological_order(self):
        """Compute topological ordering of variables"""
        # Simple topological sort using Kahn's algorithm
        in_degree = {var: 0 for var in self.variables}
        
        for var_name, var in self.variables.items():
            for parent in var.parents:
                if parent in in_degree:
                    in_degree[var_name] += 1
        
        queue = [var for var, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            var = queue.pop(0)
            order.append(var)
            
            # Find children
            for child_name, child in self.variables.items():
                if var in child.parents:
                    in_degree[child_name] -= 1
                    if in_degree[child_name] == 0:
                        queue.append(child_name)
        
        self.topological_order = order
        
    def fit(self, data: pd.DataFrame, method: str = 'linear'):
        """
        Fit structural equations from data
        
        Args:
            data: DataFrame with columns matching variable names
            method: 'linear', 'ridge', or 'rf' (random forest)
        """
        self.data = data.copy()
        
        # Fit each endogenous variable
        for var_name in self.topological_order:
            var = self.variables[var_name]
            
            if var.var_type == 'exogenous':
                continue
            
            # Get parent data
            if not var.parents:
                continue
            
            # Filter parents that exist in data
            available_parents = [p for p in var.parents if p in data.columns]
            
            if not available_parents:
                warnings.warn(f"No parent data available for {var_name}")
                continue
            
            X = data[available_parents].values
            y = data[var_name].values
            
            # Remove NaN rows
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                warnings.warn(f"No valid data for {var_name}")
                continue
            
            # Fit model
            if method == 'linear':
                model = LinearRegression()
            elif method == 'ridge':
                model = Ridge(alpha=1.0)
            elif method == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            model.fit(X, y)
            self.fitted_equations[var_name] = {
                'model': model,
                'parents': available_parents,
                'residual_std': np.std(y - model.predict(X))
            }
            
    def predict(self, data: pd.DataFrame, var_name: str) -> np.ndarray:
        """Predict values for a variable"""
        if var_name not in self.fitted_equations:
            raise ValueError(f"No fitted equation for {var_name}")
        
        eq = self.fitted_equations[var_name]
        X = data[eq['parents']].values
        
        return eq['model'].predict(X)
    
    def intervene(
        self,
        data: pd.DataFrame,
        interventions: Dict[str, float],
        n_samples: int = 1
    ) -> pd.DataFrame:
        """
        Perform do-calculus intervention
        
        Args:
            data: Observational data
            interventions: Dict mapping variable names to intervention values
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with counterfactual outcomes
        """
        # Start with observational data
        result = data.copy()
        
        # Apply interventions (set values directly)
        for var_name, value in interventions.items():
            if var_name in result.columns:
                result[var_name] = value
        
        # Propagate through causal graph
        for var_name in self.topological_order:
            var = self.variables[var_name]
            
            # Skip exogenous and intervened variables
            if var.var_type == 'exogenous' or var_name in interventions:
                continue
            
            # Predict based on parents
            if var_name in self.fitted_equations:
                eq = self.fitted_equations[var_name]
                
                # Check if all parents are available
                if all(p in result.columns for p in eq['parents']):
                    X = result[eq['parents']].values
                    pred = eq['model'].predict(X)
                    
                    # Add noise if generating samples
                    if n_samples > 1:
                        noise = np.random.normal(0, eq['residual_std'], size=pred.shape)
                        pred = pred + noise
                    
                    result[var_name] = pred
        
        return result
    
    def estimate_ate(
        self,
        data: pd.DataFrame,
        intervention_var: str,
        intervention_values: Tuple[float, float],
        outcome_var: str = 'groundwater',
        n_bootstrap: int = 100
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect (ATE)
        
        Args:
            data: Observational data
            intervention_var: Variable to intervene on
            intervention_values: (control_value, treatment_value)
            outcome_var: Outcome variable
            n_bootstrap: Number of bootstrap samples for uncertainty
            
        Returns:
            Dictionary with ATE, standard error, and confidence interval
        """
        control_val, treatment_val = intervention_values
        
        # Counterfactual under control
        cf_control = self.intervene(data, {intervention_var: control_val})
        
        # Counterfactual under treatment
        cf_treatment = self.intervene(data, {intervention_var: treatment_val})
        
        # ATE
        ate = (cf_treatment[outcome_var] - cf_control[outcome_var]).mean()
        
        # Bootstrap for uncertainty
        ates = []
        for _ in range(n_bootstrap):
            # Resample data
            sample_idx = np.random.choice(len(data), size=len(data), replace=True)
            data_boot = data.iloc[sample_idx]
            
            # Compute ATE on bootstrap sample
            cf_control_boot = self.intervene(data_boot, {intervention_var: control_val})
            cf_treatment_boot = self.intervene(data_boot, {intervention_var: treatment_val})
            
            ate_boot = (cf_treatment_boot[outcome_var] - cf_control_boot[outcome_var]).mean()
            ates.append(ate_boot)
        
        ates = np.array(ates)
        
        return {
            'ate': ate,
            'std_error': np.std(ates),
            'ci_lower': np.percentile(ates, 2.5),
            'ci_upper': np.percentile(ates, 97.5)
        }
    
    def counterfactual_trajectory(
        self,
        initial_state: pd.DataFrame,
        interventions: Dict[str, float],
        n_steps: int = 12,
        exogenous_forecast: Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Generate counterfactual trajectory over time
        
        Args:
            initial_state: Initial state (single row DataFrame)
            interventions: Interventions to apply
            n_steps: Number of time steps to forecast
            exogenous_forecast: Forecasted exogenous variables (e.g., rainfall)
            
        Returns:
            DataFrame with trajectory
        """
        trajectory = []
        current_state = initial_state.copy()
        
        for t in range(n_steps):
            # Update exogenous variables if forecast provided
            if exogenous_forecast:
                for var, values in exogenous_forecast.items():
                    if var in current_state.columns and t < len(values):
                        current_state[var] = values[t]
            
            # Apply intervention
            next_state = self.intervene(current_state, interventions, n_samples=1)
            
            # Update lag variable
            if 'gw_lag' in next_state.columns and 'groundwater' in next_state.columns:
                next_state['gw_lag'] = next_state['groundwater'].values
            
            trajectory.append(next_state.copy())
            current_state = next_state
        
        return pd.concat(trajectory, ignore_index=True)


if __name__ == "__main__":
    # Test SCM
    print("Testing Structural Causal Model...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'rainfall': np.random.normal(100, 30, n_samples),
        'gw_lag': np.random.normal(5, 2, n_samples)
    })
    
    # Generate endogenous variables
    data['pumping'] = 10 - 0.05 * data['rainfall'] + 0.3 * data['gw_lag'] + np.random.normal(0, 2, n_samples)
    data['recharge'] = 0.1 * data['rainfall'] + np.random.normal(0, 1, n_samples)
    data['crop_intensity'] = 50 - 0.02 * data['rainfall'] + 0.2 * data['gw_lag'] + np.random.normal(0, 3, n_samples)
    
    # Groundwater (outcome)
    data['groundwater'] = (
        0.8 * data['gw_lag'] +
        0.05 * data['rainfall'] -
        0.15 * data['pumping'] +
        0.10 * data['recharge'] -
        0.05 * data['crop_intensity'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Create and fit SCM
    scm = StructuralCausalModel()
    scm.define_default_groundwater_scm()
    scm.fit(data, method='linear')
    
    print(f"Fitted {len(scm.fitted_equations)} structural equations")
    
    # Test intervention
    test_data = data.head(10)
    
    # Baseline (no intervention)
    baseline = scm.intervene(test_data, {})
    
    # Intervention: increase recharge by 50%
    intervention = scm.intervene(test_data, {'recharge': test_data['recharge'].mean() * 1.5})
    
    print(f"\nBaseline GW mean: {baseline['groundwater'].mean():.2f}")
    print(f"Intervention GW mean: {intervention['groundwater'].mean():.2f}")
    print(f"Effect: {(intervention['groundwater'] - baseline['groundwater']).mean():.2f}")
    
    # Estimate ATE
    ate_results = scm.estimate_ate(
        data,
        intervention_var='recharge',
        intervention_values=(data['recharge'].mean(), data['recharge'].mean() * 1.5),
        n_bootstrap=50
    )
    
    print(f"\nAverage Treatment Effect:")
    print(f"  ATE: {ate_results['ate']:.3f}")
    print(f"  95% CI: [{ate_results['ci_lower']:.3f}, {ate_results['ci_upper']:.3f}]")
    
    print("\nSCM test successful!")
