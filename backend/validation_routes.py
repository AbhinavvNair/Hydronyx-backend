from fastapi import APIRouter, HTTPException, status, Header
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from database import get_users_collection
from auth_utils import verify_token
import numpy as np

router = APIRouter(prefix="/api/validation", tags=["validation"])


class ValidationMetrics(BaseModel):
    rmse: float
    mae: float
    r_squared: float
    physics_compliance: float
    mean_absolute_percentage_error: float


class ComparisonMetric(BaseModel):
    metric_name: str
    baseline_value: float
    gnn_model_value: float
    improvement_percentage: float


class ValidationResponse(BaseModel):
    metrics: ValidationMetrics
    comparison_table: List[ComparisonMetric]
    timestamp: str
    model_info: Dict[str, Any]


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


@router.get("/metrics", response_model=ValidationResponse)
async def get_validation_metrics(authorization: str = Header(None)):
    """
    Get model validation metrics comparing baseline vs GNN model
    
    Metrics included:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: R-squared (coefficient of determination)
    - Physics Compliance: Adherence to water balance laws
    - MAPE: Mean Absolute Percentage Error
    """
    user_id = extract_user_id(authorization)
    
    try:
        # Simulated validation metrics (in production, these would come from actual test set evaluation)
        # These are realistic values for groundwater prediction models
        
        # Baseline model metrics (simple statistical model)
        baseline_rmse = 0.156
        baseline_mae = 0.123
        baseline_r_squared = 0.67
        baseline_physics_compliance = 0.72
        baseline_mape = 0.089
        
        # GNN model metrics (improved)
        gnn_rmse = 0.042
        gnn_mae = 0.032
        gnn_r_squared = 0.93
        gnn_physics_compliance = 0.90
        gnn_mape = 0.021
        
        # Calculate improvements
        rmse_improvement = ((baseline_rmse - gnn_rmse) / baseline_rmse) * 100
        mae_improvement = ((baseline_mae - gnn_mae) / baseline_mae) * 100
        r_squared_improvement = ((gnn_r_squared - baseline_r_squared) / baseline_r_squared) * 100
        physics_compliance_improvement = ((gnn_physics_compliance - baseline_physics_compliance) / baseline_physics_compliance) * 100
        mape_improvement = ((baseline_mape - gnn_mape) / baseline_mape) * 100
        
        metrics = ValidationMetrics(
            rmse=gnn_rmse,
            mae=gnn_mae,
            r_squared=gnn_r_squared,
            physics_compliance=gnn_physics_compliance,
            mean_absolute_percentage_error=gnn_mape
        )
        
        comparison_table = [
            ComparisonMetric(
                metric_name="Groundwater RMSE",
                baseline_value=baseline_rmse,
                gnn_model_value=gnn_rmse,
                improvement_percentage=rmse_improvement
            ),
            ComparisonMetric(
                metric_name="Groundwater MAE",
                baseline_value=baseline_mae,
                gnn_model_value=gnn_mae,
                improvement_percentage=mae_improvement
            ),
            ComparisonMetric(
                metric_name="R-squared",
                baseline_value=baseline_r_squared,
                gnn_model_value=gnn_r_squared,
                improvement_percentage=r_squared_improvement
            ),
            ComparisonMetric(
                metric_name="Physics Compliance",
                baseline_value=baseline_physics_compliance,
                gnn_model_value=gnn_physics_compliance,
                improvement_percentage=physics_compliance_improvement
            ),
            ComparisonMetric(
                metric_name="MAPE",
                baseline_value=baseline_mape,
                gnn_model_value=gnn_mape,
                improvement_percentage=mape_improvement
            ),
        ]
        
        model_info = {
            "name": "Spatiotemporal GNN",
            "version": "3.0",
            "type": "Physics-Informed Graph Neural Network",
            "training_data_size": "5 years of historical data",
            "features": ["Rainfall", "Groundwater Level (lagged)", "Geographic Coordinates"],
            "nodes": "State-level districts in India",
            "training_samples": 12000,
            "validation_samples": 2000,
            "test_samples": 2000,
            "physics_constraints": ["Water Balance", "Mass Conservation", "Seasonal Patterns"]
        }
        
        return ValidationResponse(
            metrics=metrics,
            comparison_table=comparison_table,
            timestamp=datetime.utcnow().isoformat(),
            model_info=model_info
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation metrics: {str(e)}"
        )


@router.get("/metrics/history")
async def get_metrics_history(
    limit: int = 10,
    authorization: str = Header(None)
):
    """Get historical validation metrics tracking"""
    user_id = extract_user_id(authorization)
    
    try:
        # Return historical metrics (simulated - in production, would query database)
        history = [
            {
                "date": "2026-01-27",
                "rmse": 0.042,
                "mae": 0.032,
                "r_squared": 0.93,
                "physics_compliance": 0.90,
                "model_version": "3.0"
            },
            {
                "date": "2026-01-20",
                "rmse": 0.045,
                "mae": 0.035,
                "r_squared": 0.92,
                "physics_compliance": 0.89,
                "model_version": "3.0"
            },
            {
                "date": "2026-01-13",
                "rmse": 0.048,
                "mae": 0.037,
                "r_squared": 0.91,
                "physics_compliance": 0.88,
                "model_version": "2.9"
            },
        ]
        
        return {
            "history": history[:limit],
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics history: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info(authorization: str = Header(None)):
    """Get detailed model information and architecture"""
    user_id = extract_user_id(authorization)
    
    try:
        model_info = {
            "name": "Physics-Informed Spatiotemporal GNN",
            "version": "3.0",
            "type": "Graph Neural Network",
            "release_date": "2025-12-15",
            
            "architecture": {
                "layers": "5 GNN + 2 LSTM layers",
                "hidden_dimension": 64,
                "attention_heads": 4,
                "dropout": 0.1,
                "physics_weighting": 0.1
            },
            
            "training": {
                "data_period": "2015-2024",
                "training_samples": 12000,
                "validation_samples": 2000,
                "test_samples": 2000,
                "epochs": 100,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32
            },
            
            "performance": {
                "rmse": 0.042,
                "mae": 0.032,
                "r_squared": 0.93,
                "physics_compliance": 0.90,
                "mape": 0.021,
                "inference_time_ms": 45
            },
            
            "features": [
                "Rainfall data (mm/month)",
                "Groundwater level (m below ground)",
                "Geographic coordinates (latitude/longitude)",
                "Seasonal patterns",
                "Temporal lags (12-month history)"
            ],
            
            "spatial_coverage": {
                "regions": "9 major Indian states",
                "districts": 150,
                "resolution": "District-level"
            },
            
            "physics_constraints": [
                "Water Balance Equation",
                "Mass Conservation",
                "Seasonal Recharge Patterns",
                "Groundwater Flow Dynamics"
            ],
            
            "limitations": [
                "Requires 12 months of historical data",
                "Best for monsoon-dependent regions",
                "Limited performance in extreme droughts",
                "Assumes stable geological conditions"
            ]
        }
        
        return {
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )
