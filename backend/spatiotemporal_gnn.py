"""
Physics-Informed Spatiotemporal GNN for Groundwater Prediction
================================================================
Implements graph neural network with water balance constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TemporalGraphAttention(nn.Module):
    """Temporal Graph Attention Layer"""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads
        
        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"
        
        # Multi-head attention components
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.W_o = nn.Linear(out_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [batch, n_nodes, in_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Updated node features [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Mask attention to graph structure
        adj_mask = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, n_nodes, n_nodes]
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # [batch, n_heads, n_nodes, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, n_nodes, self.out_features)
        
        # Output projection
        out = self.W_o(out)
        out = self.layer_norm(out)
        
        return out


class SpatioTemporalGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network with Physics Constraints
    
    Architecture:
    - Temporal encoding (GRU/LSTM)
    - Graph attention layers
    - Physics-based regularization
    - Uncertainty estimation via MC Dropout
    """
    
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int = 64,
        n_gnn_layers: int = 2,
        n_heads: int = 4,
        forecast_horizon: int = 12,
        dropout: float = 0.1,
        use_physics: bool = True
    ):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_physics = use_physics
        
        # Temporal encoder (GRU for efficiency)
        self.temporal_encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            TemporalGraphAttention(
                in_features=hidden_dim if i > 0 else hidden_dim,
                out_features=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for i in range(n_gnn_layers)
        ])
        
        # Decoder for multi-step forecasting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon)
        )
        
        # Physics-based components
        if use_physics:
            # Learnable coefficients for water balance
            self.recharge_coef = nn.Parameter(torch.tensor(0.15))  # Recharge coefficient
            self.discharge_coef = nn.Parameter(torch.tensor(0.10))  # Discharge coefficient
            self.storage_coef = nn.Parameter(torch.tensor(0.80))   # Storage coefficient
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rainfall: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        
        Args:
            x: Input features [batch, seq_len, n_nodes, n_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            rainfall: Rainfall data for physics constraints [batch, seq_len, n_nodes]
            
        Returns:
            predictions: [batch, n_nodes, forecast_horizon]
            aux_outputs: Dictionary with auxiliary outputs (physics residuals, etc.)
        """
        batch_size, seq_len, n_nodes, n_features = x.shape
        
        # Reshape for temporal encoding: [batch * n_nodes, seq_len, n_features]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * n_nodes, seq_len, n_features)
        
        # Temporal encoding
        temporal_out, h_n = self.temporal_encoder(x_reshaped)
        
        # Take last hidden state: [batch * n_nodes, hidden_dim]
        h_last = temporal_out[:, -1, :]
        
        # Reshape back: [batch, n_nodes, hidden_dim]
        h = h_last.view(batch_size, n_nodes, self.hidden_dim)
        
        # Graph attention layers
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer(h, adj)
            h = h + h_new  # Residual connection
            h = self.dropout(h)
        
        # Decode to predictions
        predictions = self.decoder(h)  # [batch, n_nodes, forecast_horizon]
        
        # Auxiliary outputs
        aux_outputs = {}
        
        # Compute physics residuals if enabled
        if self.use_physics and rainfall is not None:
            physics_residuals = self.compute_physics_residuals(
                predictions, rainfall, x[:, -1, :, 0]  # Last observed GW level
            )
            aux_outputs['physics_residuals'] = physics_residuals
        
        return predictions, aux_outputs
    
    def compute_physics_residuals(
        self,
        gw_pred: torch.Tensor,
        rainfall: torch.Tensor,
        gw_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute water balance residuals for physics constraint
        
        Water balance: ΔS = Recharge - Discharge
        Where:
        - Recharge ≈ α * Rainfall
        - Discharge ≈ β * GW_level (proxy for pumping + baseflow)
        - ΔS = GW_pred - GW_current
        
        Args:
            gw_pred: Predicted GW levels [batch, n_nodes, horizon]
            rainfall: Rainfall data [batch, seq_len, n_nodes] (historical)
            gw_current: Current GW level [batch, n_nodes]
            
        Returns:
            Residuals [batch, n_nodes, horizon]
        """
        batch_size, n_nodes, horizon = gw_pred.shape
        
        # Use mean historical rainfall as proxy for future rainfall
        # rainfall shape: [batch, seq_len, n_nodes]
        if rainfall.dim() == 3:
            # Take mean over sequence length and expand to horizon
            rainfall_mean = rainfall.mean(dim=1)  # [batch, n_nodes]
            rainfall_expanded = rainfall_mean.unsqueeze(-1).expand(-1, -1, horizon)
        elif rainfall.dim() == 2:
            # Already [batch, n_nodes]
            rainfall_expanded = rainfall.unsqueeze(-1).expand(-1, -1, horizon)
        else:
            # Assume [batch, n_nodes, horizon]
            rainfall_expanded = rainfall
        
        # Compute components
        recharge = torch.abs(self.recharge_coef) * rainfall_expanded
        
        # For discharge, use predicted GW levels
        discharge = torch.abs(self.discharge_coef) * gw_pred
        
        # Storage change
        gw_current_expanded = gw_current.unsqueeze(-1).expand(-1, -1, horizon)
        delta_storage = gw_pred - gw_current_expanded
        
        # Water balance residual: ΔS - (Recharge - Discharge)
        residuals = delta_storage - (recharge - discharge)
        
        return residuals
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rainfall: Optional[torch.Tensor] = None,
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using MC Dropout
        
        Args:
            x: Input features
            adj: Adjacency matrix
            rainfall: Rainfall data
            n_samples: Number of MC samples
            
        Returns:
            mean_pred: Mean predictions [batch, n_nodes, horizon]
            std_pred: Standard deviation [batch, n_nodes, horizon]
            quantiles: (lower, upper) quantiles [batch, n_nodes, horizon]
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            pred, _ = self.forward(x, adj, rainfall)
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)  # [n_samples, batch, n_nodes, horizon]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Compute quantiles (90% confidence interval)
        lower_quantile = torch.quantile(predictions, 0.05, dim=0)
        upper_quantile = torch.quantile(predictions, 0.95, dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred, (lower_quantile, upper_quantile)


class PhysicsInformedLoss(nn.Module):
    """Loss function with physics constraints"""
    
    def __init__(self, lambda_physics: float = 0.1, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_residuals: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss
        
        Args:
            predictions: Predicted values
            targets: Ground truth
            physics_residuals: Water balance residuals
            adj: Adjacency matrix for spatial smoothness
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss terms
        """
        # Prediction loss (MSE)
        pred_loss = F.mse_loss(predictions, targets)
        
        loss_components = {'prediction_loss': pred_loss.item()}
        total_loss = pred_loss
        
        # Physics constraint loss
        if physics_residuals is not None:
            physics_loss = torch.mean(physics_residuals ** 2)
            total_loss = total_loss + self.lambda_physics * physics_loss
            loss_components['physics_loss'] = physics_loss.item()
        
        # Spatial smoothness loss (neighboring nodes should have similar predictions)
        if adj is not None and self.lambda_smooth > 0:
            batch_size, n_nodes, horizon = predictions.shape
            
            # Compute pairwise differences weighted by adjacency
            smooth_loss = 0
            for t in range(horizon):
                pred_t = predictions[:, :, t]  # [batch, n_nodes]
                
                # Compute differences for all pairs
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if adj[i, j] > 0:
                            diff = (pred_t[:, i] - pred_t[:, j]) ** 2
                            smooth_loss += adj[i, j] * diff.mean()
            
            smooth_loss = smooth_loss / (n_nodes * horizon)
            total_loss = total_loss + self.lambda_smooth * smooth_loss
            loss_components['smoothness_loss'] = smooth_loss.item()
        
        return total_loss, loss_components


if __name__ == "__main__":
    # Test model
    print("Testing Spatiotemporal GNN...")
    
    n_nodes = 10
    n_features = 3
    seq_len = 24
    batch_size = 4
    forecast_horizon = 12
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, n_nodes, n_features)
    adj = torch.rand(n_nodes, n_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = (adj > 0.5).float()  # Binarize
    rainfall = torch.randn(batch_size, seq_len, n_nodes)
    
    # Create model
    model = SpatioTemporalGNN(
        n_nodes=n_nodes,
        n_features=n_features,
        hidden_dim=64,
        n_gnn_layers=2,
        forecast_horizon=forecast_horizon,
        use_physics=True
    )
    
    # Forward pass
    predictions, aux = model(x, adj, rainfall)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Physics residuals shape: {aux['physics_residuals'].shape}")
    
    # Test uncertainty
    mean, std, (lower, upper) = model.predict_with_uncertainty(x, adj, rainfall, n_samples=10)
    print(f"Mean predictions shape: {mean.shape}")
    print(f"Std predictions shape: {std.shape}")
    
    print("\nModel test successful!")
