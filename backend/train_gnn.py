"""
Training Script for Physics-Informed Spatiotemporal GNN
========================================================
Trains the GNN model on historical groundwater data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict

from graph_builder import DistrictGraphBuilder
from spatiotemporal_gnn import SpatioTemporalGNN, PhysicsInformedLoss
from data_preparation import load_and_prepare_data


class GNNTrainer:
    """Trainer for spatiotemporal GNN"""
    
    def __init__(
        self,
        model: SpatioTemporalGNN,
        dataset,
        adj_matrix: torch.Tensor,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        lambda_physics: float = 0.1,
        lambda_smooth: float = 0.01
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.adj_matrix = adj_matrix.to(device)
        self.device = device
        
        # Loss and optimizer
        self.criterion = PhysicsInformedLoss(
            lambda_physics=lambda_physics,
            lambda_smooth=lambda_smooth
        )
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pred_loss': [],
            'train_physics_loss': [],
            'val_pred_loss': []
        }
    
    def train_epoch(self, batch_size: int = 8) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        n_batches = len(self.dataset.train_data) // batch_size
        epoch_losses = []
        epoch_pred_losses = []
        epoch_physics_losses = []
        
        for _ in range(n_batches):
            # Get batch
            x, rainfall, targets = self.dataset.get_batch(
                self.dataset.train_data,
                batch_size,
                len(self.dataset.node_mapping)
            )
            
            x = x.to(self.device)
            rainfall = rainfall.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, aux = self.model(x, self.adj_matrix, rainfall)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                predictions,
                targets,
                physics_residuals=aux.get('physics_residuals'),
                adj=self.adj_matrix
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            epoch_losses.append(loss.item())
            epoch_pred_losses.append(loss_dict.get('prediction_loss', 0))
            epoch_physics_losses.append(loss_dict.get('physics_loss', 0))
        
        return {
            'loss': np.mean(epoch_losses),
            'pred_loss': np.mean(epoch_pred_losses),
            'physics_loss': np.mean(epoch_physics_losses)
        }
    
    def validate(self, batch_size: int = 8) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        n_batches = max(1, len(self.dataset.val_data) // batch_size)
        val_losses = []
        val_pred_losses = []
        
        with torch.no_grad():
            for _ in range(n_batches):
                # Get batch
                x, rainfall, targets = self.dataset.get_batch(
                    self.dataset.val_data,
                    batch_size,
                    len(self.dataset.node_mapping)
                )
                
                x = x.to(self.device)
                rainfall = rainfall.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions, aux = self.model(x, self.adj_matrix, rainfall)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    predictions,
                    targets,
                    physics_residuals=aux.get('physics_residuals'),
                    adj=self.adj_matrix
                )
                
                val_losses.append(loss.item())
                val_pred_losses.append(loss_dict.get('prediction_loss', 0))
        
        return {
            'loss': np.mean(val_losses),
            'pred_loss': np.mean(val_pred_losses)
        }
    
    def train(
        self,
        n_epochs: int = 100,
        batch_size: int = 8,
        patience: int = 10,
        save_path: str = "../models/gnn_model.pth"
    ):
        """
        Train model with early stopping
        
        Args:
            n_epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            save_path: Path to save best model
        """
        print(f"Training GNN for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Lambda physics: {self.criterion.lambda_physics}")
        print(f"Lambda smooth: {self.criterion.lambda_smooth}")
        print()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(batch_size)
            
            # Validate
            val_metrics = self.validate(batch_size)
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_pred_loss'].append(train_metrics['pred_loss'])
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['val_pred_loss'].append(val_metrics['pred_loss'])
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f} "
                      f"(Pred: {train_metrics['pred_loss']:.4f}, "
                      f"Physics: {train_metrics['physics_loss']:.4f})")
                print(f"  Val Loss: {val_metrics['loss']:.4f} "
                      f"(Pred: {val_metrics['pred_loss']:.4f})")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': self.history
                }, save_path)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {save_path}")
    
    def plot_history(self, save_path: str = "../models/training_history.png"):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Component losses
        axes[1].plot(self.history['train_pred_loss'], label='Train Pred Loss', alpha=0.8)
        axes[1].plot(self.history['train_physics_loss'], label='Train Physics Loss', alpha=0.8)
        axes[1].plot(self.history['val_pred_loss'], label='Val Pred Loss', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Component Losses')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.close()


def main():
    """Main training function"""
    print("=" * 60)
    print("Physics-Informed Spatiotemporal GNN Training")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        'sequence_length': 12,
        'forecast_horizon': 6,
        'hidden_dim': 64,
        'n_gnn_layers': 2,
        'n_heads': 4,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'lambda_physics': 0.1,
        'lambda_smooth': 0.01,
        'batch_size': 8,
        'n_epochs': 100,
        'patience': 15,
        'use_physics': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 1: Build graph
    print("Step 1: Building district graph...")
    builder = DistrictGraphBuilder("../data/regions.geojson")
    builder.load_geojson()
    
    # Use k-NN to ensure connectivity
    G = builder.build_adjacency_graph(method='knn', k=5)
    stats = builder.validate_graph()
    
    print(f"  Nodes: {stats['n_nodes']}")
    print(f"  Edges: {stats['n_edges']}")
    print(f"  Connected: {stats['is_connected']}")
    print()
    
    # Get adjacency matrix
    adj_matrix = torch.FloatTensor(builder.get_normalized_adjacency())
    
    # Step 2: Prepare data
    print("Step 2: Preparing data...")
    dataset = load_and_prepare_data(
        rainfall_path="../data/rainfall.csv",
        groundwater_path="../data/groundwater.csv",
        node_mapping=builder.node_to_idx,
        sequence_length=config['sequence_length'],
        forecast_horizon=config['forecast_horizon']
    )
    print()
    
    # Step 3: Initialize model
    print("Step 3: Initializing model...")
    model = SpatioTemporalGNN(
        n_nodes=stats['n_nodes'],
        n_features=3,  # gw_level, rainfall, gw_lag
        hidden_dim=config['hidden_dim'],
        n_gnn_layers=config['n_gnn_layers'],
        n_heads=config['n_heads'],
        forecast_horizon=config['forecast_horizon'],
        dropout=config['dropout'],
        use_physics=config['use_physics']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print()
    
    # Step 4: Train model
    print("Step 4: Training model...")
    trainer = GNNTrainer(
        model=model,
        dataset=dataset,
        adj_matrix=adj_matrix,
        device=config['device'],
        learning_rate=config['learning_rate'],
        lambda_physics=config['lambda_physics'],
        lambda_smooth=config['lambda_smooth']
    )
    
    trainer.train(
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        patience=config['patience'],
        save_path="../models/gnn_model.pth"
    )
    print()
    
    # Step 5: Plot training history
    print("Step 5: Plotting training history...")
    trainer.plot_history()
    print()
    
    # Step 6: Save configuration
    print("Step 6: Saving configuration...")
    config_path = "../models/gnn_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Configuration saved to: {config_path}")
    print()
    
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Check training_history.png for loss curves")
    print("  2. Load model in api_advanced.py for inference")
    print("  3. Test predictions with uncertainty")


if __name__ == "__main__":
    main()
