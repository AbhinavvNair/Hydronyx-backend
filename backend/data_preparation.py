"""
Data Preparation for Spatiotemporal GNN Training
=================================================
Prepares sequences of groundwater and rainfall data for GNN training.
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SpatiotemporalDataset:
    """Dataset for spatiotemporal GNN training"""
    
    def __init__(
        self,
        rainfall_df: pd.DataFrame,
        groundwater_df: pd.DataFrame,
        node_mapping: Dict[str, int],
        sequence_length: int = 24,
        forecast_horizon: int = 12,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Initialize dataset
        
        Args:
            rainfall_df: Rainfall data
            groundwater_df: Groundwater data
            node_mapping: Mapping from state names to node indices
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        self.rainfall_df = rainfall_df.copy()
        self.groundwater_df = groundwater_df.copy()
        self.node_mapping = node_mapping
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Clean data
        self._clean_data()
        
        # Merge data
        self.merged_df = self._merge_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Split data
        self.train_data, self.val_data, self.test_data = self._split_data()
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Fit scalers on training data
        self._fit_scalers()
        
    def _clean_data(self):
        """Clean and standardize data"""
        # Standardize state names
        self.rainfall_df['state_name'] = self.rainfall_df['state_name'].str.strip().str.lower()
        self.groundwater_df['state_name'] = self.groundwater_df['state_name'].str.strip().str.lower()
        
        # Convert year_month to datetime
        self.rainfall_df['date'] = pd.to_datetime(self.rainfall_df['year_month'], format='%Y-%m', errors='coerce')
        self.groundwater_df['date'] = pd.to_datetime(self.groundwater_df['year_month'], format='%Y-%m', errors='coerce')
        
        # Drop rows with invalid dates
        self.rainfall_df = self.rainfall_df.dropna(subset=['date'])
        self.groundwater_df = self.groundwater_df.dropna(subset=['date'])
        
        # Sort by date
        self.rainfall_df = self.rainfall_df.sort_values(['state_name', 'date'])
        self.groundwater_df = self.groundwater_df.sort_values(['state_name', 'date'])
        
    def _merge_data(self) -> pd.DataFrame:
        """Merge rainfall and groundwater data"""
        merged = pd.merge(
            self.groundwater_df[['state_name', 'date', 'gw_level_m_bgl']],
            self.rainfall_df[['state_name', 'date', 'rainfall_actual_mm']],
            on=['state_name', 'date'],
            how='inner'
        )
        
        # Add lag features
        merged = merged.sort_values(['state_name', 'date'])
        merged['gw_lag_1'] = merged.groupby('state_name')['gw_level_m_bgl'].shift(1)
        merged['rainfall_lag_1'] = merged.groupby('state_name')['rainfall_actual_mm'].shift(1)
        
        # Drop NaN
        merged = merged.dropna()
        
        return merged
    
    def _create_sequences(self) -> List[Dict]:
        """Create sequences for training"""
        sequences = []
        
        for state in self.merged_df['state_name'].unique():
            if state not in self.node_mapping:
                continue
            
            state_data = self.merged_df[self.merged_df['state_name'] == state].sort_values('date')
            
            if len(state_data) < self.sequence_length + self.forecast_horizon:
                continue
            
            # Create sliding windows
            for i in range(len(state_data) - self.sequence_length - self.forecast_horizon + 1):
                seq_data = state_data.iloc[i:i + self.sequence_length]
                target_data = state_data.iloc[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                
                sequences.append({
                    'state': state,
                    'node_idx': self.node_mapping[state],
                    'features': seq_data[['gw_level_m_bgl', 'rainfall_actual_mm', 'gw_lag_1']].values,
                    'rainfall': seq_data['rainfall_actual_mm'].values,
                    'target': target_data['gw_level_m_bgl'].values,
                    'target_rainfall': target_data['rainfall_actual_mm'].values,
                    'date': seq_data['date'].iloc[-1]
                })
        
        return sequences
    
    def _split_data(self) -> Tuple[List, List, List]:
        """Split data into train/val/test"""
        n_samples = len(self.sequences)
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        train_data = self.sequences[:n_train]
        val_data = self.sequences[n_train:n_train + n_val]
        test_data = self.sequences[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def _fit_scalers(self):
        """Fit scalers on training data"""
        # Collect all training features and targets
        train_features = np.vstack([seq['features'] for seq in self.train_data])
        train_targets = np.concatenate([seq['target'] for seq in self.train_data])
        
        self.feature_scaler.fit(train_features)
        self.target_scaler.fit(train_targets.reshape(-1, 1))
    
    def get_batch(self, data: List[Dict], batch_size: int, n_nodes: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a batch of data
        
        Args:
            data: List of sequences
            batch_size: Batch size
            n_nodes: Total number of nodes in graph
            
        Returns:
            (x, rainfall, targets) as tensors
        """
        # Sample batch
        if len(data) < batch_size:
            batch_indices = np.arange(len(data))
        else:
            batch_indices = np.random.choice(len(data), batch_size, replace=False)
        
        batch = [data[i] for i in batch_indices]
        
        # Initialize tensors
        x = torch.zeros(batch_size, self.sequence_length, n_nodes, 3)
        rainfall = torch.zeros(batch_size, self.sequence_length, n_nodes)
        targets = torch.zeros(batch_size, n_nodes, self.forecast_horizon)
        
        # Fill tensors
        for i, seq in enumerate(batch):
            node_idx = seq['node_idx']
            
            # Scale features
            features_scaled = self.feature_scaler.transform(seq['features'])
            x[i, :, node_idx, :] = torch.FloatTensor(features_scaled)
            
            # Rainfall (not scaled, used for physics)
            rainfall[i, :, node_idx] = torch.FloatTensor(seq['rainfall'])
            
            # Scale targets
            targets_scaled = self.target_scaler.transform(seq['target'].reshape(-1, 1)).flatten()
            targets[i, node_idx, :] = torch.FloatTensor(targets_scaled)
        
        return x, rainfall, targets
    
    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets"""
        original_shape = targets_scaled.shape
        targets_flat = targets_scaled.reshape(-1, 1)
        targets_original = self.target_scaler.inverse_transform(targets_flat)
        return targets_original.reshape(original_shape)


def load_and_prepare_data(
    rainfall_path: str,
    groundwater_path: str,
    node_mapping: Dict[str, int],
    sequence_length: int = 24,
    forecast_horizon: int = 12
) -> SpatiotemporalDataset:
    """
    Load and prepare data for training
    
    Args:
        rainfall_path: Path to rainfall CSV
        groundwater_path: Path to groundwater CSV
        node_mapping: Mapping from state names to node indices
        sequence_length: Input sequence length
        forecast_horizon: Forecast horizon
        
    Returns:
        SpatiotemporalDataset instance
    """
    rainfall_df = pd.read_csv(rainfall_path)
    groundwater_df = pd.read_csv(groundwater_path)
    
    dataset = SpatiotemporalDataset(
        rainfall_df=rainfall_df,
        groundwater_df=groundwater_df,
        node_mapping=node_mapping,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )
    
    print(f"Dataset prepared:")
    print(f"  Total sequences: {len(dataset.sequences)}")
    print(f"  Train: {len(dataset.train_data)}")
    print(f"  Val: {len(dataset.val_data)}")
    print(f"  Test: {len(dataset.test_data)}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Forecast horizon: {forecast_horizon}")
    
    return dataset


if __name__ == "__main__":
    # Test data preparation
    from graph_builder import DistrictGraphBuilder
    
    # Build graph
    builder = DistrictGraphBuilder("../data/regions.geojson")
    builder.load_geojson()
    builder.build_adjacency_graph(method='knn', k=5)
    
    # Prepare data
    dataset = load_and_prepare_data(
        rainfall_path="../data/rainfall.csv",
        groundwater_path="../data/groundwater.csv",
        node_mapping=builder.node_to_idx,
        sequence_length=12,
        forecast_horizon=6
    )
    
    # Test batch generation
    x, rainfall, targets = dataset.get_batch(dataset.train_data, batch_size=4, n_nodes=len(builder.node_to_idx))
    print(f"\nBatch shapes:")
    print(f"  x: {x.shape}")
    print(f"  rainfall: {rainfall.shape}")
    print(f"  targets: {targets.shape}")
