"""
Graph Builder for Spatiotemporal GNN
=====================================
Builds district-level adjacency graph from GeoJSON boundaries.
Supports both geometric adjacency and k-nearest neighbors.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point
from scipy.spatial import distance_matrix
from typing import Dict, List, Tuple, Optional
import networkx as nx
import pickle


class DistrictGraphBuilder:
    """Build and manage district adjacency graph for spatiotemporal modeling"""
    
    def __init__(self, geojson_path: str):
        """
        Initialize graph builder
        
        Args:
            geojson_path: Path to regions.geojson file
        """
        self.geojson_path = geojson_path
        self.gdf = None
        self.graph = None
        self.node_features = {}
        self.node_to_idx = {}
        self.idx_to_node = {}
        
    def load_geojson(self) -> gpd.GeoDataFrame:
        """Load and parse GeoJSON file"""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        # Convert to GeoDataFrame
        features = []
        for feature in data['features']:
            geom = shape(feature['geometry'])
            props = feature['properties']
            features.append({
                'geometry': geom,
                'state_name': props.get('state_name', '').lower().strip(),
                'feature_id': props.get('feature_id', -1)
            })
        
        self.gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        return self.gdf
    
    def build_adjacency_graph(self, method: str = 'geometric', k: int = 5) -> nx.Graph:
        """
        Build adjacency graph between districts/states
        
        Args:
            method: 'geometric' (touching boundaries) or 'knn' (k-nearest neighbors)
            k: Number of neighbors for knn method
            
        Returns:
            NetworkX graph with district nodes and adjacency edges
        """
        if self.gdf is None:
            self.load_geojson()
        
        G = nx.Graph()
        
        # Add nodes with centroids
        for idx, row in self.gdf.iterrows():
            node_id = f"{row['state_name']}"
            centroid = row['geometry'].centroid
            
            G.add_node(node_id, 
                      state_name=row['state_name'],
                      feature_id=row['feature_id'],
                      lon=centroid.x,
                      lat=centroid.y,
                      area=row['geometry'].area,
                      idx=idx)
            
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id
        
        # Add edges based on method
        if method == 'geometric':
            # Add edges for geometrically adjacent regions
            for i, row_i in self.gdf.iterrows():
                node_i = f"{row_i['state_name']}"
                for j, row_j in self.gdf.iterrows():
                    if i >= j:
                        continue
                    
                    # Check if geometries touch or intersect
                    if row_i['geometry'].touches(row_j['geometry']) or \
                       row_i['geometry'].intersects(row_j['geometry']):
                        node_j = f"{row_j['state_name']}"
                        
                        # Calculate edge weight (inverse distance between centroids)
                        dist = row_i['geometry'].centroid.distance(row_j['geometry'].centroid)
                        weight = 1.0 / (dist + 1e-6)
                        
                        G.add_edge(node_i, node_j, weight=weight, distance=dist)
        
        elif method == 'knn':
            # Build k-nearest neighbor graph based on centroids
            centroids = np.array([[row['geometry'].centroid.x, row['geometry'].centroid.y] 
                                 for _, row in self.gdf.iterrows()])
            
            # Compute distance matrix
            dist_matrix = distance_matrix(centroids, centroids)
            
            # For each node, connect to k nearest neighbors
            for i in range(len(self.gdf)):
                node_i = self.idx_to_node[i]
                
                # Get k nearest neighbors (excluding self)
                nearest_indices = np.argsort(dist_matrix[i])[1:k+1]
                
                for j in nearest_indices:
                    node_j = self.idx_to_node[j]
                    dist = dist_matrix[i, j]
                    weight = 1.0 / (dist + 1e-6)
                    
                    G.add_edge(node_i, node_j, weight=weight, distance=dist)
        
        self.graph = G
        return G
    
    def get_adjacency_matrix(self, weighted: bool = True) -> np.ndarray:
        """
        Get adjacency matrix representation
        
        Args:
            weighted: If True, use edge weights; otherwise binary adjacency
            
        Returns:
            Adjacency matrix as numpy array
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_adjacency_graph() first.")
        
        n_nodes = len(self.graph.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Create contiguous index mapping
        node_list = list(self.graph.nodes())
        node_to_contiguous_idx = {node: i for i, node in enumerate(node_list)}
        
        for node_i, node_j, data in self.graph.edges(data=True):
            i = node_to_contiguous_idx[node_i]
            j = node_to_contiguous_idx[node_j]
            
            if weighted:
                weight = data.get('weight', 1.0)
            else:
                weight = 1.0
            
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Symmetric
        
        return adj_matrix
    
    def get_normalized_adjacency(self) -> np.ndarray:
        """
        Get normalized adjacency matrix (D^{-1/2} A D^{-1/2})
        Used in GCN-style graph convolutions
        
        Returns:
            Normalized adjacency matrix
        """
        A = self.get_adjacency_matrix(weighted=True)
        
        # Add self-loops
        A_hat = A + np.eye(A.shape[0])
        
        # Compute degree matrix
        D = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        
        # Normalize: D^{-1/2} A D^{-1/2}
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        return A_norm
    
    def get_laplacian(self, normalized: bool = True) -> np.ndarray:
        """
        Get graph Laplacian matrix
        
        Args:
            normalized: If True, return normalized Laplacian
            
        Returns:
            Laplacian matrix
        """
        A = self.get_adjacency_matrix(weighted=True)
        D = np.diag(np.sum(A, axis=1))
        
        if normalized:
            # Handle isolated nodes (degree = 0) by using pseudo-inverse
            d = np.sum(A, axis=1)
            d_inv_sqrt = np.zeros_like(d)
            
            # Only compute inverse for non-zero degrees
            non_zero = d > 1e-10
            d_inv_sqrt[non_zero] = 1.0 / np.sqrt(d[non_zero])
            
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            L = D - A
        
        return L
    
    def validate_graph(self) -> Dict[str, any]:
        """
        Validate graph properties and return statistics
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        stats = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'is_connected': nx.is_connected(self.graph),
            'n_components': nx.number_connected_components(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'density': nx.density(self.graph),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None
        }
        
        # Find isolated nodes
        isolated = list(nx.isolates(self.graph))
        stats['isolated_nodes'] = isolated
        stats['n_isolated'] = len(isolated)
        
        return stats
    
    def save_graph(self, output_path: str):
        """Save graph to file"""
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self, input_path: str):
        """Load graph from file"""
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Rebuild node mappings
        for node, data in self.graph.nodes(data=True):
            idx = data['idx']
            self.node_to_idx[node] = idx
            self.idx_to_node[idx] = node


if __name__ == "__main__":
    # Test graph builder
    builder = DistrictGraphBuilder("../data/regions.geojson")
    builder.load_geojson()
    
    print("Building geometric adjacency graph...")
    G = builder.build_adjacency_graph(method='geometric')
    
    stats = builder.validate_graph()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get adjacency matrix
    A = builder.get_adjacency_matrix(weighted=True)
    print(f"\nAdjacency matrix shape: {A.shape}")
    print(f"Non-zero entries: {np.count_nonzero(A)}")
    
    # Save graph
    builder.save_graph("../models/district_graph.pkl")
    print("\nGraph saved to ../models/district_graph.pkl")
