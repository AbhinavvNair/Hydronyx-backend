"""
Test script for graph builder
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from graph_builder import DistrictGraphBuilder
import numpy as np

def test_graph_builder():
    """Test graph construction and validation"""
    
    print("=" * 60)
    print("Testing District Graph Builder")
    print("=" * 60)
    
    # Initialize builder
    geojson_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'regions.geojson')
    builder = DistrictGraphBuilder(geojson_path)
    
    # Load GeoJSON
    print("\n1. Loading GeoJSON...")
    gdf = builder.load_geojson()
    print(f"   ✓ Loaded {len(gdf)} regions")
    print(f"   States: {gdf['state_name'].unique()[:5]}... ({len(gdf['state_name'].unique())} total)")
    
    # Build geometric adjacency graph
    print("\n2. Building geometric adjacency graph...")
    G = builder.build_adjacency_graph(method='geometric')
    print(f"   ✓ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Validate graph
    print("\n3. Validating graph...")
    stats = builder.validate_graph()
    
    print(f"   Nodes: {stats['n_nodes']}")
    print(f"   Edges: {stats['n_edges']}")
    print(f"   Connected: {stats['is_connected']}")
    print(f"   Components: {stats['n_components']}")
    print(f"   Avg Degree: {stats['avg_degree']:.2f}")
    print(f"   Density: {stats['density']:.4f}")
    print(f"   Isolated Nodes: {stats['n_isolated']}")
    
    if stats['n_isolated'] > 0:
        print(f"   ⚠ Warning: {stats['n_isolated']} isolated nodes found:")
        for node in stats['isolated_nodes'][:5]:
            print(f"      - {node}")
    
    # Get adjacency matrix
    print("\n4. Computing adjacency matrix...")
    A = builder.get_adjacency_matrix(weighted=True)
    print(f"   ✓ Adjacency matrix shape: {A.shape}")
    print(f"   Non-zero entries: {np.count_nonzero(A)}")
    print(f"   Sparsity: {1 - np.count_nonzero(A) / A.size:.4f}")
    
    # Get normalized adjacency
    print("\n5. Computing normalized adjacency...")
    A_norm = builder.get_normalized_adjacency()
    print(f"   ✓ Normalized adjacency shape: {A_norm.shape}")
    print(f"   Eigenvalue range: [{np.linalg.eigvals(A_norm).min():.3f}, {np.linalg.eigvals(A_norm).max():.3f}]")
    
    # Get Laplacian
    print("\n6. Computing graph Laplacian...")
    L = builder.get_laplacian(normalized=True)
    print(f"   ✓ Laplacian shape: {L.shape}")
    eigenvalues = np.linalg.eigvals(L)
    print(f"   Eigenvalue range: [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    
    # Save graph
    print("\n7. Saving graph...")
    output_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'district_graph.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    builder.save_graph(output_path)
    print(f"   ✓ Graph saved to {output_path}")
    
    # Test k-NN graph
    print("\n8. Building k-NN graph (k=5)...")
    G_knn = builder.build_adjacency_graph(method='knn', k=5)
    print(f"   ✓ k-NN graph: {G_knn.number_of_nodes()} nodes, {G_knn.number_of_edges()} edges")
    
    stats_knn = builder.validate_graph()
    print(f"   Connected: {stats_knn['is_connected']}")
    print(f"   Avg Degree: {stats_knn['avg_degree']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return builder, stats


if __name__ == "__main__":
    builder, stats = test_graph_builder()
