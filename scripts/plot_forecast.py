import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure backend modules are importable when running this script directly
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from graph_builder import DistrictGraphBuilder
from spatiotemporal_gnn import SpatioTemporalGNN
from data_preparation import load_and_prepare_data

def main(state="maharashtra"):
    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecast_with_uncertainty.png"

    # Load graph
    builder = DistrictGraphBuilder("../data/regions.geojson")
    builder.load_geojson()
    builder.build_adjacency_graph(method='knn', k=5)
    adj = torch.FloatTensor(builder.get_normalized_adjacency())

    # Data
    dataset = load_and_prepare_data(
        rainfall_path="../data/rainfall.csv",
        groundwater_path="../data/groundwater.csv",
        node_mapping=builder.node_to_idx,
        sequence_length=12,
        forecast_horizon=6
    )
    x, rainfall, targets = dataset.get_batch(dataset.test_data, batch_size=1, n_nodes=len(builder.node_to_idx))

    # Model
    checkpoint = torch.load("../models/gnn_model.pth", map_location="cpu", weights_only=False)
    model = SpatioTemporalGNN(
        n_nodes=len(builder.node_to_idx),
        n_features=3,
        hidden_dim=64,
        n_gnn_layers=2,
        n_heads=4,
        forecast_horizon=6,
        dropout=0.1,
        use_physics=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        mean, std, (lower, upper) = model.predict_with_uncertainty(
            x, adj, rainfall, n_samples=50
        )  # [B, N, H]

    # Pick this state's node index
    idx = builder.node_to_idx.get(state, 0)
    mean = mean.numpy()[0, idx, :]
    lower = lower.numpy()[0, idx, :]
    upper = upper.numpy()[0, idx, :]

    months = np.arange(1, mean.shape[0] + 1)
    plt.figure(figsize=(8,4))
    plt.fill_between(months, lower, upper, color='steelblue', alpha=0.2, label='95% CI')
    plt.plot(months, mean, color='steelblue', lw=2, label='Prediction (mean)')
    plt.xlabel("Months Ahead")
    plt.ylabel("Groundwater Level (m bgl)")
    plt.title(f"Forecast with Uncertainty - {state.title()}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()