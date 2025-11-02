import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("../models")
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = pd.read_csv("../data/rainfall.csv")
    gw = pd.read_csv("../data/groundwater.csv")
    rf["state_name"] = rf["state_name"].str.strip().str.lower()
    gw["state_name"] = gw["state_name"].str.strip().str.lower()

    top_rf = rf.groupby("state_name")["rainfall_actual_mm"].mean().sort_values(ascending=False).head(10)[::-1]
    plt.figure(figsize=(8,4.5))
    plt.barh(top_rf.index, top_rf.values, color="#99cfff")
    plt.xlabel("Rainfall (mm)")
    plt.title("Top 10 States by Average Rainfall")
    plt.tight_layout()
    plt.savefig(out_dir/"top10_rainfall.png", dpi=150)

    top_gw = gw.groupby("state_name")["gw_level_m_bgl"].mean().sort_values(ascending=False).head(10)[::-1]
    plt.figure(figsize=(8,4.5))
    plt.barh(top_gw.index, top_gw.values, color="#ff8a80")
    plt.xlabel("GW Level (m bgl)")
    plt.title("Top 10 States by Average Groundwater Level")
    plt.tight_layout()
    plt.savefig(out_dir/"top10_groundwater.png", dpi=150)

if __name__ == "__main__":
    main()