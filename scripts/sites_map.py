import requests
import folium
import numpy as np
from pathlib import Path

API = "http://localhost:8001"
OUT = Path("../models")
OUT.mkdir(parents=True, exist_ok=True)

def main(state="tamil nadu", nl_query="Find 10 high-impact sites with low cost", n_sites=10):
    resp = requests.post(f"{API}/api/recharge_sites", json={
        "state": state,
        "nl_query": nl_query,
        "n_sites": n_sites,
        "n_candidates": 100
    })
    resp.raise_for_status()
    data = resp.json()
    sites = data.get("selected_sites", [])
    if not sites:
        print("No sites returned.")
        return

    lats = [s["latitude"] for s in sites]
    lons = [s["longitude"] for s in sites]
    m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=6)

    for i, s in enumerate(sites):
        score = s["total_score"]
        color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"
        folium.CircleMarker(
            location=[s["latitude"], s["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Site {i+1}<br>Score: {score:.3f}"
        ).add_to(m)

    path = OUT / "sites_map.html"
    m.save(str(path))
    print(f"Saved: {path} (open in browser; screenshot if PNG needed)")

if __name__ == "__main__":
    main()