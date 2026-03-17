import requests

# Test without auth to see if the data loading works
response = requests.post(
    "http://localhost:8000/api/location/groundwater",
    json={
        "latitude": 26.9124,
        "longitude": 75.7873,
        "months_ahead": 12,
        "k": 8,
        "power": 2.0
    },
    headers={"Authorization": "Bearer test-token"}
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Current Level: {data['current_level_m_bgl']:.1f} m bgl")
    print(f"Confidence: {data['confidence']}")
    print(f"Trend: {data['trend_m_per_month']:.3f} m/month")
    print(f"Stations used: {len(data['nearest_stations'])}")
    print("Top 3 stations:")
    for station in data['nearest_stations'][:3]:
        print(f"  {station['station_name']}: {station['distance_km']:.1f} km, GWL: {station['gw_latest']:.1f} m")
else:
    print(f"Error: {response.text}")
