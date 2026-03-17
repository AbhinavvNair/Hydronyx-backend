from backend.data_loader import estimate_gwl

# Test estimation
gwl, wells = estimate_gwl(26.9124, 75.7873, k=5)
print(f'Estimated GWL: {gwl} m bgl')
print(f'Used {len(wells)} wells')
print('Top 3 wells:')
for w in wells[:3]:
    print(f'  {w["station_name"]}: {w["distance_km"]:.1f} km, GWL: {w["gw_latest"]:.1f} m')
