import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import reverse_geocoder as rg

# 1. Load your raw GIS file (the one with lon/lat, feature_id)
df = pd.read_csv("C:\Users\gauri\.vscode\bhai ke codes\groundwater-backend\data\gis_boundaries.geojson")

# 2. Build polygons by grouping lon/lat by feature_id
polygons = []
for fid, group in df.groupby("feature_id"):
    coords = list(zip(group["lon"], group["lat"]))
    poly = Polygon(coords)
    polygons.append({"feature_id": fid, "geometry": poly})

# 3. Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")

# 4. Compute centroid for each polygon
gdf["centroid"] = gdf.geometry.centroid
gdf["lon"] = gdf.centroid.x
gdf["lat"] = gdf.centroid.y

# 5. Use reverse geocoder to get state names
def get_state(lat, lon):
    try:
        result = rg.search((lat, lon))[0]
        return result["admin1"]  # State/region name
    except:
        return None

gdf["state_name"] = gdf.apply(lambda row: get_state(row["lat"], row["lon"]), axis=1)

# 6. Save new GeoJSON
output_path = "data/gis_boundaries_with_names.geojson"
gdf[["feature_id", "state_name", "geometry"]].to_file(output_path, driver="GeoJSON")

print(f"✅ Fixed GeoJSON saved at {output_path}")