import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    rainfall = pd.read_csv("data/rainfall.csv")
    groundwater = pd.read_csv("data/groundwater.csv")
    regions = gpd.read_file("data/regions.geojson")   # GeoJSON with state boundaries
    return rainfall, groundwater, regions

rainfall, groundwater, regions = load_data()

# -------------------------------
# Title + Sidebar
# -------------------------------
st.title("💧 Groundwater & Rainfall Dashboard - Prototype")
st.sidebar.header("Filters")

# --- State Selector ---
states = sorted(rainfall["state_name"].unique())
selected_state = st.sidebar.selectbox("Select State", states)

# --- Year Selector ---
years = sorted(rainfall["year_month"].str[:4].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

# --- Month Selector ---
months = sorted(rainfall[rainfall["year_month"].str[:4] == selected_year]["year_month"].unique())
selected_month = st.sidebar.selectbox("Select Month", months)

# --- District Selector ---
districts = sorted(groundwater[groundwater["state_name"] == selected_state]["district_name"].unique())
selected_district = st.sidebar.selectbox("Select District", districts)

# -------------------------------
# Filter Data for State + District
# -------------------------------
rainfall_state = rainfall[rainfall["state_name"] == selected_state].copy()
groundwater_state = groundwater[groundwater["state_name"] == selected_state].copy()

district_data = groundwater[
    (groundwater["state_name"] == selected_state) &
    (groundwater["district_name"] == selected_district)
].copy()

# Convert year_month to datetime
rainfall_state["date"] = pd.to_datetime(rainfall_state["year_month"])
groundwater_state["date"] = pd.to_datetime(groundwater_state["year_month"])
district_data["date"] = pd.to_datetime(district_data["year_month"])

# -------------------------------
# Aggregate Monthly Data
# -------------------------------
rainfall_monthly = (
    rainfall_state.groupby("year_month", as_index=False)["rainfall_actual_mm"].mean()
)
groundwater_monthly = (
    groundwater_state.groupby("year_month", as_index=False)["gw_level_m_bgl"].mean()
)

# -------------------------------
# State-level Time Series
# -------------------------------
st.subheader(f"🌧️ Rainfall Trends in {selected_state}")
fig_rain = px.line(
    rainfall_monthly,
    x="year_month", y="rainfall_actual_mm",
    title=f"Average Monthly Rainfall - {selected_state}",
    labels={"rainfall_actual_mm": "Rainfall (mm)", "year_month": "Month"}
)
st.plotly_chart(fig_rain, use_container_width=True)

st.subheader(f"💦 Groundwater Levels in {selected_state}")
fig_gw = px.line(
    groundwater_monthly,
    x="year_month", y="gw_level_m_bgl",
    title=f"Average Monthly Groundwater Levels - {selected_state}",
    labels={"gw_level_m_bgl": "Groundwater Level (m bgl)", "year_month": "Month"}
)
st.plotly_chart(fig_gw, use_container_width=True)

# -------------------------------
# District-level Time Series
# -------------------------------
st.subheader(f"📍 District-Level Trends in {selected_district}, {selected_state}")

if not district_data.empty:
    fig_dist = px.line(
        district_data,
        x="year_month", y="gw_level_m_bgl",
        title=f"Groundwater Levels - {selected_district}",
        labels={"gw_level_m_bgl": "Groundwater Level (m bgl)", "year_month": "Month"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.warning(f"No district-level data available for {selected_district}")

# -------------------------------
# Choropleth Maps (Selected Month) with Folium
# -------------------------------
st.subheader(f"🗺️ Rainfall & Groundwater by State - {selected_month}")

# Compute centroids for mapping
regions["centroid"] = regions.geometry.centroid
regions["lon"] = regions.centroid.x
regions["lat"] = regions.centroid.y

# Aggregate Rainfall (Selected Month)
rainfall_latest = (
    rainfall[rainfall["year_month"] == selected_month]
    .groupby("state_name", as_index=False)["rainfall_actual_mm"].sum()
    .merge(regions[["state_name", "lat", "lon"]], on="state_name", how="left")
)

# Aggregate Groundwater (Selected Month)
groundwater_latest = (
    groundwater[groundwater["year_month"] == selected_month]
    .groupby("state_name", as_index=False)["gw_level_m_bgl"].mean()
    .merge(regions[["state_name", "lat", "lon"]], on="state_name", how="left")
)

# Folium Map
m = folium.Map(location=[22, 78], zoom_start=5, tiles="cartodbpositron")

# Rainfall Markers
for _, row in rainfall_latest.iterrows():
    if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"{row['state_name']}<br>Rainfall: {row['rainfall_actual_mm']:.2f} mm",
        ).add_to(m)

# Groundwater Markers
for _, row in groundwater_latest.iterrows():
    if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.6,
            popup=f"{row['state_name']}<br>GW Level: {row['gw_level_m_bgl']:.2f} m bgl",
        ).add_to(m)

# Render in Streamlit
st_folium(m, width=900, height=600)