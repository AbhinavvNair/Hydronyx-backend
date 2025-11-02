import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the chatbot
from chatbot import GroundwaterChatbot

# Page configuration
st.set_page_config(
    page_title="💧 Groundwater Monitoring System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application with modern interactive dashboard"""
    
    # Custom CSS for modern dashboard styling
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        /* Navbar Styles */
        .navbar {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem 1.5rem;
            margin: -1rem -1rem 2rem -1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            border-radius: 0 0 15px 15px;
        }
        
        .navbar-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .navbar-brand {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .navbar-nav {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        
        /* Style navbar buttons specifically */
        .navbar .stButton > button {
            background: rgba(255,255,255,0.15) !important;
            color: white !important;
            border: 2px solid rgba(255,255,255,0.3) !important;
            border-radius: 25px !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            height: auto !important;
            margin: 0 !important;
            min-width: 120px !important;
            text-align: center !important;
        }
        
        .navbar .stButton > button:hover {
            background: rgba(255,255,255,0.25) !important;
            border-color: rgba(255,255,255,0.5) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        }
        
        .navbar .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Stats Cards */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-left: 4px solid;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .stat-card.confirmed { border-left-color: #e74c3c; }
        .stat-card.active { border-left-color: #3498db; }
        .stat-card.recovered { border-left-color: #27ae60; }
        .stat-card.deceased { border-left-color: #95a5a6; }
        
        .stat-title {
            font-size: 0.9rem;
            color: #7f8c8d;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .stat-change {
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .stat-change.positive { color: #27ae60; }
        .stat-change.negative { color: #e74c3c; }
        
        /* Interactive Elements */
        .interactive-card {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 1.5rem 0;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .card-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #3498db;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2980b9;
        }
        
        .btn-secondary {
            background: #ecf0f1;
            color: #2c3e50;
        }
        
        .btn-secondary:hover {
            background: #d5dbdb;
        }
        
        /* Footer */
        .footer {
            background: #2c3e50;
            color: white;
            padding: 2rem;
            margin: 3rem -1rem -1rem -1rem;
            text-align: center;
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1rem 0;
        }
        
        .footer-link {
            color: #bdc3c7;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .footer-link:hover {
            color: white;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .navbar-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .navbar-nav {
                gap: 1rem;
            }
            
            .stats-container {
                grid-template-columns: 1fr;
            }
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "📊 Dashboard"
    
    # Navbar with brand and navigation buttons
    st.markdown("""
    <div class="navbar">
        <div class="navbar-content">
            <div class="navbar-brand">🌊 Groundwater System</div>
            <div class="navbar-nav">
                <div id="navbar-buttons"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navbar buttons using Streamlit columns
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.nav_page = "📊 Dashboard"
            st.rerun()
    
    with col2:
        if st.button("🤖 AI Assistant", key="nav_chatbot", use_container_width=True):
            st.session_state.nav_page = "🤖 AI Assistant"
            st.rerun()
    
    with col3:
        if st.button("📈 Analytics", key="nav_analytics", use_container_width=True):
            st.session_state.nav_page = "📈 Analytics"
            st.rerun()
    
    with col4:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.nav_page = "ℹ️ About"
            st.rerun()
    
    # JavaScript to move buttons into navbar
    st.markdown("""
    <script>
    // Move navbar buttons into the navbar
    setTimeout(function() {
        const navbarButtons = document.querySelectorAll('.stButton');
        const navbarNav = document.getElementById('navbar-buttons');
        
        if (navbarNav && navbarButtons.length >= 4) {
            // Clear existing content
            navbarNav.innerHTML = '';
            
            // Move the first 4 buttons (our navbar buttons)
            for (let i = 0; i < 4; i++) {
                if (navbarButtons[i]) {
                    navbarNav.appendChild(navbarButtons[i]);
                }
            }
        }
    }, 100);
    </script>
    """, unsafe_allow_html=True)
    
    
    # Page content
    current_page = st.session_state.nav_page
    if current_page == "📊 Dashboard":
        show_modern_dashboard()
    elif current_page == "🤖 AI Assistant":
        show_chatbot()
    elif current_page == "📈 Analytics":
        show_analytics()
    elif current_page == "ℹ️ About":
        show_about()
    else:
        # Fallback to dashboard if page is not recognized
        st.session_state.nav_page = "📊 Dashboard"
        show_modern_dashboard()
    
    # Footer
    show_footer()

def show_modern_dashboard():
    """Display the modern interactive dashboard"""
    
    # Load data
    @st.cache_data
    def load_data():
        rainfall = pd.read_csv("../data/rainfall.csv")
        groundwater = pd.read_csv("../data/groundwater.csv")
        return rainfall, groundwater
    
    rainfall, groundwater = load_data()
    
    # Calculate summary statistics
    total_states = len(rainfall["state_name"].unique())
    total_districts = len(groundwater["district_name"].unique())
    total_records = len(rainfall) + len(groundwater)
    latest_date = rainfall["year_month"].max()
    
    # Display stats cards
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card confirmed">
            <div class="stat-title">Total States</div>
            <div class="stat-value" style="color: #e74c3c;">""" + str(total_states) + """</div>
            <div class="stat-change positive">Coverage: 100%</div>
        </div>
        <div class="stat-card active">
            <div class="stat-title">Total Districts</div>
            <div class="stat-value" style="color: #3498db;">""" + str(total_districts) + """</div>
            <div class="stat-change positive">Active monitoring</div>
        </div>
        <div class="stat-card recovered">
            <div class="stat-title">Data Records</div>
            <div class="stat-value" style="color: #27ae60;">""" + f"{total_records:,}" + """</div>
            <div class="stat-change positive">Historical data</div>
        </div>
        <div class="stat-card deceased">
            <div class="stat-title">Latest Update</div>
            <div class="stat-value" style="color: #95a5a6;">""" + latest_date + """</div>
            <div class="stat-change positive">Real-time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        states = sorted(rainfall["state_name"].unique())
        selected_state = st.selectbox("🏛️ Select State", states, key="state_filter")
    
    with col2:
        years = sorted(rainfall["year_month"].str[:4].unique())
        selected_year = st.selectbox("📅 Select Year", years, key="year_filter")
    
    with col3:
        months = sorted(rainfall[rainfall["year_month"].str[:4] == selected_year]["year_month"].unique())
        selected_month = st.selectbox("📆 Select Month", months, key="month_filter")
    
    with col4:
        districts = sorted(groundwater[groundwater["state_name"] == selected_state]["district_name"].unique())
        selected_district = st.selectbox("🏘️ Select District", districts, key="district_filter")
    
    # Filter data
    rainfall_state = rainfall[rainfall["state_name"] == selected_state].copy()
    groundwater_state = groundwater[groundwater["state_name"] == selected_state].copy()
    
    district_data = groundwater[
        (groundwater["state_name"] == selected_state) &
        (groundwater["district_name"] == selected_district)
    ].copy()
    
    # Convert to datetime
    rainfall_state["date"] = pd.to_datetime(rainfall_state["year_month"])
    groundwater_state["date"] = pd.to_datetime(groundwater_state["year_month"])
    district_data["date"] = pd.to_datetime(district_data["year_month"])
    
    # Aggregate monthly data
    rainfall_monthly = (
        rainfall_state.groupby("year_month", as_index=False)["rainfall_actual_mm"].mean()
    )
    groundwater_monthly = (
        groundwater_state.groupby("year_month", as_index=False)["gw_level_m_bgl"].mean()
    )
    
    # Create interactive charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="interactive-card">
            <div class="card-header">
                <div class="card-title">🌧️ Rainfall Trends</div>
                <div class="card-actions">
                    <span class="btn btn-secondary">Interactive Chart</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_rain = px.line(
            rainfall_monthly,
            x="year_month", y="rainfall_actual_mm",
            title=f"Average Monthly Rainfall - {selected_state}",
            labels={"rainfall_actual_mm": "Rainfall (mm)", "year_month": "Month"},
            color_discrete_sequence=['#e74c3c']
        )
        fig_rain.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_rain, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="interactive-card">
            <div class="card-header">
                <div class="card-title">💧 Groundwater Levels</div>
                <div class="card-actions">
                    <span class="btn btn-secondary">Interactive Chart</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_gw = px.line(
            groundwater_monthly,
            x="year_month", y="gw_level_m_bgl",
            title=f"Average Monthly Groundwater Levels - {selected_state}",
            labels={"gw_level_m_bgl": "Groundwater Level (m bgl)", "year_month": "Month"},
            color_discrete_sequence=['#3498db']
        )
        fig_gw.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_gw, use_container_width=True)
    
    # District-level analysis
    if not district_data.empty:
        st.markdown("""
        <div class="interactive-card">
            <div class="card-header">
                <div class="card-title">📍 District-Level Analysis</div>
                <div class="card-actions">
                    <span class="btn btn-secondary">District Analysis</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_dist = px.line(
            district_data,
            x="year_month", y="gw_level_m_bgl",
            title=f"Groundwater Levels - {selected_district}",
            labels={"gw_level_m_bgl": "Groundwater Level (m bgl)", "year_month": "Month"},
            color_discrete_sequence=['#27ae60']
        )
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning(f"No district-level data available for {selected_district}")
    
    # Map section
    st.markdown("""
    <div class="interactive-card">
        <div class="card-header">
            <div class="card-title">🗺️ Geographic Visualization</div>
            <div class="card-actions">
                <span class="btn btn-secondary">Interactive Map</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load regions data for map
    @st.cache_data
    def load_regions():
        import geopandas as gpd
        return gpd.read_file("../data/regions.geojson")
    
    try:
        regions = load_regions()
        
        # Compute centroids
        regions["centroid"] = regions.geometry.centroid
        regions["lon"] = regions.centroid.x
        regions["lat"] = regions.centroid.y
        
        # Aggregate data for map
        rainfall_latest = (
            rainfall[rainfall["year_month"] == selected_month]
            .groupby("state_name", as_index=False)["rainfall_actual_mm"].sum()
            .merge(regions[["state_name", "lat", "lon"]], on="state_name", how="left")
        )
        
        groundwater_latest = (
            groundwater[groundwater["year_month"] == selected_month]
            .groupby("state_name", as_index=False)["gw_level_m_bgl"].mean()
            .merge(regions[["state_name", "lat", "lon"]], on="state_name", how="left")
        )
        
        # Create map
        import folium
        from streamlit_folium import st_folium
        
        m = folium.Map(location=[22, 78], zoom_start=5, tiles="cartodbpositron")
        
        # Add rainfall markers
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
        
        # Add groundwater markers
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
        
        st_folium(m, width=900, height=600)
        
    except Exception as e:
        st.warning(f"Map not available: {e}")
        st.info("Please ensure the regions.geojson file is available in the data directory.")
    
    # Quick actions
    st.markdown("""
    <div class="interactive-card">
        <div class="card-header">
            <div class="card-title">⚡ Quick Actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Get AI Prediction", use_container_width=True):
            st.session_state.nav_page = "🤖 AI Assistant"
            st.rerun()
    
    with col2:
        if st.button("📈 View Analytics", use_container_width=True):
            st.session_state.nav_page = "📈 Analytics"
            st.rerun()
    
    with col3:
        if st.button("ℹ️ About System", use_container_width=True):
            st.session_state.nav_page = "ℹ️ About"
            st.rerun()

def show_analytics():
    """Display advanced analytics page"""
    
    # Load data for analytics
    @st.cache_data
    def load_analytics_data():
        rainfall = pd.read_csv("../data/rainfall.csv")
        groundwater = pd.read_csv("../data/groundwater.csv")
        return rainfall, groundwater
    
    rainfall, groundwater = load_analytics_data()
    
    st.markdown("""
    <div class="interactive-card">
        <div class="card-header">
            <div class="card-title">📈 Advanced Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total States", len(rainfall["state_name"].unique()))
    
    with col2:
        st.metric("Total Districts", len(groundwater["district_name"].unique()))
    
    with col3:
        avg_rainfall = rainfall["rainfall_actual_mm"].mean()
        st.metric("Avg Rainfall (mm)", f"{avg_rainfall:.1f}")
    
    with col4:
        avg_gw = groundwater["gw_level_m_bgl"].mean()
        st.metric("Avg GW Level (m)", f"{avg_gw:.2f}")
    
    # State-wise analysis
    st.markdown("### 🏛️ State-wise Analysis")
    
    # Top states by rainfall
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌧️ Top 10 States by Average Rainfall**")
        state_rainfall = rainfall.groupby("state_name")["rainfall_actual_mm"].mean().sort_values(ascending=False).head(10)
        
        fig_rainfall = px.bar(
            x=state_rainfall.values,
            y=state_rainfall.index,
            orientation='h',
            title="Average Rainfall by State",
            labels={'x': 'Rainfall (mm)', 'y': 'State'},
            color=state_rainfall.values,
            color_continuous_scale='Blues'
        )
        fig_rainfall.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rainfall, use_container_width=True)
    
    with col2:
        st.markdown("**💧 Top 10 States by Average Groundwater Level**")
        state_gw = groundwater.groupby("state_name")["gw_level_m_bgl"].mean().sort_values(ascending=True).head(10)
        
        fig_gw = px.bar(
            x=state_gw.values,
            y=state_gw.index,
            orientation='h',
            title="Average Groundwater Level by State",
            labels={'x': 'GW Level (m bgl)', 'y': 'State'},
            color=state_gw.values,
            color_continuous_scale='Reds'
        )
        fig_gw.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_gw, use_container_width=True)
    
    # Time series analysis
    st.markdown("### 📊 National Trends")
    
    # National average over time
    national_rainfall = rainfall.groupby("year_month")["rainfall_actual_mm"].mean().reset_index()
    national_gw = groundwater.groupby("year_month")["gw_level_m_bgl"].mean().reset_index()
    
    fig_national = go.Figure()
    
    # Add rainfall trace
    fig_national.add_trace(go.Scatter(
        x=national_rainfall["year_month"],
        y=national_rainfall["rainfall_actual_mm"],
        mode='lines+markers',
        name='Rainfall (mm)',
        line=dict(color='#3498db', width=3),
        yaxis='y'
    ))
    
    # Add groundwater trace
    fig_national.add_trace(go.Scatter(
        x=national_gw["year_month"],
        y=national_gw["gw_level_m_bgl"],
        mode='lines+markers',
        name='Groundwater Level (m bgl)',
        line=dict(color='#e74c3c', width=3),
        yaxis='y2'
    ))
    
    # Update layout
    fig_national.update_layout(
        title="National Average Trends Over Time",
        xaxis_title="Time Period",
        yaxis=dict(title="Rainfall (mm)", side="left"),
        yaxis2=dict(title="Groundwater Level (m bgl)", side="right", overlaying="y"),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_national, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Merge data for correlation
    merged_data = pd.merge(
        rainfall.groupby(["state_name", "year_month"])["rainfall_actual_mm"].mean().reset_index(),
        groundwater.groupby(["state_name", "year_month"])["gw_level_m_bgl"].mean().reset_index(),
        on=["state_name", "year_month"],
        how="inner"
    )
    
    correlation = merged_data["rainfall_actual_mm"].corr(merged_data["gw_level_m_bgl"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rainfall-GW Correlation", f"{correlation:.3f}")
    
    with col2:
        st.metric("Data Points", len(merged_data))
    
    with col3:
        st.metric("States Analyzed", merged_data["state_name"].nunique())
    
    # Scatter plot
    fig_scatter = px.scatter(
        merged_data,
        x="rainfall_actual_mm",
        y="gw_level_m_bgl",
        title="Rainfall vs Groundwater Level Correlation",
        labels={"rainfall_actual_mm": "Rainfall (mm)", "gw_level_m_bgl": "Groundwater Level (m bgl)"},
        color_discrete_sequence=['#27ae60']
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Quick actions for analytics
    st.markdown("""
    <div class="interactive-card">
        <div class="card-header">
            <div class="card-title">⚡ Quick Actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Dashboard", use_container_width=True, key="analytics_to_dashboard"):
            st.session_state.nav_page = "📊 Dashboard"
            st.rerun()
    
    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True, key="analytics_to_chatbot"):
            st.session_state.nav_page = "🤖 AI Assistant"
            st.rerun()
    
    with col3:
        if st.button("ℹ️ About System", use_container_width=True, key="analytics_to_about"):
            st.session_state.nav_page = "ℹ️ About"
            st.rerun()

def show_footer():
    """Display footer"""
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <h3>🌊 Groundwater Monitoring System</h3>
            <p>Comprehensive Water Resource Management & AI-Powered Insights</p>
            <div class="footer-links">
                <a href="#" class="footer-link">Privacy Policy</a>
                <a href="#" class="footer-link">Terms of Service</a>
                <a href="#" class="footer-link">Contact Us</a>
                <a href="#" class="footer-link">Documentation</a>
            </div>
            <p style="margin-top: 1rem; color: #7f8c8d;">
                © 2024 Groundwater Monitoring System. All rights reserved.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_chatbot():
    """Display the AI chatbot"""
    st.markdown("## 🤖 AI Groundwater Assistant")
    st.markdown("Ask me anything about groundwater levels, rainfall data, and predictions!")
    
    try:
        # Initialize chatbot
        chatbot = GroundwaterChatbot()
        chatbot.display_chat_interface()
        
        # Quick actions for AI Assistant
        st.markdown("""
        <div class="interactive-card">
            <div class="card-header">
                <div class="card-title">⚡ Quick Actions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Dashboard", use_container_width=True, key="chatbot_to_dashboard"):
                st.session_state.nav_page = "📊 Dashboard"
                st.rerun()
        
        with col2:
            if st.button("📈 View Analytics", use_container_width=True, key="chatbot_to_analytics"):
                st.session_state.nav_page = "📈 Analytics"
                st.rerun()
        
        with col3:
            if st.button("ℹ️ About System", use_container_width=True, key="chatbot_to_about"):
                st.session_state.nav_page = "ℹ️ About"
                st.rerun()
                
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.info("Please make sure all data files and the model are available in the correct directories.")

def show_about():
    """Display about information"""
    st.markdown("## ℹ️ About Groundwater Monitoring System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Dashboard Features</h3>
            <ul>
                <li>Interactive time series charts</li>
                <li>Geographic mapping with Folium</li>
                <li>State and district-level analysis</li>
                <li>Rainfall and groundwater visualization</li>
                <li>Real-time data filtering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Assistant Features</h3>
            <ul>
                <li>Natural language queries</li>
                <li>Groundwater level predictions</li>
                <li>District-wise analysis</li>
                <li>Year-specific forecasts</li>
                <li>Interactive chat interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 Technical Stack</h3>
            <ul>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>Backend:</strong> FastAPI</li>
                <li><strong>ML:</strong> Scikit-learn</li>
                <li><strong>Data:</strong> Pandas, NumPy</li>
                <li><strong>Visualization:</strong> Plotly, Folium</li>
                <li><strong>Language:</strong> Python</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Data Sources</h3>
            <ul>
                <li>Rainfall data (2009-2024)</li>
                <li>Groundwater levels (2013-2024)</li>
                <li>Geographic boundaries</li>
                <li>37 states coverage</li>
                <li>565+ districts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Use Cases</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
            <div>
                <h4>👨‍🔬 Researchers</h4>
                <p>Data analysis, trend identification, hypothesis testing</p>
            </div>
            <div>
                <h4>🏛️ Policy Makers</h4>
                <p>Resource planning, impact assessment, decision support</p>
            </div>
            <div>
                <h4>🎓 Students</h4>
                <p>Learning about water resources, data science, ML applications</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions for About page
    st.markdown("""
    <div class="interactive-card">
        <div class="card-header">
            <div class="card-title">⚡ Quick Actions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Dashboard", use_container_width=True, key="about_to_dashboard"):
            st.session_state.nav_page = "📊 Dashboard"
            st.rerun()
    
    with col2:
        if st.button("🤖 AI Assistant", use_container_width=True, key="about_to_chatbot"):
            st.session_state.nav_page = "🤖 AI Assistant"
            st.rerun()
    
    with col3:
        if st.button("📈 View Analytics", use_container_width=True, key="about_to_analytics"):
            st.session_state.nav_page = "📈 Analytics"
            st.rerun()

if __name__ == "__main__":
    main()