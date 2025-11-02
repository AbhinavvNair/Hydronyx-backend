import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="💬 Groundwater Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.bot {
        background-color: #475063;
        color: white;
    }
    .chat-message-container {
        display: flex;
        align-items: center;
        margin: 1rem 0;
    }
    .chat-message-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    .user-avatar {
        background-color: #007bff;
    }
    .bot-avatar {
        background-color: #28a745;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GroundwaterChatbot:
    def __init__(self):
        self.load_data()
        self.load_model()
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'user_context' not in st.session_state:
            st.session_state.user_context = {
                'current_state': None,
                'current_district': None,
                'current_year': None,
                'current_month': None
            }
    
    def load_data(self):
        """Load groundwater and rainfall data"""
        try:
            # Try different path combinations
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try relative path first
            rainfall_path = "../data/rainfall.csv"
            groundwater_path = "../data/groundwater.csv"
            
            if not os.path.exists(rainfall_path):
                # Try absolute path
                rainfall_path = os.path.join(current_dir, "..", "data", "rainfall.csv")
                groundwater_path = os.path.join(current_dir, "..", "data", "groundwater.csv")
            
            self.rainfall = pd.read_csv(rainfall_path)
            self.groundwater = pd.read_csv(groundwater_path)
            
            # Clean data - handle potential NaN values
            self.rainfall["state_name"] = self.rainfall["state_name"].fillna("").astype(str).str.strip().str.lower()
            self.groundwater["state_name"] = self.groundwater["state_name"].fillna("").astype(str).str.strip().str.lower()
            self.groundwater["district_name"] = self.groundwater["district_name"].fillna("").astype(str).str.strip().str.lower()
            
            # Convert year_month to datetime
            self.rainfall["date"] = pd.to_datetime(self.rainfall["year_month"])
            self.groundwater["date"] = pd.to_datetime(self.groundwater["year_month"])
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.rainfall = pd.DataFrame()
            self.groundwater = pd.DataFrame()
    
    def load_model(self):
        """Load the trained ML model"""
        try:
            # Try different path combinations
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try relative path first
            model_path = "../models/groundwater_predictor.pkl"
            
            if not os.path.exists(model_path):
                # Try absolute path
                model_path = os.path.join(current_dir, "..", "models", "groundwater_predictor.pkl")
            
            self.model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None
    
    def get_available_states(self) -> List[str]:
        """Get list of available states"""
        if self.rainfall.empty:
            return []
        # Filter out empty strings and get unique states
        states = self.rainfall["state_name"].unique()
        states = [state for state in states if state and state.strip()]
        return sorted(states)
    
    def get_available_districts(self, state: str) -> List[str]:
        """Get list of available districts for a state"""
        if state and not self.groundwater.empty:
            districts = self.groundwater[self.groundwater["state_name"] == state]["district_name"].unique()
            # Filter out empty strings and NaN values
            districts = [d for d in districts if pd.notna(d) and d and d.strip()]
            return sorted(districts)
        return []
    
    def get_available_months(self) -> List[str]:
        """Get list of available months"""
        return sorted(self.rainfall["year_month"].unique().tolist())
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from user input"""
        text_lower = text.lower()
        entities = {
            'state': None,
            'district': None,
            'year': None,
            'month': None,
            'number': None
        }
        
        # Extract state
        for state in self.get_available_states():
            if state in text_lower:
                entities['state'] = state
                break
        
        # Extract district (only if state is found)
        if entities['state']:
            districts = self.get_available_districts(entities['state'])
            for district in districts:
                if district in text_lower:
                    entities['district'] = district
                    break
        
        # Extract year (including year ranges like 2016-17)
        year_pattern = r'\b(20\d{2})(?:-(\d{2}))?\b'
        year_match = re.search(year_pattern, text)
        if year_match:
            entities['year'] = year_match.group(1)
            # If it's a range like 2016-17, use the first year
        
        # Extract month
        month_pattern = r'\b(0[1-9]|1[0-2])-(20\d{2})\b'
        month_match = re.search(month_pattern, text)
        if month_match:
            entities['month'] = f"{month_match.group(2)}-{month_match.group(1)}"
        
        # Extract numbers (but not years) - handle cases like "120mm"
        number_pattern = r'(\d+\.?\d*)(?:mm|m|cm|km)?'
        number_matches = re.findall(number_pattern, text)
        for match in number_matches:
            num = float(match)
            # Don't treat years as numbers, but allow rainfall values
            if not (2000 <= num <= 2030):
                entities['number'] = num
                break
        
        return entities
    
    def get_rainfall_data(self, state: str = None, year_month: str = None) -> pd.DataFrame:
        """Get rainfall data with optional filters"""
        data = self.rainfall.copy()
        if state:
            data = data[data["state_name"] == state]
        if year_month:
            data = data[data["year_month"] == year_month]
        return data
    
    def get_groundwater_data(self, state: str = None, district: str = None, year_month: str = None) -> pd.DataFrame:
        """Get groundwater data with optional filters"""
        data = self.groundwater.copy()
        if state:
            data = data[data["state_name"] == state]
        if district:
            data = data[data["district_name"] == district]
        if year_month:
            data = data[data["year_month"] == year_month]
        return data
    
    def predict_groundwater(self, state: str, rainfall_value: float, lag_gw: float) -> float:
        """Predict groundwater level using the ML model"""
        if self.model is None:
            return None
        
        try:
            X = np.array([[rainfall_value, lag_gw]])
            prediction = self.model.predict(X)[0]
            return round(float(prediction), 2)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def analyze_trends(self, data: pd.DataFrame, value_col: str, date_col: str) -> Dict[str, float]:
        """Analyze trends in historical data"""
        if data.empty:
            return {"trend": 0, "seasonality": 0, "volatility": 0}
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(date_col)
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(data))
        y = data[value_col].values
        if len(x) > 1:
            trend = np.polyfit(x, y, 1)[0]  # Linear trend
        else:
            trend = 0
        
        # Calculate seasonality (monthly patterns)
        data['month'] = data[date_col].dt.month
        monthly_avg = data.groupby('month')[value_col].mean()
        if len(monthly_avg) > 1:
            seasonality = monthly_avg.std()
        else:
            seasonality = 0
        
        # Calculate volatility (standard deviation)
        volatility = data[value_col].std()
        
        return {
            "trend": round(trend, 4),
            "seasonality": round(seasonality, 2),
            "volatility": round(volatility, 2)
        }
    
    def predict_future_values(self, data: pd.DataFrame, value_col: str, date_col: str, 
                            months_ahead: int = 12) -> Dict[str, Any]:
        """Predict future values based on historical trends"""
        if data.empty:
            return {"predictions": [], "confidence": 0, "trend_analysis": {}}
        
        # Convert to datetime and sort
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(date_col)
        
        # Analyze trends
        trend_analysis = self.analyze_trends(data, value_col, date_col)
        
        # Get latest value and date
        latest_value = data[value_col].iloc[-1]
        latest_date = data[date_col].iloc[-1]
        
        # Generate future predictions
        predictions = []
        current_date = latest_date
        current_value = latest_value
        
        for i in range(1, months_ahead + 1):
            # Calculate next month
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1)
            
            # Apply trend
            predicted_value = current_value + (trend_analysis["trend"] * i)
            
            # Apply seasonal adjustment (simplified)
            seasonal_factor = 1 + (trend_analysis["seasonality"] * 0.1 * np.sin(2 * np.pi * next_date.month / 12))
            predicted_value *= seasonal_factor
            
            predictions.append({
                "date": next_date.strftime("%Y-%m"),
                "value": round(predicted_value, 2),
                "confidence": max(0.3, 1 - (i * 0.05))  # Decreasing confidence over time
            })
            
            current_date = next_date
            current_value = predicted_value
        
        return {
            "predictions": predictions,
            "confidence": round(np.mean([p["confidence"] for p in predictions]), 2),
            "trend_analysis": trend_analysis
        }
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on user input"""
        user_input_lower = user_input.lower()
        entities = self.extract_entities(user_input)
        
        # Greeting responses
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return self.get_greeting_response()
        
        # Help responses
        if any(word in user_input_lower for word in ['help', 'what can you do', 'commands', 'features']):
            return self.get_help_response()
        
        # Future prediction queries (check first for trend-based predictions)
        if any(word in user_input_lower for word in ['future', 'forecast', 'next year', 'coming months', 'estimate', 'projection']):
            return self.handle_future_prediction_query(entities, user_input_lower)
        
        # ML prediction queries (for specific rainfall values)
        if any(word in user_input_lower for word in ['predict', 'what will be']):
            return self.handle_prediction_query(entities, user_input_lower)
        
        # Data queries (only if not a prediction query)
        if any(word in user_input_lower for word in ['rainfall', 'rain', 'precipitation']) and not any(word in user_input_lower for word in ['future', 'forecast', 'next year', 'coming months', 'estimate', 'projection']):
            return self.handle_rainfall_query(entities, user_input_lower)
        
        if any(word in user_input_lower for word in ['groundwater', 'water level', 'gw level']):
            return self.handle_groundwater_query(entities, user_input_lower)
        
        # Comparison queries
        if any(word in user_input_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return self.handle_comparison_query(entities, user_input_lower)
        
        # Trend queries
        if any(word in user_input_lower for word in ['trend', 'trends', 'over time', 'change']):
            return self.handle_trend_query(entities, user_input_lower)
        
        # District queries
        if any(word in user_input_lower for word in ['district', 'districts', 'show me districts']):
            return self.handle_district_query(entities, user_input_lower)
        
        # Future prediction queries
        if any(word in user_input_lower for word in ['future', 'forecast', 'next year', 'coming months', 'estimate', 'projection']):
            return self.handle_future_prediction_query(entities, user_input_lower)
        
        # Default response
        return self.get_default_response()
    
    def get_greeting_response(self) -> str:
        """Generate greeting response"""
        greetings = [
            "Hello! I'm your Groundwater Assistant. I can help you with groundwater and rainfall data analysis, predictions, and insights. What would you like to know?",
            "Hi there! I'm here to help you understand groundwater levels, rainfall patterns, and make predictions. How can I assist you today?",
            "Welcome! I'm your AI assistant for groundwater monitoring. I can provide data insights, predictions, and answer questions about water resources. What's on your mind?"
        ]
        return np.random.choice(greetings)
    
    def get_help_response(self) -> str:
        """Generate help response"""
        return """
🤖 **I can help you with:**

📊 **Data Queries:**
- "What's the rainfall in Maharashtra?"
- "Show me groundwater levels in Delhi"
- "Rainfall data for 2023"

🔮 **Predictions:**
- "Predict groundwater level for next month"
- "What will be the water level if rainfall is 100mm?"

🔮 **Future Estimates:**
- "Future rainfall forecast for Maharashtra"
- "What will be the groundwater levels in Karnataka next year?"
- "Estimate rainfall for Delhi for 2025"
- "Coming months forecast for Mumbai"

📈 **Trends & Analysis:**
- "Show rainfall trends in Karnataka"
- "Compare groundwater levels between states"
- "How has rainfall changed over time?"

🗺️ **Geographic Queries:**
- "Which states have the highest rainfall?"
- "Groundwater levels by district"
- "Show me districts in Maharashtra"
- "Groundwater in Mumbai district"

💡 **Tips:**
- Be specific about states, districts, or time periods
- Ask for comparisons between different regions
- Request predictions with rainfall values

**Example queries:**
- "Rainfall in Tamil Nadu for 2023"
- "Predict groundwater level for Maharashtra with 150mm rainfall"
- "Compare groundwater levels between Delhi and Mumbai"
        """
    
    def handle_rainfall_query(self, entities: Dict, user_input: str) -> str:
        """Handle rainfall-related queries"""
        state = entities.get('state')
        year_month = entities.get('month')
        
        if state:
            data = self.get_rainfall_data(state=state, year_month=year_month)
            if not data.empty:
                if year_month:
                    avg_rainfall = data['rainfall_actual_mm'].mean()
                    return f"🌧️ **Rainfall in {state.title()} for {year_month}:**\n\nAverage rainfall: **{avg_rainfall:.2f} mm**"
                else:
                    latest_data = data.sort_values('year_month').tail(12)
                    avg_rainfall = latest_data['rainfall_actual_mm'].mean()
                    return f"🌧️ **Recent Rainfall in {state.title()}:**\n\nAverage rainfall (last 12 months): **{avg_rainfall:.2f} mm**\n\nLatest data available: {latest_data['year_month'].iloc[-1]}"
            else:
                return f"❌ No rainfall data found for {state.title()}"
        else:
            return "🌧️ **Rainfall Query:**\n\nPlease specify a state. For example:\n- 'Rainfall in Maharashtra'\n- 'Show rainfall data for Karnataka'\n- 'Rainfall in Delhi for 2023-06'"
    
    def handle_groundwater_query(self, entities: Dict, user_input: str) -> str:
        """Handle groundwater-related queries"""
        state = entities.get('state')
        district = entities.get('district')
        year_month = entities.get('month')
        
        if state:
            data = self.get_groundwater_data(state=state, district=district, year_month=year_month)
            if not data.empty:
                if year_month:
                    avg_gw = data['gw_level_m_bgl'].mean()
                    location = f"{district.title()}, {state.title()}" if district else state.title()
                    return f"💧 **Groundwater Level in {location} for {year_month}:**\n\nAverage groundwater level: **{avg_gw:.2f} m bgl**"
                else:
                    latest_data = data.sort_values('year_month').tail(12)
                    avg_gw = latest_data['gw_level_m_bgl'].mean()
                    location = f"{district.title()}, {state.title()}" if district else state.title()
                    return f"💧 **Recent Groundwater Levels in {location}:**\n\nAverage groundwater level (last 12 months): **{avg_gw:.2f} m bgl**\n\nLatest data available: {latest_data['year_month'].iloc[-1]}"
            else:
                location = f"{district.title()}, {state.title()}" if district else state.title()
                return f"❌ No groundwater data found for {location}"
        else:
            return "💧 **Groundwater Query:**\n\nPlease specify a state or district. For example:\n- 'Groundwater levels in Maharashtra'\n- 'Show groundwater data for Delhi'\n- 'Groundwater in Mumbai for 2023-06'"
    
    def handle_prediction_query(self, entities: Dict, user_input: str) -> str:
        """Handle prediction queries"""
        if self.model is None:
            return "❌ **Prediction Model Unavailable:**\n\nThe prediction model is not loaded. Please check if the model file exists."
        
        state = entities.get('state')
        district = entities.get('district')
        rainfall_value = entities.get('number')
        year = entities.get('year')
        
        if state and rainfall_value:
            # Get latest groundwater level for the state/district
            latest_gw_data = self.get_groundwater_data(state=state, district=district).sort_values('year_month').tail(1)
            if not latest_gw_data.empty:
                lag_gw = latest_gw_data['gw_level_m_bgl'].iloc[0]
                prediction = self.predict_groundwater(state, rainfall_value, lag_gw)
                
                if prediction is not None:
                    location = f"{district.title()}, {state.title()}" if district else state.title()
                    year_info = f" for {year}" if year else ""
                    
                    response = f"🔮 **Groundwater Prediction for {location}{year_info}:**\n\n"
                    response += f"**Input Parameters:**\n"
                    response += f"- Rainfall: {rainfall_value} mm\n"
                    response += f"- Current groundwater level: {lag_gw:.2f} m bgl\n"
                    if district:
                        response += f"- Location: {district.title()} district\n"
                    if year:
                        response += f"- Year: {year}\n"
                    
                    response += f"\n**Predicted groundwater level:** **{prediction} m bgl**\n\n"
                    response += "*Note: This is a machine learning prediction based on historical data.*"
                    
                    return response
                else:
                    return "❌ **Prediction Failed:**\n\nUnable to generate prediction. Please check the input parameters."
            else:
                location = f"{district.title()}, {state.title()}" if district else state.title()
                return f"❌ **No Historical Data:**\n\nNo groundwater data found for {location} to use as baseline for prediction."
        else:
            return "🔮 **Prediction Query:**\n\nPlease provide both state and rainfall value. For example:\n- 'Predict groundwater level for Maharashtra with 150mm rainfall'\n- 'What will be the water level in Delhi if rainfall is 100mm?'\n- 'Predict groundwater for Mumbai district with 120mm rainfall for 2024'"
    
    def handle_comparison_query(self, entities: Dict, user_input: str) -> str:
        """Handle comparison queries"""
        # This is a simplified version - in a real implementation, you'd parse multiple states
        return "📊 **Comparison Feature:**\n\nTo compare data between states or districts, please ask specific questions like:\n- 'Compare rainfall between Maharashtra and Karnataka'\n- 'Which state has higher groundwater levels?'\n\n*Note: Advanced comparison features are being developed.*"
    
    def handle_trend_query(self, entities: Dict, user_input: str) -> str:
        """Handle trend queries"""
        state = entities.get('state')
        
        if state:
            # Get recent data for trend analysis
            rainfall_data = self.get_rainfall_data(state=state).sort_values('year_month').tail(24)
            gw_data = self.get_groundwater_data(state=state).sort_values('year_month').tail(24)
            
            if not rainfall_data.empty and not gw_data.empty:
                # Calculate trends
                rainfall_trend = rainfall_data['rainfall_actual_mm'].diff().mean()
                gw_trend = gw_data['gw_level_m_bgl'].diff().mean()
                
                trend_analysis = f"📈 **Trend Analysis for {state.title()}:**\n\n"
                trend_analysis += f"**Rainfall Trend:** {rainfall_trend:+.2f} mm/month\n"
                trend_analysis += f"**Groundwater Trend:** {gw_trend:+.2f} m bgl/month\n\n"
                
                if rainfall_trend > 0:
                    trend_analysis += "📈 Rainfall is increasing over time\n"
                else:
                    trend_analysis += "📉 Rainfall is decreasing over time\n"
                
                if gw_trend > 0:
                    trend_analysis += "📈 Groundwater levels are rising (good for water availability)\n"
                else:
                    trend_analysis += "📉 Groundwater levels are declining (concerning for water availability)\n"
                
                return trend_analysis
            else:
                return f"❌ **Insufficient Data:**\n\nNot enough data available for trend analysis in {state.title()}"
        else:
            return "📈 **Trend Analysis:**\n\nPlease specify a state for trend analysis. For example:\n- 'Show rainfall trends in Maharashtra'\n- 'Groundwater trends in Karnataka'"
    
    def handle_district_query(self, entities: Dict, user_input: str) -> str:
        """Handle district-related queries"""
        state = entities.get('state')
        
        if state:
            districts = self.get_available_districts(state)
            if districts:
                # Show first 10 districts
                district_list = districts[:10]
                district_text = "\n".join([f"- {d.title()}" for d in district_list])
                
                response = f"📍 **Districts in {state.title()}:**\n\n{district_text}"
                
                if len(districts) > 10:
                    response += f"\n\n... and {len(districts) - 10} more districts"
                
                response += f"\n\n**Total districts:** {len(districts)}"
                response += f"\n\n💡 **Tip:** Ask for specific district data like:\n- 'Groundwater levels in {districts[0].title()}'"
                
                return response
            else:
                return f"❌ **No Districts Found:**\n\nNo district data available for {state.title()}"
        else:
            return "📍 **District Query:**\n\nPlease specify a state to see its districts. For example:\n- 'Show me districts in Maharashtra'\n- 'What districts are in Karnataka?'"
    
    def handle_future_prediction_query(self, entities: Dict, user_input: str) -> str:
        """Handle future prediction queries based on historical trends"""
        state = entities.get('state')
        district = entities.get('district')
        year = entities.get('year')
        
        # Determine months ahead based on year or default to 12
        months_ahead = 12
        if year:
            current_year = 2024  # Assuming current year
            target_year = int(year)
            months_ahead = (target_year - current_year) * 12
            months_ahead = max(1, min(months_ahead, 60))  # Limit to 5 years
        
        if state:
            # Get historical data
            rainfall_data = self.get_rainfall_data(state=state)
            groundwater_data = self.get_groundwater_data(state=state, district=district)
            
            if not rainfall_data.empty and not groundwater_data.empty:
                # Prepare data for analysis
                rainfall_data['date'] = pd.to_datetime(rainfall_data['year_month'])
                groundwater_data['date'] = pd.to_datetime(groundwater_data['year_month'])
                
                # Get recent data (last 3 years for better trend analysis)
                recent_rainfall = rainfall_data.tail(36)  # 3 years
                recent_groundwater = groundwater_data.tail(36)
                
                # Predict future rainfall
                rainfall_predictions = self.predict_future_values(
                    recent_rainfall, 'rainfall_actual_mm', 'date', months_ahead
                )
                
                # Predict future groundwater
                groundwater_predictions = self.predict_future_values(
                    recent_groundwater, 'gw_level_m_bgl', 'date', months_ahead
                )
                
                location = f"{district.title()}, {state.title()}" if district else state.title()
                
                response = f"🔮 **Future Predictions for {location}:**\n\n"
                
                # Add trend analysis
                if rainfall_predictions['trend_analysis']:
                    trend_info = rainfall_predictions['trend_analysis']
                    response += f"**📊 Trend Analysis:**\n"
                    response += f"- Rainfall trend: {trend_info['trend']:+.4f} mm/month\n"
                    response += f"- Seasonality: {trend_info['seasonality']:.2f}\n"
                    response += f"- Volatility: {trend_info['volatility']:.2f}\n\n"
                
                # Show next 6 months predictions
                response += f"**📅 Next 6 Months Forecast:**\n"
                for i, (rain_pred, gw_pred) in enumerate(zip(
                    rainfall_predictions['predictions'][:6],
                    groundwater_predictions['predictions'][:6]
                )):
                    response += f"- **{rain_pred['date']}:**\n"
                    response += f"  • Rainfall: {rain_pred['value']:.1f} mm (confidence: {rain_pred['confidence']:.1%})\n"
                    response += f"  • Groundwater: {gw_pred['value']:.2f} m bgl (confidence: {gw_pred['confidence']:.1%})\n"
                
                # Add long-term projection if requested
                if months_ahead > 6:
                    response += f"\n**📈 Long-term Projection ({months_ahead} months):**\n"
                    final_rain = rainfall_predictions['predictions'][-1]
                    final_gw = groundwater_predictions['predictions'][-1]
                    response += f"- **{final_rain['date']}:**\n"
                    response += f"  • Expected rainfall: {final_rain['value']:.1f} mm\n"
                    response += f"  • Expected groundwater: {final_gw['value']:.2f} m bgl\n"
                
                # Add confidence and warnings
                avg_confidence = (rainfall_predictions['confidence'] + groundwater_predictions['confidence']) / 2
                response += f"\n**⚠️ Confidence Level:** {avg_confidence:.1%}\n"
                response += f"*Note: Predictions are based on historical trends and may not account for extreme weather events or human interventions.*"
                
                return response
            else:
                location = f"{district.title()}, {state.title()}" if district else state.title()
                return f"❌ **Insufficient Data:**\n\nNot enough historical data available for future predictions in {location}."
        else:
            return "🔮 **Future Prediction Query:**\n\nPlease specify a state for future predictions. For example:\n- 'Future rainfall forecast for Maharashtra'\n- 'What will be the groundwater levels in Karnataka next year?'\n- 'Estimate rainfall for Delhi for 2025'"
    
    def get_default_response(self) -> str:
        """Generate default response for unrecognized queries"""
        return """
🤔 **I didn't quite understand that query.**

Here are some things I can help you with:

🌧️ **Rainfall Data:** Ask about rainfall in specific states or time periods
💧 **Groundwater Levels:** Query groundwater data by state or district  
🔮 **Predictions:** Get groundwater level predictions based on rainfall
📊 **Comparisons:** Compare data between different regions
📈 **Trends:** Analyze how water levels change over time

**Try asking:**
- "What's the rainfall in Maharashtra?"
- "Show me groundwater levels in Delhi"
- "Predict groundwater level for Karnataka with 120mm rainfall"

Type **'help'** for more detailed information about my capabilities!
        """
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.title("🤖 Groundwater Assistant")
        st.markdown("Ask me anything about groundwater levels, rainfall data, and predictions!")
        
        # Sidebar with quick actions
        with st.sidebar:
            st.header("🎯 Quick Actions")
            
            # State selector
            states = self.get_available_states()
            selected_state = st.selectbox("Select State", ["All States"] + states, key="chatbot_state_selector")
            
            if selected_state != "All States":
                districts = self.get_available_districts(selected_state)
                if districts:
                    selected_district = st.selectbox("Select District", ["All Districts"] + districts, key="chatbot_district_selector")
                else:
                    selected_district = "All Districts"
            else:
                selected_district = "All Districts"
            
            # Quick query buttons
            st.subheader("💡 Quick Queries")
            
            if st.button("🌧️ Rainfall Data", key="quick_rainfall"):
                if selected_state != "All States":
                    query = f"Show rainfall data for {selected_state}"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    # Generate bot response
                    response = self.generate_response(query)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    st.rerun()
                else:
                    st.warning("Please select a state first!")
            
            if st.button("💧 Groundwater Levels", key="quick_groundwater"):
                if selected_state != "All States":
                    query = f"Show groundwater levels in {selected_state}"
                    if selected_district != "All Districts":
                        query += f", {selected_district}"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    # Generate bot response
                    response = self.generate_response(query)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    st.rerun()
                else:
                    st.warning("Please select a state first!")
            
            if st.button("🔮 Get Prediction", key="quick_prediction"):
                if selected_state != "All States":
                    query = f"Predict groundwater level for {selected_state} with 100mm rainfall"
                    if selected_district != "All Districts":
                        query = f"Predict groundwater level for {selected_district}, {selected_state} with 100mm rainfall"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    # Generate bot response
                    response = self.generate_response(query)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    st.rerun()
                else:
                    st.warning("Please select a state first!")
            
            if st.button("📈 Show Trends", key="quick_trends"):
                if selected_state != "All States":
                    query = f"Show trends for {selected_state}"
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": query
                    })
                    # Generate bot response
                    response = self.generate_response(query)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    st.rerun()
                else:
                    st.warning("Please select a state first!")
            
            # Data summary
            st.subheader("📊 Data Summary")
            st.metric("Total States", len(states))
            st.metric("Total Records", len(self.groundwater) + len(self.rainfall))
            
            if selected_state != "All States":
                state_data = self.groundwater[self.groundwater["state_name"] == selected_state]
                st.metric("Districts in State", len(state_data["district_name"].unique()))
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about groundwater, rainfall, or predictions..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and add bot response
            with st.chat_message("assistant"):
                response = self.generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

def main():
    """Main function to run the chatbot"""
    try:
        chatbot = GroundwaterChatbot()
        chatbot.display_chat_interface()
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.info("Please make sure all data files and the model are available in the correct directories.")

if __name__ == "__main__":
    main()