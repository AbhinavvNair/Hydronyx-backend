# 🤖 Groundwater Assistant Chatbot

An intelligent chatbot for your groundwater monitoring system that can answer questions about rainfall, groundwater levels, and provide predictions using machine learning.

## ✨ Features

### 🗣️ Natural Language Processing
- Understands natural language queries about groundwater and rainfall
- Extracts entities like states, districts, years, and numbers from user input
- Provides contextual responses based on available data

### 📊 Data Queries
- **Rainfall Data**: Query rainfall information by state and time period
- **Groundwater Levels**: Get groundwater level data by state/district
- **Historical Analysis**: Access historical trends and patterns

### 🔮 AI Predictions
- **Groundwater Predictions**: Predict future groundwater levels based on rainfall
- **ML Integration**: Uses your trained scikit-learn model for accurate predictions
- **Parameter-based**: Accepts rainfall values and current groundwater levels

### 🎯 Smart Features
- **Quick Actions**: Sidebar with one-click queries for common requests
- **Context Awareness**: Remembers user selections and preferences
- **Interactive UI**: Modern, responsive design with real-time chat interface

## 🚀 Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
# From the project root directory
python groundwater-backend/run_chatbot.py
```

### Option 2: Direct Streamlit Launch
```bash
# Navigate to the frontend directory
cd groundwater-backend/frontend

# Launch the chatbot
streamlit run chatbot.py --server.port 8502
```

## 💬 Example Queries

### 📊 Data Queries
```
"What's the rainfall in Maharashtra?"
"Show me groundwater levels in Delhi"
"Rainfall data for Karnataka in 2023"
"Groundwater levels in Mumbai district"
```

### 🔮 Predictions
```
"Predict groundwater level for Maharashtra with 150mm rainfall"
"What will be the water level in Delhi if rainfall is 100mm?"
"Forecast groundwater for Karnataka with 200mm rain"
```

### 📈 Trends & Analysis
```
"Show rainfall trends in Tamil Nadu"
"How has groundwater changed in Maharashtra?"
"Compare rainfall between Delhi and Mumbai"
```

### 🗺️ Geographic Queries
```
"Which states have the highest rainfall?"
"Groundwater levels by district in Karnataka"
"Show me all available states"
```

## 🎛️ Interface Features

### Sidebar Quick Actions
- **State/District Selector**: Choose your area of interest
- **Quick Query Buttons**: One-click access to common queries
- **Data Summary**: Overview of available data
- **Metrics**: Key statistics about your dataset

### Chat Interface
- **Real-time Chat**: Natural conversation flow
- **Message History**: Persistent chat history during session
- **Rich Responses**: Formatted responses with emojis and styling
- **Clear Chat**: Reset conversation anytime

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning model integration
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

### Data Integration
- **Rainfall Data**: `data/rainfall.csv`
- **Groundwater Data**: `data/groundwater.csv`
- **Geographic Data**: `data/regions.geojson`
- **ML Model**: `models/groundwater_predictor.pkl`

### Architecture
```
GroundwaterChatbot
├── Data Loading & Processing
├── Entity Extraction (NLP)
├── Query Classification
├── Response Generation
└── UI Rendering
```

## 🎨 Customization

### Adding New Query Types
1. Add new keywords to the `generate_response()` method
2. Create a new handler method (e.g., `handle_new_query_type()`)
3. Implement the logic in the handler method

### Modifying Responses
- Edit the response templates in each handler method
- Customize the styling in the CSS section
- Add new emojis and formatting as needed

### Extending Data Sources
- Modify the `load_data()` method to include new datasets
- Update entity extraction for new data fields
- Add new query handlers for additional data types

## 🐛 Troubleshooting

### Common Issues

**"Error loading data"**
- Ensure CSV files are in the `data/` directory
- Check file permissions and format

**"Model not found"**
- Run `train_model.py` to generate the ML model
- Verify the model file exists in `models/` directory

**"Dependencies missing"**
- Run `pip install -r backend/requirements.txt`
- Use the launcher script for automatic installation

**"Port already in use"**
- Change the port: `streamlit run chatbot.py --server.port 8503`
- Or stop other Streamlit instances

### Performance Tips
- The chatbot caches data for better performance
- Large datasets may take time to load initially
- Use specific queries rather than broad requests for faster responses

## 🔮 Future Enhancements

### Planned Features
- **Voice Input**: Speech-to-text integration
- **Advanced Visualizations**: Interactive charts and maps
- **Multi-language Support**: Support for regional languages
- **Export Functionality**: Download query results
- **API Integration**: REST API for external applications

### Advanced AI Features
- **Sentiment Analysis**: Understand user intent better
- **Contextual Memory**: Remember conversation context
- **Smart Suggestions**: Proactive data insights
- **Anomaly Detection**: Identify unusual patterns

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure data files are in the correct format
4. Check the console for error messages

## 🎯 Use Cases

### For Researchers
- Quick data access and analysis
- Hypothesis testing with predictions
- Trend analysis across regions

### For Policy Makers
- Regional water resource assessment
- Impact analysis of rainfall patterns
- Planning and decision support

### For Students
- Learning about water resources
- Understanding data analysis
- Exploring machine learning applications

---

**Happy Chatting! 💬🤖**

*The Groundwater Assistant is here to help you understand and analyze water resources data with the power of AI.*

