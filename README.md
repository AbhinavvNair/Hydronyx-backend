# HydroAI: Groundwater Management System

A comprehensive groundwater forecasting, policy simulation, and optimization platform with real-time alerts and geospatial analytics.

## 🌊 Features

- **Forecasting**: Spatiotemporal GNN-based groundwater level predictions
- **Policy Simulation**: Counterfactual analysis for intervention scenarios
- **Optimization**: Geospatial placement of recharge structures
- **Real-time Alerts**: Live groundwater stress notifications
- **Location Insights**: Point-and-click groundwater analytics
- **Drivers Analysis**: Causal modeling of groundwater factors
- **Validation**: Model performance metrics and benchmarking
- **Authentication**: Secure JWT-based user system

## 🏗️ Architecture

```
Hydronix/
├── Hydronyx-backend/          # FastAPI Python backend
│   ├── backend/
│   │   ├── app.py            # Main FastAPI application
│   │   ├── auth_routes.py    # Authentication endpoints
│   │   ├── forecast_routes.py # Forecasting API
│   │   ├── policy_routes.py  # Policy simulation
│   │   ├── optimizer_routes.py # Geospatial optimization
│   │   ├── alerts_routes.py  # Real-time alerts
│   │   ├── location_routes.py # Location-based queries
│   │   ├── drivers_routes.py # Causal drivers
│   │   ├── validation_routes.py # Model validation
│   │   ├── database.py       # MongoDB connection
│   │   ├── models.py         # ML models
│   │   ├── spatiotemporal_gnn.py # GNN implementation
│   │   └── requirements.txt  # Python dependencies
│   ├── data/                 # Static datasets (CSV, GeoJSON)
│   ├── models/               # Trained model artifacts
│   ├── Dockerfile            # Backend container
│   └── .env.example          # Environment template
└── Hydronyx-frontend/        # Next.js TypeScript frontend
    ├── app/
    │   ├── alerts/           # Alerts dashboard
    │   ├── forecast/         # Forecasting UI
    │   ├── policy/           # Policy simulation UI
    │   ├── optimizer/        # Optimization interface
    │   ├── location-gw/      # Location insights
    │   ├── drivers/          # Drivers analysis
    │   ├── validation/       # Validation metrics
    │   ├── login/            # Auth pages
    │   └── context/          # Auth context
    ├── lib/                  # API client utilities
    ├── components/           # Reusable UI components
    ├── package.json          # Node dependencies
    ├── Dockerfile            # Frontend container
    └── .env.local.example    # Frontend env template
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **MongoDB 6+** (local or Docker)
- **Git**

### 1. Clone & Setup

```bash
git clone <repository-url>
cd Hydronix
```

### 2. Backend Setup

```bash
cd Hydronyx-backend

# Create virtual environment
python -3.11 -m venv .venv
.venv\Scripts\activate   # Windows
# .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

# Environment configuration
cp backend/.env.example backend/.env
# Edit backend/.env with your settings (MongoDB URL, secrets, etc.)
```

### 3. Frontend Setup

```bash
cd ../Hydronyx-frontend

# Install dependencies
npm install

# Environment configuration
cp .env.local.example .env.local
# Edit .env.local with API URL if needed
```

### 4. Start MongoDB (Docker example)

```bash
docker run -d --name mongo -p 27017:27017 -v mongo_data:/data/db mongo:6
```

### 5. Run Services

#### Backend (Terminal 1)

```bash
cd Hydronyx-backend/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (Terminal 2)

```bash
cd Hydronyx-frontend
npm run dev
```

Visit:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- API Health: http://localhost:8000/api/health

## 🐳 Docker Deployment

### Backend

```bash
cd Hydronyx-backend
docker build -t hydronyx-backend .
docker run --rm -p 8000:8000 --env-file backend/.env hydronyx-backend
```

### Frontend

```bash
cd Hydronyx-frontend
docker build -t hydronyx-frontend .
docker run --rm -p 3000:3000 hydronyx-frontend
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  backend:
    build: ./Hydronyx-backend
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=mongodb://mongo:27017/
    depends_on:
      - mongo

  frontend:
    build: ./Hydronyx-frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  mongo_data:
```

Run:

```bash
docker-compose up -d
```

## ⚙️ Configuration

### Backend Environment (.env)

```bash
# MongoDB
MONGODB_URL=mongodb://localhost:27017/
DATABASE_NAME=hydroai_db

# Authentication
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# URLs
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000

# Email (optional)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=noreply@hydronyx.local

# Rate limiting
RATE_LIMIT_WINDOW_SECONDS=60
RATE_LIMIT_MAX_PER_WINDOW=120

# Optional services
REDIS_URL=
SENTRY_DSN=

# GIS layers (optional)
LULC_LAYER_PATH=
PROTECTED_AREAS_PATH=
SOIL_PERMEABILITY_PATH=
DEM_SLOPE_PATH=
```

### Frontend Environment (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📚 API Documentation

### Authentication

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/refresh` - Token refresh
- `POST /api/auth/logout` - User logout

### Core Endpoints

- `GET /api/forecast` - Groundwater forecasts
- `POST /api/policy/simulate` - Policy simulation
- `POST /api/optimizer/run` - Geospatial optimization
- `GET /api/alerts` - Real-time alerts
- `POST /api/location/groundwater` - Location-based queries
- `GET /api/drivers` - Causal drivers analysis
- `GET /api/validation` - Model validation metrics

### Interactive Docs

Visit http://localhost:8000/docs for interactive OpenAPI documentation.

## 🧠 Models & Data

### Data Sources

- **Groundwater levels**: `data/groundwater.csv`
- **Rainfall**: `data/rainfall.csv`
- **GIS layers**: Optional GeoJSON/raster files for constraints

### Models

- **Spatiotemporal GNN**: `models/spatiotemporal_gnn.pkl`
- **Causal model**: `models/causal_model.pkl`
- **Forecasting models**: Various scikit-learn models

### Training

```bash
cd Hydronyx-backend/backend

# Train GNN
python train_gnn.py

# Train forecasting models
python train_model.py

# Build causal graph
python graph_builder.py
```

## 🔒 Security

- JWT-based authentication with refresh tokens
- Password hashing with Argon2
- Rate limiting per IP
- CORS configuration
- Environment-based secrets
- Optional Redis for distributed rate limiting

## 🚨 Alerts System

The alerts endpoint provides real-time groundwater stress notifications:

- **Live updates**: Simulated variations every minute
- **Severity levels**: Critical (>25m), High (>15m), Medium (>10m)
- **Trend analysis**: Improving/declining/stable
- **Bottom notifications**: Critical alerts popup

Configure polling frequency in the frontend (default: 60 seconds).

## 📊 Monitoring & Validation

- Model performance metrics via `/api/validation`
- Rate limiting logs
- Database connection health checks
- Optional Sentry integration for error tracking

## 🛠️ Development

### Code Style

- **Backend**: Python, Black formatter, type hints
- **Frontend**: TypeScript, ESLint, Prettier
- **UI**: Tailwind CSS, Lucide icons

### Testing

```bash
# Backend (placeholder)
cd Hydronyx-backend
pytest

# Frontend (placeholder)
cd Hydronyx-frontend
npm test
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB connection failed**
   - Ensure MongoDB is running on localhost:27017
   - Check MONGODB_URL in .env

2. **Module import errors**
   - Activate virtual environment
   - Install all requirements.txt dependencies

3. **Frontend API errors**
   - Verify backend is running on port 8000
   - Check NEXT_PUBLIC_API_URL in .env.local

4. **Missing data files**
   - Ensure data/ directory contains required CSV files
   - Check file paths in configuration

### Debug Mode

Enable debug logging by setting:

```bash
# Backend
LOG_LEVEL=DEBUG

# Frontend
NEXT_PUBLIC_DEBUG=true
```

## 📄 License

[Add your license information here]

## 🤝 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at /docs
- Review logs for error details

## 🧪 Validation & Reliability Methodology

- **Historical Backtesting:**  
  Compare predictions with historical CGWB groundwater data.

- **Spatial Consistency Checks:**  
  Ensure smooth and realistic predictions across nearby locations.

- **Physics Constraint Verification:**  
  Enforce hydrological laws (mass balance, realistic depletion).

- **Uncertainty Estimation:**  
  Provide confidence bands with each forecast.

- **Scenario Validation:**  
  Verify logical consistency of policy simulation outputs.

## 📊 Datasets Used

- **CGWB Groundwater Level Data (2000–2022)**  
- **Well-level monitoring data with latitude & longitude**
- **Seasonal groundwater observations (Jan, May, Aug, Nov)**

> Data sources include India WRIS and CGWB publications.

## 🚀 Deployment Status

- Backend APIs: ✅ Implemented  
- Frontend Dashboard: ✅ Implemented  
- PI-GNN Model: ✅ Integrated  
- SCM Simulation: ✅ Integrated  
- Coordinate-based Estimation: 🚧 In Progress  
- Real-time Data Integration: 🚧 Planned  

## 🔮 Future Enhancements

- Real-time sensor data integration
- Mobile application support
- User-specific groundwater alerts
- Advanced aquifer-level modeling
- API access for third-party platforms

## 👥 Target Users

- Farmers
- Policymakers
- Water Resource Planners
- Researchers & Academicians

## 📌 Project Status

**Development Project / Research-Oriented System**  
Designed for academic presentation, research evaluation, and scalable real-world deployment.

## 📜 License

This project is developed for academic and research purposes.  
Licensing details will be added upon public release.

## 🙌 Acknowledgements

- Central Ground Water Board (CGWB)
- India WRIS
- Open-source scientific Python ecosystem

## 📫 Contact

For collaboration or research discussions, please open an issue or contact the repository owner.

---

**Built with ❤️ for sustainable groundwater management**
