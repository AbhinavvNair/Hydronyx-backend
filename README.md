# 🌊 HydroAI - Physics-Informed Groundwater Management System

**Comprehensive AI-powered platform for groundwater forecasting, policy simulation, and geospatial optimization**

A full-stack application combining machine learning, causal inference, and geospatial analytics to provide intelligent decision support for sustainable groundwater management across India.

🔗 **Live Demo**: https://hydroai-pi.vercel.app  

---

## ✨ Complete Feature Set

### Backend Features (API)

#### 🔮 **Forecasting & Prediction**
- Spatiotemporal GNN-based groundwater level predictions with uncertainty quantification
- Multi-step forecasting (7-365 days ahead)
- Physics-informed constraints (water balance compliance scoring)
- Batch prediction endpoints for bulk analysis
- Confidence scoring with uncertainty bands

#### 📊 **Policy Simulation & Counterfactual Analysis**
- Structural causal model for what-if scenario analysis
- Policy intervention effect estimation
- Saved simulation history and PDF report generation
- Compare baseline vs. intervention trajectories
- Multi-variable intervention support (recharge, pumping, crop mix)

#### 📍 **Geospatial Optimization**
- Multi-objective optimization for recharge site placement
- Natural language query parsing for objectives
- Pareto frontier generation with impact/cost/equity/accessibility trade-offs
- Interactive map visualization with confidence scoring
- Website map-generated site prioritization

#### 🚨 **Real-time Alerts**
- Critical groundwater stress detection and classification
- State-level alert aggregation
- Severity trends and anomaly detection
- Alert subscription system with notifications
- Live stress indicator dashboard

#### 🎯 **Location-based Analytics**
- Point-and-click groundwater queries via IDW interpolation
- Nearby monitoring station identification
- Trend analysis at specific coordinates
- PDF report generation for locations
- Coordinate-based sharing

#### 🧠 **Causal Attribution & Drivers**
- Rainfall vs. pumping impact decomposition
- Recharge contribution analysis
- Factor attribution using causal inference
- Sensitivity analysis for robustness
- Decomposition visualization

#### ✅ **Model Validation & Benchmarking**
- Comprehensive performance metrics (RMSE, MAE, R², Physics Compliance Score)
- Data quality assessment
- Baseline vs. GNN model comparison
- Model performance time-series tracking

#### 🔐 **Enterprise Security**
- JWT-based authentication with refresh tokens
- Argon2 password hashing
- Role-based access control (RBAC): Admin, Analyst, Viewer
- Email verification and secure password reset
- IP-based rate limiting (configurable, default 120 req/min)

---

### Frontend Features (UI/UX)

#### 🔐 **Authentication & User Management**
- Secure JWT-based login/signup system
- Email verification workflow
- Password recovery and reset flow
- Role-based dashboard views (Admin/Analyst/Viewer)
- User profile and preferences management
- Persistent session management with token refresh

#### 📊 **Dashboard & Analytics**
- Real-time groundwater level metrics
- Forecast history with interactive charts
- Trend analysis visualization
- Model performance indicators
- Quick-access to key metrics and KPIs
- Responsive grid layout adapts to all screen sizes

#### 🔮 **Forecasting Interface**
- Interactive location picker (map-based)
- Temporal range selector (7-365 days ahead)
- Uncertainty visualization with confidence bands
- Real-time forecast generation
- Historical forecast comparison
- Physics compliance scoring display
- Batch prediction support

#### 📋 **Policy Simulator**
- Counterfactual analysis interface
- Intervention scenario builder
- What-if analysis for policy decisions
- Baseline vs. intervention trajectory comparison
- PDF report generation
- Saved simulations library
- Multi-variable intervention support

#### 📍 **Geospatial Optimizer**
- Interactive map-based site selection
- Multi-objective optimization criteria (Impact/Cost/Equity/Accessibility)
- Real-time optimization results
- Confidence heatmap visualization
- Site priority ranking
- Budget and constraint specification

#### 🗺️ **Location Insights (Point-and-Click)**
- Interactive map for location selection
- IDW interpolation-based queries
- Nearby monitoring station identification
- Local trend analysis
- Spatial statistics (mean, std, trend)
- PDF location report generation

#### 🚨 **Real-time Alerts Dashboard**
- Critical groundwater stress notifications
- State-level alert aggregation
- Severity classification (Critical/Warning/Normal)
- Trend indicators (improving/stable/declining)
- Historical alert patterns
- Alert filtering and search

#### 🧠 **Drivers Analysis**
- Causal attribution of groundwater changes
- Rainfall impact quantification
- Pumping contribution analysis
- Recharge effect decomposition
- Factor importance ranking
- Sensitivity analysis visualization

#### ✅ **Model Validation & Benchmarking**
- Comprehensive performance metrics display
- Data quality assessment dashboard
- Model comparison charts
- Temporal performance trends
- Download metric reports

#### 🏡 **My Farm/Property Management**
- User farm/property location tracking
- Property-specific analytics
- Historical performance on owned properties
- Alert subscriptions for managed locations
- Custom boundary drawing tools

---

## 🏗️ Complete Architecture & Project Structure

```
Hydronix2/
├── Hydronyx-backend/                 # Python FastAPI Backend
│   ├── backend/                      # Core application
│   │   ├── app.py                   # FastAPI entry point & middleware
│   │   ├── database.py              # MongoDB connection & lifecycle
│   │   ├── models.py                # Data schemas & models
│   │   │
│   │   ├── auth_routes.py           # Authentication endpoints
│   │   ├── auth_utils.py            # Auth helpers (token generation, password hashing)
│   │   ├── auth.py                  # Auth schemes (JWT bearer)
│   │   │
│   │   ├── forecast_routes.py       # Forecasting API
│   │   ├── policy_routes.py         # Policy simulation
│   │   ├── optimizer_routes.py      # Geospatial optimization
│   │   ├── alerts_routes.py         # Real-time alerts
│   │   ├── drivers_routes.py        # Causal drivers
│   │   ├── location_routes.py       # Location-based queries
│   │   ├── rainfall_routes.py       # Rainfall data
│   │   ├── validation_routes.py     # Model validation
│   │   │
│   │   ├── spatiotemporal_gnn.py    # GNN model architecture & inference
│   │   ├── causal_model.py          # Structural causal model
│   │   ├── geospatial_optimizer.py  # Multi-objective optimization
│   │   ├── gis_constraints.py       # Geospatial constraints
│   │   ├── graph_builder.py         # Station network graph
│   │   ├── model_utils.py           # ML utilities
│   │   │
│   │   ├── data_loader.py           # Data loading & caching
│   │   ├── data_preparation.py      # Data preprocessing
│   │   ├── validation_schemas.yaml  # Request/response schemas
│   │   │
│   │   ├── email_service.py         # Email notifications
│   │   ├── rainfall_service.py      # Rainfall data aggregation
│   │   ├── policy_pdf.py            # PDF generation
│   │   ├── compute_accuracy.py      # Performance metrics
│   │   ├── fix_gis.py               # Geospatial fixes
│   │   │
│   │   ├── users.json               # Demo user data
│   │   └── requirements.txt         # Python dependencies
│   │
│   ├── data/                        # Datasets
│   │   ├── groundwater.csv          # Historical groundwater levels
│   │   ├── rainfall.csv             # Historical rainfall data
│   │   ├── stations.csv             # Monitoring station metadata
│   │   └── regions.geojson          # State & district boundaries
│   │
│   ├── models/                      # Trained model artifacts
│   │   ├── gnn_model.pth            # Trained GNN (PyTorch)
│   │   ├── gnn_config.json          # GNN architecture config
│   │   └── sites_map.html           # Optimization visualization
│   │
│   ├── scripts/                     # Utility scripts
│   │   ├── train_gnn.py             # GNN training
│   │   ├── train_model.py           # Model training
│   │   ├── compute_accuracy.py      # Accuracy computation
│   │   ├── plot_forecast.py         # Forecast visualization
│   │   ├── sites_map.py             # Map generation
│   │   └── state_analysis_plots.py  # Regional analysis
│   │
│   ├── demo_chatbot.py              # Interactive demo chatbot
│   ├── run_chatbot.py               # Chatbot runner
│   ├── test_api.py                  # API tests
│   ├── test_data.py                 # Data validation tests
│   │
│   ├── Dockerfile                   # Docker definition
│   ├── Procfile                     # Heroku/Render config
│   ├── render.yaml                  # Render.com config
│   ├── requirements.txt             # Dependencies
│   └── README.md                    # Backend documentation
│
├── Hydronyx-frontend/               # Next.js 14 React Frontend
│   ├── app/                         # Next.js app directory
│   │   ├── layout.tsx               # Root layout
│   │   ├── page.tsx                 # Home page
│   │   ├── globals.css              # Global styles
│   │   │
│   │   ├── login/page.tsx           # Login page
│   │   ├── signup/page.tsx          # Registration page
│   │   ├── forgot-password/page.tsx # Password recovery
│   │   ├── reset-password/page.tsx  # Password reset
│   │   ├── verify-email/page.tsx    # Email verification
│   │   │
│   │   ├── dashboard/page.tsx       # Main dashboard
│   │   ├── forecast/page.tsx        # Forecasting UI
│   │   ├── policy/page.tsx          # Policy simulator
│   │   ├── policy/limitations/page.tsx # Policy limitations
│   │   ├── optimizer/page.tsx       # Optimizer UI
│   │   ├── optimizer/OptimizerMap.tsx # Optimization map
│   │   │
│   │   ├── location-gw/page.tsx     # Location queries
│   │   ├── alerts/page.tsx          # Alerts dashboard
│   │   ├── drivers/page.tsx         # Drivers analysis
│   │   ├── validation/page.tsx      # Validation metrics
│   │   │
│   │   ├── my-farm/page.tsx         # Farm management
│   │   ├── my-farm/MyFarmMap.tsx    # Farm map
│   │   ├── simulator/page.tsx       # Simulation tools
│   │   ├── personas/page.tsx        # User personas
│   │   │
│   │   └── context/
│   │       ├── AuthContext.tsx      # Auth state management
│   │       └── PersonaContext.tsx   # Persona management
│   │
│   ├── components/                  # Reusable components
│   │   ├── AppShell.tsx             # Main layout
│   │   ├── ProtectedRoute.tsx       # Route protection
│   │   ├── ConfidenceMap.tsx        # Confidence visualization
│   │   ├── ConfidenceMapMap.tsx     # Confidence heatmap
│   │   ├── UncertaintyChart.tsx     # Uncertainty bands
│   │   ├── UncertaintyWarning.tsx   # Uncertainty alerts
│   │   ├── RiskDisclaimer.tsx       # Risk warnings
│   │   └── Footer.tsx               # Footer
│   │
│   ├── lib/
│   │   └── api.ts                   # Axios API client
│   │
│   ├── package.json                 # Dependencies & scripts
│   ├── tsconfig.json                # TypeScript config
│   ├── tailwind.config.js           # Tailwind theming
│   ├── postcss.config.js            # PostCSS config
│   ├── next.config.js               # Next.js config
│   │
│   ├── Dockerfile                   # Docker definition
│   ├── LICENSE                      # License
│   └── README.md                    # Frontend documentation
│
├── PROJECT_SUMMARY.md               # Comprehensive project overview
├── MASTER_README.md                 # This file
└── [Root package.json]              # Project root config
```

---

## 🚀 Quick Start (Full Stack)

### Prerequisites
- **Python**: 3.11 or 3.12
- **Node.js**: 18.17+ LTS
- **MongoDB**: 6.0+ (local or Atlas)
- **npm/yarn**: Package manager

### Backend Setup

1. **Navigate to Backend**
```bash
cd Hydronyx-backend
```

2. **Create Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Setup Environment Variables**
```bash
cp .env.example .env
# Edit .env with your MongoDB URL and configuration
```

Required `.env` variables:
```bash
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=hydroai_db
SECRET_KEY=your-secret-key
RATE_LIMIT_MAX_PER_WINDOW=120
RATE_LIMIT_WINDOW_SECONDS=60
```

5. **Run Backend Server**
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Backend available at: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Frontend Setup

1. **Navigate to Frontend**
```bash
cd Hydronyx-frontend
```

2. **Install Dependencies**
```bash
npm install
```

3. **Setup Environment Variables**
```bash
# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

4. **Run Frontend Server**
```bash
npm run dev
```

Frontend available at: http://localhost:3000

---

## 📚 Comprehensive API Reference

### Authentication Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/register` | Register new user | ✅ |
| POST | `/api/auth/login` | Login & get token | ✅ |
| POST | `/api/auth/logout` | Logout | ✅ |
| GET | `/api/auth/verify-email` | Verify email token | ❌ |
| POST | `/api/auth/refresh-token` | Refresh JWT token | ✅ |
| POST | `/api/auth/forgot-password` | Request password reset | ✅ |
| POST | `/api/auth/reset-password` | Confirm password reset | ✅ |

### Forecasting Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/forecast/predict` | Single location forecast | ✅ |
| POST | `/api/forecast/predict-batch` | Batch predictions | ✅ |
| GET | `/api/forecast/history` | User's forecast history | ✅ |
| GET | `/api/forecast/metrics` | Model performance metrics | ✅ |
| GET | `/api/forecast/states` | Available states list | ✅ |

### Policy Simulation Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/policy/counterfactual` | Run policy scenario | ✅ |
| POST | `/api/policy/intervention-effect` | Analyze intervention impact | ✅ |
| GET | `/api/policy/saved-interventions` | Retrieve saved simulations | ✅ |
| POST | `/api/policy/generate-report` | Generate PDF report | ✅ |

### Optimization Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/optimizer/recharge-sites` | Find optimal site locations | ✅ |
| POST | `/api/optimizer/multi-objective` | Multi-objective optimization | ✅ |
| GET | `/api/optimizer/map-visualization` | Optimization map data | ✅ |

### Alerts Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/alerts/critical` | Critical groundwater alerts | ✅ |
| GET | `/api/alerts/by-state` | State-specific alerts | ✅ |
| POST | `/api/alerts/subscribe` | Subscribe to notifications | ✅ |

### Drivers/Attribution Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/drivers/attribution` | Causal attribution | ✅ |
| GET | `/api/drivers/rainfall-impact` | Rainfall contribution | ✅ |
| GET | `/api/drivers/pumping-impact` | Pumping effect analysis | ✅ |

### Location Query Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/location/forecast` | Point-and-click forecast | ✅ |
| GET | `/api/location/nearby-stations` | Nearest stations (IDW) | ✅ |
| GET | `/api/location/trend-analysis` | Location trend analysis | ✅ |
| POST | `/api/location/report-pdf` | Generate location report | ✅ |

### Rainfall Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/rainfall/forecast` | Rainfall forecast | ✅ |
| GET | `/api/rainfall/historical` | Historical rainfall | ✅ |

### Validation Endpoints
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/validation/metrics` | Performance metrics | ✅ |
| GET | `/api/validation/data-quality` | Data quality report | ✅ |
| GET | `/api/validation/benchmark` | Model benchmarks | ✅ |

---

## 🛠️ Technology Stack

### Backend Stack
| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | FastAPI | 0.104+ |
| **Server** | Uvicorn | Latest |
| **Database** | MongoDB | 6.0+ |
| **Async Driver** | Motor | Latest |
| **Deep Learning** | PyTorch | 2.0+ |
| **ML** | scikit-learn | Latest |
| **Geospatial** | GeoPandas, Shapely | Latest |
| **Data Processing** | Pandas, NumPy | Latest |
| **Visualization** | Plotly | Latest |
| **PDF Generation** | ReportLab | Latest |
| **Authentication** | PyJWT, Passlib | Latest |
| **Async** | asyncio, aiohttp | Latest |

### Frontend Stack
| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | Next.js | 14.2.0 |
| **UI Library** | React | 18.2.0 |
| **Language** | TypeScript | 5.3.3 |
| **Styling** | Tailwind CSS | 3.4.1 |
| **HTTP Client** | Axios | 1.6.2 |
| **Maps** | React Leaflet | 4.2.1 |
| **Charts** | Recharts | 2.10.3 |
| **3D Graphics** | Three.js | 0.182.0 |
| **Icons** | Lucide React | 0.263.1 |
| **Build** | Next.js | Built-in |

### DevOps Stack
| Component | Technology |
|-----------|-----------|
| **Containerization** | Docker |
| **Frontend Hosting** | Vercel |
| **Backend Hosting** | Render.com |
| **Database Hosting** | MongoDB Atlas |
| **Version Control** | Git |

---

## 🧠 Machine Learning Models

### Spatiotemporal Graph Neural Network (GNN)
**File**: `backend/spatiotemporal_gnn.py`

- **Purpose**: Multi-step groundwater forecasting with physics constraints
- **Architecture**: Multi-head temporal attention + spatial graph convolutions
- **Inputs**: Historical GW levels, rainfall, station network topology
- **Outputs**: Predicted levels + uncertainty bounds + physics compliance score
- **Physics Constraints**: Water balance equations embedded in loss function
- **Model File**: `models/gnn_model.pth` (PyTorch checkpoint)

### Structural Causal Model (SCM)
**File**: `backend/causal_model.py`

- **Purpose**: Counterfactual policy analysis and causal attribution
- **Variables**: Rainfall → Recharge, Pumping, Crop Mix → GW Level
- **Mechanism**: Do-calculus for intervention simulation
- **Use Cases**: What-if analysis, policy effect estimation, sensitivity analysis

### Geospatial Multi-Objective Optimizer
**File**: `backend/geospatial_optimizer.py`

- **Purpose**: Multi-criteria optimized recharge site prioritization
- **Objectives**: Impact (effectiveness), Cost, Equity, Accessibility
- **Algorithm**: Weighted sum method with Pareto frontier
- **Constraints**: Budget, search radius, water balance feasibility

### Classical Baselines
- **Linear/Ridge Regression**: Simple baseline forecasts
- **Random Forest**: Feature importance & non-linear patterns
- **ARIMA**: Time-series statistical model
- **IDW Interpolation**: Spatial estimation at arbitrary coordinates

---

## 💾 Database Schema (MongoDB)

### Collections

**users**
```javascript
{
  email: String,
  password_hash: String,
  role: "admin|analyst|viewer",
  is_verified: Boolean,
  created_at: Date,
  updated_at: Date
}
```

**predictions**
```javascript
{
  user_id: ObjectId,
  location: {lat: Number, lng: Number},
  predicted_level: Number,
  confidence: Number,
  physics_score: Number,
  timestamp: Date
}
```

**forecast_history**
```javascript
{
  user_id: ObjectId,
  params: Object,
  result: {predictions: Array, metrics: Object},
  created_at: Date
}
```

**policy_simulations**
```javascript
{
  user_id: ObjectId,
  intervention: String,
  baseline_trajectory: Array,
  counterfactual_trajectory: Array,
  effect_size: Number,
  created_at: Date
}
```

**validation_runs**
```javascript
{
  metric: String,
  value: Number,
  model: String,
  timestamp: Date
}
```

---

## 📊 Datasets

Located in `Hydronyx-backend/data/`:

**groundwater.csv**
- Columns: state, district, year_month, gw_level_m_bgl
- Coverage: 29+ Indian states, multi-year records
- Format: Monthly aggregates
- Source: CGWB groundwater monitoring

**rainfall.csv**
- Columns: state, year_month, rainfall_actual_mm, rainfall_normal_mm, deviation_pct
- Coverage: Multi-decadal rainfall records
- Source: IMD (India Meteorological Department)

**stations.csv**
- Columns: station_code, name, state, district, block, village, latitude, longitude, aquifer_type, well_depth
- Purpose: Spatial reference for IDW interpolation and location queries

**regions.geojson**
- Format: GeoJSON with state and district boundaries
- Geometry: Polygons for spatial querying
- Purpose: Geospatial filtering, boundary constraints in optimization

---

## 🚀 Deployment

### Backend Deployment (Render.com)

```yaml
# render.yaml
services:
  - type: web
    name: hydronyx-backend
    runtime: python
    buildCommand: pip install --upgrade pip && pip install geopandas && pip install -r requirements.txt
    startCommand: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
      - key: MONGODB_URL
        value: $MONGODB_URL
```

Deploy:
```bash
git push origin main  # Triggers Render auto-deployment
```

### Frontend Deployment (Vercel)

```bash
# Deploy with Vercel CLI
npm i -g vercel
vercel

# Or connect GitHub repo to Vercel dashboard for auto-deploy
```

**Live Deployment**: https://hydroai-pi.vercel.app

### Docker Deployment

```bash
# Backend
cd Hydronyx-backend
docker build -t hydroai-backend .
docker run -p 8000:8000 \
  -e MONGODB_URL="mongodb://host.docker.internal:27017" \
  -e SECRET_KEY="your-secret-key" \
  hydroai-backend

# Frontend
cd Hydronyx-frontend
docker build -t hydroai-frontend .
docker run -p 3000:3000 hydroai-frontend
```

---

## ⚙️ Configuration

### Backend Environment (.env)

```bash
# MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=hydroai_db

# Authentication
SECRET_KEY=your-super-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate Limiting
RATE_LIMIT_MAX_PER_WINDOW=120
RATE_LIMIT_WINDOW_SECONDS=60

# CORS (Development allows all)
# Production: allow_origins=["https://yourdomain.com"]
```

### Frontend Environment (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🧪 Testing & Demo

### Run Backend Tests
```bash
cd Hydronyx-backend

# Unit tests
python -m pytest backend/

# API endpoint tests
python test_api.py

# Data validation tests
python test_data.py
```

### Interactive Demo Chatbot
```bash
python run_chatbot.py
```

### Check API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📝 Frontend Route Reference

### Public Routes (No Auth)
- `/` - Landing page
- `/login` - Login
- `/signup` - Registration
- `/forgot-password` - Password recovery
- `/reset-password` - Password reset
- `/verify-email` - Email verification

### Protected Routes (Require Auth)
- `/dashboard` - Main dashboard
- `/forecast` - Forecasting interface
- `/policy` - Policy simulator
- `/policy/limitations` - Policy limitations
- `/optimizer` - Geospatial optimizer
- `/location-gw` - Location-based queries
- `/alerts` - Real-time alerts
- `/drivers` - Causal drivers analysis
- `/validation` - Model validation
- `/my-farm` - Farm management
- `/simulator` - Simulation tools
- `/personas` - User personas

---

## 🔐 Security Features

- **JWT Tokens**: Secure token-based authentication with refresh capability
- **Password Hashing**: Argon2 algorithm for secure password storage
- **RBAC**: Role-based access control (Admin, Analyst, Viewer)
- **Email Verification**: Email confirmation for new accounts
- **Rate Limiting**: IP-based sliding window rate limiting
- **CORS**: Configurable cross-origin resource sharing
- **Environment Secrets**: Sensitive configuration via .env files

---

## 🛠️ Troubleshooting

### MongoDB Connection
```bash
# Ensure MongoDB is running locally
mongod --dbpath /path/to/data

# Or use MongoDB Atlas
# Update .env: MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
```

### Frontend API Errors
```bash
# Verify backend is running on port 8000
curl http://localhost:8000/docs

# Check .env.local has correct API URL
cat .env.local
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Port Already in Use
```bash
# Backend (8000)
lsof -i :8000
kill -9 <PID>

# Frontend (3000)
lsof -i :3000
kill -9 <PID>
```

---

## 📊 Project Statistics

- **Backend Routes**: 30+ endpoints
- **Frontend Pages**: 18 pages
- **API Documentation**: Full Swagger/ReDoc
- **ML Models**: 4 advanced models (GNN, SCM, Optimizer, Baselines)
- **Datasets**: 4 comprehensive datasets (29+ states)
- **Database Collections**: 5 MongoDB collections
- **UI Components**: 50+ React components
- **Tech Stack**: 25+ dependencies

---

## 🎯 Use Cases

### For Farmers
- Check groundwater levels and trends at your location
- Get forecasts for water availability planning
- Understand rainfall and pumping impacts
- Participate in policy discussions

### For Policymakers
- Simulate policy interventions (what-if scenarios)
- Optimize recharge site placement
- Analyze causal factors driving groundwater
- Generate reports for stakeholder communication

### For Researchers
- Access to physics-informed deep learning models
- Causal inference framework for groundwater
- Geospatial optimization algorithms
- Multi-state groundwater dataset

### For Water Resource Planners
- Real-time alerts for groundwater stress
- Predictive forecasting for planning
- Multi-objective optimization for constraints
- Comprehensive validation and benchmarking

---

## 🔍 Development Tips

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Database Inspection
```bash
mongosh mongodb://localhost:27017/hydroai_db
db.users.find()
db.predictions.find()
```

### API Testing
```bash
curl -X POST "http://localhost:8000/api/forecast/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 28.7041, "longitude": 77.1025, "days_ahead": 30}'
```

---

## 🚀 Performance Optimization

- **GNN Inference**: Cached model weights for fast predictions
- **Batch Processing**: `/predict-batch` for bulk queries
- **MongoDB Indexing**: Automatic index creation on startup
- **Rate Limiting**: IP-based sliding window
- **Code Splitting**: Next.js automatic route-based splitting
- **Image Optimization**: Next.js built-in image optimization

---

## 📦 Key Dependencies

### Python (Backend)
FastAPI, Uvicorn, MongoDB, Motor, PyTorch, scikit-learn, GeoPandas, Pandas, NumPy, Plotly, ReportLab, PyJWT, Passlib

### Node.js (Frontend)
Next.js, React, TypeScript, Tailwind CSS, Axios, React Leaflet, Recharts, Three.js, Lucide React

See `requirements.txt` (backend) and `package.json` (frontend) for complete versions.

---

## 👥 Contributing

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Commit changes: `git commit -m 'Add amazing feature'`
3. Push to branch: `git push origin feature/amazing-feature`
4. Open a Pull Request

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact the development team
- Check the API documentation at `/docs`

---

## 📄 License

This project is developed for academic and research purposes.

---

## 🙏 Acknowledgements

- Central Ground Water Board (CGWB)
- India WRIS
- Open-source scientific Python ecosystem
- React and Next.js communities

---

## 🌟 Project Highlights

✅ **Full-stack application** - Complete frontend + backend + database  
✅ **Advanced ML models** - Physics-informed GNN + Causal models  
✅ **Production-ready** - Deployed on Vercel + Render  
✅ **Comprehensive APIs** - 30+ endpoints fully documented  
✅ **Beautiful UI** - Modern React + Tailwind CSS interface  
✅ **Secure** - JWT auth, rate limiting, RBAC  
✅ **Scalable** - MongoDB, async/await, containerized  
✅ **Research-oriented** - Causal inference, multi-objective optimization  

---

**Built with ❤️ for sustainable groundwater management**

**Last Updated**: March 29, 2026  
**Version**: 2.0  
**Status**: Production Ready ✅
