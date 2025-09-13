# Ceramic Armor ML API

A production-ready FastAPI-based web application for predicting ceramic armor material properties using machine learning. This system provides REST API endpoints for mechanical and ballistic property predictions, batch processing capabilities, and a modern web interface.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
# Double-click setup.bat or run in Command Prompt:
setup.bat
```

**macOS/Linux:**
```bash
# Make executable and run:
chmod +x setup.sh
./setup.sh
```

**Any Platform:**
```bash
# Run Python setup script:
python setup.py
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   ```

4. **Start the application:**
   ```bash
   python start_server.py
   ```

5. **Open in browser:** http://localhost:8000

## 📚 Complete Documentation

- **[📖 Getting Started Guide](GETTING_STARTED.md)** - Comprehensive setup and deployment guide
- **[🚀 Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions  
- **[🔌 API Integration Guide](docs/API_INTEGRATION_GUIDE.md)** - How to use the API
- **[📊 Batch Processing Guide](docs/batch_processing_guide.md)** - Bulk prediction processing

## 📋 Prerequisites

- Python 3.10+
- Materials Project API key ([Get one here](https://materialsproject.org/api)) - Optional for basic functionality
- 4GB+ RAM (8GB+ recommended for production)

### Local Development Setup

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd ceramic-armor-ml
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```
   
   Edit `.env` file with your API keys:
   ```bash
   MATERIALS_PROJECT_API_KEY=your_api_key_here
   NIST_API_KEY=your_nist_key_here  # Optional
   ```

4. **Run Application**
   ```bash
   # Development server with hot reload
   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   
   # Or using the start script
   python start_server.py
   ```

5. **Access Application**
   - **Web Interface**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

### Production Deployment on Render

1. **Fork Repository** and connect to Render
2. **Set Environment Variables** in Render dashboard:
   ```bash
   MATERIALS_PROJECT_API_KEY=your_api_key_here
   NIST_API_KEY=your_nist_key_here
   ```
3. **Deploy** - Render will automatically use `render.yaml` configuration
4. **Validate Deployment**:
   ```bash
   python scripts/validate_deployment.py https://your-app.onrender.com
   ```

## 📁 Project Structure

```
ceramic_armor_ml/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app initialization
│   │   ├── routes/              # API route handlers
│   │   │   ├── predictions.py   # ML prediction endpoints
│   │   │   ├── health.py        # Health check endpoints
│   │   │   └── upload.py        # File upload endpoints
│   │   ├── models/              # Pydantic request/response models
│   │   └── middleware/          # CORS, logging, rate limiting
│   ├── ml/                      # ML prediction services
│   │   ├── predictor.py         # Core prediction logic
│   │   ├── feature_extractor.py # Feature engineering
│   │   └── model_loader.py      # Model loading utilities
│   ├── static/                  # Web interface files
│   │   ├── index.html           # Main web interface
│   │   ├── css/styles.css       # Custom styling
│   │   └── js/app.js            # Frontend JavaScript
│   └── config.py                # Configuration management
├── models/                      # Pre-trained ML models (8 models)
├── data/                        # Sample data and processing
├── examples/                    # API usage examples
├── tests/                       # Comprehensive test suite
├── scripts/                     # Deployment and validation scripts
├── docs/                        # Additional documentation
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
├── docker-compose.yml           # Docker deployment
└── .env.example                 # Environment variables template
```

## 🔧 API Endpoints

### Core Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/mechanical` | POST | Predict mechanical properties |
| `/api/v1/predict/ballistic` | POST | Predict ballistic properties |
| `/api/v1/predict/batch` | POST | Batch processing via file upload |

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for load balancers |
| `/api/v1/status` | GET | Detailed system status |
| `/api/v1/models/info` | GET | ML model information |
| `/docs` | GET | Interactive API documentation |

### Example API Usage

```python
import requests

# Predict mechanical properties
response = requests.post(
    "https://your-app.onrender.com/api/v1/predict/mechanical",
    json={
        "composition": {
            "SiC": 0.85,
            "B4C": 0.10,
            "Al2O3": 0.05
        },
        "processing": {
            "sintering_temperature": 2100,
            "pressure": 80,
            "grain_size": 5.0
        },
        "microstructure": {
            "porosity": 0.01,
            "phase_distribution": "uniform"
        }
    }
)

predictions = response.json()
print(f"Fracture Toughness: {predictions['predictions']['fracture_toughness']['value']} MPa·m^0.5")
```

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MATERIALS_PROJECT_API_KEY` | Yes | - | Materials Project API key |
| `NIST_API_KEY` | No | - | NIST database API key |
| `DEBUG` | No | `false` | Enable debug mode |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `CORS_ORIGINS` | No | See config | Allowed CORS origins |
| `RATE_LIMIT_REQUESTS` | No | `100` | Requests per hour |
| `MAX_FILE_SIZE` | No | `10MB` | Max upload file size |

### Production Configuration

For production deployment, ensure these settings:

```bash
# Security
DEBUG=false
ENVIRONMENT=production
ENABLE_SECURITY_HEADERS=true

# Performance
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
MODEL_CACHE_SIZE=10

# CORS (update with your domain)
CORS_ORIGINS=https://your-domain.com
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_api_models.py -v          # API model tests
pytest tests/test_integration.py -v         # Integration tests
pytest tests/test_performance.py -v         # Performance tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### API Testing

```bash
# Test API endpoints directly
python examples/api_examples.py

# Validate deployment
python scripts/validate_deployment.py http://localhost:8000
```

## 📊 Supported Materials

The system supports predictions for ceramic armor materials including:

- **Silicon Carbide (SiC)**: High-performance armor applications
- **Boron Carbide (B₄C)**: Lightweight armor systems
- **Aluminum Oxide (Al₂O₃)**: Cost-effective armor solutions
- **Tungsten Carbide (WC)**: High-density applications
- **Titanium Carbide (TiC)**: Specialized composites
- **Multi-phase Composites**: Custom material combinations

### Predicted Properties

**Mechanical Properties:**
- Fracture Toughness (MPa·m^0.5)
- Vickers Hardness (HV)
- Density (g/cm³)
- Elastic Modulus (GPa)

**Ballistic Properties:**
- V50 Ballistic Limit Velocity (m/s)
- Penetration Resistance
- Back-Face Deformation (mm)
- Multi-Hit Capability

## 🔍 Model Performance

| Property | R² Score | RMSE | Confidence Intervals |
|----------|----------|------|---------------------|
| Fracture Toughness | 0.87 | 0.45 MPa·m^0.5 | ±15% |
| Vickers Hardness | 0.89 | 180 HV | ±12% |
| Density | 0.92 | 0.08 g/cm³ | ±8% |
| Elastic Modulus | 0.85 | 25 GPa | ±10% |
| V50 Velocity | 0.82 | 45 m/s | ±18% |
| Penetration Resistance | 0.80 | 0.12 | ±20% |

## 📚 Documentation

### 🚀 Getting Started
- **[Quick Start Guide](QUICK_START.md)**: Get running in 5 minutes
- **[Setup Script](scripts/setup_deployment.py)**: Automated environment setup

### 📖 Comprehensive Guides
- **[API Integration Guide](docs/API_INTEGRATION_GUIDE.md)**: Complete integration reference
- **[Environment Configuration](docs/ENVIRONMENT_CONFIGURATION.md)**: Configuration management
- **[Deployment Process](docs/DEPLOYMENT_PROCESS.md)**: Step-by-step deployment
- **[Deployment Guide](README_DEPLOYMENT.md)**: Platform-specific instructions

### 🔧 Technical Documentation
- **[API Documentation](docs/api_documentation_guide.md)**: Detailed API reference
- **[Batch Processing Guide](docs/batch_processing_guide.md)**: File upload and batch processing
- **[Performance Optimizations](docs/performance_optimizations.md)**: Performance tuning guide
- **[Logging Middleware Guide](docs/logging_middleware_guide.md)**: Monitoring and logging

### 💻 Examples and Usage
- **[API Examples](examples/api_examples.py)**: Complete usage examples
- **[Sample Data](examples/sample_batch_materials.csv)**: Example material data

## 🚀 Development

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Add Route Handler**
   ```python
   # src/api/routes/new_feature.py
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.post("/new-endpoint")
   async def new_endpoint():
       return {"message": "New feature"}
   ```

3. **Register Route**
   ```python
   # src/api/main.py
   from src.api.routes import new_feature
   
   app.include_router(new_feature.router, prefix="/api/v1")
   ```

4. **Add Tests**
   ```python
   # tests/test_new_feature.py
   def test_new_endpoint():
       response = client.post("/api/v1/new-endpoint")
       assert response.status_code == 200
   ```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 🔒 Security

- **Input Validation**: Comprehensive Pydantic model validation
- **Rate Limiting**: Configurable per-endpoint rate limits
- **CORS Protection**: Strict origin validation
- **Security Headers**: HSTS, CSP, and other security headers
- **File Upload Security**: Size limits and type validation
- **API Key Management**: Secure environment variable handling

## 📈 Monitoring

### Health Checks

```bash
# Basic health check
curl https://your-app.onrender.com/health

# Detailed system status
curl https://your-app.onrender.com/api/v1/status

# Model information
curl https://your-app.onrender.com/api/v1/models/info
```

### Performance Metrics

The API tracks and reports:
- Request/response times
- Memory usage
- CPU utilization
- Model prediction accuracy
- Error rates and types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` directory
- **Examples**: See `examples/api_examples.py`
- **Issues**: Create an issue on GitHub
- **API Status**: Check `/health` and `/api/v1/status` endpoints

## 🏆 Acknowledgments

- Materials Project for crystal structure data
- NIST for ceramic materials database
- FastAPI framework for excellent API development
- Render platform for seamless deployment