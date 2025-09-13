# Getting Started Guide - Ceramic Armor ML API

This comprehensive guide will help you get the Ceramic Armor ML API running locally and deployed to production on Render.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Running the Application Locally](#running-the-application-locally)
4. [Testing the Application](#testing-the-application)
5. [Deployment to Render](#deployment-to-render)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **Python 3.10 or 3.11** (recommended: 3.11)
- **Git** for version control
- **128GB RAM** (recommended for optimal ML model performance)
- **2GB+ free disk space** for models and dependencies

### Required Accounts
- **GitHub account** (for code repository)
- **Render account** (for deployment) - Sign up at [render.com](https://render.com)
- **Materials Project API key** (optional, for enhanced features)

### Development Tools (Optional but Recommended)
- **VS Code** with Python extension
- **Postman** or similar API testing tool
- **Git Bash** or terminal of choice

---

## 🚀 Local Development Setup

### Step 1: Clone and Navigate to Repository

```bash
# If you haven't cloned yet
git clone <your-repository-url>
cd data-driven-ballistic-ceramic-armor-property-prediction

# Or if you're already in the project directory
pwd  # Should show your project path
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation (should show venv in prompt)
which python  # Should point to venv/Scripts/python or venv/bin/python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify critical packages are installed
python -c "import fastapi, uvicorn, sklearn, pandas, numpy; print('✅ All critical packages installed')"
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example environment file
copy .env.example .env  # Windows
# cp .env.example .env    # macOS/Linux

# Edit .env file with your preferred text editor
notepad .env  # Windows
# nano .env     # macOS/Linux
```

**Edit the `.env` file with these values:**

```env
# Application Configuration
APP_NAME=Ceramic Armor ML API
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# Server Configuration
HOST=0.0.0.0
PORT=8000

# API Configuration
API_V1_PREFIX=/api/v1

# External API Keys (optional for basic functionality)
MATERIALS_PROJECT_API_KEY=your_api_key_here
NIST_API_KEY=your_nist_key_here

# CORS Configuration
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=*

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Security Configuration
ENABLE_SECURITY_HEADERS=true
ENABLE_INPUT_SANITIZATION=true
MAX_REQUEST_SIZE=10485760

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Model Configuration
MODEL_CACHE_SIZE=10
MODEL_PATH=models/

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=.csv,.xlsx,.json
```

### Step 5: Verify ML Models

```bash
# Check if ML models exist
dir models  # Windows
# ls -la models/  # macOS/Linux

# You should see these model files:
# - ballistic_back_face_deformation.joblib
# - ballistic_multi_hit_capability.joblib
# - ballistic_penetration_resistance.pkl
# - ballistic_v50_velocity.joblib
# - mechanical_density.pkl
# - mechanical_elastic_modulus.joblib
# - mechanical_fracture_toughness.joblib
# - mechanical_vickers_hardness.joblib
```

**If models are missing, they should be automatically created when you first run the application.**

---

## 🏃‍♂️ Running the Application Locally

### Method 1: Using the Start Script (Recommended)

```bash
# Make sure you're in the project root and venv is activated
python start_server.py
```

You should see output like:
```
Starting Ceramic Armor ML API server...
API will be available at: http://localhost:8000
API documentation at: http://localhost:8000/docs
Press Ctrl+C to stop the server
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Method 2: Using Uvicorn Directly

```bash
# Run with uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or with specific configuration
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

### Method 3: Using Python Module

```bash
# Run as Python module
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Verify the Application is Running

1. **Open your browser** and go to: http://localhost:8000
2. **Check the health endpoint**: http://localhost:8000/health
3. **View API documentation**: http://localhost:8000/docs
4. **Check detailed status**: http://localhost:8000/api/v1/status

You should see:
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ API documentation loads successfully
- ✅ Status endpoint shows system information

---

## 🧪 Testing the Application

### Quick Health Check

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","timestamp":"2025-09-14T...","version":"1.0.0"}
```

### Test Prediction Endpoints

#### Test Mechanical Prediction

```bash
# Create a test request file
cat > test_request.json << 'EOF'
{
  "composition": {
    "SiC": 0.6,
    "B4C": 0.3,
    "Al2O3": 0.1,
    "WC": 0.0,
    "TiC": 0.0
  },
  "processing": {
    "sintering_temperature": 1800,
    "pressure": 50,
    "grain_size": 10,
    "holding_time": 120,
    "heating_rate": 15,
    "atmosphere": "argon"
  },
  "microstructure": {
    "porosity": 0.02,
    "phase_distribution": "uniform",
    "interface_quality": "good",
    "pore_size": 1.0,
    "connectivity": 0.1
  },
  "include_uncertainty": true,
  "include_feature_importance": true
}
EOF

# Test mechanical prediction
curl -X POST http://localhost:8000/api/v1/predict/mechanical \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

#### Test Ballistic Prediction

```bash
# Test ballistic prediction
curl -X POST http://localhost:8000/api/v1/predict/ballistic \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Run Automated Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_main.py -v                    # API tests
python -m pytest tests/test_integration.py -v            # Integration tests
python -m pytest tests/test_performance.py -v            # Performance tests

# Run deployment validation tests
python -m pytest tests/test_deployment_validation.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Run Deployment Checklist

```bash
# Check deployment readiness
python scripts/deployment_checklist.py

# Should show: "Status: ✅ READY FOR DEPLOYMENT"
```

---

## 🚀 Deployment to Render

### Step 1: Prepare Repository

```bash
# Make sure all changes are committed
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Create Render Account and Service

1. **Sign up at [render.com](https://render.com)**
2. **Connect your GitHub account**
3. **Create a new Web Service**
4. **Select your repository**
5. **Configure the service:**

   - **Name**: `ceramic-armor-ml-api` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt && python -c "import src.api.main; print('Build validation successful')"
     ```
   - **Start Command**: 
     ```bash
     uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 30
     ```
   - **Plan**: Start with `Starter` (can upgrade later)

### Step 3: Configure Environment Variables in Render

In the Render dashboard, add these environment variables:

```env
# Application Configuration
APP_NAME=Ceramic Armor ML API
APP_VERSION=1.0.0
ENVIRONMENT=production
DEBUG=false

# Server Configuration
HOST=0.0.0.0
# PORT is automatically set by Render

# API Configuration
API_V1_PREFIX=/api/v1

# External API Keys (set your actual keys)
MATERIALS_PROJECT_API_KEY=your_actual_api_key_here
NIST_API_KEY=your_actual_nist_key_here

# CORS Configuration (update with your actual domain)
CORS_ORIGINS=https://your-app-name.onrender.com,https://www.your-app-name.onrender.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=*

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Security Configuration
ENABLE_SECURITY_HEADERS=true
ENABLE_INPUT_SANITIZATION=true
MAX_REQUEST_SIZE=10485760

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Model Configuration
MODEL_CACHE_SIZE=10
MODEL_PATH=models/

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=.csv,.xlsx,.json
```

### Step 4: Deploy

1. **Click "Create Web Service"**
2. **Monitor the build logs** - should take 5-10 minutes
3. **Wait for deployment to complete**
4. **Note your deployment URL**: `https://your-app-name.onrender.com`

---

## ✅ Post-Deployment Validation

### Automated Validation

```bash
# Run comprehensive deployment validation
python scripts/validate_render_deployment.py https://your-app-name.onrender.com

# Run load testing
python scripts/run_integration_tests.py --url https://your-app-name.onrender.com

# Expected output: "✅ DEPLOYMENT READY" with detailed test results
```

### Manual Validation

1. **Health Check**: Visit `https://your-app-name.onrender.com/health`
2. **API Status**: Visit `https://your-app-name.onrender.com/api/v1/status`
3. **Frontend**: Visit `https://your-app-name.onrender.com`
4. **API Docs**: Visit `https://your-app-name.onrender.com/docs` (may be disabled in production)

### Test API Endpoints

```bash
# Replace YOUR_DEPLOYMENT_URL with your actual Render URL
export DEPLOYMENT_URL="https://your-app-name.onrender.com"

# Test health
curl $DEPLOYMENT_URL/health

# Test mechanical prediction
curl -X POST $DEPLOYMENT_URL/api/v1/predict/mechanical \
  -H "Content-Type: application/json" \
  -d @test_request.json

# Test ballistic prediction
curl -X POST $DEPLOYMENT_URL/api/v1/predict/ballistic \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

---

## 🔧 Troubleshooting

### Common Local Issues

#### Issue: "Module not found" errors
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Issue: "Port already in use"
```bash
# Solution: Kill process using port 8000 or use different port
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Or use different port:
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

#### Issue: ML models not loading
```bash
# Solution: Check models directory exists and has correct files
dir models
# If missing, models should be created automatically on first run
```

#### Issue: Environment variables not loading
```bash
# Solution: Ensure .env file exists and is properly formatted
type .env  # Windows
# cat .env   # macOS/Linux

# Check if variables are loaded:
python -c "from src.config import get_settings; print(get_settings().app_name)"
```

### Common Deployment Issues

#### Issue: Build fails on Render
- **Check build logs** in Render dashboard
- **Verify requirements.txt** has all dependencies
- **Ensure Python version compatibility** (use Python 3.10 or 3.11)

#### Issue: Application crashes on startup
- **Check deployment logs** in Render dashboard
- **Verify environment variables** are set correctly
- **Check start command** matches exactly: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

#### Issue: API endpoints return 500 errors
- **Check application logs** for detailed error messages
- **Verify ML models** are loading correctly
- **Check external API keys** if using Materials Project integration

#### Issue: Slow response times
- **Upgrade Render plan** for more resources
- **Check model loading** - models should be cached after first load
- **Monitor memory usage** - may need more RAM for ML models

### Getting Help

#### Check Logs
```bash
# Local development logs
# Check console output where you started the server

# Render deployment logs
# Check "Logs" tab in Render dashboard
```

#### Run Diagnostics
```bash
# Run deployment checklist
python scripts/deployment_checklist.py

# Run health checks
python scripts/validate_render_deployment.py http://localhost:8000  # Local
python scripts/validate_render_deployment.py https://your-app.onrender.com  # Production
```

#### Debug Mode
```bash
# Enable debug mode locally
# In .env file, set:
DEBUG=true
LOG_LEVEL=DEBUG

# Restart the application
```

---

## 🎉 Success Indicators

### Local Development Success
- ✅ Server starts without errors
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ API documentation loads at `/docs`
- ✅ Prediction endpoints return valid responses
- ✅ All tests pass: `python -m pytest tests/ -v`

### Deployment Success
- ✅ Render build completes successfully
- ✅ Application starts and stays running
- ✅ Health endpoint accessible via public URL
- ✅ Prediction endpoints work with sample data
- ✅ Deployment validation script passes: `python scripts/validate_render_deployment.py <your-url>`

### Performance Success
- ✅ Response times < 5 seconds for predictions
- ✅ Health endpoint responds < 1 second
- ✅ Application handles concurrent requests
- ✅ Memory usage remains stable

---

## 📚 Additional Resources

- **API Documentation**: Available at `/docs` when running locally
- **Project Documentation**: See `README.md` and `docs/` folder
- **Deployment Guide**: See `DEPLOYMENT.md`
- **API Integration Guide**: See `docs/API_INTEGRATION_GUIDE.md`
- **Render Documentation**: [render.com/docs](https://render.com/docs)

---

## 🚀 Quick Start Summary

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env with your settings

# 3. Run locally
python start_server.py

# 4. Test
curl http://localhost:8000/health

# 5. Deploy to Render
# - Push code to GitHub
# - Create Render service
# - Configure environment variables
# - Deploy

# 6. Validate deployment
python scripts/validate_render_deployment.py https://your-app.onrender.com
```

**🎯 You're now ready to run the Ceramic Armor ML API locally and deploy it to production!**

---

*For additional support or questions, refer to the troubleshooting section or check the project documentation.*