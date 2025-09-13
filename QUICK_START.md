# Quick Start Guide

Get the Ceramic Armor ML API up and running in minutes with this streamlined guide.

## 🚀 5-Minute Setup

### Prerequisites

- Python 3.10+ installed
- Materials Project API key ([Get one free here](https://materialsproject.org/api))
- 4GB+ RAM available

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd ceramic-armor-ml

# Run automated setup
python scripts/setup_deployment.py
```

The setup script will:
- ✅ Check system requirements
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Configure environment
- ✅ Validate model files
- ✅ Test configuration

### Step 2: Configure API Key

Edit the `.env` file created by setup:

```bash
# Required: Get your free API key from https://materialsproject.org/api
MATERIALS_PROJECT_API_KEY=mp-your-actual-key-here

# Optional: For enhanced data
NIST_API_KEY=your-nist-key-here
```

### Step 3: Start the Server

```bash
# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Start development server
python -m uvicorn src.api.main:app --reload
```

### Step 4: Test the API

Open your browser and visit:

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🧪 Quick Test

Test a prediction with curl:

```bash
curl -X POST http://localhost:8000/api/v1/predict/mechanical \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Expected response:
```json
{
  "status": "success",
  "predictions": {
    "fracture_toughness": {
      "value": 4.6,
      "unit": "MPa·m^0.5",
      "confidence_interval": [4.2, 5.0]
    },
    "vickers_hardness": {
      "value": 2800,
      "unit": "HV",
      "confidence_interval": [2650, 2950]
    }
  }
}
```

## 🚀 Deploy to Production

### Option 1: Render (Recommended - Free Tier Available)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Render**
   - Go to [render.com](https://render.com)
   - Connect GitHub repository
   - Set environment variables in dashboard
   - Deploy automatically with `render.yaml`

3. **Validate Deployment**
   ```bash
   python scripts/validate_deployment.py https://your-app.onrender.com
   ```

### Option 2: Docker

```bash
# Build and run with Docker
docker build -t ceramic-armor-ml .
docker run -p 8000:8000 \
  -e MATERIALS_PROJECT_API_KEY=your_key \
  ceramic-armor-ml
```

## 📚 What's Included

### 🤖 ML Models (8 Pre-trained Models)

**Mechanical Properties:**
- Fracture Toughness (R² = 0.87)
- Vickers Hardness (R² = 0.89)
- Density (R² = 0.92)
- Elastic Modulus (R² = 0.85)

**Ballistic Properties:**
- V50 Velocity (R² = 0.82)
- Penetration Resistance (R² = 0.80)
- Back-Face Deformation (R² = 0.78)
- Multi-Hit Capability (R² = 0.75)

### 🎯 Supported Materials

- **Silicon Carbide (SiC)**: High-performance armor
- **Boron Carbide (B₄C)**: Lightweight applications
- **Aluminum Oxide (Al₂O₃)**: Cost-effective solutions
- **Tungsten Carbide (WC)**: High-density applications
- **Titanium Carbide (TiC)**: Specialized composites
- **Multi-phase Composites**: Custom combinations

### 🔧 API Features

- **REST API**: Full OpenAPI/Swagger documentation
- **Web Interface**: User-friendly material input forms
- **Batch Processing**: CSV/Excel file uploads
- **Uncertainty Quantification**: Confidence intervals
- **Feature Importance**: SHAP-based explanations
- **Rate Limiting**: Built-in abuse protection
- **Health Monitoring**: Comprehensive status endpoints

## 🆘 Troubleshooting

### Common Issues

**❌ "Materials Project API key invalid"**
```bash
# Check your API key format (should start with 'mp-')
echo $MATERIALS_PROJECT_API_KEY

# Test the key
python -c "
from mp_api.client import MPRester
import os
with MPRester(os.getenv('MATERIALS_PROJECT_API_KEY')) as mpr:
    print('✅ API key works')
"
```

**❌ "Model files not found"**
```bash
# Check model files
ls -la models/
# Should show 8 .pkl/.joblib files

# Re-run setup if models are missing
python scripts/setup_deployment.py
```

**❌ "Port already in use"**
```bash
# Use different port
python -m uvicorn src.api.main:app --port 8001

# Or kill existing process
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -ti:8000 | xargs kill
```

### Get Help

- 📖 **Full Documentation**: [README.md](README.md)
- 🔧 **Configuration Guide**: [docs/ENVIRONMENT_CONFIGURATION.md](docs/ENVIRONMENT_CONFIGURATION.md)
- 🚀 **Deployment Guide**: [docs/DEPLOYMENT_PROCESS.md](docs/DEPLOYMENT_PROCESS.md)
- 🔌 **API Integration**: [docs/API_INTEGRATION_GUIDE.md](docs/API_INTEGRATION_GUIDE.md)
- 💻 **Usage Examples**: [examples/api_examples.py](examples/api_examples.py)

### Validation Commands

```bash
# Validate configuration
python scripts/validate_config.py

# Test deployment
python scripts/validate_deployment.py http://localhost:8000

# Run comprehensive examples
python examples/api_examples.py
```

## 🎯 Next Steps

1. **Explore the Web Interface** at http://localhost:8000
2. **Try the API Examples** in `examples/api_examples.py`
3. **Read the Full Documentation** for advanced features
4. **Deploy to Production** using Render or Docker
5. **Integrate with Your Applications** using the REST API

## 🏆 Success Metrics

After setup, you should achieve:
- ⚡ **Response Times**: < 500ms for predictions
- 🎯 **Accuracy**: R² > 0.80 for all properties
- 🔒 **Security**: Rate limiting and input validation
- 📊 **Monitoring**: Health checks and status endpoints
- 🚀 **Scalability**: Ready for production deployment

---

**🎉 Congratulations!** You now have a production-ready ML API for ceramic armor property prediction. Start making predictions and building amazing materials science applications!