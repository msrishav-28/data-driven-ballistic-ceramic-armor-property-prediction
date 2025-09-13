# 🚀 START HERE - Ceramic Armor ML API

Welcome! This guide will get you up and running quickly.

## 🎯 What do you want to do?

### 🏃‍♂️ I want to run this locally RIGHT NOW
**→ Use the automated setup:**

**Windows:** Double-click `setup.bat`  
**macOS/Linux:** Run `./setup.sh`  
**Any platform:** Run `python setup.py`

Then:
1. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
2. Start server: `python start_server.py`
3. Open browser: http://localhost:8000

### 📚 I want detailed instructions
**→ Read the [Getting Started Guide](GETTING_STARTED.md)**

This comprehensive guide covers:
- Prerequisites and system requirements
- Step-by-step local setup
- Testing the application
- Deployment to production
- Troubleshooting

### 🚀 I want to deploy to production
**→ Follow the [Deployment Guide](DEPLOYMENT.md)**

Or use the quick deployment checklist:
1. Run: `python scripts/deployment_checklist.py`
2. Should show: "✅ READY FOR DEPLOYMENT"
3. Deploy to Render using the configuration in `render.yaml`
4. Validate: `python scripts/validate_render_deployment.py <your-url>`

### 🔌 I want to use the API
**→ Check the [API Integration Guide](docs/API_INTEGRATION_GUIDE.md)**

Quick API test:
```bash
# Health check
curl http://localhost:8000/health

# Prediction example
curl -X POST http://localhost:8000/api/v1/predict/mechanical \
  -H "Content-Type: application/json" \
  -d '{
    "composition": {"SiC": 0.6, "B4C": 0.3, "Al2O3": 0.1, "WC": 0.0, "TiC": 0.0},
    "processing": {"sintering_temperature": 1800, "pressure": 50, "grain_size": 10, "holding_time": 120, "heating_rate": 15, "atmosphere": "argon"},
    "microstructure": {"porosity": 0.02, "phase_distribution": "uniform", "interface_quality": "good", "pore_size": 1.0, "connectivity": 0.1}
  }'
```

### 🧪 I want to run tests
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run all tests
python -m pytest tests/ -v

# Run deployment validation
python scripts/deployment_checklist.py
```

### ❓ I need help
1. **Check [GETTING_STARTED.md](GETTING_STARTED.md)** - Most comprehensive guide
2. **Check the troubleshooting section** in the Getting Started guide
3. **Run diagnostics:** `python scripts/deployment_checklist.py`
4. **Check logs** where you started the server

## 🎯 Success Indicators

You'll know everything is working when:
- ✅ Server starts without errors
- ✅ http://localhost:8000/health returns `{"status": "healthy"}`
- ✅ http://localhost:8000/docs shows API documentation
- ✅ Prediction endpoints return valid responses

## 📁 Key Files

- `setup.py` / `setup.bat` / `setup.sh` - Automated setup scripts
- `start_server.py` - Start the application
- `requirements.txt` - Python dependencies
- `.env.example` - Environment configuration template
- `render.yaml` - Deployment configuration
- `GETTING_STARTED.md` - Comprehensive guide
- `tests/` - Test suites
- `scripts/` - Utility scripts

## 🚀 Quick Commands

```bash
# Setup (choose one)
python setup.py        # Any platform
setup.bat              # Windows
./setup.sh             # macOS/Linux

# Activate environment
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux

# Start application
python start_server.py

# Test application
python -m pytest tests/ -v

# Check deployment readiness
python scripts/deployment_checklist.py

# Validate deployment
python scripts/validate_render_deployment.py <url>
```

---

**🎉 That's it! Choose your path above and get started!**