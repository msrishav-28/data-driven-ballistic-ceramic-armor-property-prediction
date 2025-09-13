# Deployment Process Guide

This guide provides step-by-step instructions for deploying the Ceramic Armor ML API to various platforms, with detailed procedures for Render (recommended), Docker, and other cloud platforms.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Render Deployment (Recommended)](#render-deployment-recommended)
3. [Docker Deployment](#docker-deployment)
4. [Alternative Cloud Platforms](#alternative-cloud-platforms)
5. [Post-Deployment Validation](#post-deployment-validation)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### 1. Code Preparation

```bash
# Ensure all tests pass
pytest tests/ -v --cov=src

# Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/

# Security scan
bandit -r src/

# Validate configuration
python scripts/validate_config.py
```

### 2. Dependencies and Requirements

```bash
# Update requirements.txt
pip-compile requirements.in

# Verify all dependencies install cleanly
pip install -r requirements.txt --dry-run

# Check for security vulnerabilities
pip-audit
```

### 3. Model Files Verification

```bash
# Verify all model files are present
python -c "
from pathlib import Path
model_path = Path('models')
required_models = [
    'ballistic_back_face_deformation.joblib',
    'ballistic_multi_hit_capability.joblib',
    'ballistic_penetration_resistance.pkl',
    'ballistic_v50_velocity.joblib',
    'mechanical_density.pkl',
    'mechanical_elastic_modulus.joblib',
    'mechanical_fracture_toughness.joblib',
    'mechanical_vickers_hardness.joblib'
]

missing = []
for model in required_models:
    if not (model_path / model).exists():
        missing.append(model)

if missing:
    print(f'❌ Missing models: {missing}')
else:
    print('✅ All model files present')
"
```

### 4. Environment Configuration

```bash
# Validate environment configuration
python scripts/validate_config.py

# Check API keys are valid
python -c "
from mp_api.client import MPRester
import os
try:
    with MPRester(os.getenv('MATERIALS_PROJECT_API_KEY')) as mpr:
        print('✅ Materials Project API key valid')
except Exception as e:
    print(f'❌ API key invalid: {e}')
"
```

### 5. Documentation Updates

- [ ] Update version in `src/config.py`
- [ ] Update CHANGELOG.md
- [ ] Verify README.md is current
- [ ] Update API documentation if needed

## Render Deployment (Recommended)

Render provides the easiest deployment with automatic scaling, SSL, and monitoring.

### Step 1: Repository Setup

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Verify render.yaml Configuration**
   ```yaml
   # render.yaml should include:
   services:
     - type: web
       name: ceramic-armor-ml-api
       env: python
       runtime: python-3.10
       buildCommand: |
         pip install --upgrade pip &&
         pip install -r requirements.txt &&
         python -c "import src.api.main; print('Build validation successful')"
       startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1
       healthCheckPath: /health
   ```

### Step 2: Create Render Service

1. **Login to Render Dashboard**
   - Go to [render.com](https://render.com)
   - Sign in with GitHub account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Select repository: `ceramic-armor-ml`

3. **Configure Service**
   - **Name**: `ceramic-armor-ml-api`
   - **Environment**: `Python`
   - **Build Command**: (auto-detected from render.yaml)
   - **Start Command**: (auto-detected from render.yaml)

### Step 3: Environment Variables Configuration

Set these variables in Render dashboard:

```bash
# Required Variables
MATERIALS_PROJECT_API_KEY=your_actual_api_key_here
NIST_API_KEY=your_nist_api_key_here

# Production Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# CORS Configuration (update with your domain)
CORS_ORIGINS=https://ceramic-armor-ml-api.onrender.com

# Performance Settings
MODEL_CACHE_SIZE=10
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### Step 4: Deploy

1. **Initial Deployment**
   - Click "Create Web Service"
   - Monitor build logs in real-time
   - Wait for "Live" status

2. **Verify Deployment**
   ```bash
   # Check health endpoint
   curl https://ceramic-armor-ml-api.onrender.com/health
   
   # Run deployment validation
   python scripts/validate_deployment.py https://ceramic-armor-ml-api.onrender.com
   ```

### Step 5: Custom Domain (Optional)

1. **Add Custom Domain**
   - Go to service settings
   - Click "Custom Domains"
   - Add your domain (e.g., api.your-domain.com)

2. **Update CORS Configuration**
   ```bash
   CORS_ORIGINS=https://api.your-domain.com,https://ceramic-armor-ml-api.onrender.com
   ```

### Step 6: Scaling Configuration

For production workloads, consider upgrading:

1. **Resource Plans**
   - **Starter**: Free (0.5 CPU, 512MB RAM)
   - **Standard**: $7/month (1 CPU, 2GB RAM)
   - **Pro**: $25/month (2 CPU, 4GB RAM)

2. **Auto-Scaling** (Pro plan)
   ```yaml
   # In render.yaml
   scaling:
     minInstances: 2
     maxInstances: 10
     targetMemoryPercent: 70
     targetCPUPercent: 70
   ```

## Docker Deployment

For containerized deployment on any Docker-compatible platform.

### Step 1: Build Docker Image

1. **Create Production Dockerfile**
   ```dockerfile
   FROM python:3.10-slim

   # Set working directory
   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir --upgrade pip && \
       pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY models/ ./models/
   COPY start_server.py .

   # Create non-root user
   RUN useradd --create-home --shell /bin/bash app && \
       chown -R app:app /app
   USER app

   # Expose port
   EXPOSE 8000

   # Health check
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:8000/health || exit 1

   # Start command
   CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build Image**
   ```bash
   # Build production image
   docker build -t ceramic-armor-ml:latest .
   
   # Build with version tag
   docker build -t ceramic-armor-ml:v1.0.0 .
   
   # Test image locally
   docker run -p 8000:8000 \
     -e MATERIALS_PROJECT_API_KEY=your_key \
     ceramic-armor-ml:latest
   ```

### Step 2: Docker Compose Deployment

1. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   
   services:
     ceramic-armor-api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - MATERIALS_PROJECT_API_KEY=${MATERIALS_PROJECT_API_KEY}
         - NIST_API_KEY=${NIST_API_KEY}
         - ENVIRONMENT=production
         - DEBUG=false
         - LOG_LEVEL=INFO
       volumes:
         - ./models:/app/models:ro
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
         - ./ssl:/etc/nginx/ssl:ro
       depends_on:
         - ceramic-armor-api
       restart: unless-stopped
   ```

2. **Deploy with Docker Compose**
   ```bash
   # Create environment file
   cp .env.production .env
   
   # Deploy services
   docker-compose up -d
   
   # Check status
   docker-compose ps
   
   # View logs
   docker-compose logs -f ceramic-armor-api
   ```

### Step 3: Container Registry

1. **Push to Docker Hub**
   ```bash
   # Tag image
   docker tag ceramic-armor-ml:latest your-username/ceramic-armor-ml:latest
   
   # Push to registry
   docker push your-username/ceramic-armor-ml:latest
   ```

2. **Deploy from Registry**
   ```bash
   # Pull and run from registry
   docker run -d \
     --name ceramic-armor-ml \
     -p 8000:8000 \
     -e MATERIALS_PROJECT_API_KEY=your_key \
     your-username/ceramic-armor-ml:latest
   ```

## Alternative Cloud Platforms

### AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize Application**
   ```bash
   eb init ceramic-armor-ml --platform python-3.10
   ```

3. **Create Environment**
   ```bash
   eb create production --instance-type t3.medium
   ```

4. **Set Environment Variables**
   ```bash
   eb setenv MATERIALS_PROJECT_API_KEY=your_key \
            ENVIRONMENT=production \
            DEBUG=false
   ```

5. **Deploy**
   ```bash
   eb deploy
   ```

### Google Cloud Run

1. **Build and Push Image**
   ```bash
   # Configure gcloud
   gcloud auth configure-docker
   
   # Build image
   docker build -t gcr.io/your-project/ceramic-armor-ml .
   
   # Push image
   docker push gcr.io/your-project/ceramic-armor-ml
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ceramic-armor-ml \
     --image gcr.io/your-project/ceramic-armor-ml \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars MATERIALS_PROJECT_API_KEY=your_key,ENVIRONMENT=production
   ```

### Heroku

1. **Install Heroku CLI**
   ```bash
   # Install CLI and login
   heroku login
   ```

2. **Create Application**
   ```bash
   heroku create ceramic-armor-ml-api
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set MATERIALS_PROJECT_API_KEY=your_key
   heroku config:set ENVIRONMENT=production
   heroku config:set DEBUG=false
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

## Post-Deployment Validation

### Automated Validation Script

Run the comprehensive validation script:

```bash
# Validate deployment
python scripts/validate_deployment.py https://your-app-url.com

# Save validation report
python scripts/validate_deployment.py https://your-app-url.com --output validation_report.json
```

### Manual Validation Checklist

1. **Health Check**
   ```bash
   curl https://your-app-url.com/health
   # Expected: {"status": "healthy", "version": "1.0.0", ...}
   ```

2. **API Documentation**
   ```bash
   curl https://your-app-url.com/docs
   # Should return HTML documentation page
   ```

3. **System Status**
   ```bash
   curl https://your-app-url.com/api/v1/status
   # Expected: {"overall_status": "healthy", ...}
   ```

4. **Model Information**
   ```bash
   curl https://your-app-url.com/api/v1/models/info
   # Expected: Model metadata for all 8 models
   ```

5. **Prediction Endpoints**
   ```bash
   # Test mechanical prediction
   curl -X POST https://your-app-url.com/api/v1/predict/mechanical \
     -H "Content-Type: application/json" \
     -d '{
       "composition": {"SiC": 0.85, "B4C": 0.10, "Al2O3": 0.05},
       "processing": {"sintering_temperature": 2100, "pressure": 80, "grain_size": 5.0},
       "microstructure": {"porosity": 0.01, "phase_distribution": "uniform"}
     }'
   ```

6. **Performance Testing**
   ```bash
   # Test response time
   time curl https://your-app-url.com/health
   
   # Load testing with Apache Bench
   ab -n 100 -c 10 https://your-app-url.com/health
   ```

### Validation Metrics

| Metric | Target | Command |
|--------|--------|---------|
| Health Check Response Time | < 200ms | `time curl /health` |
| Prediction Response Time | < 500ms | `time curl /api/v1/predict/mechanical` |
| Memory Usage | < 80% | Check platform dashboard |
| CPU Usage | < 70% | Check platform dashboard |
| Error Rate | < 1% | Monitor logs |

## Monitoring and Maintenance

### Health Monitoring

1. **Set Up Monitoring**
   ```bash
   # Create monitoring script
   cat > scripts/monitor.sh << 'EOF'
   #!/bin/bash
   URL="https://your-app-url.com"
   
   # Health check
   if curl -f "$URL/health" > /dev/null 2>&1; then
       echo "✅ Health check passed"
   else
       echo "❌ Health check failed"
       exit 1
   fi
   
   # Performance check
   RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$URL/health")
   if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
       echo "✅ Response time OK: ${RESPONSE_TIME}s"
   else
       echo "⚠️  Slow response time: ${RESPONSE_TIME}s"
   fi
   EOF
   
   chmod +x scripts/monitor.sh
   ```

2. **Set Up Cron Job**
   ```bash
   # Add to crontab for monitoring every 5 minutes
   */5 * * * * /path/to/scripts/monitor.sh
   ```

### Log Management

1. **View Logs**
   ```bash
   # Render
   render logs --service ceramic-armor-ml-api --tail
   
   # Docker
   docker-compose logs -f ceramic-armor-api
   
   # Heroku
   heroku logs --tail
   ```

2. **Log Analysis**
   ```bash
   # Error analysis
   grep "ERROR" logs/app.log | tail -20
   
   # Performance analysis
   grep "prediction_time" logs/app.log | awk '{print $NF}' | sort -n
   ```

### Performance Monitoring

1. **Key Metrics to Track**
   - Response times
   - Memory usage
   - CPU utilization
   - Request rate
   - Error rate
   - Model prediction accuracy

2. **Alerting Setup**
   ```bash
   # Example alert script
   cat > scripts/alert.sh << 'EOF'
   #!/bin/bash
   THRESHOLD=1.0
   RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "$URL/health")
   
   if (( $(echo "$RESPONSE_TIME > $THRESHOLD" | bc -l) )); then
       # Send alert (email, Slack, etc.)
       echo "ALERT: High response time: ${RESPONSE_TIME}s"
   fi
   EOF
   ```

## Rollback Procedures

### Render Rollback

1. **Via Dashboard**
   - Go to service dashboard
   - Click "Deploys" tab
   - Click "Rollback" on previous successful deploy

2. **Via Git**
   ```bash
   # Revert to previous commit
   git revert HEAD
   git push origin main
   # Render will auto-deploy the revert
   ```

### Docker Rollback

1. **Tag-Based Rollback**
   ```bash
   # Stop current version
   docker-compose down
   
   # Update docker-compose.yml to use previous tag
   # image: ceramic-armor-ml:v1.0.0  # Previous version
   
   # Deploy previous version
   docker-compose up -d
   ```

2. **Image-Based Rollback**
   ```bash
   # Pull previous image
   docker pull ceramic-armor-ml:v1.0.0
   
   # Stop and remove current container
   docker stop ceramic-armor-ml
   docker rm ceramic-armor-ml
   
   # Start with previous image
   docker run -d --name ceramic-armor-ml ceramic-armor-ml:v1.0.0
   ```

### Emergency Rollback Checklist

- [ ] Identify the issue and last known good version
- [ ] Notify stakeholders of rollback
- [ ] Execute rollback procedure
- [ ] Verify rollback success
- [ ] Monitor system stability
- [ ] Document incident and lessons learned

## Troubleshooting

### Common Deployment Issues

#### 1. Build Failures

**Symptoms:**
- Build process fails during pip install
- Missing dependencies errors

**Solutions:**
```bash
# Check requirements.txt
pip install -r requirements.txt --dry-run

# Update pip and setuptools
pip install --upgrade pip setuptools

# Clear pip cache
pip cache purge

# Use specific Python version
python3.10 -m pip install -r requirements.txt
```

#### 2. Model Loading Failures

**Symptoms:**
- 503 Service Unavailable
- "Model not found" errors

**Solutions:**
```bash
# Verify model files
ls -la models/
du -sh models/

# Check model loading
python -c "
from src.ml.model_loader import ModelLoader
loader = ModelLoader()
print(f'Loaded models: {list(loader.models.keys())}')
"

# Verify model file integrity
python -c "
import joblib
import pickle
from pathlib import Path

model_path = Path('models')
for model_file in model_path.glob('*.joblib'):
    try:
        joblib.load(model_file)
        print(f'✅ {model_file.name}')
    except Exception as e:
        print(f'❌ {model_file.name}: {e}')
"
```

#### 3. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow response times
- Container restarts

**Solutions:**
```bash
# Reduce model cache size
export MODEL_CACHE_SIZE=5

# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# Optimize model loading
# Consider lazy loading or model compression
```

#### 4. API Key Issues

**Symptoms:**
- 401 Unauthorized errors
- Materials Project API failures

**Solutions:**
```bash
# Verify API key format
echo $MATERIALS_PROJECT_API_KEY | grep -E '^mp-[a-zA-Z0-9]+$'

# Test API key
python -c "
from mp_api.client import MPRester
import os
try:
    with MPRester(os.getenv('MATERIALS_PROJECT_API_KEY')) as mpr:
        result = mpr.get_structure_by_material_id('mp-149')
        print('✅ API key works')
except Exception as e:
    print(f'❌ API key error: {e}')
"
```

### Debug Mode Deployment

For troubleshooting, temporarily enable debug mode:

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG

# Redeploy with debug settings
# Remember to disable debug mode in production!
```

### Getting Help

1. **Check Logs First**
   - Application logs
   - Platform logs (Render, Docker, etc.)
   - System logs

2. **Use Validation Scripts**
   ```bash
   python scripts/validate_config.py
   python scripts/validate_deployment.py https://your-app-url.com
   ```

3. **Test Locally**
   ```bash
   # Reproduce issue locally
   python -m uvicorn src.api.main:app --reload
   ```

4. **Check Platform Status**
   - Render status page
   - Docker daemon status
   - Cloud provider status

This comprehensive deployment process guide ensures reliable, repeatable deployments across all supported platforms.