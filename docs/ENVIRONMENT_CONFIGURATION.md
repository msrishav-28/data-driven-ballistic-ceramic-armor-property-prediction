# Environment Configuration Guide

This guide provides comprehensive instructions for configuring environment variables for the Ceramic Armor ML API across different deployment scenarios.

## Table of Contents

1. [Overview](#overview)
2. [Environment Files](#environment-files)
3. [Configuration Variables](#configuration-variables)
4. [Deployment-Specific Configuration](#deployment-specific-configuration)
5. [Security Best Practices](#security-best-practices)
6. [Validation and Testing](#validation-and-testing)
7. [Troubleshooting](#troubleshooting)

## Overview

The Ceramic Armor ML API uses environment variables for all configuration to ensure security, flexibility, and deployment portability. Configuration is managed through Pydantic Settings with automatic validation and type conversion.

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Environment Files** (.env, .env.production, etc.)
3. **Default Values** (lowest priority)

## Environment Files

### File Structure

```
ceramic_armor_ml/
├── .env.example          # Template with all variables and documentation
├── .env                  # Local development (not in git)
├── .env.production       # Production template
├── .env.testing          # Testing configuration
└── .env.docker          # Docker-specific configuration
```

### .env.example (Template)

```bash
# =============================================================================
# Ceramic Armor ML API - Environment Configuration Template
# =============================================================================
# Copy this file to .env and update values for your environment
# For production deployment, set these variables in your deployment platform

# Application Configuration
# ========================
APP_NAME=Ceramic Armor ML API
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# Server Configuration
# ===================
HOST=0.0.0.0
PORT=8000

# API Configuration
# ================
API_V1_PREFIX=/api/v1

# External API Keys
# ================
# Materials Project API key (required)
# Get your key at: https://materialsproject.org/api
MATERIALS_PROJECT_API_KEY=your_materials_project_api_key_here

# NIST API key (optional, for enhanced data)
NIST_API_KEY=your_nist_api_key_here

# CORS Configuration
# =================
# Comma-separated list of allowed origins
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=*

# Rate Limiting Configuration
# ==========================
# Number of requests per window
RATE_LIMIT_REQUESTS=100
# Window duration in seconds
RATE_LIMIT_WINDOW=3600

# Security Configuration
# =====================
ENABLE_SECURITY_HEADERS=true
ENABLE_INPUT_SANITIZATION=true
MAX_REQUEST_SIZE=10485760
TRUSTED_IPS=

# Content Security Policy
CSP_SCRIPT_SRC='self','unsafe-inline','unsafe-eval',https://cdn.jsdelivr.net,https://cdnjs.cloudflare.com
CSP_STYLE_SRC='self','unsafe-inline',https://cdn.jsdelivr.net,https://cdnjs.cloudflare.com

# Logging Configuration
# ====================
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Model Configuration
# ==================
# Number of models to keep in cache
MODEL_CACHE_SIZE=10
# Path to ML model files
MODEL_PATH=models/

# File Upload Configuration
# ========================
# Maximum file size in bytes (10MB default)
MAX_FILE_SIZE=10485760
# Allowed file extensions
ALLOWED_FILE_TYPES=.csv,.xlsx,.json

# Database Configuration (Future Use)
# ==================================
# DATABASE_URL=postgresql://user:password@localhost/dbname

# Monitoring Configuration
# =======================
# Enable performance monitoring
ENABLE_MONITORING=true
# Metrics collection interval in seconds
METRICS_INTERVAL=60

# Cache Configuration
# ==================
# Enable response caching
ENABLE_CACHING=true
# Cache TTL in seconds
CACHE_TTL=300
# Maximum cache size in MB
CACHE_MAX_SIZE=100
```

### Local Development (.env)

```bash
# Local Development Configuration
APP_NAME=Ceramic Armor ML API (Dev)
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Use localhost for development
HOST=127.0.0.1
PORT=8000

# Relaxed CORS for development
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000

# Higher rate limits for development
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Your actual API keys
MATERIALS_PROJECT_API_KEY=mp-your-actual-key-here
NIST_API_KEY=your-nist-key-here

# Disable security features for easier development
ENABLE_SECURITY_HEADERS=false
ENABLE_INPUT_SANITIZATION=false
```

### Production (.env.production)

```bash
# Production Configuration Template
APP_NAME=Ceramic Armor ML API
DEBUG=false
ENVIRONMENT=production
LOG_LEVEL=INFO

# Production security settings
ENABLE_SECURITY_HEADERS=true
ENABLE_INPUT_SANITIZATION=true
MAX_REQUEST_SIZE=10485760

# Strict CORS policy
CORS_ORIGINS=https://ceramic-armor-ml-api.onrender.com

# Production rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Performance optimizations
MODEL_CACHE_SIZE=15
ENABLE_CACHING=true
CACHE_TTL=600

# Note: Sensitive values should be set in deployment platform
# MATERIALS_PROJECT_API_KEY=set_in_render_dashboard
# NIST_API_KEY=set_in_render_dashboard
```

### Testing (.env.testing)

```bash
# Testing Configuration
APP_NAME=Ceramic Armor ML API (Test)
DEBUG=false
ENVIRONMENT=testing
LOG_LEVEL=WARNING

# Use test database
DATABASE_URL=sqlite:///test.db

# Disable external API calls in tests
MATERIALS_PROJECT_API_KEY=test_key
NIST_API_KEY=test_key

# Minimal caching for tests
MODEL_CACHE_SIZE=2
ENABLE_CACHING=false

# High rate limits for testing
RATE_LIMIT_REQUESTS=10000
RATE_LIMIT_WINDOW=60
```

## Configuration Variables

### Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | str | "Ceramic Armor ML API" | Application name |
| `APP_VERSION` | str | "1.0.0" | Application version |
| `DEBUG` | bool | false | Enable debug mode |
| `ENVIRONMENT` | str | "production" | Environment name |

### Server Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOST` | str | "0.0.0.0" | Server bind address |
| `PORT` | int | 8000 | Server port |
| `API_V1_PREFIX` | str | "/api/v1" | API version prefix |

### External API Settings

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `MATERIALS_PROJECT_API_KEY` | str | Yes | Materials Project API key |
| `NIST_API_KEY` | str | No | NIST database API key |

### CORS Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ORIGINS` | str | "*" | Allowed origins (comma-separated) |
| `CORS_ALLOW_CREDENTIALS` | bool | true | Allow credentials |
| `CORS_ALLOW_METHODS` | str | "GET,POST,PUT,DELETE,OPTIONS" | Allowed methods |
| `CORS_ALLOW_HEADERS` | str | "*" | Allowed headers |

### Rate Limiting Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | int | 100 | Requests per window |
| `RATE_LIMIT_WINDOW` | int | 3600 | Window duration (seconds) |

### Security Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_SECURITY_HEADERS` | bool | true | Enable security headers |
| `ENABLE_INPUT_SANITIZATION` | bool | true | Enable input sanitization |
| `MAX_REQUEST_SIZE` | int | 10485760 | Max request size (bytes) |
| `TRUSTED_IPS` | str | "" | Trusted IP addresses |

### Logging Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | str | "INFO" | Logging level |
| `LOG_FORMAT` | str | Standard format | Log message format |

### Model Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_CACHE_SIZE` | int | 10 | Number of cached models |
| `MODEL_PATH` | str | "models/" | Model files directory |

### File Upload Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_FILE_SIZE` | int | 10485760 | Max upload size (bytes) |
| `ALLOWED_FILE_TYPES` | str | ".csv,.xlsx,.json" | Allowed extensions |

## Deployment-Specific Configuration

### Render Deployment

#### Setting Environment Variables in Render

1. **Via Render Dashboard:**
   - Go to your service dashboard
   - Click "Environment" tab
   - Add variables one by one

2. **Via render.yaml:**
   ```yaml
   envVars:
     - key: MATERIALS_PROJECT_API_KEY
       sync: false  # Set manually in dashboard
     - key: DEBUG
       value: false
     - key: LOG_LEVEL
       value: INFO
   ```

#### Required Render Variables

```bash
# Essential for Render deployment
MATERIALS_PROJECT_API_KEY=your_actual_key
CORS_ORIGINS=https://your-app.onrender.com
ENVIRONMENT=production
DEBUG=false
```

#### Render-Specific Settings

```bash
# Render automatically provides PORT
# Don't set PORT in environment variables

# Use Render's provided domain
CORS_ORIGINS=https://ceramic-armor-ml-api.onrender.com

# Optimize for Render's infrastructure
MODEL_CACHE_SIZE=8  # Adjust based on plan
RATE_LIMIT_REQUESTS=100
```

### Docker Deployment

#### Docker Environment File (.env.docker)

```bash
# Docker-specific configuration
APP_NAME=Ceramic Armor ML API (Docker)
ENVIRONMENT=production
DEBUG=false

# Docker networking
HOST=0.0.0.0
PORT=8000

# Volume paths
MODEL_PATH=/app/models/

# Docker-optimized settings
MODEL_CACHE_SIZE=12
ENABLE_CACHING=true
```

#### Docker Compose Configuration

```yaml
version: '3.8'
services:
  ceramic-armor-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env.docker
    environment:
      - MATERIALS_PROJECT_API_KEY=${MATERIALS_PROJECT_API_KEY}
      - NIST_API_KEY=${NIST_API_KEY}
    volumes:
      - ./models:/app/models:ro
```

### Local Development

#### Development-Specific Settings

```bash
# Enable development features
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# Relaxed security for development
ENABLE_SECURITY_HEADERS=false
CORS_ORIGINS=*

# Higher limits for testing
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Fast iteration
MODEL_CACHE_SIZE=5
ENABLE_CACHING=false
```

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
   ```bash
   # Add to .gitignore
   .env
   .env.local
   .env.production
   ```

2. **Use different keys for different environments**
   ```bash
   # Development
   MATERIALS_PROJECT_API_KEY=mp-dev-key-here
   
   # Production
   MATERIALS_PROJECT_API_KEY=mp-prod-key-here
   ```

3. **Rotate keys regularly**
   - Set up key rotation schedule
   - Monitor key usage
   - Have backup keys ready

### Environment Variable Security

1. **Use deployment platform secret management**
   ```bash
   # Render: Set sync: false for sensitive variables
   - key: MATERIALS_PROJECT_API_KEY
     sync: false
   ```

2. **Validate sensitive variables**
   ```python
   # In config.py
   @validator('materials_project_api_key')
   def validate_api_key(cls, v):
       if not v or v == 'your_api_key_here':
           raise ValueError('Valid Materials Project API key required')
       return v
   ```

3. **Use environment-specific validation**
   ```python
   @validator('debug')
   def validate_debug_in_production(cls, v, values):
       if values.get('environment') == 'production' and v:
           raise ValueError('Debug mode cannot be enabled in production')
       return v
   ```

### CORS Security

1. **Restrict origins in production**
   ```bash
   # Development (permissive)
   CORS_ORIGINS=*
   
   # Production (restrictive)
   CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
   ```

2. **Use specific methods and headers**
   ```bash
   CORS_ALLOW_METHODS=GET,POST,OPTIONS
   CORS_ALLOW_HEADERS=Content-Type,Authorization
   ```

## Validation and Testing

### Configuration Validation Script

Create `scripts/validate_config.py`:

```python
#!/usr/bin/env python3
"""
Validate environment configuration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Settings

def validate_configuration():
    """Validate current configuration."""
    try:
        settings = Settings()
        print("✅ Configuration validation successful")
        
        # Check required settings
        if not settings.materials_project_api_key or settings.materials_project_api_key == 'your_api_key_here':
            print("❌ Materials Project API key not set")
            return False
        
        # Check environment-specific settings
        if settings.environment == 'production':
            if settings.debug:
                print("❌ Debug mode enabled in production")
                return False
            
            if settings.cors_origins == ['*']:
                print("⚠️  CORS origins set to wildcard in production")
        
        # Check model path
        model_path = Path(settings.model_path)
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            return False
        
        print(f"Environment: {settings.environment}")
        print(f"Debug mode: {settings.debug}")
        print(f"Log level: {settings.log_level}")
        print(f"Model cache size: {settings.model_cache_size}")
        print(f"Rate limit: {settings.rate_limit_requests}/{settings.rate_limit_window}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_configuration()
    sys.exit(0 if success else 1)
```

### Testing Configuration

```bash
# Validate current configuration
python scripts/validate_config.py

# Test with specific environment file
ENV_FILE=.env.production python scripts/validate_config.py

# Test configuration loading
python -c "from src.config import Settings; print(Settings().dict())"
```

### Environment-Specific Testing

```python
# tests/test_config.py
import pytest
import os
from src.config import Settings

def test_development_config():
    """Test development configuration."""
    os.environ.update({
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'MATERIALS_PROJECT_API_KEY': 'test-key'
    })
    
    settings = Settings()
    assert settings.environment == 'development'
    assert settings.debug is True

def test_production_config():
    """Test production configuration."""
    os.environ.update({
        'ENVIRONMENT': 'production',
        'DEBUG': 'false',
        'MATERIALS_PROJECT_API_KEY': 'prod-key'
    })
    
    settings = Settings()
    assert settings.environment == 'production'
    assert settings.debug is False

def test_invalid_config():
    """Test invalid configuration raises errors."""
    os.environ.update({
        'MATERIALS_PROJECT_API_KEY': 'your_api_key_here'  # Invalid placeholder
    })
    
    with pytest.raises(ValueError):
        Settings()
```

## Troubleshooting

### Common Configuration Issues

#### 1. API Key Not Working

**Symptoms:**
- 401 Unauthorized errors
- "Invalid API key" messages

**Solutions:**
```bash
# Check if key is set
echo $MATERIALS_PROJECT_API_KEY

# Validate key format
python -c "
import os
key = os.getenv('MATERIALS_PROJECT_API_KEY')
if key and key.startswith('mp-'):
    print('✅ Key format looks correct')
else:
    print('❌ Key format incorrect or missing')
"

# Test key with Materials Project
python -c "
from mp_api.client import MPRester
import os
try:
    with MPRester(os.getenv('MATERIALS_PROJECT_API_KEY')) as mpr:
        print('✅ API key is valid')
except:
    print('❌ API key is invalid')
"
```

#### 2. CORS Issues

**Symptoms:**
- Browser blocks requests
- "CORS policy" errors in console

**Solutions:**
```bash
# Check current CORS settings
python -c "from src.config import Settings; print(f'CORS Origins: {Settings().cors_origins}')"

# Test CORS headers
curl -H "Origin: https://your-domain.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     https://your-app.onrender.com/api/v1/predict/mechanical
```

#### 3. Rate Limiting Issues

**Symptoms:**
- 429 Too Many Requests errors
- Requests being blocked

**Solutions:**
```bash
# Check rate limit settings
python -c "
from src.config import Settings
s = Settings()
print(f'Rate limit: {s.rate_limit_requests} requests per {s.rate_limit_window} seconds')
"

# Check rate limit headers in response
curl -I https://your-app.onrender.com/health
```

#### 4. Model Loading Issues

**Symptoms:**
- 503 Service Unavailable
- "Model not found" errors

**Solutions:**
```bash
# Check model path
python -c "
from src.config import Settings
from pathlib import Path
model_path = Path(Settings().model_path)
print(f'Model path: {model_path}')
print(f'Exists: {model_path.exists()}')
if model_path.exists():
    models = list(model_path.glob('*.pkl')) + list(model_path.glob('*.joblib'))
    print(f'Found {len(models)} model files')
"

# Validate model loading
python -c "
from src.ml.model_loader import ModelLoader
loader = ModelLoader()
print(f'Loaded models: {list(loader.models.keys())}')
"
```

### Configuration Debugging

#### Enable Debug Logging

```bash
# Temporary debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m uvicorn src.api.main:app --reload

# Check configuration loading
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.config import Settings
settings = Settings()
print('Configuration loaded successfully')
"
```

#### Environment Variable Debugging

```bash
# List all environment variables
env | grep -E "(API_KEY|DEBUG|ENVIRONMENT|CORS|RATE_LIMIT)"

# Check specific variable
echo "Materials Project API Key: ${MATERIALS_PROJECT_API_KEY:0:10}..."

# Validate environment file loading
python -c "
from dotenv import load_dotenv
import os
load_dotenv('.env')
print('Environment variables loaded from .env')
for key in ['DEBUG', 'ENVIRONMENT', 'MATERIALS_PROJECT_API_KEY']:
    value = os.getenv(key, 'NOT SET')
    if 'API_KEY' in key and value != 'NOT SET':
        value = value[:10] + '...'
    print(f'{key}: {value}')
"
```

### Production Deployment Checklist

Before deploying to production, verify:

- [ ] `DEBUG=false`
- [ ] `ENVIRONMENT=production`
- [ ] Valid `MATERIALS_PROJECT_API_KEY` set
- [ ] Restrictive `CORS_ORIGINS` configured
- [ ] Appropriate rate limits set
- [ ] Security headers enabled
- [ ] Log level set to `INFO` or `WARNING`
- [ ] Model files accessible
- [ ] Configuration validation passes

```bash
# Run complete validation
python scripts/validate_config.py
python scripts/validate_deployment.py https://your-app.onrender.com
```

This comprehensive environment configuration guide ensures secure, flexible, and maintainable deployment across all environments.