# Deployment Guide - Ceramic Armor ML API

This document provides comprehensive deployment instructions for the Ceramic Armor ML API on various platforms.

## Table of Contents

1. [Render Deployment (Recommended)](#render-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Local Development](#local-development)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)

## Render Deployment (Recommended)

Render provides the easiest deployment option with automatic scaling, SSL certificates, and integrated monitoring.

### Prerequisites

- GitHub repository with your code
- Render account ([render.com](https://render.com))
- Materials Project API key
- NIST API key (optional)

### Quick Start

1. **Fork/Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd ceramic-armor-ml
   ```

2. **Run Pre-deployment Checks**
   ```bash
   python scripts/deploy.py
   ```

3. **Connect to Render**
   - Log in to Render dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

4. **Configure Environment Variables**
   Set these in Render dashboard:
   ```bash
   MATERIALS_PROJECT_API_KEY=your_api_key_here
   NIST_API_KEY=your_nist_key_here
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Monitor build logs
   - Access your deployed API at the provided URL

### Advanced Configuration

#### Custom Domain
1. Add custom domain in Render dashboard
2. Update CORS_ORIGINS environment variable:
   ```bash
   CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com
   ```

#### Scaling Configuration
Edit `render.yaml` for custom scaling:
```yaml
scaling:
  minInstances: 2
  maxInstances: 10
  targetMemoryPercent: 80
  targetCPUPercent: 80
```

#### Resource Allocation
Upgrade plan for better performance:
- **Starter**: 0.5 CPU, 512MB RAM (Free tier)
- **Standard**: 1 CPU, 2GB RAM ($7/month)
- **Pro**: 2 CPU, 4GB RAM ($25/month)

### Post-Deployment Validation

```bash
# Validate deployment
python scripts/validate_deployment.py https://your-app.onrender.com

# Save validation report
python scripts/validate_deployment.py https://your-app.onrender.com --output validation_report.json
```

## Docker Deployment

For containerized deployment on any platform supporting Docker.

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 2GB+ available RAM
- 5GB+ available disk space

### Production Deployment

1. **Build Production Image**
   ```bash
   docker build -t ceramic-armor-ml:latest .
   ```

2. **Run Production Container**
   ```bash
   docker run -d \
     --name ceramic-armor-ml \
     -p 8000:8000 \
     -e MATERIALS_PROJECT_API_KEY=your_key \
     -e ENVIRONMENT=production \
     -v $(pwd)/models:/app/models:ro \
     ceramic-armor-ml:latest
   ```

3. **Using Docker Compose**
   ```bash
   # Production deployment
   docker-compose up -d ceramic-armor-api
   
   # With Nginx reverse proxy
   docker-compose --profile nginx up -d
   ```

### Development with Docker

```bash
# Development with hot reload
docker-compose --profile dev up -d ceramic-armor-dev

# Access development server
curl http://localhost:8001/health
```

### Container Management

```bash
# View logs
docker-compose logs -f ceramic-armor-api

# Scale service
docker-compose up -d --scale ceramic-armor-api=3

# Update and restart
docker-compose pull && docker-compose up -d
```

## Local Development

For development and testing without containers.

### Prerequisites

- Python 3.10+
- pip or conda
- 4GB+ available RAM

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run Development Server**
   ```bash
   # Direct execution
   python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   
   # Or using the main module
   python src/api/main.py
   ```

4. **Access Application**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Development Tools

```bash
# Run tests
pytest tests/ -v

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Environment Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MATERIALS_PROJECT_API_KEY` | Materials Project API key | `mp-1234567890abcdef` |
| `ENVIRONMENT` | Deployment environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `CORS_ORIGINS` | See config | Allowed CORS origins |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per window |
| `RATE_LIMIT_WINDOW` | `3600` | Rate limit window (seconds) |
| `MAX_FILE_SIZE` | `10485760` | Max upload size (bytes) |

### Environment Files

- `.env.example`: Template with all variables
- `.env.production`: Production-specific settings
- `.env`: Local development (not in git)

## Monitoring and Maintenance

### Health Monitoring

```bash
# Basic health check
curl https://your-app.onrender.com/health

# Detailed status
curl https://your-app.onrender.com/api/v1/status

# Model information
curl https://your-app.onrender.com/api/v1/models/info
```

### Log Analysis

```bash
# View recent logs (Render)
render logs --service ceramic-armor-ml-api --tail

# Docker logs
docker-compose logs -f --tail=100 ceramic-armor-api
```

### Performance Monitoring

Key metrics to monitor:
- Response time (target: <500ms)
- Memory usage (target: <80%)
- CPU usage (target: <70%)
- Error rate (target: <1%)
- Request rate

### Maintenance Tasks

```bash
# Update dependencies
pip-compile requirements.in
pip install -r requirements.txt

# Model updates
# 1. Upload new models to models/ directory
# 2. Update model version in config
# 3. Restart service

# Security updates
# 1. Review dependency vulnerabilities
# 2. Update packages
# 3. Test and deploy
```

## Troubleshooting

### Common Issues

#### Build Failures

**Symptom**: Build fails during pip install
```bash
# Solution: Check requirements.txt
pip install -r requirements.txt --dry-run

# Fix dependency conflicts
pip-tools compile requirements.in
```

#### Health Check Failures

**Symptom**: Health check returns 503
```bash
# Check application logs
# Verify model loading
# Check memory usage
```

#### CORS Errors

**Symptom**: Browser blocks requests
```bash
# Update CORS_ORIGINS
export CORS_ORIGINS="https://your-domain.com"

# Check preflight requests
curl -X OPTIONS https://your-app.com/api/v1/status \
  -H "Origin: https://your-domain.com" \
  -H "Access-Control-Request-Method: GET"
```

#### Rate Limiting Issues

**Symptom**: 429 Too Many Requests
```bash
# Increase rate limits
export RATE_LIMIT_REQUESTS=200
export RATE_LIMIT_WINDOW=3600

# Check rate limit headers
curl -I https://your-app.com/health
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG

# Restart service
# Check detailed logs
```

### Performance Issues

```bash
# Check resource usage
curl https://your-app.com/api/v1/status

# Monitor response times
time curl https://your-app.com/health

# Profile memory usage
# Add memory profiling to application
```

### Getting Help

1. **Check Logs**: Always start with application logs
2. **Validate Configuration**: Use deployment validation script
3. **Test Locally**: Reproduce issues in development
4. **Check Dependencies**: Verify all requirements are met
5. **Monitor Resources**: Check CPU, memory, and disk usage

### Support Resources

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)
- **Application Logs**: Available in deployment platform
- **Health Endpoints**: `/health`, `/api/v1/status`

## Security Considerations

### API Keys
- Store securely in environment variables
- Never commit to version control
- Rotate regularly
- Use least-privilege access

### Network Security
- Use HTTPS in production
- Configure CORS appropriately
- Implement rate limiting
- Monitor for abuse patterns

### Container Security
- Use non-root user in containers
- Keep base images updated
- Scan for vulnerabilities
- Limit container permissions

### Data Security
- Validate all inputs
- Sanitize file uploads
- Log security events
- Implement proper error handling

---

For additional support or questions, please refer to the main project documentation or create an issue in the repository.