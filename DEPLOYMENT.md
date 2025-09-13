# Render Deployment Guide

This guide covers the deployment of the Ceramic Armor ML API to Render.

## Prerequisites

1. **Render Account**: Create an account at [render.com](https://render.com)
2. **GitHub Repository**: Code must be in a GitHub repository
3. **API Keys**: Obtain required API keys for external services

## Deployment Steps

### 1. Connect Repository to Render

1. Log in to your Render dashboard
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Select the repository containing this project

### 2. Configure Service Settings

Render will automatically detect the `render.yaml` file and use its configuration. Key settings include:

- **Name**: `ceramic-armor-ml-api`
- **Environment**: `python`
- **Runtime**: `python-3.10`
- **Build Command**: Automatically installs dependencies and validates build
- **Start Command**: Runs FastAPI with uvicorn
- **Health Check**: `/health` endpoint

### 3. Set Environment Variables

The following environment variables need to be set manually in the Render dashboard:

#### Required API Keys
```bash
MATERIALS_PROJECT_API_KEY=your_materials_project_api_key_here
NIST_API_KEY=your_nist_api_key_here
```

#### Optional Configuration Overrides
```bash
# Only set these if you need to override defaults
CORS_ORIGINS=https://your-custom-domain.com
RATE_LIMIT_REQUESTS=200  # Increase if needed
LOG_LEVEL=DEBUG  # For debugging issues
```

### 4. Configure Custom Domain (Optional)

1. In Render dashboard, go to your service settings
2. Navigate to "Custom Domains"
3. Add your domain and configure DNS
4. Update `CORS_ORIGINS` environment variable with your domain

### 5. Monitor Deployment

1. **Build Logs**: Monitor the build process in Render dashboard
2. **Deploy Logs**: Check deployment logs for any issues
3. **Health Check**: Verify `/health` endpoint responds correctly
4. **API Documentation**: Access `/docs` for API documentation (disabled in production)

## Configuration Details

### Auto-Scaling

The service is configured with basic auto-scaling:
- **Min Instances**: 1
- **Max Instances**: 3
- **CPU Target**: 70%
- **Memory Target**: 70%

### Persistent Storage

A 2GB disk is mounted at `/opt/render/project/models` for ML model storage.

### Security Features

- **CORS**: Configured for production domains
- **Rate Limiting**: 100 requests per hour per IP
- **Security Headers**: Enabled for production
- **Input Sanitization**: Enabled for all endpoints

## Environment-Specific Configuration

### Production Environment
- Debug mode disabled
- API documentation disabled
- Enhanced security headers
- Structured logging
- Error tracking

### Development Environment
- Debug mode enabled
- API documentation available at `/docs`
- Relaxed CORS settings
- Verbose logging

## Monitoring and Maintenance

### Health Checks
- **Endpoint**: `/health`
- **Frequency**: Every 30 seconds
- **Timeout**: 10 seconds

### Logging
- **Level**: INFO (configurable via `LOG_LEVEL`)
- **Format**: Structured JSON logging
- **Retention**: Managed by Render

### Performance Monitoring
- **Response Times**: Logged for all requests
- **Error Rates**: Tracked and logged
- **Resource Usage**: Monitored by Render

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for dependency conflicts
   - Verify Python version compatibility
   - Review build logs in Render dashboard

2. **Health Check Failures**
   - Ensure `/health` endpoint is accessible
   - Check application startup logs
   - Verify port configuration

3. **API Key Issues**
   - Verify API keys are set in Render dashboard
   - Check key permissions and quotas
   - Review external API service status

4. **CORS Issues**
   - Update `CORS_ORIGINS` with correct domains
   - Verify domain configuration
   - Check browser developer tools for CORS errors

### Debug Mode

To enable debug mode temporarily:
1. Set `DEBUG=true` in environment variables
2. Set `LOG_LEVEL=DEBUG`
3. Redeploy the service
4. Remember to disable debug mode in production

### Scaling

To handle increased traffic:
1. Upgrade to Standard or Pro plan
2. Increase `maxInstances` in render.yaml
3. Monitor resource usage and adjust as needed

## Security Considerations

### API Keys
- Never commit API keys to version control
- Use Render's environment variable management
- Rotate keys regularly

### CORS Configuration
- Restrict origins to known domains
- Avoid using wildcards in production
- Regularly review and update allowed origins

### Rate Limiting
- Monitor for abuse patterns
- Adjust limits based on usage patterns
- Consider implementing user-based rate limiting

## Support and Resources

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Application Logs**: Available in Render dashboard
- **Health Status**: Monitor via `/health` endpoint

## Deployment Checklist

- [ ] Repository connected to Render
- [ ] Environment variables configured
- [ ] API keys set securely
- [ ] Health check endpoint working
- [ ] CORS origins configured correctly
- [ ] Custom domain configured (if applicable)
- [ ] Monitoring and alerting set up
- [ ] Documentation updated with deployment URLs