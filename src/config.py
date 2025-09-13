"""
Configuration management for Ceramic Armor ML API.
Handles environment variables and application settings.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="Ceramic Armor ML API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # API settings
    api_v1_prefix: str = Field(default="/api/v1", env="API_V1_PREFIX")
    
    # External API keys
    materials_project_api_key: Optional[str] = Field(default=None, env="MATERIALS_PROJECT_API_KEY")
    nist_api_key: Optional[str] = Field(default=None, env="NIST_API_KEY")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )
    
    # Rate limiting settings
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Security settings
    trusted_ips: List[str] = Field(default=[], env="TRUSTED_IPS")
    enable_security_headers: bool = Field(default=True, env="ENABLE_SECURITY_HEADERS")
    enable_input_sanitization: bool = Field(default=True, env="ENABLE_INPUT_SANITIZATION")
    max_request_size: int = Field(default=10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    
    # Content Security Policy settings
    csp_script_src: List[str] = Field(
        default=["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com"],
        env="CSP_SCRIPT_SRC"
    )
    csp_style_src: List[str] = Field(
        default=["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net", "https://cdnjs.cloudflare.com"],
        env="CSP_STYLE_SRC"
    )
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Model settings
    model_cache_size: int = Field(default=10, env="MODEL_CACHE_SIZE")
    model_path: str = Field(default="models/", env="MODEL_PATH")
    
    # Performance optimization settings
    prediction_workers: int = Field(default=4, env="PREDICTION_WORKERS")
    feature_extraction_workers: int = Field(default=4, env="FEATURE_EXTRACTION_WORKERS")
    enable_response_caching: bool = Field(default=True, env="ENABLE_RESPONSE_CACHING")
    enable_prediction_caching: bool = Field(default=True, env="ENABLE_PREDICTION_CACHING")
    enable_feature_caching: bool = Field(default=True, env="ENABLE_FEATURE_CACHING")
    response_cache_size: int = Field(default=1000, env="RESPONSE_CACHE_SIZE")
    response_cache_ttl: int = Field(default=3600, env="RESPONSE_CACHE_TTL")  # 1 hour
    prediction_cache_ttl: int = Field(default=1800, env="PREDICTION_CACHE_TTL")  # 30 minutes
    feature_cache_ttl: int = Field(default=3600, env="FEATURE_CACHE_TTL")  # 1 hour
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=[".csv", ".xlsx", ".json"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        protected_namespaces = ('settings_',)  # Avoid conflicts with model_ fields
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            """Parse environment variables with special handling for lists."""
            list_fields = [
                "cors_origins", "cors_allow_methods", "cors_allow_headers", 
                "allowed_file_types", "trusted_ips", "csp_script_src", "csp_style_src"
            ]
            if field_name in list_fields:
                return [item.strip() for item in raw_val.split(",")]
            return cls.json_loads(raw_val)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.environment.lower() == "development"


def get_cors_origins() -> List[str]:
    """Get CORS origins with environment-specific defaults."""
    if is_production():
        # In production, only allow specific domains
        return settings.cors_origins
    else:
        # In development, allow localhost variants
        return settings.cors_origins + [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ]