#!/usr/bin/env python3
"""
Verification script for FastAPI project setup.
Tests basic functionality without external dependencies.
"""

import sys
import os

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from src.config import get_settings, is_production, is_development
        from src.api.main import create_app
        from src.api.middleware.logging import LoggingMiddleware
        from src.api.middleware.rate_limiting import RateLimitMiddleware
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Test basic settings
        assert settings.app_name == "Ceramic Armor ML API"
        assert settings.app_version == "1.0.0"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        
        print("✓ Configuration loading successful")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation."""
    try:
        from src.api.main import create_app
        app = create_app()
        
        # Check app properties
        assert app.title == "Ceramic Armor ML API"
        assert app.version == "1.0.0"
        
        print("✓ FastAPI app creation successful")
        return True
    except Exception as e:
        print(f"✗ App creation error: {e}")
        return False

def test_directory_structure():
    """Test that all required directories exist."""
    required_dirs = [
        "src",
        "src/api",
        "src/api/routes",
        "src/api/models", 
        "src/api/middleware",
        "src/ml",
        "src/static",
        "tests"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ Directory structure correct")
        return True

def test_required_files():
    """Test that all required files exist."""
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/api/middleware/__init__.py",
        "src/api/middleware/logging.py",
        "src/api/middleware/rate_limiting.py",
        "requirements.txt",
        "render.yaml",
        ".env.example",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def main():
    """Run all verification tests."""
    print("Verifying FastAPI project setup...")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_required_files,
        test_imports,
        test_configuration,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FastAPI project setup is complete.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())