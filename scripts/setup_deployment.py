#!/usr/bin/env python3
"""
Comprehensive setup script for Ceramic Armor ML API deployment.

This script automates the setup process for local development and deployment
preparation, including environment configuration, dependency installation,
and validation.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class DeploymentSetup:
    """Handles deployment setup and configuration."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.src_path = self.project_root / "src"
        
        # Add src to Python path
        sys.path.insert(0, str(self.src_path))
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        print(f"Running: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            if check:
                raise
            return e
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 10:
            print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_system_dependencies(self) -> bool:
        """Check for required system dependencies."""
        print("🔧 Checking system dependencies...")
        
        dependencies = {
            "git": ["git", "--version"],
            "curl": ["curl", "--version"],
        }
        
        missing = []
        for name, command in dependencies.items():
            try:
                result = subprocess.run(command, capture_output=True, check=True)
                print(f"✅ {name} available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"❌ {name} not found")
                missing.append(name)
        
        if missing:
            print(f"Please install missing dependencies: {', '.join(missing)}")
            return False
        
        return True
    
    def setup_virtual_environment(self) -> bool:
        """Set up Python virtual environment."""
        print("🏗️  Setting up virtual environment...")
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            print("✅ Virtual environment already exists")
            return True
        
        try:
            # Create virtual environment
            self.run_command([sys.executable, "-m", "venv", "venv"])
            print("✅ Virtual environment created")
            
            # Provide activation instructions
            if os.name == 'nt':  # Windows
                activate_cmd = r"venv\Scripts\activate"
            else:  # Unix/Linux/Mac
                activate_cmd = "source venv/bin/activate"
            
            print(f"💡 Activate with: {activate_cmd}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            self.run_command([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_file)
            ])
            
            print("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Set up environment configuration file."""
        print("⚙️  Setting up environment configuration...")
        
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if not env_example.exists():
            print("❌ .env.example not found")
            return False
        
        if env_file.exists():
            print("✅ .env file already exists")
            return True
        
        try:
            # Copy example file
            shutil.copy2(env_example, env_file)
            print("✅ .env file created from template")
            
            # Provide configuration instructions
            print("💡 Please edit .env file with your configuration:")
            print("   - Set MATERIALS_PROJECT_API_KEY")
            print("   - Set NIST_API_KEY (optional)")
            print("   - Adjust other settings as needed")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def validate_model_files(self) -> bool:
        """Validate that all required model files are present."""
        print("🤖 Validating ML model files...")
        
        model_path = self.project_root / "models"
        if not model_path.exists():
            print("❌ Models directory not found")
            return False
        
        required_models = [
            "ballistic_back_face_deformation.joblib",
            "ballistic_multi_hit_capability.joblib", 
            "ballistic_penetration_resistance.pkl",
            "ballistic_v50_velocity.joblib",
            "mechanical_density.pkl",
            "mechanical_elastic_modulus.joblib",
            "mechanical_fracture_toughness.joblib",
            "mechanical_vickers_hardness.joblib"
        ]
        
        missing_models = []
        for model_file in required_models:
            model_path_full = model_path / model_file
            if not model_path_full.exists():
                missing_models.append(model_file)
            else:
                # Check file size (should be > 1KB)
                size = model_path_full.stat().st_size
                if size < 1024:
                    print(f"⚠️  {model_file} seems too small ({size} bytes)")
        
        if missing_models:
            print(f"❌ Missing model files: {missing_models}")
            print("💡 Please ensure all model files are in the models/ directory")
            return False
        
        print(f"✅ All {len(required_models)} model files present")
        return True
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        print("🔍 Validating configuration...")
        
        try:
            # Import and validate settings
            from config import Settings
            
            settings = Settings()
            print("✅ Configuration loaded successfully")
            
            # Check critical settings
            if not settings.materials_project_api_key or settings.materials_project_api_key == "your_materials_project_api_key_here":
                print("⚠️  Materials Project API key not set")
                print("💡 Set MATERIALS_PROJECT_API_KEY in .env file")
                return False
            
            # Environment-specific validation
            if self.environment == "production":
                if settings.debug:
                    print("❌ Debug mode enabled in production")
                    return False
                
                if settings.cors_origins == ["*"]:
                    print("⚠️  CORS origins set to wildcard in production")
            
            print(f"✅ Configuration valid for {self.environment} environment")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def test_api_connectivity(self) -> bool:
        """Test connectivity to external APIs."""
        print("🌐 Testing API connectivity...")
        
        try:
            # Test Materials Project API
            from mp_api.client import MPRester
            import os
            
            api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
            if not api_key or api_key == "your_materials_project_api_key_here":
                print("⚠️  Materials Project API key not configured")
                return False
            
            with MPRester(api_key) as mpr:
                # Test with a simple query
                result = mpr.get_structure_by_material_id("mp-149")  # Silicon
                print("✅ Materials Project API connection successful")
            
            return True
            
        except Exception as e:
            print(f"❌ API connectivity test failed: {e}")
            print("💡 Check your API key and internet connection")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite."""
        print("🧪 Running test suite...")
        
        try:
            # Check if pytest is available
            result = self.run_command([sys.executable, "-m", "pytest", "--version"], check=False)
            if result.returncode != 0:
                print("⚠️  pytest not available, skipping tests")
                return True
            
            # Run tests
            result = self.run_command([
                sys.executable, "-m", "pytest", 
                "tests/", "-v", "--tb=short"
            ], check=False)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("❌ Some tests failed")
                return False
                
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
            return False
    
    def create_deployment_checklist(self) -> None:
        """Create a deployment checklist file."""
        print("📋 Creating deployment checklist...")
        
        checklist = {
            "pre_deployment": {
                "code_quality": [
                    "All tests pass",
                    "Code formatted with black",
                    "Linting passes (flake8)",
                    "Type checking passes (mypy)",
                    "Security scan passes (bandit)"
                ],
                "configuration": [
                    "Environment variables configured",
                    "API keys valid and set",
                    "CORS origins configured for production",
                    "Debug mode disabled for production",
                    "Log level appropriate for environment"
                ],
                "models": [
                    "All 8 model files present",
                    "Model files validated and loadable",
                    "Model cache size configured"
                ]
            },
            "deployment": {
                "render": [
                    "GitHub repository connected",
                    "render.yaml configuration verified",
                    "Environment variables set in dashboard",
                    "Build and deployment successful",
                    "Health check endpoint responding"
                ],
                "docker": [
                    "Dockerfile optimized for production",
                    "Image builds successfully",
                    "Container runs without errors",
                    "Health check configured",
                    "Resource limits set"
                ]
            },
            "post_deployment": {
                "validation": [
                    "Health endpoint returns 200",
                    "API documentation accessible",
                    "Prediction endpoints functional",
                    "Response times within targets",
                    "Error rates acceptable"
                ],
                "monitoring": [
                    "Logging configured and working",
                    "Monitoring alerts set up",
                    "Performance metrics tracked",
                    "Backup and recovery tested"
                ]
            }
        }
        
        checklist_file = self.project_root / "deployment_checklist.json"
        with open(checklist_file, 'w') as f:
            json.dump(checklist, f, indent=2)
        
        print(f"✅ Deployment checklist created: {checklist_file}")
    
    def print_next_steps(self) -> None:
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🎉 Setup completed successfully!")
        print("="*60)
        
        print("\n📋 Next Steps:")
        
        if self.environment == "development":
            print("1. Activate virtual environment:")
            if os.name == 'nt':  # Windows
                print("   venv\\Scripts\\activate")
            else:  # Unix/Linux/Mac
                print("   source venv/bin/activate")
            
            print("\n2. Configure environment variables:")
            print("   - Edit .env file")
            print("   - Set MATERIALS_PROJECT_API_KEY")
            print("   - Set NIST_API_KEY (optional)")
            
            print("\n3. Start development server:")
            print("   python -m uvicorn src.api.main:app --reload")
            
            print("\n4. Access application:")
            print("   - Web Interface: http://localhost:8000")
            print("   - API Docs: http://localhost:8000/docs")
            print("   - Health Check: http://localhost:8000/health")
        
        elif self.environment == "production":
            print("1. Review deployment checklist:")
            print("   deployment_checklist.json")
            
            print("\n2. Choose deployment platform:")
            print("   - Render (recommended): See docs/DEPLOYMENT_PROCESS.md")
            print("   - Docker: See docker-compose.yml")
            print("   - Other platforms: See docs/DEPLOYMENT_PROCESS.md")
            
            print("\n3. Set production environment variables")
            print("\n4. Deploy and validate")
        
        print("\n📚 Documentation:")
        print("   - README.md - Main documentation")
        print("   - docs/API_INTEGRATION_GUIDE.md - API usage")
        print("   - docs/ENVIRONMENT_CONFIGURATION.md - Configuration")
        print("   - docs/DEPLOYMENT_PROCESS.md - Deployment guide")
        print("   - examples/api_examples.py - Usage examples")
        
        print("\n🆘 Need help?")
        print("   - Check documentation in docs/ directory")
        print("   - Run validation: python scripts/validate_config.py")
        print("   - Test deployment: python scripts/validate_deployment.py")
    
    def setup(self) -> bool:
        """Run the complete setup process."""
        print("🚀 Starting Ceramic Armor ML API Setup")
        print(f"Environment: {self.environment}")
        print("="*60)
        
        steps = [
            ("Python Version", self.check_python_version),
            ("System Dependencies", self.check_system_dependencies),
            ("Virtual Environment", self.setup_virtual_environment),
            ("Python Dependencies", self.install_dependencies),
            ("Environment Configuration", self.setup_environment_file),
            ("Model Files", self.validate_model_files),
            ("Configuration Validation", self.validate_configuration),
        ]
        
        # Add environment-specific steps
        if self.environment == "development":
            steps.extend([
                ("API Connectivity", self.test_api_connectivity),
                ("Test Suite", self.run_tests),
            ])
        
        # Run all steps
        failed_steps = []
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                failed_steps.append(step_name)
        
        # Create deployment checklist
        self.create_deployment_checklist()
        
        # Print results
        print("\n" + "="*60)
        print("📊 Setup Results")
        print("="*60)
        
        if failed_steps:
            print(f"❌ {len(failed_steps)} step(s) failed:")
            for step in failed_steps:
                print(f"   - {step}")
            print("\n💡 Please resolve the issues above and run setup again.")
            return False
        else:
            print("✅ All setup steps completed successfully!")
            self.print_next_steps()
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup script for Ceramic Armor ML API deployment"
    )
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "production", "testing"],
        default="development",
        help="Target environment (default: development)"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true",
        help="Skip running tests during setup"
    )
    
    args = parser.parse_args()
    
    # Create setup instance and run
    setup = DeploymentSetup(environment=args.environment)
    success = setup.setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()