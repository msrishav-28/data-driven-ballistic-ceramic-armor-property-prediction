#!/usr/bin/env python3
"""
Quick Setup Script for Ceramic Armor ML API

This script automates the initial setup process for local development.
Run this script to quickly get your development environment ready.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_step(step_num, description):
    """Print a formatted step description."""
    print(f"\n🔧 Step {step_num}: {description}")
    print("=" * 50)


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.10+ required.")
        print("Please install Python 3.10 or 3.11 and try again.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected - Compatible!")
    return True


def setup_virtual_environment():
    """Set up Python virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    success = run_command("python -m venv venv", "Create virtual environment")
    
    if success:
        print("✅ Virtual environment created successfully")
        print("\n💡 To activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # macOS/Linux
            print("   source venv/bin/activate")
    
    return success


def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies from requirements.txt...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:  # macOS/Linux
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    # Upgrade pip first
    success = run_command(f"{pip_path} install --upgrade pip", "Upgrade pip")
    if not success:
        return False
    
    # Install dependencies
    success = run_command(f"{pip_path} install -r requirements.txt", "Install dependencies")
    if not success:
        return False
    
    # Verify critical packages
    print("Verifying critical packages...")
    verify_cmd = f'{python_path} -c "import fastapi, uvicorn, sklearn, pandas, numpy; print(\\"✅ All critical packages installed\\")"'
    success = run_command(verify_cmd, "Verify packages")
    
    return success


def setup_environment_file():
    """Set up environment configuration file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("\n💡 Please edit .env file with your specific configuration:")
        print("   - Set MATERIALS_PROJECT_API_KEY if you have one")
        print("   - Adjust other settings as needed")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def check_ml_models():
    """Check if ML model files exist."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("⚠️  Models directory not found - will be created automatically")
        return True
    
    model_files = [
        "ballistic_back_face_deformation.joblib",
        "ballistic_multi_hit_capability.joblib", 
        "ballistic_penetration_resistance.pkl",
        "ballistic_v50_velocity.joblib",
        "mechanical_density.pkl",
        "mechanical_elastic_modulus.joblib",
        "mechanical_fracture_toughness.joblib",
        "mechanical_vickers_hardness.joblib"
    ]
    
    existing_models = []
    missing_models = []
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            existing_models.append(model_file)
        else:
            missing_models.append(model_file)
    
    print(f"✅ Found {len(existing_models)} ML model files")
    
    if missing_models:
        print(f"⚠️  Missing {len(missing_models)} model files:")
        for model in missing_models[:3]:  # Show first 3
            print(f"   - {model}")
        if len(missing_models) > 3:
            print(f"   ... and {len(missing_models) - 3} more")
        print("💡 Missing models will be created automatically when you first run the application")
    
    return True


def test_application():
    """Test if the application can start."""
    print("Testing application import...")
    
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # macOS/Linux
        python_path = "venv/bin/python"
    
    test_cmd = f'{python_path} -c "from src.api.main import app; print(\\"✅ Application imports successfully\\")"'
    success = run_command(test_cmd, "Test application import", check=False)
    
    if success:
        print("✅ Application is ready to run!")
    else:
        print("⚠️  Application import test failed - check the error messages above")
        print("💡 This might be normal if ML models need to be initialized")
    
    return success


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "🎉" * 50)
    print("SETUP COMPLETE!")
    print("🎉" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # macOS/Linux
        print("   source venv/bin/activate")
    
    print("\n2. Edit .env file with your configuration:")
    print("   - Set your API keys (optional)")
    print("   - Adjust settings as needed")
    
    print("\n3. Start the application:")
    print("   python start_server.py")
    
    print("\n4. Test the application:")
    print("   Open browser to: http://localhost:8000")
    print("   Check health: http://localhost:8000/health")
    print("   View docs: http://localhost:8000/docs")
    
    print("\n5. Run tests:")
    print("   python -m pytest tests/ -v")
    
    print("\n6. Check deployment readiness:")
    print("   python scripts/deployment_checklist.py")
    
    print("\n📚 For detailed instructions, see GETTING_STARTED.md")
    print("\n🚀 Happy coding!")


def main():
    """Main setup function."""
    print("🚀 Ceramic Armor ML API - Quick Setup")
    print("=" * 50)
    print("This script will set up your development environment automatically.")
    print("Make sure you're in the project root directory.")
    
    # Confirm setup
    response = input("\nProceed with setup? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    success_count = 0
    total_steps = 6
    
    # Step 1: Check Python version
    print_step(1, "Checking Python Version")
    if check_python_version():
        success_count += 1
    
    # Step 2: Set up virtual environment
    print_step(2, "Setting Up Virtual Environment")
    if setup_virtual_environment():
        success_count += 1
    
    # Step 3: Install dependencies
    print_step(3, "Installing Dependencies")
    if install_dependencies():
        success_count += 1
    
    # Step 4: Set up environment file
    print_step(4, "Setting Up Environment Configuration")
    if setup_environment_file():
        success_count += 1
    
    # Step 5: Check ML models
    print_step(5, "Checking ML Models")
    if check_ml_models():
        success_count += 1
    
    # Step 6: Test application
    print_step(6, "Testing Application")
    if test_application():
        success_count += 1
    
    # Summary
    print(f"\n📊 Setup Summary: {success_count}/{total_steps} steps completed successfully")
    
    if success_count == total_steps:
        print("✅ All setup steps completed successfully!")
        print_next_steps()
    elif success_count >= total_steps - 1:
        print("⚠️  Setup mostly complete with minor issues")
        print("💡 Check the messages above and refer to GETTING_STARTED.md for troubleshooting")
        print_next_steps()
    else:
        print("❌ Setup encountered significant issues")
        print("💡 Please check the error messages above and refer to GETTING_STARTED.md")
        print("💡 You may need to resolve dependencies or configuration issues manually")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("💡 Please refer to GETTING_STARTED.md for manual setup instructions")
        sys.exit(1)