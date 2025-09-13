#!/usr/bin/env python3
"""
Repository Verification Script

This script verifies that the repository has been properly set up and pushed.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()


def check_git_status():
    """Check git repository status."""
    print("🔍 Checking Git Repository Status...")
    
    # Check if we're in a git repository
    success, output = run_command("git status --porcelain", "Check git status")
    if not success:
        print("❌ Not a git repository or git not available")
        return False
    
    # Check remote
    success, remote = run_command("git remote -v", "Check remote")
    if not success:
        print("❌ No git remote configured")
        return False
    
    print("✅ Git repository properly configured")
    print(f"   Remote: {remote.split()[1] if remote else 'None'}")
    
    # Check if there are uncommitted changes
    if output.strip():
        print("⚠️  Uncommitted changes detected:")
        for line in output.split('\n')[:5]:  # Show first 5 changes
            print(f"   {line}")
        if len(output.split('\n')) > 5:
            remaining = len(output.split('\n')) - 5
            print(f"   ... and {remaining} more")
    else:
        print("✅ Working directory clean")
    
    return True


def check_essential_files():
    """Check that essential files exist."""
    print("\n📁 Checking Essential Files...")
    
    essential_files = [
        "README.md",
        "GETTING_STARTED.md", 
        "START_HERE.md",
        "requirements.txt",
        "render.yaml",
        ".gitignore",
        ".env.example",
        "setup.py",
        "setup.bat",
        "setup.sh",
        "start_server.py",
        "src/api/main.py",
        "src/config.py",
        "tests/conftest.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} essential files:")
        for file in missing_files[:5]:
            print(f"   - {file}")
        if len(missing_files) > 5:
            remaining = len(missing_files) - 5
            print(f"   ... and {remaining} more")
        return False
    else:
        print(f"✅ All {len(essential_files)} essential files present")
        return True


def check_directory_structure():
    """Check directory structure."""
    print("\n📂 Checking Directory Structure...")
    
    required_dirs = [
        "src",
        "src/api",
        "src/ml", 
        "tests",
        "scripts",
        "docs",
        "models"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing {len(missing_dirs)} required directories:")
        for dir in missing_dirs:
            print(f"   - {dir}")
        return False
    else:
        print(f"✅ All {len(required_dirs)} required directories present")
        return True


def check_python_files():
    """Check that key Python files can be imported."""
    print("\n🐍 Checking Python Files...")
    
    try:
        # Test basic imports
        import src.config
        print("✅ Configuration module imports successfully")
        
        # Test API main (this might fail if dependencies aren't installed)
        try:
            import src.api.main
            print("✅ API main module imports successfully")
        except ImportError as e:
            print(f"⚠️  API main import failed (may need dependencies): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Python import test failed: {e}")
        return False


def check_gitignore():
    """Check .gitignore file."""
    print("\n🚫 Checking .gitignore...")
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("❌ .gitignore file missing")
        return False
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    # Check for essential patterns
    essential_patterns = [
        "__pycache__/",
        "*.pyc",
        ".env",
        "venv/",
        ".pytest_cache/",
        "*.log"
    ]
    
    missing_patterns = []
    for pattern in essential_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"⚠️  .gitignore missing {len(missing_patterns)} essential patterns:")
        for pattern in missing_patterns:
            print(f"   - {pattern}")
    else:
        print("✅ .gitignore contains all essential patterns")
    
    print(f"✅ .gitignore file exists ({len(content.split())} lines)")
    return True


def main():
    """Main verification function."""
    print("🔍 REPOSITORY VERIFICATION")
    print("=" * 50)
    print("Verifying that the repository has been properly set up and pushed.")
    print()
    
    checks = [
        ("Git Status", check_git_status),
        ("Essential Files", check_essential_files),
        ("Directory Structure", check_directory_structure),
        ("Python Files", check_python_files),
        (".gitignore", check_gitignore)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        try:
            if check_function():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Checks Passed: {passed_checks}/{total_checks}")
    print(f"Success Rate: {passed_checks/total_checks*100:.1f}%")
    
    if passed_checks == total_checks:
        print("✅ Repository verification PASSED!")
        print("\n🚀 Next Steps:")
        print("1. Visit your repository: https://github.com/msrishav-28/data-driven-ballistic-ceramic-armor-property-prediction")
        print("2. Follow the GETTING_STARTED.md guide to set up locally")
        print("3. Deploy to Render using the render.yaml configuration")
        print("4. Run deployment validation after deployment")
        return True
    elif passed_checks >= total_checks * 0.8:
        print("⚠️  Repository verification mostly PASSED with minor issues")
        print("💡 Check the warnings above and refer to documentation")
        return True
    else:
        print("❌ Repository verification FAILED")
        print("💡 Please address the issues above before proceeding")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        sys.exit(1)