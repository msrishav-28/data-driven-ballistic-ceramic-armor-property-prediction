#!/usr/bin/env python3
"""
Deployment helper script for Ceramic Armor ML API.
Assists with pre-deployment checks and post-deployment validation.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any


class DeploymentHelper:
    """Helper class for deployment operations."""
    
    def __init__(self, project_root: str = "."):
        """Initialize deployment helper."""
        self.project_root = Path(project_root).resolve()
        self.required_files = [
            "render.yaml",
            "requirements.txt",
            "src/api/main.py",
            "src/config.py"
        ]
        self.optional_files = [
            ".env.example",
            ".env.production",
            "DEPLOYMENT.md"
        ]
    
    def check_required_files(self) -> Dict[str, Any]:
        """Check if all required files exist."""
        print("🔍 Checking required files...")
        
        missing_files = []
        present_files = []
        
        for file_path in self.required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                present_files.append(file_path)
                print(f"✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"❌ {file_path} - MISSING")
        
        return {
            "status": "pass" if not missing_files else "fail",
            "present_files": present_files,
            "missing_files": missing_files
        }
    
    def validate_render_config(self) -> Dict[str, Any]:
        """Validate render.yaml configuration."""
        print("\n🔍 Validating render.yaml configuration...")
        
        render_file = self.project_root / "render.yaml"
        if not render_file.exists():
            return {
                "status": "fail",
                "error": "render.yaml not found"
            }
        
        try:
            import yaml
            with open(render_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["services"]
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                return {
                    "status": "fail",
                    "error": f"Missing sections: {missing_sections}"
                }
            
            # Check service configuration
            services = config.get("services", [])
            if not services:
                return {
                    "status": "fail",
                    "error": "No services defined"
                }
            
            service = services[0]
            required_service_fields = ["name", "type", "env", "buildCommand", "startCommand"]
            missing_fields = []
            
            for field in required_service_fields:
                if field not in service:
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    "status": "fail",
                    "error": f"Missing service fields: {missing_fields}"
                }
            
            print("✅ render.yaml configuration is valid")
            return {
                "status": "pass",
                "service_name": service.get("name"),
                "service_type": service.get("type"),
                "environment": service.get("env")
            }
            
        except ImportError:
            print("⚠️  PyYAML not installed, skipping detailed validation")
            return {
                "status": "partial",
                "warning": "PyYAML not available for detailed validation"
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Error parsing render.yaml: {str(e)}"
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all dependencies can be installed."""
        print("\n🔍 Checking Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            return {
                "status": "fail",
                "error": "requirements.txt not found"
            }
        
        try:
            # Try to parse requirements.txt
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            dependencies = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    dependencies.append(line)
            
            print(f"✅ Found {len(dependencies)} dependencies")
            return {
                "status": "pass",
                "dependency_count": len(dependencies),
                "dependencies": dependencies[:10]  # Show first 10
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Error reading requirements.txt: {str(e)}"
            }
    
    def validate_environment_config(self) -> Dict[str, Any]:
        """Validate environment configuration."""
        print("\n🔍 Validating environment configuration...")
        
        try:
            # Try to import and validate config
            sys.path.insert(0, str(self.project_root))
            from src.config import Settings
            
            # Create settings instance to validate
            settings = Settings()
            
            # Check critical settings
            critical_settings = [
                "app_name",
                "app_version",
                "environment",
                "host",
                "port"
            ]
            
            missing_settings = []
            for setting in critical_settings:
                if not hasattr(settings, setting):
                    missing_settings.append(setting)
            
            if missing_settings:
                return {
                    "status": "fail",
                    "error": f"Missing critical settings: {missing_settings}"
                }
            
            print("✅ Environment configuration is valid")
            return {
                "status": "pass",
                "app_name": settings.app_name,
                "app_version": settings.app_version,
                "environment": settings.environment
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Error validating config: {str(e)}"
            }
    
    def check_api_structure(self) -> Dict[str, Any]:
        """Check API structure and imports."""
        print("\n🔍 Checking API structure...")
        
        try:
            # Check if main API file can be imported
            sys.path.insert(0, str(self.project_root))
            
            # Try importing main components
            from src.api.main import create_app
            from src.config import get_settings
            
            # Try creating app instance
            app = create_app()
            
            print("✅ API structure is valid")
            return {
                "status": "pass",
                "app_created": True
            }
            
        except Exception as e:
            return {
                "status": "fail",
                "error": f"Error validating API structure: {str(e)}"
            }
    
    def generate_deployment_checklist(self) -> List[str]:
        """Generate deployment checklist."""
        return [
            "✅ Repository pushed to GitHub",
            "✅ Render account created and connected to GitHub",
            "✅ Service created in Render dashboard",
            "✅ Environment variables configured in Render",
            "✅ API keys added securely (MATERIALS_PROJECT_API_KEY, NIST_API_KEY)",
            "✅ CORS origins updated for production domain",
            "✅ Health check endpoint verified",
            "✅ Custom domain configured (if applicable)",
            "✅ Monitoring and alerting set up",
            "✅ Deployment validation completed"
        ]
    
    def run_pre_deployment_checks(self) -> Dict[str, Any]:
        """Run all pre-deployment checks."""
        print("🚀 Running pre-deployment checks...")
        print("=" * 60)
        
        checks = [
            ("Required Files", self.check_required_files),
            ("Render Configuration", self.validate_render_config),
            ("Dependencies", self.check_dependencies),
            ("Environment Config", self.validate_environment_config),
            ("API Structure", self.check_api_structure)
        ]
        
        results = {}
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                results[check_name.lower().replace(" ", "_")] = result
                
                if result["status"] != "pass":
                    all_passed = False
                    
            except Exception as e:
                results[check_name.lower().replace(" ", "_")] = {
                    "status": "error",
                    "error": str(e)
                }
                all_passed = False
        
        print("\n" + "=" * 60)
        print("📊 PRE-DEPLOYMENT CHECK SUMMARY")
        print("=" * 60)
        
        for check_name, result in results.items():
            status = result["status"]
            if status == "pass":
                print(f"✅ {check_name.replace('_', ' ').title()}")
            elif status == "partial":
                print(f"⚠️  {check_name.replace('_', ' ').title()} (partial)")
            else:
                print(f"❌ {check_name.replace('_', ' ').title()}")
                if "error" in result:
                    print(f"   Error: {result['error']}")
        
        if all_passed:
            print("\n🎉 All pre-deployment checks passed!")
            print("\n📋 Deployment Checklist:")
            for item in self.generate_deployment_checklist():
                print(f"  {item}")
        else:
            print("\n⚠️  Some checks failed. Please fix issues before deploying.")
        
        return {
            "overall_status": "pass" if all_passed else "fail",
            "checks": results,
            "checklist": self.generate_deployment_checklist()
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ceramic Armor ML API Deployment Helper")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Run pre-deployment checks
    helper = DeploymentHelper(args.project_root)
    results = helper.run_pre_deployment_checks()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "pass" else 1)


if __name__ == "__main__":
    main()