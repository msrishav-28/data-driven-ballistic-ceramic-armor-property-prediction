"""
Deployment Checklist and Validation

This script provides a comprehensive checklist for deployment readiness
and validates all requirements before deploying to Render platform.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import requests


class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self, project_root: Path = None):
        """Initialize deployment checker."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.checklist_results = {
            'timestamp': time.time(),
            'project_root': str(self.project_root),
            'checks': {},
            'summary': {}
        }
    
    def check_file_exists(self, file_path: str, description: str) -> bool:
        """Check if a required file exists."""
        full_path = self.project_root / file_path
        exists = full_path.exists()
        
        self.checklist_results['checks'][f'file_{file_path.replace("/", "_")}'] = {
            'description': f'File exists: {description}',
            'file_path': file_path,
            'exists': exists,
            'full_path': str(full_path)
        }
        
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {file_path}")
        
        return exists
    
    def check_directory_structure(self) -> bool:
        """Check required directory structure."""
        print("\n📁 Checking Directory Structure...")
        
        required_dirs = [
            ('src', 'Source code directory'),
            ('src/api', 'API module directory'),
            ('src/ml', 'ML module directory'),
            ('src/static', 'Static files directory'),
            ('tests', 'Tests directory'),
            ('models', 'ML models directory'),
            ('scripts', 'Scripts directory')
        ]
        
        all_exist = True
        for dir_path, description in required_dirs:
            exists = (self.project_root / dir_path).exists()
            all_exist = all_exist and exists
            
            status = "✅" if exists else "❌"
            print(f"  {status} {description}: {dir_path}")
        
        self.checklist_results['checks']['directory_structure'] = {
            'description': 'Required directory structure',
            'all_exist': all_exist,
            'directories': required_dirs
        }
        
        return all_exist
    
    def check_required_files(self) -> bool:
        """Check all required files exist."""
        print("\n📄 Checking Required Files...")
        
        required_files = [
            ('requirements.txt', 'Python dependencies'),
            ('render.yaml', 'Render deployment configuration'),
            ('src/api/main.py', 'FastAPI main application'),
            ('src/config.py', 'Application configuration'),
            ('start_server.py', 'Server startup script'),
            ('.env.example', 'Environment variables example'),
            ('README.md', 'Project documentation'),
            ('src/static/index.html', 'Frontend HTML'),
            ('src/static/css/styles.css', 'Frontend CSS'),
            ('src/static/js/app.js', 'Frontend JavaScript')
        ]
        
        all_exist = True
        for file_path, description in required_files:
            exists = self.check_file_exists(file_path, description)
            all_exist = all_exist and exists
        
        return all_exist
    
    def check_ml_models(self) -> bool:
        """Check ML model files exist."""
        print("\n🤖 Checking ML Model Files...")
        
        model_files = [
            'ballistic_back_face_deformation.joblib',
            'ballistic_multi_hit_capability.joblib',
            'ballistic_penetration_resistance.pkl',
            'ballistic_v50_velocity.joblib',
            'mechanical_density.pkl',
            'mechanical_elastic_modulus.joblib',
            'mechanical_fracture_toughness.joblib',
            'mechanical_vickers_hardness.joblib'
        ]
        
        models_dir = self.project_root / 'models'
        all_exist = True
        
        for model_file in model_files:
            model_path = models_dir / model_file
            exists = model_path.exists()
            all_exist = all_exist and exists
            
            status = "✅" if exists else "❌"
            print(f"  {status} ML Model: {model_file}")
            
            if exists:
                # Check file size (models should not be empty)
                size_mb = model_path.stat().st_size / (1024 * 1024)
                if size_mb < 0.001:  # Less than 1KB
                    print(f"    ⚠️  Warning: Model file is very small ({size_mb:.3f} MB)")
                else:
                    print(f"    📊 Size: {size_mb:.2f} MB")
        
        self.checklist_results['checks']['ml_models'] = {
            'description': 'ML model files',
            'all_exist': all_exist,
            'model_files': model_files
        }
        
        return all_exist
    
    def check_python_dependencies(self) -> bool:
        """Check Python dependencies can be installed."""
        print("\n🐍 Checking Python Dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("  ❌ requirements.txt not found")
            return False
        
        try:
            # Read requirements
            with open(requirements_file, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            # Filter out empty lines and comments
            requirements = [req.strip() for req in requirements 
                          if req.strip() and not req.strip().startswith('#')]
            
            print(f"  📦 Found {len(requirements)} dependencies")
            
            # Check for critical dependencies
            critical_deps = [
                'fastapi', 'uvicorn', 'pydantic', 'requests',
                'scikit-learn', 'pandas', 'numpy', 'joblib'
            ]
            
            missing_critical = []
            for dep in critical_deps:
                found = any(dep.lower() in req.lower() for req in requirements)
                if not found:
                    missing_critical.append(dep)
            
            if missing_critical:
                print(f"  ❌ Missing critical dependencies: {', '.join(missing_critical)}")
                return False
            else:
                print(f"  ✅ All critical dependencies found")
            
            # Try to validate requirements syntax
            for req in requirements[:5]:  # Check first 5 for syntax
                if '==' in req or '>=' in req or '<=' in req:
                    print(f"  ✅ Dependency format OK: {req}")
                else:
                    print(f"  ⚠️  Dependency format unclear: {req}")
            
            self.checklist_results['checks']['python_dependencies'] = {
                'description': 'Python dependencies',
                'requirements_count': len(requirements),
                'critical_deps_found': len(critical_deps) - len(missing_critical),
                'missing_critical': missing_critical,
                'valid': len(missing_critical) == 0
            }
            
            return len(missing_critical) == 0
            
        except Exception as e:
            print(f"  ❌ Error checking dependencies: {e}")
            return False
    
    def check_render_configuration(self) -> bool:
        """Check Render deployment configuration."""
        print("\n🚀 Checking Render Configuration...")
        
        render_config = self.project_root / 'render.yaml'
        if not render_config.exists():
            print("  ❌ render.yaml not found")
            return False
        
        try:
            import yaml
            with open(render_config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required configuration sections
            required_sections = ['services']
            for section in required_sections:
                if section not in config:
                    print(f"  ❌ Missing required section: {section}")
                    return False
            
            # Check service configuration
            services = config.get('services', [])
            if not services:
                print("  ❌ No services defined")
                return False
            
            web_service = services[0]  # Assume first service is web service
            
            # Check required service fields
            required_fields = ['type', 'name', 'env', 'buildCommand', 'startCommand']
            missing_fields = []
            
            for field in required_fields:
                if field not in web_service:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"  ❌ Missing service fields: {', '.join(missing_fields)}")
                return False
            
            # Validate specific configurations
            if web_service.get('type') != 'web':
                print(f"  ❌ Service type should be 'web', got '{web_service.get('type')}'")
                return False
            
            if web_service.get('env') != 'python':
                print(f"  ❌ Environment should be 'python', got '{web_service.get('env')}'")
                return False
            
            # Check build and start commands
            build_command = web_service.get('buildCommand', '')
            if 'pip install' not in build_command:
                print(f"  ⚠️  Build command may not install dependencies")
            
            start_command = web_service.get('startCommand', '')
            if 'uvicorn' not in start_command:
                print(f"  ⚠️  Start command may not use uvicorn")
            
            print(f"  ✅ Render configuration valid")
            print(f"    Service: {web_service.get('name')}")
            print(f"    Type: {web_service.get('type')}")
            print(f"    Environment: {web_service.get('env')}")
            
            self.checklist_results['checks']['render_configuration'] = {
                'description': 'Render deployment configuration',
                'config_valid': True,
                'service_name': web_service.get('name'),
                'service_type': web_service.get('type'),
                'environment': web_service.get('env')
            }
            
            return True
            
        except ImportError:
            print("  ⚠️  PyYAML not available, cannot validate render.yaml syntax")
            # Still check basic file structure
            with open(render_config, 'r') as f:
                content = f.read()
            
            if 'services:' in content and 'buildCommand:' in content and 'startCommand:' in content:
                print("  ✅ Basic render.yaml structure appears valid")
                return True
            else:
                print("  ❌ render.yaml missing required sections")
                return False
                
        except Exception as e:
            print(f"  ❌ Error validating render.yaml: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """Check environment variable configuration."""
        print("\n🔧 Checking Environment Variables...")
        
        env_example = self.project_root / '.env.example'
        if not env_example.exists():
            print("  ❌ .env.example not found")
            return False
        
        try:
            with open(env_example, 'r') as f:
                env_content = f.read()
            
            # Check for required environment variables
            required_env_vars = [
                'MATERIALS_PROJECT_API_KEY',
                'LOG_LEVEL',
                'ENVIRONMENT',
                'CORS_ORIGINS'
            ]
            
            missing_vars = []
            for var in required_env_vars:
                if var not in env_content:
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"  ❌ Missing environment variables in .env.example: {', '.join(missing_vars)}")
                return False
            
            print(f"  ✅ All required environment variables documented")
            
            # Check if actual environment variables are set (for local testing)
            set_vars = []
            unset_vars = []
            
            for var in required_env_vars:
                if os.getenv(var):
                    set_vars.append(var)
                else:
                    unset_vars.append(var)
            
            if set_vars:
                print(f"  ✅ Currently set: {', '.join(set_vars)}")
            if unset_vars:
                print(f"  ⚠️  Not set locally: {', '.join(unset_vars)}")
            
            self.checklist_results['checks']['environment_variables'] = {
                'description': 'Environment variable configuration',
                'required_vars': required_env_vars,
                'documented_vars': len(required_env_vars) - len(missing_vars),
                'set_locally': set_vars,
                'unset_locally': unset_vars,
                'valid': len(missing_vars) == 0
            }
            
            return len(missing_vars) == 0
            
        except Exception as e:
            print(f"  ❌ Error checking environment variables: {e}")
            return False
    
    def check_api_structure(self) -> bool:
        """Check API code structure."""
        print("\n🔌 Checking API Structure...")
        
        api_files = [
            ('src/api/main.py', 'FastAPI main application'),
            ('src/api/routes/predictions.py', 'Prediction routes'),
            ('src/api/routes/health.py', 'Health check routes'),
            ('src/api/models/request_models.py', 'Request models'),
            ('src/api/models/response_models.py', 'Response models'),
            ('src/ml/predictor.py', 'ML predictor'),
            ('src/config.py', 'Configuration module')
        ]
        
        all_exist = True
        for file_path, description in api_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            all_exist = all_exist and exists
            
            status = "✅" if exists else "❌"
            print(f"  {status} {description}: {file_path}")
            
            # Check file is not empty
            if exists:
                size = full_path.stat().st_size
                if size < 100:  # Less than 100 bytes
                    print(f"    ⚠️  Warning: File is very small ({size} bytes)")
        
        self.checklist_results['checks']['api_structure'] = {
            'description': 'API code structure',
            'all_files_exist': all_exist,
            'api_files': api_files
        }
        
        return all_exist
    
    def check_test_coverage(self) -> bool:
        """Check test coverage."""
        print("\n🧪 Checking Test Coverage...")
        
        test_files = [
            ('tests/test_main.py', 'Main API tests'),
            ('tests/test_integration.py', 'Integration tests'),
            ('tests/test_performance.py', 'Performance tests'),
            ('tests/conftest.py', 'Test configuration'),
            ('tests/test_deployment_validation.py', 'Deployment validation tests'),
            ('tests/test_load_testing.py', 'Load testing')
        ]
        
        existing_tests = 0
        for file_path, description in test_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            
            status = "✅" if exists else "❌"
            print(f"  {status} {description}: {file_path}")
            
            if exists:
                existing_tests += 1
        
        coverage_percentage = (existing_tests / len(test_files)) * 100
        coverage_good = coverage_percentage >= 80
        
        print(f"  📊 Test coverage: {existing_tests}/{len(test_files)} files ({coverage_percentage:.1f}%)")
        
        self.checklist_results['checks']['test_coverage'] = {
            'description': 'Test coverage',
            'existing_tests': existing_tests,
            'total_tests': len(test_files),
            'coverage_percentage': coverage_percentage,
            'coverage_good': coverage_good
        }
        
        return coverage_good
    
    def check_documentation(self) -> bool:
        """Check documentation completeness."""
        print("\n📚 Checking Documentation...")
        
        doc_files = [
            ('README.md', 'Main project documentation'),
            ('DEPLOYMENT.md', 'Deployment documentation'),
            ('docs/API_INTEGRATION_GUIDE.md', 'API integration guide'),
            ('docs/batch_processing_guide.md', 'Batch processing guide')
        ]
        
        existing_docs = 0
        for file_path, description in doc_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            
            status = "✅" if exists else "❌"
            print(f"  {status} {description}: {file_path}")
            
            if exists:
                existing_docs += 1
                # Check file is not empty
                size = full_path.stat().st_size
                if size < 500:  # Less than 500 bytes
                    print(f"    ⚠️  Warning: Documentation file is very small ({size} bytes)")
        
        doc_coverage = (existing_docs / len(doc_files)) * 100
        doc_good = doc_coverage >= 75
        
        print(f"  📊 Documentation coverage: {existing_docs}/{len(doc_files)} files ({doc_coverage:.1f}%)")
        
        self.checklist_results['checks']['documentation'] = {
            'description': 'Documentation completeness',
            'existing_docs': existing_docs,
            'total_docs': len(doc_files),
            'doc_coverage': doc_coverage,
            'doc_good': doc_good
        }
        
        return doc_good
    
    def run_syntax_check(self) -> bool:
        """Run Python syntax check on main files."""
        print("\n🔍 Running Syntax Check...")
        
        python_files = [
            'src/api/main.py',
            'src/config.py',
            'start_server.py'
        ]
        
        all_valid = True
        for file_path in python_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                continue
            
            try:
                # Try to compile the Python file
                with open(full_path, 'r') as f:
                    source = f.read()
                
                compile(source, str(full_path), 'exec')
                print(f"  ✅ Syntax OK: {file_path}")
                
            except SyntaxError as e:
                print(f"  ❌ Syntax Error in {file_path}: {e}")
                all_valid = False
            except Exception as e:
                print(f"  ⚠️  Could not check {file_path}: {e}")
        
        self.checklist_results['checks']['syntax_check'] = {
            'description': 'Python syntax validation',
            'all_valid': all_valid,
            'checked_files': python_files
        }
        
        return all_valid
    
    def run_comprehensive_checklist(self) -> Dict[str, Any]:
        """Run comprehensive deployment checklist."""
        print("🎯 DEPLOYMENT READINESS CHECKLIST")
        print("=" * 60)
        
        # Run all checks
        checks = [
            ('Directory Structure', self.check_directory_structure),
            ('Required Files', self.check_required_files),
            ('ML Models', self.check_ml_models),
            ('Python Dependencies', self.check_python_dependencies),
            ('Render Configuration', self.check_render_configuration),
            ('Environment Variables', self.check_environment_variables),
            ('API Structure', self.check_api_structure),
            ('Test Coverage', self.check_test_coverage),
            ('Documentation', self.check_documentation),
            ('Syntax Check', self.run_syntax_check)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_function in checks:
            try:
                result = check_function()
                if result:
                    passed_checks += 1
            except Exception as e:
                print(f"  ❌ Error running {check_name}: {e}")
        
        # Calculate overall readiness
        readiness_percentage = (passed_checks / total_checks) * 100
        deployment_ready = readiness_percentage >= 90  # 90% of checks must pass
        
        self.checklist_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'readiness_percentage': readiness_percentage,
            'deployment_ready': deployment_ready
        }
        
        return self.checklist_results
    
    def print_summary(self):
        """Print deployment readiness summary."""
        summary = self.checklist_results['summary']
        
        print("\n" + "=" * 60)
        print("🎯 DEPLOYMENT READINESS SUMMARY")
        print("=" * 60)
        
        print(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']}")
        print(f"Readiness: {summary['readiness_percentage']:.1f}%")
        
        if summary['deployment_ready']:
            print("Status: ✅ READY FOR DEPLOYMENT")
            print("\n🚀 Next Steps:")
            print("  1. Commit and push all changes to your repository")
            print("  2. Connect your repository to Render")
            print("  3. Configure environment variables in Render dashboard")
            print("  4. Deploy and monitor the deployment logs")
            print("  5. Run post-deployment validation tests")
        else:
            print("Status: ❌ NOT READY FOR DEPLOYMENT")
            print("\n💡 Required Actions:")
            
            # List failed checks
            failed_checks = []
            for check_name, check_data in self.checklist_results['checks'].items():
                if isinstance(check_data, dict):
                    # Determine if check failed based on various success indicators
                    success_indicators = ['exists', 'all_exist', 'valid', 'all_valid', 'coverage_good', 'doc_good']
                    failed = True
                    
                    for indicator in success_indicators:
                        if indicator in check_data and check_data[indicator]:
                            failed = False
                            break
                    
                    if failed:
                        failed_checks.append(check_data.get('description', check_name))
            
            for i, check in enumerate(failed_checks[:5], 1):  # Show top 5 issues
                print(f"  {i}. Fix: {check}")
            
            if len(failed_checks) > 5:
                print(f"  ... and {len(failed_checks) - 5} more issues")
    
    def save_results(self, filename: Optional[str] = None):
        """Save checklist results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"deployment_checklist_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.checklist_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed checklist saved to: {filename}")
        return filename


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run deployment readiness checklist')
    parser.add_argument('--project-root', help='Project root directory (default: current directory parent)')
    parser.add_argument('--output', help='Output file for results (default: auto-generated)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent
    
    if not project_root.exists():
        print(f"❌ Error: Project root directory does not exist: {project_root}")
        sys.exit(1)
    
    # Create checker and run checklist
    checker = DeploymentChecker(project_root)
    
    try:
        results = checker.run_comprehensive_checklist()
        
        if not args.quiet:
            checker.print_summary()
        
        # Save results
        output_file = checker.save_results(args.output)
        
        # Exit with appropriate code
        deployment_ready = results['summary']['deployment_ready']
        exit_code = 0 if deployment_ready else 1
        
        if args.quiet:
            # Print minimal output for CI/CD
            status = "READY" if deployment_ready else "NOT_READY"
            readiness = results['summary']['readiness_percentage']
            print(f"{status}: {readiness:.1f}% deployment readiness")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Checklist interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Checklist failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()