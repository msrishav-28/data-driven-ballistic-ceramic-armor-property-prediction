"""
Comprehensive Integration Test Runner

This script orchestrates the complete integration testing and deployment validation
for the Ceramic Armor ML API. It runs all test suites and provides comprehensive
reporting for deployment readiness assessment.
"""

import os
import sys
import subprocess
import time
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class IntegrationTestRunner:
    """Orchestrates comprehensive integration testing."""
    
    def __init__(self, deployment_url: str = None, local_testing: bool = False):
        """Initialize test runner."""
        self.deployment_url = deployment_url or os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
        self.local_testing = local_testing
        self.project_root = Path(__file__).parent.parent
        self.test_results = {
            'test_run_timestamp': time.time(),
            'deployment_url': self.deployment_url,
            'local_testing': local_testing,
            'test_suites': {},
            'overall_summary': {}
        }
    
    def run_pytest_suite(self, test_file: str, test_name: str, markers: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Run a pytest test suite and return results."""
        print(f"\n🧪 Running {test_name}...")
        print(f"   Test file: {test_file}")
        
        # Construct pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.project_root / test_file),
            '-v',
            '--tb=short',
            '--json-report',
            f'--json-report-file={test_name.lower().replace(" ", "_")}_results.json'
        ]
        
        # Add markers if specified
        if markers:
            for marker in markers:
                cmd.extend(['-m', marker])
        
        # Set environment variables
        env = os.environ.copy()
        env['DEPLOYMENT_URL'] = self.deployment_url
        env['PYTHONPATH'] = str(self.project_root)
        
        try:
            # Run pytest
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            end_time = time.time()
            
            # Parse results
            success = result.returncode == 0
            duration = end_time - start_time
            
            # Try to load JSON report
            json_report_file = f"{test_name.lower().replace(' ', '_')}_results.json"
            json_data = None
            
            if os.path.exists(json_report_file):
                try:
                    with open(json_report_file, 'r') as f:
                        json_data = json.load(f)
                except Exception as e:
                    print(f"   Warning: Could not parse JSON report: {e}")
            
            test_result = {
                'success': success,
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'json_report': json_data
            }
            
            # Extract summary from JSON report if available
            if json_data and 'summary' in json_data:
                summary = json_data['summary']
                test_result['test_summary'] = {
                    'total': summary.get('total', 0),
                    'passed': summary.get('passed', 0),
                    'failed': summary.get('failed', 0),
                    'skipped': summary.get('skipped', 0),
                    'error': summary.get('error', 0)
                }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   Status: {status} ({duration:.1f}s)")
            
            if json_data and 'summary' in json_data:
                summary = json_data['summary']
                print(f"   Tests: {summary.get('passed', 0)}/{summary.get('total', 0)} passed")
            
            return success, test_result
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: Test suite exceeded 10 minute limit")
            return False, {
                'success': False,
                'error': 'Test suite timeout',
                'duration_seconds': 600
            }
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False, {
                'success': False,
                'error': str(e),
                'duration_seconds': 0
            }
    
    def run_deployment_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Run deployment validation script."""
        print(f"\n🚀 Running Deployment Validation...")
        print(f"   Target URL: {self.deployment_url}")
        
        try:
            # Run deployment validation script
            cmd = [
                sys.executable,
                str(self.project_root / 'scripts' / 'validate_render_deployment.py'),
                self.deployment_url,
                '--timeout', '30'
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            success = result.returncode == 0
            duration = end_time - start_time
            
            # Try to find and load validation results file
            validation_data = None
            for file in os.listdir('.'):
                if file.startswith('render_deployment_validation_') and file.endswith('.json'):
                    try:
                        with open(file, 'r') as f:
                            validation_data = json.load(f)
                        break
                    except Exception as e:
                        print(f"   Warning: Could not parse validation file {file}: {e}")
            
            validation_result = {
                'success': success,
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'validation_data': validation_data
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   Status: {status} ({duration:.1f}s)")
            
            if validation_data:
                passed = validation_data.get('passed_tests', 0)
                total = validation_data.get('total_tests', 0)
                print(f"   Validation: {passed}/{total} tests passed")
            
            return success, validation_result
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: Deployment validation exceeded 5 minute limit")
            return False, {
                'success': False,
                'error': 'Deployment validation timeout',
                'duration_seconds': 300
            }
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False, {
                'success': False,
                'error': str(e),
                'duration_seconds': 0
            }
    
    def run_load_testing(self) -> Tuple[bool, Dict[str, Any]]:
        """Run load testing script."""
        print(f"\n⚡ Running Load Testing...")
        print(f"   Target URL: {self.deployment_url}")
        
        try:
            # Run load testing script
            cmd = [
                sys.executable,
                str(self.project_root / 'tests' / 'test_load_testing.py'),
                self.deployment_url
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            end_time = time.time()
            
            success = result.returncode == 0
            duration = end_time - start_time
            
            # Try to find and load load test results file
            load_test_data = None
            for file in os.listdir('.'):
                if file.startswith('load_test_results_') and file.endswith('.json'):
                    try:
                        with open(file, 'r') as f:
                            load_test_data = json.load(f)
                        break
                    except Exception as e:
                        print(f"   Warning: Could not parse load test file {file}: {e}")
            
            load_test_result = {
                'success': success,
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'load_test_data': load_test_data
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   Status: {status} ({duration:.1f}s)")
            
            if load_test_data:
                # Calculate overall load test success
                test_count = len(load_test_data)
                successful_tests = sum(1 for test in load_test_data.values() 
                                     if test.get('success_rate', 0) >= 0.8)
                print(f"   Load Tests: {successful_tests}/{test_count} passed performance criteria")
            
            return success, load_test_result
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT: Load testing exceeded 10 minute limit")
            return False, {
                'success': False,
                'error': 'Load testing timeout',
                'duration_seconds': 600
            }
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False, {
                'success': False,
                'error': str(e),
                'duration_seconds': 0
            }
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        print("🎯 Starting Comprehensive Integration Testing")
        print("=" * 60)
        print(f"Target URL: {self.deployment_url}")
        print(f"Local Testing: {self.local_testing}")
        print(f"Project Root: {self.project_root}")
        
        test_suites = []
        
        # Define test suites to run
        if self.local_testing:
            # Local testing - run all unit and integration tests
            test_suites = [
                ('tests/test_main.py', 'API Main Tests', None),
                ('tests/test_api_models.py', 'API Models Tests', None),
                ('tests/test_mechanical_prediction.py', 'Mechanical Prediction Tests', None),
                ('tests/test_ballistic_prediction.py', 'Ballistic Prediction Tests', None),
                ('tests/test_batch_processing.py', 'Batch Processing Tests', None),
                ('tests/test_integration.py', 'Integration Tests', None),
                ('tests/test_performance.py', 'Performance Tests', ['not slow']),
                ('tests/test_logging_middleware.py', 'Logging Middleware Tests', None),
                ('tests/test_security_middleware.py', 'Security Middleware Tests', None),
                ('tests/test_monitoring_service.py', 'Monitoring Service Tests', None)
            ]
        else:
            # Deployment testing - focus on integration and deployment tests
            test_suites = [
                ('tests/test_integration.py', 'Integration Tests', None),
                ('tests/test_deployment_validation.py', 'Deployment Validation Tests', None),
                ('tests/test_performance.py', 'Performance Tests', ['not slow']),
                ('tests/test_load_testing.py', 'Load Testing', ['not comprehensive'])
            ]
        
        # Run pytest test suites
        for test_file, test_name, markers in test_suites:
            success, result = self.run_pytest_suite(test_file, test_name, markers)
            self.test_results['test_suites'][test_name] = result
        
        # Run deployment validation (for deployment testing)
        if not self.local_testing:
            success, result = self.run_deployment_validation()
            self.test_results['test_suites']['Deployment Validation'] = result
            
            # Run load testing
            success, result = self.run_load_testing()
            self.test_results['test_suites']['Load Testing'] = result
        
        # Calculate overall results
        self.calculate_overall_results()
        
        return self.test_results
    
    def calculate_overall_results(self):
        """Calculate overall test results."""
        total_suites = len(self.test_results['test_suites'])
        passed_suites = sum(1 for result in self.test_results['test_suites'].values() 
                           if result.get('success', False))
        
        # Calculate total test counts from JSON reports
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for suite_name, suite_result in self.test_results['test_suites'].items():
            if 'test_summary' in suite_result:
                summary = suite_result['test_summary']
                total_tests += summary.get('total', 0)
                passed_tests += summary.get('passed', 0)
                failed_tests += summary.get('failed', 0)
                skipped_tests += summary.get('skipped', 0)
        
        # Calculate success rates
        suite_success_rate = passed_suites / total_suites if total_suites > 0 else 0
        test_success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Determine overall success
        overall_success = (
            suite_success_rate >= 0.8 and  # 80% of test suites must pass
            test_success_rate >= 0.85      # 85% of individual tests must pass
        )
        
        self.test_results['overall_summary'] = {
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'suite_success_rate': suite_success_rate,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'test_success_rate': test_success_rate,
            'overall_success': overall_success,
            'deployment_ready': overall_success and not self.local_testing
        }
    
    def print_summary(self):
        """Print comprehensive test summary."""
        summary = self.test_results['overall_summary']
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        print(f"Test Run: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_results['test_run_timestamp']))}")
        print(f"Target URL: {self.test_results['deployment_url']}")
        print(f"Test Mode: {'Local Development' if self.test_results['local_testing'] else 'Deployment Validation'}")
        
        print(f"\n📊 Overall Results:")
        print(f"  Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed ({summary['suite_success_rate']:.1%})")
        print(f"  Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['test_success_rate']:.1%})")
        print(f"  Failed Tests: {summary['failed_tests']}")
        print(f"  Skipped Tests: {summary['skipped_tests']}")
        
        overall_status = "✅ SUCCESS" if summary['overall_success'] else "❌ FAILURE"
        print(f"\nOverall Status: {overall_status}")
        
        if not self.local_testing:
            deployment_status = "🚀 DEPLOYMENT READY" if summary['deployment_ready'] else "⚠️  DEPLOYMENT NOT READY"
            print(f"Deployment Status: {deployment_status}")
        
        # Print individual suite results
        print(f"\n📋 Test Suite Results:")
        for suite_name, suite_result in self.test_results['test_suites'].items():
            status = "✅" if suite_result.get('success', False) else "❌"
            duration = suite_result.get('duration_seconds', 0)
            
            print(f"  {status} {suite_name} ({duration:.1f}s)")
            
            # Show test counts if available
            if 'test_summary' in suite_result:
                summary = suite_result['test_summary']
                passed = summary.get('passed', 0)
                total = summary.get('total', 0)
                if total > 0:
                    print(f"      Tests: {passed}/{total} passed")
        
        # Print recommendations if there are failures
        if not summary['overall_success']:
            print(f"\n💡 Recommendations:")
            failed_suites = [name for name, result in self.test_results['test_suites'].items() 
                           if not result.get('success', False)]
            
            for suite_name in failed_suites:
                print(f"  • Review and fix failures in: {suite_name}")
            
            if not self.local_testing:
                print(f"  • Check deployment logs and configuration")
                print(f"  • Verify all environment variables are set correctly")
                print(f"  • Ensure ML models are properly loaded")
    
    def save_results(self, filename: Optional[str] = None):
        """Save test results to file."""
        if filename is None:
            timestamp = int(time.time())
            mode = "local" if self.local_testing else "deployment"
            filename = f"integration_test_results_{mode}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        return filename
    
    def cleanup_temp_files(self):
        """Clean up temporary test files."""
        temp_files = [
            f for f in os.listdir('.')
            if (f.endswith('_results.json') or 
                f.startswith('render_deployment_validation_') or
                f.startswith('load_test_results_'))
        ]
        
        for file in temp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {file}: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Run comprehensive integration tests')
    parser.add_argument('--url', help='Deployment URL for testing (default: http://localhost:8000)')
    parser.add_argument('--local', action='store_true', help='Run local development tests')
    parser.add_argument('--output', help='Output file for results (default: auto-generated)')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary test files')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Determine deployment URL
    deployment_url = args.url or os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
    
    # Create test runner
    runner = IntegrationTestRunner(deployment_url, args.local)
    
    try:
        # Run comprehensive test suite
        results = runner.run_comprehensive_test_suite()
        
        if not args.quiet:
            runner.print_summary()
        
        # Save results
        output_file = runner.save_results(args.output)
        
        # Cleanup temporary files unless requested to keep them
        if not args.keep_temp:
            runner.cleanup_temp_files()
        
        # Exit with appropriate code
        overall_success = results['overall_summary']['overall_success']
        exit_code = 0 if overall_success else 1
        
        if args.quiet:
            # Print minimal output for CI/CD
            status = "PASS" if overall_success else "FAIL"
            summary = results['overall_summary']
            print(f"{status}: {summary['passed_tests']}/{summary['total_tests']} tests passed, "
                  f"{summary['passed_suites']}/{summary['total_suites']} suites passed")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        if not args.keep_temp:
            runner.cleanup_temp_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        if not args.keep_temp:
            runner.cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    main()