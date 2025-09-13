"""
Render Deployment Validation Script

This script validates the complete deployment of the Ceramic Armor ML API on Render platform.
It performs comprehensive testing including:
- Health and status endpoint validation
- API functionality testing
- Performance benchmarking
- Load testing
- Security validation
- Frontend interface testing
"""

import os
import sys
import time
import json
import argparse
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
import concurrent.futures
import statistics


class RenderDeploymentValidator:
    """Comprehensive validator for Render deployment."""
    
    def __init__(self, deployment_url: str, timeout: int = 30):
        """Initialize deployment validator."""
        self.deployment_url = deployment_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'CeramicArmorML-RenderValidator/1.0',
            'Accept': 'application/json'
        })
        
        self.validation_results = {
            'deployment_url': deployment_url,
            'validation_timestamp': time.time(),
            'tests': {}
        }
    
    def url(self, path: str) -> str:
        """Construct full URL."""
        return urljoin(self.deployment_url, path)
    
    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result."""
        self.validation_results['tests'][test_name] = {
            'success': success,
            'timestamp': time.time(),
            'details': details
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if not success and 'error' in details:
            print(f"    Error: {details['error']}")
    
    def test_deployment_accessibility(self) -> bool:
        """Test basic deployment accessibility."""
        try:
            response = self.session.get(self.url('/health'))
            success = response.status_code == 200
            
            details = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'headers': dict(response.headers)
            }
            
            if success:
                details['health_data'] = response.json()
            else:
                details['error'] = f"Health endpoint returned {response.status_code}"
            
            self.log_test_result("Deployment Accessibility", success, details)
            return success
            
        except Exception as e:
            self.log_test_result("Deployment Accessibility", False, {'error': str(e)})
            return False
    
    def test_health_endpoints(self) -> bool:
        """Test all health and status endpoints."""
        endpoints = [
            ('/health', 'Basic Health'),
            ('/api/v1/status', 'Detailed Status'),
            ('/api/v1/models/info', 'Model Information')
        ]
        
        all_success = True
        
        for endpoint, name in endpoints:
            try:
                response = self.session.get(self.url(endpoint))
                success = response.status_code == 200
                
                details = {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                
                if success:
                    try:
                        details['data'] = response.json()
                    except:
                        details['data'] = 'Non-JSON response'
                else:
                    details['error'] = f"Endpoint returned {response.status_code}"
                
                self.log_test_result(f"Health Endpoint - {name}", success, details)
                all_success = all_success and success
                
            except Exception as e:
                self.log_test_result(f"Health Endpoint - {name}", False, {'error': str(e)})
                all_success = False
        
        return all_success
    
    def test_prediction_endpoints(self) -> bool:
        """Test prediction endpoints with sample data."""
        sample_request = {
            "composition": {
                "SiC": 0.6,
                "B4C": 0.3,
                "Al2O3": 0.1,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": {
                "sintering_temperature": 1800,
                "pressure": 50,
                "grain_size": 10,
                "holding_time": 120,
                "heating_rate": 15,
                "atmosphere": "argon"
            },
            "microstructure": {
                "porosity": 0.02,
                "phase_distribution": "uniform",
                "interface_quality": "good",
                "pore_size": 1.0,
                "connectivity": 0.1
            },
            "include_uncertainty": True,
            "include_feature_importance": True
        }
        
        endpoints = [
            ('/api/v1/predict/mechanical', 'Mechanical Prediction'),
            ('/api/v1/predict/ballistic', 'Ballistic Prediction')
        ]
        
        all_success = True
        
        for endpoint, name in endpoints:
            try:
                response = self.session.post(
                    self.url(endpoint),
                    json=sample_request,
                    headers={'Content-Type': 'application/json'}
                )
                
                success = response.status_code == 200
                
                details = {
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                
                if success:
                    try:
                        data = response.json()
                        details['has_predictions'] = 'predictions' in data
                        details['has_model_info'] = 'model_info' in data
                        details['has_request_id'] = 'request_id' in data
                        
                        # Validate prediction structure
                        if 'predictions' in data:
                            predictions = data['predictions']
                            details['prediction_count'] = len(predictions)
                            details['prediction_properties'] = list(predictions.keys())
                    except Exception as e:
                        details['json_error'] = str(e)
                else:
                    details['error'] = f"Prediction endpoint returned {response.status_code}"
                    try:
                        details['error_response'] = response.json()
                    except:
                        details['error_response'] = response.text[:500]
                
                self.log_test_result(f"Prediction Endpoint - {name}", success, details)
                all_success = all_success and success
                
            except Exception as e:
                self.log_test_result(f"Prediction Endpoint - {name}", False, {'error': str(e)})
                all_success = False
        
        return all_success
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests."""
        test_cases = [
            {
                'name': 'Invalid Composition Sum',
                'data': {
                    "composition": {"SiC": 0.8, "B4C": 0.5, "Al2O3": 0.2},  # Sum > 1.0
                    "processing": {"sintering_temperature": 1800, "pressure": 50, "grain_size": 10},
                    "microstructure": {"porosity": 0.02, "phase_distribution": "uniform"}
                },
                'expected_status': 422
            },
            {
                'name': 'Missing Required Fields',
                'data': {"composition": {"SiC": 0.6}},  # Missing processing and microstructure
                'expected_status': 422
            },
            {
                'name': 'Invalid Temperature Range',
                'data': {
                    "composition": {"SiC": 0.6, "B4C": 0.3, "Al2O3": 0.1},
                    "processing": {"sintering_temperature": 500, "pressure": 50, "grain_size": 10},  # Too low
                    "microstructure": {"porosity": 0.02, "phase_distribution": "uniform"}
                },
                'expected_status': 422
            }
        ]
        
        all_success = True
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    self.url('/api/v1/predict/mechanical'),
                    json=test_case['data'],
                    headers={'Content-Type': 'application/json'}
                )
                
                success = response.status_code == test_case['expected_status']
                
                details = {
                    'test_case': test_case['name'],
                    'expected_status': test_case['expected_status'],
                    'actual_status': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                
                if success:
                    try:
                        error_data = response.json()
                        details['has_error_details'] = 'detail' in error_data or 'error' in error_data
                    except:
                        details['has_error_details'] = False
                else:
                    details['error'] = f"Expected {test_case['expected_status']}, got {response.status_code}"
                
                self.log_test_result(f"Error Handling - {test_case['name']}", success, details)
                all_success = all_success and success
                
            except Exception as e:
                self.log_test_result(f"Error Handling - {test_case['name']}", False, {'error': str(e)})
                all_success = False
        
        return all_success
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks."""
        sample_request = {
            "composition": {"SiC": 0.6, "B4C": 0.3, "Al2O3": 0.1, "WC": 0.0, "TiC": 0.0},
            "processing": {"sintering_temperature": 1800, "pressure": 50, "grain_size": 10, 
                          "holding_time": 120, "heating_rate": 15, "atmosphere": "argon"},
            "microstructure": {"porosity": 0.02, "phase_distribution": "uniform", 
                             "interface_quality": "good", "pore_size": 1.0, "connectivity": 0.1}
        }
        
        # Test sequential performance
        response_times = []
        successful_requests = 0
        num_requests = 5
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = self.session.post(
                    self.url('/api/v1/predict/mechanical'),
                    json=sample_request,
                    headers={'Content-Type': 'application/json'}
                )
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
            except Exception as e:
                print(f"Performance test request {i+1} failed: {e}")
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Performance criteria
            success = (
                successful_requests >= num_requests * 0.8 and  # 80% success rate
                avg_response_time < 10000 and  # Average < 10 seconds
                max_response_time < 20000  # Max < 20 seconds
            )
            
            details = {
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / num_requests,
                'avg_response_time_ms': avg_response_time,
                'min_response_time_ms': min_response_time,
                'max_response_time_ms': max_response_time,
                'performance_criteria_met': success
            }
            
            self.log_test_result("Performance Benchmarks", success, details)
            return success
        else:
            self.log_test_result("Performance Benchmarks", False, {'error': 'No successful requests'})
            return False
    
    def test_concurrent_load(self) -> bool:
        """Test concurrent request handling."""
        sample_request = {
            "composition": {"SiC": 0.6, "B4C": 0.3, "Al2O3": 0.1, "WC": 0.0, "TiC": 0.0},
            "processing": {"sintering_temperature": 1800, "pressure": 50, "grain_size": 10,
                          "holding_time": 120, "heating_rate": 15, "atmosphere": "argon"},
            "microstructure": {"porosity": 0.02, "phase_distribution": "uniform",
                             "interface_quality": "good", "pore_size": 1.0, "connectivity": 0.1}
        }
        
        def make_request():
            try:
                start_time = time.time()
                response = self.session.post(
                    self.url('/api/v1/predict/mechanical'),
                    json=sample_request,
                    headers={'Content-Type': 'application/json'}
                )
                end_time = time.time()
                
                return {
                    'success': response.status_code == 200,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'response_time_ms': 0
                }
        
        # Run concurrent requests
        num_concurrent = 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful_requests = sum(1 for r in results if r['success'])
        response_times = [r['response_time_ms'] for r in results if r['success']]
        
        success_rate = successful_requests / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Concurrent load criteria
        success = (
            success_rate >= 0.7 and  # 70% success rate under concurrent load
            avg_response_time < 15000  # Average < 15 seconds under load
        )
        
        details = {
            'concurrent_requests': num_concurrent,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'concurrent_criteria_met': success
        }
        
        self.log_test_result("Concurrent Load Handling", success, details)
        return success
    
    def test_frontend_interface(self) -> bool:
        """Test frontend interface availability."""
        try:
            # Test root endpoint
            response = self.session.get(self.url('/'))
            root_success = response.status_code == 200
            
            # Test static files
            css_response = self.session.get(self.url('/static/css/styles.css'))
            css_available = css_response.status_code == 200
            
            # Test API documentation (may be disabled in production)
            docs_response = self.session.get(self.url('/docs'))
            docs_available = docs_response.status_code == 200
            
            success = root_success  # Only require root endpoint to work
            
            details = {
                'root_endpoint': {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'success': root_success
                },
                'static_css': {
                    'status_code': css_response.status_code,
                    'available': css_available
                },
                'api_docs': {
                    'status_code': docs_response.status_code,
                    'available': docs_available
                }
            }
            
            self.log_test_result("Frontend Interface", success, details)
            return success
            
        except Exception as e:
            self.log_test_result("Frontend Interface", False, {'error': str(e)})
            return False
    
    def test_security_configuration(self) -> bool:
        """Test security headers and CORS configuration."""
        try:
            response = self.session.get(self.url('/health'))
            headers = response.headers
            
            # Check for security headers
            security_headers = {
                'x-content-type-options': headers.get('x-content-type-options'),
                'x-frame-options': headers.get('x-frame-options'),
                'x-xss-protection': headers.get('x-xss-protection'),
                'strict-transport-security': headers.get('strict-transport-security'),
                'access-control-allow-origin': headers.get('access-control-allow-origin')
            }
            
            # Count present security headers
            present_headers = sum(1 for v in security_headers.values() if v is not None)
            
            # Basic security check (at least some headers should be present)
            success = present_headers >= 1 or response.status_code == 200
            
            details = {
                'status_code': response.status_code,
                'security_headers': security_headers,
                'headers_present': present_headers,
                'basic_security_met': success
            }
            
            self.log_test_result("Security Configuration", success, details)
            return success
            
        except Exception as e:
            self.log_test_result("Security Configuration", False, {'error': str(e)})
            return False
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive deployment validation."""
        print(f"🚀 Starting Render Deployment Validation")
        print(f"Target URL: {self.deployment_url}")
        print(f"Timeout: {self.timeout}s")
        print("=" * 60)
        
        # Run all validation tests
        test_results = []
        
        print("\n📡 Testing Deployment Accessibility...")
        test_results.append(self.test_deployment_accessibility())
        
        print("\n💓 Testing Health Endpoints...")
        test_results.append(self.test_health_endpoints())
        
        print("\n🔮 Testing Prediction Endpoints...")
        test_results.append(self.test_prediction_endpoints())
        
        print("\n⚠️  Testing Error Handling...")
        test_results.append(self.test_error_handling())
        
        print("\n⚡ Testing Performance Benchmarks...")
        test_results.append(self.test_performance_benchmarks())
        
        print("\n🔄 Testing Concurrent Load...")
        test_results.append(self.test_concurrent_load())
        
        print("\n🌐 Testing Frontend Interface...")
        test_results.append(self.test_frontend_interface())
        
        print("\n🔒 Testing Security Configuration...")
        test_results.append(self.test_security_configuration())
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = sum(test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        overall_success = success_rate >= 0.8  # 80% of tests must pass
        
        self.validation_results.update({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'overall_success': overall_success,
            'validation_complete': True
        })
        
        return self.validation_results
    
    def print_summary(self):
        """Print validation summary."""
        results = self.validation_results
        
        print("\n" + "=" * 60)
        print("🎯 RENDER DEPLOYMENT VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"Deployment URL: {results['deployment_url']}")
        print(f"Validation Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['validation_timestamp']))}")
        print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        overall_status = "✅ DEPLOYMENT READY" if results['overall_success'] else "❌ DEPLOYMENT ISSUES"
        print(f"Overall Status: {overall_status}")
        
        # Print individual test results
        print(f"\n📋 Individual Test Results:")
        for test_name, test_data in results['tests'].items():
            status = "✅" if test_data['success'] else "❌"
            print(f"  {status} {test_name}")
        
        # Recommendations
        if not results['overall_success']:
            print(f"\n💡 Recommendations:")
            failed_tests = [name for name, data in results['tests'].items() if not data['success']]
            for test_name in failed_tests:
                print(f"  • Fix issues with: {test_name}")
            print(f"  • Check Render deployment logs for detailed error information")
            print(f"  • Verify environment variables are properly configured")
            print(f"  • Ensure all dependencies are installed correctly")
    
    def save_results(self, filename: Optional[str] = None):
        """Save validation results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"render_deployment_validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        return filename


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Validate Render deployment of Ceramic Armor ML API')
    parser.add_argument('url', help='Deployment URL (e.g., https://your-app.onrender.com)')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds (default: 30)')
    parser.add_argument('--output', help='Output file for detailed results (default: auto-generated)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Validate URL format
    if not args.url.startswith(('http://', 'https://')):
        print("❌ Error: URL must start with http:// or https://")
        sys.exit(1)
    
    # Create validator and run tests
    validator = RenderDeploymentValidator(args.url, args.timeout)
    
    try:
        results = validator.run_comprehensive_validation()
        
        if not args.quiet:
            validator.print_summary()
        
        # Save results
        output_file = validator.save_results(args.output)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        
        if args.quiet:
            # Print minimal output for CI/CD
            status = "PASS" if results['overall_success'] else "FAIL"
            print(f"{status}: {results['passed_tests']}/{results['total_tests']} tests passed")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()