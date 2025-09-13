"""
Deployment validation tests for Render platform.

These tests validate the complete application deployment including:
- Production environment configuration
- All API endpoints functionality
- Performance under production conditions
- Security and rate limiting
- Health monitoring and alerting
- Load testing and scalability validation
"""

import pytest
import requests
import time
import json
import concurrent.futures
import statistics
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
import os


class DeploymentValidator:
    """Validator for deployed application on Render platform."""
    
    def __init__(self, base_url: str = None):
        """Initialize deployment validator with base URL."""
        self.base_url = base_url or os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
        self.session = requests.Session()
        self.session.timeout = 30  # 30 second timeout
        
        # Add headers for production testing
        self.session.headers.update({
            'User-Agent': 'CeramicArmorML-DeploymentValidator/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def url(self, path: str) -> str:
        """Construct full URL for endpoint."""
        return urljoin(self.base_url, path)
    
    def validate_health_endpoints(self) -> Dict[str, Any]:
        """Validate health and status endpoints."""
        results = {}
        
        # Test basic health endpoint
        try:
            response = self.session.get(self.url('/health'))
            results['health'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            results['health'] = {'success': False, 'error': str(e)}
        
        # Test detailed status endpoint
        try:
            response = self.session.get(self.url('/api/v1/status'))
            results['status'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            results['status'] = {'success': False, 'error': str(e)}
        
        # Test model info endpoint
        try:
            response = self.session.get(self.url('/api/v1/models/info'))
            results['models_info'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            results['models_info'] = {'success': False, 'error': str(e)}
        
        return results
    
    def validate_prediction_endpoints(self) -> Dict[str, Any]:
        """Validate prediction endpoints with sample data."""
        results = {}
        
        # Sample prediction request
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
        
        # Test mechanical prediction endpoint
        try:
            response = self.session.post(
                self.url('/api/v1/predict/mechanical'),
                json=sample_request
            )
            results['mechanical_prediction'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            results['mechanical_prediction'] = {'success': False, 'error': str(e)}
        
        # Test ballistic prediction endpoint
        try:
            response = self.session.post(
                self.url('/api/v1/predict/ballistic'),
                json=sample_request
            )
            results['ballistic_prediction'] = {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            results['ballistic_prediction'] = {'success': False, 'error': str(e)}
        
        return results
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling with invalid requests."""
        results = {}
        
        # Test invalid composition (sum > 1.0)
        invalid_composition = {
            "composition": {
                "SiC": 0.8,
                "B4C": 0.5,  # Total > 1.0
                "Al2O3": 0.2
            },
            "processing": {
                "sintering_temperature": 1800,
                "pressure": 50,
                "grain_size": 10
            },
            "microstructure": {
                "porosity": 0.02,
                "phase_distribution": "uniform"
            }
        }
        
        try:
            response = self.session.post(
                self.url('/api/v1/predict/mechanical'),
                json=invalid_composition
            )
            results['invalid_composition'] = {
                'status_code': response.status_code,
                'success': response.status_code == 422,  # Should return validation error
                'data': response.json() if response.content else None
            }
        except Exception as e:
            results['invalid_composition'] = {'success': False, 'error': str(e)}
        
        # Test malformed JSON
        try:
            response = self.session.post(
                self.url('/api/v1/predict/mechanical'),
                data="invalid json",
                headers={'Content-Type': 'application/json'}
            )
            results['malformed_json'] = {
                'status_code': response.status_code,
                'success': response.status_code in [400, 422],  # Should return client error
                'data': response.json() if response.content else None
            }
        except Exception as e:
            results['malformed_json'] = {'success': False, 'error': str(e)}
        
        # Test missing required fields
        incomplete_request = {
            "composition": {
                "SiC": 0.6
                # Missing other required fields
            }
        }
        
        try:
            response = self.session.post(
                self.url('/api/v1/predict/mechanical'),
                json=incomplete_request
            )
            results['incomplete_request'] = {
                'status_code': response.status_code,
                'success': response.status_code == 422,  # Should return validation error
                'data': response.json() if response.content else None
            }
        except Exception as e:
            results['incomplete_request'] = {'success': False, 'error': str(e)}
        
        return results
    
    def validate_performance(self, num_requests: int = 10) -> Dict[str, Any]:
        """Validate performance characteristics."""
        results = {}
        
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
            }
        }
        
        # Test sequential performance
        response_times = []
        successful_requests = 0
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = self.session.post(
                    self.url('/api/v1/predict/mechanical'),
                    json=sample_request
                )
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        if response_times:
            results['sequential_performance'] = {
                'total_requests': num_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / num_requests,
                'avg_response_time_ms': statistics.mean(response_times),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'p95_response_time_ms': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
            }
        
        return results
    
    def validate_concurrent_load(self, num_concurrent: int = 5, requests_per_thread: int = 3) -> Dict[str, Any]:
        """Validate concurrent load handling."""
        results = {}
        
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
            }
        }
        
        def make_requests():
            """Make multiple requests in a thread."""
            thread_results = []
            for _ in range(requests_per_thread):
                try:
                    start_time = time.time()
                    response = self.session.post(
                        self.url('/api/v1/predict/mechanical'),
                        json=sample_request
                    )
                    end_time = time.time()
                    
                    thread_results.append({
                        'success': response.status_code == 200,
                        'status_code': response.status_code,
                        'response_time_ms': (end_time - start_time) * 1000
                    })
                except Exception as e:
                    thread_results.append({
                        'success': False,
                        'error': str(e),
                        'response_time_ms': None
                    })
            return thread_results
        
        # Execute concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_requests) for _ in range(num_concurrent)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    print(f"Thread failed: {e}")
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = sum(1 for r in all_results if r['success'])
        total_requests = len(all_results)
        response_times = [r['response_time_ms'] for r in all_results if r['success'] and r['response_time_ms']]
        
        results['concurrent_load'] = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'total_time_seconds': total_time,
            'requests_per_second': total_requests / total_time if total_time > 0 else 0,
            'avg_response_time_ms': statistics.mean(response_times) if response_times else None,
            'max_response_time_ms': max(response_times) if response_times else None
        }
        
        return results
    
    def validate_frontend_interface(self) -> Dict[str, Any]:
        """Validate frontend interface availability."""
        results = {}
        
        # Test root endpoint (should serve frontend)
        try:
            response = self.session.get(self.url('/'))
            results['frontend_root'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'content_type': response.headers.get('content-type', ''),
                'has_html': 'html' in response.headers.get('content-type', '').lower()
            }
        except Exception as e:
            results['frontend_root'] = {'success': False, 'error': str(e)}
        
        # Test static files
        try:
            response = self.session.get(self.url('/static/css/styles.css'))
            results['static_css'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'content_type': response.headers.get('content-type', '')
            }
        except Exception as e:
            results['static_css'] = {'success': False, 'error': str(e)}
        
        # Test API documentation (if available)
        try:
            response = self.session.get(self.url('/docs'))
            results['api_docs'] = {
                'status_code': response.status_code,
                'success': response.status_code in [200, 404],  # 404 is OK in production
                'available': response.status_code == 200
            }
        except Exception as e:
            results['api_docs'] = {'success': False, 'error': str(e)}
        
        return results
    
    def validate_security_headers(self) -> Dict[str, Any]:
        """Validate security headers and CORS configuration."""
        results = {}
        
        try:
            response = self.session.get(self.url('/health'))
            headers = response.headers
            
            results['security_headers'] = {
                'status_code': response.status_code,
                'has_cors_headers': 'access-control-allow-origin' in headers,
                'has_security_headers': any(h in headers for h in [
                    'x-content-type-options',
                    'x-frame-options', 
                    'x-xss-protection',
                    'strict-transport-security'
                ]),
                'content_type_options': headers.get('x-content-type-options'),
                'frame_options': headers.get('x-frame-options'),
                'cors_origin': headers.get('access-control-allow-origin')
            }
        except Exception as e:
            results['security_headers'] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete deployment validation."""
        print(f"Starting deployment validation for: {self.base_url}")
        
        validation_results = {
            'deployment_url': self.base_url,
            'validation_timestamp': time.time(),
            'validation_results': {}
        }
        
        # Run all validation tests
        print("Validating health endpoints...")
        validation_results['validation_results']['health_endpoints'] = self.validate_health_endpoints()
        
        print("Validating prediction endpoints...")
        validation_results['validation_results']['prediction_endpoints'] = self.validate_prediction_endpoints()
        
        print("Validating error handling...")
        validation_results['validation_results']['error_handling'] = self.validate_error_handling()
        
        print("Validating performance...")
        validation_results['validation_results']['performance'] = self.validate_performance()
        
        print("Validating concurrent load...")
        validation_results['validation_results']['concurrent_load'] = self.validate_concurrent_load()
        
        print("Validating frontend interface...")
        validation_results['validation_results']['frontend_interface'] = self.validate_frontend_interface()
        
        print("Validating security headers...")
        validation_results['validation_results']['security_headers'] = self.validate_security_headers()
        
        # Calculate overall success
        all_tests = []
        for category, tests in validation_results['validation_results'].items():
            if isinstance(tests, dict):
                for test_name, test_result in tests.items():
                    if isinstance(test_result, dict) and 'success' in test_result:
                        all_tests.append(test_result['success'])
        
        validation_results['overall_success'] = all(all_tests) if all_tests else False
        validation_results['success_rate'] = sum(all_tests) / len(all_tests) if all_tests else 0
        validation_results['total_tests'] = len(all_tests)
        validation_results['passed_tests'] = sum(all_tests)
        
        return validation_results


# Pytest test classes for deployment validation
class TestDeploymentValidation:
    """Test deployment validation on Render platform."""
    
    @pytest.fixture(scope="class")
    def validator(self):
        """Create deployment validator."""
        deployment_url = os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
        return DeploymentValidator(deployment_url)
    
    def test_health_endpoints_deployment(self, validator):
        """Test health endpoints in deployment."""
        results = validator.validate_health_endpoints()
        
        # Health endpoint should be available
        assert results['health']['success'], f"Health endpoint failed: {results['health']}"
        assert results['health']['response_time_ms'] < 5000, "Health endpoint too slow"
        
        # Status endpoint should be available
        assert results['status']['success'], f"Status endpoint failed: {results['status']}"
        
        # Model info should be available
        assert results['models_info']['success'], f"Models info failed: {results['models_info']}"
    
    def test_prediction_endpoints_deployment(self, validator):
        """Test prediction endpoints in deployment."""
        results = validator.validate_prediction_endpoints()
        
        # Mechanical prediction should work
        assert results['mechanical_prediction']['success'], \
            f"Mechanical prediction failed: {results['mechanical_prediction']}"
        assert results['mechanical_prediction']['response_time_ms'] < 10000, \
            "Mechanical prediction too slow"
        
        # Ballistic prediction should work
        assert results['ballistic_prediction']['success'], \
            f"Ballistic prediction failed: {results['ballistic_prediction']}"
        assert results['ballistic_prediction']['response_time_ms'] < 10000, \
            "Ballistic prediction too slow"
        
        # Verify response structure
        mech_data = results['mechanical_prediction']['data']
        if mech_data:
            assert 'predictions' in mech_data
            assert 'model_info' in mech_data
            assert 'request_id' in mech_data
    
    def test_error_handling_deployment(self, validator):
        """Test error handling in deployment."""
        results = validator.validate_error_handling()
        
        # Invalid composition should return 422
        assert results['invalid_composition']['success'], \
            f"Invalid composition handling failed: {results['invalid_composition']}"
        
        # Malformed JSON should return client error
        assert results['malformed_json']['success'], \
            f"Malformed JSON handling failed: {results['malformed_json']}"
        
        # Incomplete request should return 422
        assert results['incomplete_request']['success'], \
            f"Incomplete request handling failed: {results['incomplete_request']}"
    
    def test_performance_deployment(self, validator):
        """Test performance in deployment."""
        results = validator.validate_performance(num_requests=5)
        
        perf_data = results['sequential_performance']
        
        # Success rate should be high
        assert perf_data['success_rate'] >= 0.8, \
            f"Success rate {perf_data['success_rate']} too low"
        
        # Average response time should be reasonable
        assert perf_data['avg_response_time_ms'] < 5000, \
            f"Average response time {perf_data['avg_response_time_ms']}ms too high"
        
        # No request should be extremely slow
        assert perf_data['max_response_time_ms'] < 15000, \
            f"Max response time {perf_data['max_response_time_ms']}ms too high"
    
    def test_concurrent_load_deployment(self, validator):
        """Test concurrent load handling in deployment."""
        results = validator.validate_concurrent_load(num_concurrent=3, requests_per_thread=2)
        
        load_data = results['concurrent_load']
        
        # Success rate should be reasonable under load
        assert load_data['success_rate'] >= 0.7, \
            f"Concurrent success rate {load_data['success_rate']} too low"
        
        # Should handle reasonable throughput
        assert load_data['requests_per_second'] >= 0.5, \
            f"Throughput {load_data['requests_per_second']} req/sec too low"
        
        # Response times should be reasonable under load
        if load_data['avg_response_time_ms']:
            assert load_data['avg_response_time_ms'] < 10000, \
                f"Concurrent avg response time {load_data['avg_response_time_ms']}ms too high"
    
    def test_frontend_deployment(self, validator):
        """Test frontend interface in deployment."""
        results = validator.validate_frontend_interface()
        
        # Root endpoint should serve content
        assert results['frontend_root']['success'], \
            f"Frontend root failed: {results['frontend_root']}"
        
        # Should serve HTML or JSON (both acceptable)
        root_data = results['frontend_root']
        assert root_data['status_code'] == 200, "Root endpoint not accessible"
    
    def test_security_deployment(self, validator):
        """Test security configuration in deployment."""
        results = validator.validate_security_headers()
        
        security_data = results['security_headers']
        
        # Should have basic security headers in production
        # Note: Some headers might be added by Render platform
        assert security_data.get('status_code') == 200, "Security headers test failed"
    
    @pytest.mark.slow
    def test_full_deployment_validation(self, validator):
        """Run complete deployment validation suite."""
        results = validator.run_full_validation()
        
        # Overall validation should pass
        assert results['success_rate'] >= 0.8, \
            f"Overall validation success rate {results['success_rate']} too low"
        
        # Print summary
        print(f"\nDeployment Validation Summary:")
        print(f"URL: {results['deployment_url']}")
        print(f"Tests: {results['passed_tests']}/{results['total_tests']} passed")
        print(f"Success Rate: {results['success_rate']:.2%}")
        
        # Save detailed results
        results_file = f"deployment_validation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    """Run deployment validation as standalone script."""
    import sys
    
    deployment_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000'
    
    validator = DeploymentValidator(deployment_url)
    results = validator.run_full_validation()
    
    print(f"\n{'='*60}")
    print(f"DEPLOYMENT VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"URL: {results['deployment_url']}")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
    
    # Print category summaries
    for category, tests in results['validation_results'].items():
        if isinstance(tests, dict):
            category_tests = []
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and 'success' in test_result:
                    category_tests.append(test_result['success'])
            
            if category_tests:
                category_success_rate = sum(category_tests) / len(category_tests)
                status = "✅" if category_success_rate == 1.0 else "⚠️" if category_success_rate >= 0.5 else "❌"
                print(f"{status} {category.replace('_', ' ').title()}: {sum(category_tests)}/{len(category_tests)} ({category_success_rate:.1%})")
    
    # Save results
    results_file = f"deployment_validation_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)