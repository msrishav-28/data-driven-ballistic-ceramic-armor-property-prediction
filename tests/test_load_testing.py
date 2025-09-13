"""
Load testing for Ceramic Armor ML API.

This module provides comprehensive load testing capabilities including:
- Stress testing with high concurrent loads
- Endurance testing for sustained periods
- Spike testing with sudden load increases
- Volume testing with large datasets
- Performance benchmarking and analysis
"""

import pytest
import requests
import time
import threading
import statistics
import json
import random
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os


@dataclass
class LoadTestResult:
    """Result of a load test execution."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    duration_seconds: float
    requests_per_second: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    errors: List[str]
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class MaterialGenerator:
    """Generate realistic material compositions for load testing."""
    
    @staticmethod
    def generate_random_composition() -> Dict[str, float]:
        """Generate a random but valid material composition."""
        # Primary ceramic materials with realistic ranges
        sic = random.uniform(0.2, 0.8)
        b4c = random.uniform(0.1, min(0.6, 1.0 - sic))
        al2o3 = random.uniform(0.05, min(0.4, 1.0 - sic - b4c))
        
        # Remaining fraction for other materials
        remaining = 1.0 - sic - b4c - al2o3
        
        # Distribute remaining among WC and TiC
        wc = random.uniform(0, remaining * 0.7) if remaining > 0.01 else 0
        tic = max(0, remaining - wc)
        
        # Normalize to ensure sum = 1.0
        total = sic + b4c + al2o3 + wc + tic
        if total > 0:
            sic /= total
            b4c /= total
            al2o3 /= total
            wc /= total
            tic /= total
        
        return {
            "SiC": round(sic, 3),
            "B4C": round(b4c, 3),
            "Al2O3": round(al2o3, 3),
            "WC": round(wc, 3),
            "TiC": round(tic, 3)
        }
    
    @staticmethod
    def generate_random_processing() -> Dict[str, Any]:
        """Generate random but realistic processing parameters."""
        return {
            "sintering_temperature": random.randint(1400, 2200),
            "pressure": random.randint(20, 150),
            "grain_size": round(random.uniform(1, 50), 1),
            "holding_time": random.randint(60, 300),
            "heating_rate": round(random.uniform(5, 25), 1),
            "atmosphere": random.choice(["argon", "nitrogen", "vacuum", "air"])
        }
    
    @staticmethod
    def generate_random_microstructure() -> Dict[str, Any]:
        """Generate random but realistic microstructure parameters."""
        return {
            "porosity": round(random.uniform(0.001, 0.15), 4),
            "phase_distribution": random.choice(["uniform", "gradient", "layered"]),
            "interface_quality": random.choice(["poor", "fair", "good", "excellent"]),
            "pore_size": round(random.uniform(0.1, 5.0), 2),
            "connectivity": round(random.uniform(0.01, 0.3), 3)
        }
    
    @classmethod
    def generate_material_request(cls) -> Dict[str, Any]:
        """Generate a complete material prediction request."""
        return {
            "composition": cls.generate_random_composition(),
            "processing": cls.generate_random_processing(),
            "microstructure": cls.generate_random_microstructure(),
            "include_uncertainty": random.choice([True, False]),
            "include_feature_importance": random.choice([True, False])
        }
    
    @classmethod
    def generate_material_batch(cls, count: int) -> List[Dict[str, Any]]:
        """Generate a batch of material requests."""
        return [cls.generate_material_request() for _ in range(count)]


class LoadTester:
    """Comprehensive load testing framework."""
    
    def __init__(self, base_url: str = None):
        """Initialize load tester."""
        self.base_url = base_url or os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
        self.session = requests.Session()
        self.session.timeout = 60  # Longer timeout for load testing
        
        # Configure session for load testing
        self.session.headers.update({
            'User-Agent': 'CeramicArmorML-LoadTester/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def url(self, path: str) -> str:
        """Construct full URL for endpoint."""
        from urllib.parse import urljoin
        return urljoin(self.base_url, path)
    
    def make_prediction_request(self, endpoint: str, material_data: Dict[str, Any]) -> Tuple[bool, float, Optional[str]]:
        """Make a single prediction request and return success, response time, and error."""
        try:
            start_time = time.time()
            response = self.session.post(self.url(endpoint), json=material_data)
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            success = response.status_code == 200
            error = None if success else f"HTTP {response.status_code}: {response.text[:200]}"
            
            return success, response_time_ms, error
            
        except Exception as e:
            return False, 0.0, str(e)
    
    def run_concurrent_load_test(
        self,
        endpoint: str,
        num_threads: int,
        requests_per_thread: int,
        test_name: str = "Concurrent Load Test"
    ) -> LoadTestResult:
        """Run concurrent load test with specified parameters."""
        
        def worker_thread():
            """Worker thread function."""
            thread_results = []
            materials = MaterialGenerator.generate_material_batch(requests_per_thread)
            
            for material in materials:
                success, response_time, error = self.make_prediction_request(endpoint, material)
                thread_results.append({
                    'success': success,
                    'response_time_ms': response_time,
                    'error': error
                })
            
            return thread_results
        
        # Execute concurrent load test
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread) for _ in range(num_threads)]
            
            for future in as_completed(futures):
                try:
                    thread_results = future.result()
                    all_results.extend(thread_results)
                except Exception as e:
                    print(f"Thread failed: {e}")
        
        end_time = time.time()
        
        # Analyze results
        return self._analyze_results(all_results, end_time - start_time, test_name)
    
    def run_sustained_load_test(
        self,
        endpoint: str,
        duration_seconds: int,
        requests_per_second: float,
        test_name: str = "Sustained Load Test"
    ) -> LoadTestResult:
        """Run sustained load test for specified duration."""
        
        start_time = time.time()
        all_results = []
        request_interval = 1.0 / requests_per_second
        
        while time.time() - start_time < duration_seconds:
            material = MaterialGenerator.generate_material_request()
            success, response_time, error = self.make_prediction_request(endpoint, material)
            
            all_results.append({
                'success': success,
                'response_time_ms': response_time,
                'error': error
            })
            
            # Wait for next request (if needed)
            elapsed = time.time() - start_time - len(all_results) * request_interval
            if elapsed < 0:
                time.sleep(-elapsed)
        
        end_time = time.time()
        
        return self._analyze_results(all_results, end_time - start_time, test_name)
    
    def run_spike_load_test(
        self,
        endpoint: str,
        baseline_rps: float,
        spike_rps: float,
        baseline_duration: int,
        spike_duration: int,
        test_name: str = "Spike Load Test"
    ) -> LoadTestResult:
        """Run spike load test with sudden load increases."""
        
        start_time = time.time()
        all_results = []
        
        # Phase 1: Baseline load
        phase1_end = start_time + baseline_duration
        while time.time() < phase1_end:
            material = MaterialGenerator.generate_material_request()
            success, response_time, error = self.make_prediction_request(endpoint, material)
            
            all_results.append({
                'success': success,
                'response_time_ms': response_time,
                'error': error,
                'phase': 'baseline'
            })
            
            time.sleep(1.0 / baseline_rps)
        
        # Phase 2: Spike load
        phase2_end = time.time() + spike_duration
        while time.time() < phase2_end:
            material = MaterialGenerator.generate_material_request()
            success, response_time, error = self.make_prediction_request(endpoint, material)
            
            all_results.append({
                'success': success,
                'response_time_ms': response_time,
                'error': error,
                'phase': 'spike'
            })
            
            time.sleep(1.0 / spike_rps)
        
        # Phase 3: Return to baseline
        phase3_end = time.time() + baseline_duration
        while time.time() < phase3_end:
            material = MaterialGenerator.generate_material_request()
            success, response_time, error = self.make_prediction_request(endpoint, material)
            
            all_results.append({
                'success': success,
                'response_time_ms': response_time,
                'error': error,
                'phase': 'recovery'
            })
            
            time.sleep(1.0 / baseline_rps)
        
        end_time = time.time()
        
        return self._analyze_results(all_results, end_time - start_time, test_name)
    
    def run_volume_load_test(
        self,
        endpoint: str,
        total_requests: int,
        max_concurrent: int,
        test_name: str = "Volume Load Test"
    ) -> LoadTestResult:
        """Run volume load test with large number of requests."""
        
        def worker_batch(batch_materials):
            """Process a batch of materials."""
            batch_results = []
            for material in batch_materials:
                success, response_time, error = self.make_prediction_request(endpoint, material)
                batch_results.append({
                    'success': success,
                    'response_time_ms': response_time,
                    'error': error
                })
            return batch_results
        
        # Generate all materials
        all_materials = MaterialGenerator.generate_material_batch(total_requests)
        
        # Split into batches for concurrent processing
        batch_size = max(1, total_requests // max_concurrent)
        batches = [all_materials[i:i + batch_size] for i in range(0, len(all_materials), batch_size)]
        
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(worker_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Batch failed: {e}")
        
        end_time = time.time()
        
        return self._analyze_results(all_results, end_time - start_time, test_name)
    
    def _analyze_results(self, results: List[Dict[str, Any]], duration: float, test_name: str) -> LoadTestResult:
        """Analyze load test results and return summary."""
        
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Response time analysis
        response_times = [r['response_time_ms'] for r in results if r['success']]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            p50_response_time = sorted_times[n // 2] if n > 0 else 0
            p95_response_time = sorted_times[int(n * 0.95)] if n > 0 else 0
            p99_response_time = sorted_times[int(n * 0.99)] if n > 0 else 0
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        # Collect errors
        errors = [r['error'] for r in results if r['error']]
        unique_errors = list(set(errors))
        
        return LoadTestResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            duration_seconds=duration,
            requests_per_second=total_requests / duration if duration > 0 else 0,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            errors=unique_errors
        )
    
    def run_comprehensive_load_test_suite(self) -> Dict[str, LoadTestResult]:
        """Run comprehensive load test suite."""
        results = {}
        
        print("Starting comprehensive load test suite...")
        
        # Test 1: Light concurrent load
        print("Running light concurrent load test...")
        results['light_concurrent'] = self.run_concurrent_load_test(
            endpoint='/api/v1/predict/mechanical',
            num_threads=3,
            requests_per_thread=5,
            test_name="Light Concurrent Load"
        )
        
        # Test 2: Medium concurrent load
        print("Running medium concurrent load test...")
        results['medium_concurrent'] = self.run_concurrent_load_test(
            endpoint='/api/v1/predict/mechanical',
            num_threads=8,
            requests_per_thread=3,
            test_name="Medium Concurrent Load"
        )
        
        # Test 3: Mixed endpoint load
        print("Running mixed endpoint load test...")
        results['mixed_endpoints'] = self.run_concurrent_load_test(
            endpoint='/api/v1/predict/ballistic',
            num_threads=5,
            requests_per_thread=4,
            test_name="Mixed Endpoint Load"
        )
        
        # Test 4: Sustained load
        print("Running sustained load test...")
        results['sustained_load'] = self.run_sustained_load_test(
            endpoint='/api/v1/predict/mechanical',
            duration_seconds=60,
            requests_per_second=2.0,
            test_name="Sustained Load"
        )
        
        # Test 5: Volume test
        print("Running volume load test...")
        results['volume_load'] = self.run_volume_load_test(
            endpoint='/api/v1/predict/mechanical',
            total_requests=50,
            max_concurrent=10,
            test_name="Volume Load"
        )
        
        # Test 6: Spike test
        print("Running spike load test...")
        results['spike_load'] = self.run_spike_load_test(
            endpoint='/api/v1/predict/mechanical',
            baseline_rps=1.0,
            spike_rps=5.0,
            baseline_duration=30,
            spike_duration=20,
            test_name="Spike Load"
        )
        
        return results


# Pytest test classes for load testing
class TestLoadTesting:
    """Load testing test cases."""
    
    @pytest.fixture(scope="class")
    def load_tester(self):
        """Create load tester instance."""
        deployment_url = os.getenv('DEPLOYMENT_URL', 'http://localhost:8000')
        return LoadTester(deployment_url)
    
    def test_light_concurrent_load(self, load_tester):
        """Test light concurrent load handling."""
        result = load_tester.run_concurrent_load_test(
            endpoint='/api/v1/predict/mechanical',
            num_threads=3,
            requests_per_thread=3,
            test_name="Light Concurrent Load Test"
        )
        
        # Assertions for light load
        assert result.success_rate >= 0.9, f"Success rate {result.success_rate:.2%} too low for light load"
        assert result.avg_response_time_ms < 5000, f"Average response time {result.avg_response_time_ms:.2f}ms too high"
        assert result.requests_per_second >= 0.5, f"Throughput {result.requests_per_second:.2f} RPS too low"
        
        print(f"Light Load Results: {result.successful_requests}/{result.total_requests} success, "
              f"{result.avg_response_time_ms:.2f}ms avg, {result.requests_per_second:.2f} RPS")
    
    def test_medium_concurrent_load(self, load_tester):
        """Test medium concurrent load handling."""
        result = load_tester.run_concurrent_load_test(
            endpoint='/api/v1/predict/mechanical',
            num_threads=5,
            requests_per_thread=4,
            test_name="Medium Concurrent Load Test"
        )
        
        # Assertions for medium load
        assert result.success_rate >= 0.8, f"Success rate {result.success_rate:.2%} too low for medium load"
        assert result.avg_response_time_ms < 8000, f"Average response time {result.avg_response_time_ms:.2f}ms too high"
        assert result.p95_response_time_ms < 15000, f"95th percentile {result.p95_response_time_ms:.2f}ms too high"
        
        print(f"Medium Load Results: {result.successful_requests}/{result.total_requests} success, "
              f"{result.avg_response_time_ms:.2f}ms avg, {result.p95_response_time_ms:.2f}ms p95")
    
    def test_mixed_endpoint_load(self, load_tester):
        """Test load across different endpoints."""
        # Test mechanical endpoint
        mech_result = load_tester.run_concurrent_load_test(
            endpoint='/api/v1/predict/mechanical',
            num_threads=3,
            requests_per_thread=2,
            test_name="Mechanical Endpoint Load"
        )
        
        # Test ballistic endpoint
        ball_result = load_tester.run_concurrent_load_test(
            endpoint='/api/v1/predict/ballistic',
            num_threads=3,
            requests_per_thread=2,
            test_name="Ballistic Endpoint Load"
        )
        
        # Both endpoints should perform reasonably
        assert mech_result.success_rate >= 0.8, "Mechanical endpoint success rate too low"
        assert ball_result.success_rate >= 0.8, "Ballistic endpoint success rate too low"
        
        # Response times should be comparable
        time_diff = abs(mech_result.avg_response_time_ms - ball_result.avg_response_time_ms)
        assert time_diff < 3000, f"Response time difference {time_diff:.2f}ms too high between endpoints"
    
    @pytest.mark.slow
    def test_sustained_load(self, load_tester):
        """Test sustained load over time."""
        result = load_tester.run_sustained_load_test(
            endpoint='/api/v1/predict/mechanical',
            duration_seconds=45,
            requests_per_second=1.5,
            test_name="Sustained Load Test"
        )
        
        # Assertions for sustained load
        assert result.success_rate >= 0.85, f"Sustained success rate {result.success_rate:.2%} too low"
        assert result.avg_response_time_ms < 6000, f"Sustained avg response time {result.avg_response_time_ms:.2f}ms too high"
        
        # Should maintain reasonable throughput
        expected_requests = 45 * 1.5  # duration * RPS
        actual_requests = result.total_requests
        throughput_ratio = actual_requests / expected_requests
        assert throughput_ratio >= 0.8, f"Throughput ratio {throughput_ratio:.2f} too low for sustained load"
        
        print(f"Sustained Load Results: {result.total_requests} requests in {result.duration_seconds:.1f}s, "
              f"{result.success_rate:.2%} success rate")
    
    @pytest.mark.slow
    def test_volume_load(self, load_tester):
        """Test high volume of requests."""
        result = load_tester.run_volume_load_test(
            endpoint='/api/v1/predict/mechanical',
            total_requests=30,
            max_concurrent=8,
            test_name="Volume Load Test"
        )
        
        # Assertions for volume load
        assert result.success_rate >= 0.75, f"Volume success rate {result.success_rate:.2%} too low"
        assert result.total_requests >= 25, f"Not enough requests processed: {result.total_requests}"
        
        # Performance should degrade gracefully
        assert result.avg_response_time_ms < 10000, f"Volume avg response time {result.avg_response_time_ms:.2f}ms too high"
        assert result.p99_response_time_ms < 20000, f"Volume 99th percentile {result.p99_response_time_ms:.2f}ms too high"
        
        print(f"Volume Load Results: {result.successful_requests}/{result.total_requests} success, "
              f"{result.requests_per_second:.2f} RPS, {result.p99_response_time_ms:.2f}ms p99")
    
    def test_error_resilience_under_load(self, load_tester):
        """Test error handling under load conditions."""
        # Generate mix of valid and invalid requests
        materials = []
        
        # Add valid requests
        materials.extend(MaterialGenerator.generate_material_batch(8))
        
        # Add some invalid requests
        invalid_materials = [
            {"composition": {"SiC": 1.5}},  # Invalid composition
            {"invalid": "data"},  # Completely invalid
            {},  # Empty request
        ]
        materials.extend(invalid_materials)
        
        # Shuffle to mix valid and invalid
        random.shuffle(materials)
        
        # Test with mixed requests
        start_time = time.time()
        results = []
        
        for material in materials:
            success, response_time, error = load_tester.make_prediction_request(
                '/api/v1/predict/mechanical', material
            )
            results.append({
                'success': success,
                'response_time_ms': response_time,
                'error': error
            })
        
        duration = time.time() - start_time
        
        # Analyze results
        valid_requests = len(materials) - len(invalid_materials)
        successful_requests = sum(1 for r in results if r['success'])
        
        # Valid requests should mostly succeed
        # Invalid requests should fail gracefully (not crash the system)
        assert successful_requests >= valid_requests * 0.8, "Too many valid requests failed"
        
        # System should remain responsive
        avg_response_time = statistics.mean([r['response_time_ms'] for r in results if r['response_time_ms'] > 0])
        assert avg_response_time < 8000, f"Response time {avg_response_time:.2f}ms too high under mixed load"
    
    @pytest.mark.comprehensive
    def test_comprehensive_load_suite(self, load_tester):
        """Run comprehensive load test suite."""
        results = load_tester.run_comprehensive_load_test_suite()
        
        # Verify all tests completed
        assert len(results) >= 5, "Not all load tests completed"
        
        # Check overall performance across all tests
        overall_success_rates = [r.success_rate for r in results.values()]
        avg_success_rate = statistics.mean(overall_success_rates)
        
        assert avg_success_rate >= 0.8, f"Overall success rate {avg_success_rate:.2%} too low across all tests"
        
        # Print comprehensive summary
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE LOAD TEST RESULTS")
        print(f"{'='*60}")
        
        for test_name, result in results.items():
            print(f"\n{test_name.upper().replace('_', ' ')}:")
            print(f"  Requests: {result.successful_requests}/{result.total_requests} "
                  f"({result.success_rate:.1%} success)")
            print(f"  Performance: {result.avg_response_time_ms:.1f}ms avg, "
                  f"{result.p95_response_time_ms:.1f}ms p95")
            print(f"  Throughput: {result.requests_per_second:.2f} RPS")
            
            if result.errors:
                print(f"  Errors: {len(result.errors)} unique error types")
        
        print(f"\nOverall Success Rate: {avg_success_rate:.2%}")
        
        # Save detailed results
        results_file = f"load_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            json_results = {}
            for name, result in results.items():
                json_results[name] = {
                    'test_name': result.test_name,
                    'total_requests': result.total_requests,
                    'successful_requests': result.successful_requests,
                    'success_rate': result.success_rate,
                    'duration_seconds': result.duration_seconds,
                    'requests_per_second': result.requests_per_second,
                    'avg_response_time_ms': result.avg_response_time_ms,
                    'p95_response_time_ms': result.p95_response_time_ms,
                    'p99_response_time_ms': result.p99_response_time_ms,
                    'errors': result.errors
                }
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")


if __name__ == "__main__":
    """Run load testing as standalone script."""
    import sys
    
    deployment_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8000'
    
    load_tester = LoadTester(deployment_url)
    results = load_tester.run_comprehensive_load_test_suite()
    
    print(f"\n{'='*60}")
    print(f"LOAD TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Target URL: {deployment_url}")
    
    total_requests = sum(r.total_requests for r in results.values())
    total_successful = sum(r.successful_requests for r in results.values())
    overall_success_rate = total_successful / total_requests if total_requests > 0 else 0
    
    print(f"Total Requests: {total_successful}/{total_requests} ({overall_success_rate:.1%} success)")
    
    for test_name, result in results.items():
        status = "✅" if result.success_rate >= 0.8 else "⚠️" if result.success_rate >= 0.6 else "❌"
        print(f"{status} {test_name}: {result.success_rate:.1%} success, "
              f"{result.avg_response_time_ms:.1f}ms avg, {result.requests_per_second:.1f} RPS")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success_rate >= 0.8 else 1)