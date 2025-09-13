"""
Performance tests for response time and memory usage validation.

These tests verify that the API meets performance requirements including:
- Response time thresholds for single predictions
- Memory usage limits and stability
- Concurrent request handling
- Load testing with multiple materials
- Resource cleanup and memory leak detection
"""

import pytest
import time
import threading
import statistics
import gc
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient


class TestResponseTimePerformance:
    """Test response time performance requirements."""
    
    def test_mechanical_prediction_response_time(self, client, valid_prediction_request, performance_thresholds):
        """Test mechanical prediction meets response time requirements."""
        request_data = valid_prediction_request.model_dump()
        
        # Warm up the system with one request
        client.post("/api/v1/predict/mechanical", json=request_data)
        
        # Measure response time for actual test
        start_time = time.time()
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < performance_thresholds['max_response_time_ms'], \
            f"Mechanical prediction took {response_time_ms:.2f}ms, exceeds limit of {performance_thresholds['max_response_time_ms']}ms"
    
    def test_ballistic_prediction_response_time(self, client, valid_prediction_request, performance_thresholds):
        """Test ballistic prediction meets response time requirements."""
        request_data = valid_prediction_request.model_dump()
        
        # Warm up the system
        client.post("/api/v1/predict/ballistic", json=request_data)
        
        # Measure response time
        start_time = time.time()
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < performance_thresholds['max_response_time_ms'], \
            f"Ballistic prediction took {response_time_ms:.2f}ms, exceeds limit of {performance_thresholds['max_response_time_ms']}ms"
    
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint has fast response time."""
        # Health endpoint should be very fast (< 100ms)
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 100, f"Health endpoint took {response_time_ms:.2f}ms, should be < 100ms"
    
    def test_status_endpoint_response_time(self, client):
        """Test status endpoint response time."""
        start_time = time.time()
        response = client.get("/api/v1/status")
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 200, f"Status endpoint took {response_time_ms:.2f}ms, should be < 200ms"
    
    def test_response_time_consistency(self, client, valid_prediction_request):
        """Test response time consistency across multiple requests."""
        request_data = valid_prediction_request.model_dump()
        response_times = []
        
        # Make multiple requests and measure response times
        for _ in range(10):
            start_time = time.time()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response times should be consistent (low standard deviation)
        coefficient_of_variation = std_dev / avg_time
        assert coefficient_of_variation < 0.5, f"Response time too variable: CV={coefficient_of_variation:.3f}"
        
        # No outliers should be more than 3x the average
        assert max_time < avg_time * 3, f"Max response time {max_time:.2f}ms too high vs avg {avg_time:.2f}ms"
        
        print(f"Response time stats: avg={avg_time:.2f}ms, std={std_dev:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")


class TestMemoryUsagePerformance:
    """Test memory usage performance and limits."""
    
    def test_single_request_memory_usage(self, client, valid_prediction_request, performance_monitor, performance_thresholds):
        """Test memory usage for single prediction request."""
        request_data = valid_prediction_request.model_dump()
        
        # Force garbage collection before test
        gc.collect()
        
        performance_monitor.start_monitoring()
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        metrics = performance_monitor.get_metrics()
        
        assert response.status_code == 200
        
        # Check memory usage limits
        assert metrics['memory_usage_mb'] < performance_thresholds['max_memory_usage_mb'], \
            f"Memory usage {metrics['memory_usage_mb']:.2f}MB exceeds limit of {performance_thresholds['max_memory_usage_mb']}MB"
        
        assert metrics['memory_delta_mb'] < performance_thresholds['max_memory_delta_mb'], \
            f"Memory delta {metrics['memory_delta_mb']:.2f}MB exceeds limit of {performance_thresholds['max_memory_delta_mb']}MB"
    
    def test_memory_leak_detection(self, client, valid_prediction_request):
        """Test for memory leaks across multiple requests."""
        request_data = valid_prediction_request.model_dump()
        
        # Get baseline memory usage
        gc.collect()
        import psutil
        import os
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        num_requests = 50
        for i in range(num_requests):
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            assert response.status_code == 200
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_growth = final_memory - baseline_memory
        memory_per_request = memory_growth / num_requests
        
        # Memory growth should be minimal (< 1MB per request on average)
        assert memory_per_request < 1.0, \
            f"Memory leak detected: {memory_per_request:.3f}MB per request, total growth {memory_growth:.2f}MB"
        
        print(f"Memory usage: baseline={baseline_memory:.2f}MB, final={final_memory:.2f}MB, growth={memory_growth:.2f}MB")
    
    def test_concurrent_memory_usage(self, client, valid_prediction_request, performance_thresholds):
        """Test memory usage under concurrent load."""
        request_data = valid_prediction_request.model_dump()
        
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def make_request():
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            return response.status_code == 200
        
        # Run concurrent requests
        num_concurrent = 10
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        # Check that all requests succeeded
        assert all(results), "Some concurrent requests failed"
        
        # Check memory usage after concurrent load
        gc.collect()
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory increase should be reasonable for concurrent requests
        max_expected_increase = performance_thresholds['max_memory_delta_mb'] * num_concurrent * 0.5  # Allow some sharing
        assert memory_increase < max_expected_increase, \
            f"Concurrent memory usage {memory_increase:.2f}MB too high, expected < {max_expected_increase:.2f}MB"
    
    def test_large_batch_memory_efficiency(self, client, load_test_materials):
        """Test memory efficiency with large batch of materials."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process materials in batches to test memory efficiency
        batch_size = 10
        materials = load_test_materials[:30]  # Use subset for testing
        
        for i in range(0, len(materials), batch_size):
            batch = materials[i:i + batch_size]
            
            # Process batch
            for material in batch:
                response = client.post("/api/v1/predict/mechanical", json=material)
                assert response.status_code == 200
            
            # Check memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_material = (current_memory - baseline_memory) / (i + len(batch))
            
            # Memory per material should be reasonable (< 5MB per material)
            assert memory_per_material < 5.0, \
                f"Memory per material {memory_per_material:.3f}MB too high after {i + len(batch)} materials"
        
        # Final cleanup
        gc.collect()


class TestConcurrentPerformance:
    """Test performance under concurrent load."""
    
    def test_concurrent_request_handling(self, client, valid_prediction_request):
        """Test handling of concurrent requests."""
        request_data = valid_prediction_request.model_dump()
        
        def make_timed_request():
            start_time = time.time()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            end_time = time.time()
            
            return {
                'success': response.status_code == 200,
                'response_time_ms': (end_time - start_time) * 1000,
                'status_code': response.status_code
            }
        
        # Run concurrent requests
        num_concurrent = 8
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_timed_request) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        response_times = [r['response_time_ms'] for r in results if r['success']]
        
        # At least 80% of requests should succeed
        success_rate = successful_requests / len(results)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} too low for concurrent requests"
        
        # Response times should still be reasonable under load
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            
            # Allow higher response times under concurrent load (2x normal limit)
            assert avg_response_time < 1000, f"Average response time {avg_response_time:.2f}ms too high under load"
            assert max_response_time < 2000, f"Max response time {max_response_time:.2f}ms too high under load"
    
    def test_mixed_endpoint_concurrency(self, client, valid_prediction_request):
        """Test concurrent requests to different endpoints."""
        request_data = valid_prediction_request.model_dump()
        
        def make_mechanical_request():
            start_time = time.time()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            end_time = time.time()
            return ('mechanical', response.status_code == 200, (end_time - start_time) * 1000)
        
        def make_ballistic_request():
            start_time = time.time()
            response = client.post("/api/v1/predict/ballistic", json=request_data)
            end_time = time.time()
            return ('ballistic', response.status_code == 200, (end_time - start_time) * 1000)
        
        def make_health_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return ('health', response.status_code == 200, (end_time - start_time) * 1000)
        
        # Mix of different request types
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            
            # Submit mixed requests
            for _ in range(4):
                futures.append(executor.submit(make_mechanical_request))
                futures.append(executor.submit(make_ballistic_request))
                futures.append(executor.submit(make_health_request))
            
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results by endpoint type
        endpoint_stats = {}
        for endpoint_type, success, response_time in results:
            if endpoint_type not in endpoint_stats:
                endpoint_stats[endpoint_type] = {'successes': 0, 'total': 0, 'response_times': []}
            
            endpoint_stats[endpoint_type]['total'] += 1
            if success:
                endpoint_stats[endpoint_type]['successes'] += 1
                endpoint_stats[endpoint_type]['response_times'].append(response_time)
        
        # Verify performance for each endpoint type
        for endpoint_type, stats in endpoint_stats.items():
            success_rate = stats['successes'] / stats['total']
            assert success_rate >= 0.8, f"{endpoint_type} success rate {success_rate:.2f} too low"
            
            if stats['response_times']:
                avg_time = statistics.mean(stats['response_times'])
                expected_max_time = 100 if endpoint_type == 'health' else 1500
                assert avg_time < expected_max_time, \
                    f"{endpoint_type} avg response time {avg_time:.2f}ms too high"
    
    def test_sustained_load_performance(self, client, valid_prediction_request):
        """Test performance under sustained load."""
        request_data = valid_prediction_request.model_dump()
        
        # Run sustained load for a period of time
        duration_seconds = 30
        start_time = time.time()
        request_count = 0
        successful_requests = 0
        response_times = []
        
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            request_end = time.time()
            
            request_count += 1
            if response.status_code == 200:
                successful_requests += 1
                response_times.append((request_end - request_start) * 1000)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        # Analyze sustained load performance
        success_rate = successful_requests / request_count
        requests_per_second = request_count / duration_seconds
        
        assert success_rate >= 0.9, f"Success rate {success_rate:.2f} too low under sustained load"
        assert requests_per_second >= 2, f"Request rate {requests_per_second:.2f} RPS too low"
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            
            assert avg_response_time < 800, f"Average response time {avg_response_time:.2f}ms too high under sustained load"
            assert p95_response_time < 1500, f"95th percentile response time {p95_response_time:.2f}ms too high"
        
        print(f"Sustained load: {request_count} requests in {duration_seconds}s, "
              f"{requests_per_second:.2f} RPS, {success_rate:.2%} success rate")


class TestResourceCleanupPerformance:
    """Test resource cleanup and efficiency."""
    
    def test_request_cleanup_efficiency(self, client, valid_prediction_request):
        """Test that resources are properly cleaned up after requests."""
        request_data = valid_prediction_request.model_dump()
        
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        # Baseline measurements
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        baseline_threads = threading.active_count()
        
        # Make a series of requests
        for i in range(20):
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            assert response.status_code == 200
            
            # Periodic cleanup check
            if i % 5 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                current_threads = threading.active_count()
                
                # Memory should not grow excessively
                memory_growth = current_memory - baseline_memory
                assert memory_growth < 50, f"Memory growth {memory_growth:.2f}MB too high after {i+1} requests"
                
                # Thread count should remain stable
                thread_growth = current_threads - baseline_threads
                assert thread_growth < 10, f"Thread count grew by {thread_growth} after {i+1} requests"
        
        # Final cleanup and verification
        gc.collect()
        time.sleep(0.1)  # Allow any background cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_threads = threading.active_count()
        
        total_memory_growth = final_memory - baseline_memory
        total_thread_growth = final_threads - baseline_threads
        
        assert total_memory_growth < 30, f"Total memory growth {total_memory_growth:.2f}MB indicates resource leak"
        assert total_thread_growth <= 2, f"Thread count grew by {total_thread_growth}, indicates thread leak"
    
    def test_error_handling_resource_cleanup(self, client):
        """Test that resources are cleaned up properly even when errors occur."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make requests that will cause validation errors
        invalid_requests = [
            {"invalid": "data"},
            {"composition": {"SiC": 2.0}},  # Invalid composition
            {},  # Empty request
            {"composition": {"SiC": 0.5}, "processing": {"sintering_temperature": -100}},  # Invalid processing
        ]
        
        for invalid_data in invalid_requests * 5:  # Repeat to amplify any leaks
            response = client.post("/api/v1/predict/mechanical", json=invalid_data)
            # Should return error, not crash
            assert response.status_code in [422, 400, 500]
        
        # Check that memory is still reasonable after error handling
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        assert memory_growth < 20, f"Memory growth {memory_growth:.2f}MB too high after error handling"


class TestLoadTestingPerformance:
    """Test performance with realistic load scenarios."""
    
    def test_varied_material_load(self, client, load_test_materials):
        """Test performance with varied material compositions."""
        materials = load_test_materials[:25]  # Use subset for testing
        
        start_time = time.time()
        successful_requests = 0
        response_times = []
        
        for material in materials:
            request_start = time.time()
            response = client.post("/api/v1/predict/mechanical", json=material)
            request_end = time.time()
            
            if response.status_code == 200:
                successful_requests += 1
                response_times.append((request_end - request_start) * 1000)
        
        total_time = time.time() - start_time
        
        # Performance expectations
        success_rate = successful_requests / len(materials)
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        throughput = successful_requests / total_time
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} too low for varied materials"
        assert avg_response_time < 600, f"Average response time {avg_response_time:.2f}ms too high for load test"
        assert throughput >= 2, f"Throughput {throughput:.2f} requests/sec too low"
        
        print(f"Load test: {len(materials)} materials, {success_rate:.2%} success, "
              f"{avg_response_time:.2f}ms avg response, {throughput:.2f} req/sec")
    
    def test_mixed_prediction_types_load(self, client, load_test_materials):
        """Test performance with mixed mechanical and ballistic predictions."""
        materials = load_test_materials[:20]  # Use subset
        
        def make_mixed_requests():
            results = []
            for i, material in enumerate(materials):
                # Alternate between mechanical and ballistic predictions
                endpoint = "/api/v1/predict/mechanical" if i % 2 == 0 else "/api/v1/predict/ballistic"
                
                start_time = time.time()
                response = client.post(endpoint, json=material)
                end_time = time.time()
                
                results.append({
                    'endpoint': endpoint,
                    'success': response.status_code == 200,
                    'response_time_ms': (end_time - start_time) * 1000
                })
            
            return results
        
        # Run the mixed load test
        results = make_mixed_requests()
        
        # Analyze results
        mechanical_results = [r for r in results if 'mechanical' in r['endpoint']]
        ballistic_results = [r for r in results if 'ballistic' in r['endpoint']]
        
        # Check success rates
        mech_success_rate = sum(1 for r in mechanical_results if r['success']) / len(mechanical_results)
        ball_success_rate = sum(1 for r in ballistic_results if r['success']) / len(ballistic_results)
        
        assert mech_success_rate >= 0.9, f"Mechanical success rate {mech_success_rate:.2%} too low"
        assert ball_success_rate >= 0.9, f"Ballistic success rate {ball_success_rate:.2%} too low"
        
        # Check response times
        mech_times = [r['response_time_ms'] for r in mechanical_results if r['success']]
        ball_times = [r['response_time_ms'] for r in ballistic_results if r['success']]
        
        if mech_times:
            mech_avg = statistics.mean(mech_times)
            assert mech_avg < 700, f"Mechanical avg response time {mech_avg:.2f}ms too high in mixed load"
        
        if ball_times:
            ball_avg = statistics.mean(ball_times)
            assert ball_avg < 700, f"Ballistic avg response time {ball_avg:.2f}ms too high in mixed load"