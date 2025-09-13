"""
Test runner script for comprehensive testing suite.

This script provides utilities to run different types of tests:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance tests for response time and memory validation
- Load tests for concurrent and batch scenarios
"""

import pytest
import sys
import os
import time
import argparse
from pathlib import Path


def run_unit_tests():
    """Run unit tests for individual components."""
    print("Running unit tests...")
    
    unit_test_files = [
        "tests/test_main.py",
        "tests/test_api_models.py", 
        "tests/test_mechanical_prediction.py",
        "tests/test_ballistic_prediction.py",
        "tests/test_batch_processing.py",
        "tests/test_logging_middleware.py",
        "tests/test_security_middleware.py"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in unit_test_files if os.path.exists(f)]
    
    if not existing_files:
        print("No unit test files found!")
        return False
    
    # Run unit tests with verbose output
    result = pytest.main([
        "-v",
        "--tb=short",
        "--durations=10",
        *existing_files
    ])
    
    return result == 0


def run_integration_tests():
    """Run integration tests for complete workflows."""
    print("Running integration tests...")
    
    if not os.path.exists("tests/test_integration.py"):
        print("Integration test file not found!")
        return False
    
    result = pytest.main([
        "-v",
        "--tb=short",
        "--durations=10",
        "tests/test_integration.py"
    ])
    
    return result == 0


def run_performance_tests():
    """Run performance tests for response time and memory validation."""
    print("Running performance tests...")
    
    if not os.path.exists("tests/test_performance.py"):
        print("Performance test file not found!")
        return False
    
    result = pytest.main([
        "-v",
        "--tb=short",
        "--durations=20",
        "-s",  # Don't capture output for performance tests
        "tests/test_performance.py"
    ])
    
    return result == 0


def run_all_tests():
    """Run all test suites."""
    print("Running comprehensive test suite...")
    
    start_time = time.time()
    
    # Run all tests
    result = pytest.main([
        "-v",
        "--tb=short",
        "--durations=20",
        "tests/",
        "--ignore=tests/test_runner.py"  # Don't run this file as a test
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nTotal test execution time: {duration:.2f} seconds")
    
    return result == 0


def run_quick_tests():
    """Run a quick subset of tests for rapid feedback."""
    print("Running quick test suite...")
    
    quick_tests = [
        "tests/test_main.py::TestApplicationSetup",
        "tests/test_main.py::TestHealthEndpoints",
        "tests/test_mechanical_prediction.py::TestMechanicalPredictionEndpoint::test_endpoint_exists",
        "tests/test_ballistic_prediction.py::TestBallisticPredictionEndpoint::test_endpoint_exists"
    ]
    
    result = pytest.main([
        "-v",
        "--tb=line",
        *quick_tests
    ])
    
    return result == 0


def run_load_tests():
    """Run load and stress tests."""
    print("Running load tests...")
    
    # Run performance tests that include load testing
    if not os.path.exists("tests/test_performance.py"):
        print("Performance test file not found!")
        return False
    
    result = pytest.main([
        "-v",
        "--tb=short",
        "-s",
        "-k", "load or concurrent or sustained",
        "tests/test_performance.py"
    ])
    
    return result == 0


def run_validation_tests():
    """Run validation and error handling tests."""
    print("Running validation tests...")
    
    validation_tests = [
        "tests/test_api_models.py",
        "tests/test_mechanical_prediction.py::TestMechanicalPredictionValidation",
        "tests/test_ballistic_prediction.py::TestBallisticPredictionValidation",
        "tests/test_integration.py::TestErrorHandlingIntegration"
    ]
    
    # Filter to only existing test patterns
    existing_patterns = []
    for pattern in validation_tests:
        if "::" in pattern:
            file_path = pattern.split("::")[0]
            if os.path.exists(file_path):
                existing_patterns.append(pattern)
        elif os.path.exists(pattern):
            existing_patterns.append(pattern)
    
    if not existing_patterns:
        print("No validation test patterns found!")
        return False
    
    result = pytest.main([
        "-v",
        "--tb=short",
        *existing_patterns
    ])
    
    return result == 0


def generate_test_report():
    """Generate comprehensive test report."""
    print("Generating test report...")
    
    report_file = "test_report.html"
    
    result = pytest.main([
        "--html=" + report_file,
        "--self-contained-html",
        "-v",
        "tests/",
        "--ignore=tests/test_runner.py"
    ])
    
    if os.path.exists(report_file):
        print(f"Test report generated: {report_file}")
    
    return result == 0


def check_test_coverage():
    """Check test coverage."""
    print("Checking test coverage...")
    
    try:
        import coverage
    except ImportError:
        print("Coverage package not installed. Install with: pip install coverage")
        return False
    
    # Run tests with coverage
    result = pytest.main([
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "tests/",
        "--ignore=tests/test_runner.py"
    ])
    
    print("Coverage report generated in htmlcov/")
    
    return result == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Ceramic Armor ML API Test Runner")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "performance", "all", "quick", 
            "load", "validation", "report", "coverage"
        ],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Set up test environment
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during testing
    
    # Add src to Python path if not already there
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Run selected test type
    success = False
    
    if args.test_type == "unit":
        success = run_unit_tests()
    elif args.test_type == "integration":
        success = run_integration_tests()
    elif args.test_type == "performance":
        success = run_performance_tests()
    elif args.test_type == "all":
        success = run_all_tests()
    elif args.test_type == "quick":
        success = run_quick_tests()
    elif args.test_type == "load":
        success = run_load_tests()
    elif args.test_type == "validation":
        success = run_validation_tests()
    elif args.test_type == "report":
        success = generate_test_report()
    elif args.test_type == "coverage":
        success = check_test_coverage()
    
    # Exit with appropriate code
    if success:
        print(f"\n✅ {args.test_type.title()} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {args.test_type.title()} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()