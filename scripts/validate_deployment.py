#!/usr/bin/env python3
"""
Deployment validation script for Ceramic Armor ML API.
Validates that the deployed service is working correctly.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any, List
import httpx
import argparse


class DeploymentValidator:
    """Validates deployment health and functionality."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize validator with base URL and timeout."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []
    
    async def validate_health_check(self) -> Dict[str, Any]:
        """Validate health check endpoint."""
        print("🔍 Testing health check endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                
                result = {
                    "test": "health_check",
                    "status": "pass" if response.status_code == 200 else "fail",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": {}
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result["details"] = {
                            "response_data": data,
                            "has_status": "status" in data,
                            "has_timestamp": "timestamp" in data
                        }
                        print(f"✅ Health check passed: {data.get('status', 'unknown')}")
                    except json.JSONDecodeError:
                        result["details"]["error"] = "Invalid JSON response"
                        print("⚠️  Health check returned non-JSON response")
                else:
                    print(f"❌ Health check failed with status {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return {
                "test": "health_check",
                "status": "error",
                "error": str(e)
            }
    
    async def validate_api_info(self) -> Dict[str, Any]:
        """Validate API info endpoint."""
        print("🔍 Testing API info endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api")
                
                result = {
                    "test": "api_info",
                    "status": "pass" if response.status_code == 200 else "fail",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": {}
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result["details"] = {
                            "response_data": data,
                            "has_message": "message" in data,
                            "has_version": "version" in data
                        }
                        print(f"✅ API info passed: {data.get('message', 'unknown')}")
                    except json.JSONDecodeError:
                        result["details"]["error"] = "Invalid JSON response"
                        print("⚠️  API info returned non-JSON response")
                else:
                    print(f"❌ API info failed with status {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ API info error: {str(e)}")
            return {
                "test": "api_info",
                "status": "error",
                "error": str(e)
            }
    
    async def validate_status_endpoint(self) -> Dict[str, Any]:
        """Validate detailed status endpoint."""
        print("🔍 Testing status endpoint...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v1/status")
                
                result = {
                    "test": "status_endpoint",
                    "status": "pass" if response.status_code == 200 else "fail",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": {}
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        result["details"] = {
                            "response_data": data,
                            "has_system_info": "system" in data,
                            "has_models_info": "models" in data
                        }
                        print(f"✅ Status endpoint passed")
                    except json.JSONDecodeError:
                        result["details"]["error"] = "Invalid JSON response"
                        print("⚠️  Status endpoint returned non-JSON response")
                else:
                    print(f"❌ Status endpoint failed with status {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ Status endpoint error: {str(e)}")
            return {
                "test": "status_endpoint",
                "status": "error",
                "error": str(e)
            }
    
    async def validate_cors_headers(self) -> Dict[str, Any]:
        """Validate CORS headers."""
        print("🔍 Testing CORS headers...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Send OPTIONS request to check CORS
                response = await client.options(
                    f"{self.base_url}/api/v1/status",
                    headers={
                        "Origin": "https://example.com",
                        "Access-Control-Request-Method": "GET"
                    }
                )
                
                result = {
                    "test": "cors_headers",
                    "status": "pass" if response.status_code in [200, 204] else "fail",
                    "status_code": response.status_code,
                    "details": {
                        "cors_headers": dict(response.headers),
                        "has_cors_origin": "access-control-allow-origin" in response.headers,
                        "has_cors_methods": "access-control-allow-methods" in response.headers
                    }
                }
                
                if response.status_code in [200, 204]:
                    print("✅ CORS headers present")
                else:
                    print(f"❌ CORS validation failed with status {response.status_code}")
                
                return result
                
        except Exception as e:
            print(f"❌ CORS validation error: {str(e)}")
            return {
                "test": "cors_headers",
                "status": "error",
                "error": str(e)
            }
    
    async def validate_security_headers(self) -> Dict[str, Any]:
        """Validate security headers."""
        print("🔍 Testing security headers...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/health")
                
                security_headers = [
                    "x-content-type-options",
                    "x-frame-options",
                    "x-xss-protection",
                    "strict-transport-security"
                ]
                
                present_headers = []
                missing_headers = []
                
                for header in security_headers:
                    if header in response.headers:
                        present_headers.append(header)
                    else:
                        missing_headers.append(header)
                
                result = {
                    "test": "security_headers",
                    "status": "pass" if len(present_headers) >= 2 else "partial",
                    "details": {
                        "present_headers": present_headers,
                        "missing_headers": missing_headers,
                        "all_headers": dict(response.headers)
                    }
                }
                
                if len(present_headers) >= 2:
                    print(f"✅ Security headers present: {', '.join(present_headers)}")
                else:
                    print(f"⚠️  Limited security headers: {', '.join(present_headers)}")
                
                return result
                
        except Exception as e:
            print(f"❌ Security headers error: {str(e)}")
            return {
                "test": "security_headers",
                "status": "error",
                "error": str(e)
            }
    
    async def validate_rate_limiting(self) -> Dict[str, Any]:
        """Validate rate limiting (basic test)."""
        print("🔍 Testing rate limiting...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Make a few requests to check for rate limit headers
                response = await client.get(f"{self.base_url}/health")
                
                rate_limit_headers = [
                    "x-ratelimit-limit",
                    "x-ratelimit-remaining",
                    "x-ratelimit-reset"
                ]
                
                present_headers = []
                for header in rate_limit_headers:
                    if header in response.headers:
                        present_headers.append(header)
                
                result = {
                    "test": "rate_limiting",
                    "status": "pass" if present_headers else "unknown",
                    "details": {
                        "rate_limit_headers": present_headers,
                        "all_headers": dict(response.headers)
                    }
                }
                
                if present_headers:
                    print(f"✅ Rate limiting headers present: {', '.join(present_headers)}")
                else:
                    print("⚠️  No rate limiting headers detected (may still be active)")
                
                return result
                
        except Exception as e:
            print(f"❌ Rate limiting test error: {str(e)}")
            return {
                "test": "rate_limiting",
                "status": "error",
                "error": str(e)
            }
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print(f"🚀 Starting deployment validation for: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all validation tests
        validations = [
            self.validate_health_check(),
            self.validate_api_info(),
            self.validate_status_endpoint(),
            self.validate_cors_headers(),
            self.validate_security_headers(),
            self.validate_rate_limiting()
        ]
        
        results = await asyncio.gather(*validations, return_exceptions=True)
        
        # Process results
        passed = 0
        failed = 0
        errors = 0
        
        for result in results:
            if isinstance(result, Exception):
                errors += 1
                self.results.append({
                    "test": "unknown",
                    "status": "error",
                    "error": str(result)
                })
            else:
                self.results.append(result)
                if result["status"] == "pass":
                    passed += 1
                elif result["status"] == "fail":
                    failed += 1
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errors: {errors}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        
        overall_status = "pass" if failed == 0 and errors == 0 else "fail"
        
        summary = {
            "overall_status": overall_status,
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total_time_seconds": total_time,
            "base_url": self.base_url,
            "timestamp": time.time(),
            "detailed_results": self.results
        }
        
        if overall_status == "pass":
            print("\n🎉 All validations passed! Deployment is healthy.")
        else:
            print("\n⚠️  Some validations failed. Check the detailed results.")
        
        return summary


async def main():
    """Main function to run deployment validation."""
    parser = argparse.ArgumentParser(description="Validate Ceramic Armor ML API deployment")
    parser.add_argument(
        "url",
        help="Base URL of the deployed service (e.g., https://ceramic-armor-ml-api.onrender.com)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = DeploymentValidator(args.url, args.timeout)
    summary = await validator.run_all_validations()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_status"] == "pass" else 1)


if __name__ == "__main__":
    asyncio.run(main())