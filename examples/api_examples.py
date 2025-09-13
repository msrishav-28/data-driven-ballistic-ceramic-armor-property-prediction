"""
Comprehensive examples for the Ceramic Armor ML API.

This module provides practical examples of how to use all API endpoints
with realistic material data and proper error handling.
"""

import requests
import json
import time
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


class CeramicArmorAPIClient:
    """
    Client class for interacting with the Ceramic Armor ML API.
    
    Provides convenient methods for all API endpoints with built-in
    error handling, rate limiting, and response validation.
    """
    
    def __init__(self, base_url: str = "https://ceramic-armor-ml-api.onrender.com"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CeramicArmorML-Client/1.0'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                response = self.session.request(method, url, **kwargs)
            
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        response = self._make_request('GET', '/health')
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get detailed system status.
        
        Returns:
            Comprehensive system status information
        """
        response = self._make_request('GET', '/api/v1/status')
        return response.json()
    
    def get_models_info(self) -> Dict[str, Any]:
        """
        Get ML models information.
        
        Returns:
            Model status and metadata
        """
        response = self._make_request('GET', '/api/v1/models/info')
        return response.json()
    
    def predict_mechanical_properties(
        self,
        composition: Dict[str, float],
        processing: Dict[str, Any],
        microstructure: Dict[str, Any],
        include_uncertainty: bool = True,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Predict mechanical properties.
        
        Args:
            composition: Material composition fractions
            processing: Processing parameters
            microstructure: Microstructure characteristics
            include_uncertainty: Include uncertainty quantification
            include_feature_importance: Include feature importance analysis
            
        Returns:
            Mechanical property predictions
        """
        data = {
            "composition": composition,
            "processing": processing,
            "microstructure": microstructure,
            "include_uncertainty": include_uncertainty,
            "include_feature_importance": include_feature_importance,
            "prediction_type": "mechanical"
        }
        
        response = self._make_request('POST', '/api/v1/predict/mechanical', json=data)
        return response.json()
    
    def predict_ballistic_properties(
        self,
        composition: Dict[str, float],
        processing: Dict[str, Any],
        microstructure: Dict[str, Any],
        include_uncertainty: bool = True,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Predict ballistic properties.
        
        Args:
            composition: Material composition fractions
            processing: Processing parameters
            microstructure: Microstructure characteristics
            include_uncertainty: Include uncertainty quantification
            include_feature_importance: Include feature importance analysis
            
        Returns:
            Ballistic property predictions
        """
        data = {
            "composition": composition,
            "processing": processing,
            "microstructure": microstructure,
            "include_uncertainty": include_uncertainty,
            "include_feature_importance": include_feature_importance,
            "prediction_type": "ballistic"
        }
        
        response = self._make_request('POST', '/api/v1/predict/ballistic', json=data)
        return response.json()
    
    def upload_batch_file(
        self,
        file_path: str,
        output_format: str = "csv",
        include_uncertainty: bool = True,
        prediction_type: str = "both",
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Upload file for batch processing.
        
        Args:
            file_path: Path to CSV/Excel file
            output_format: Desired output format (csv, xlsx, json)
            include_uncertainty: Include uncertainty quantification
            prediction_type: Type of predictions (mechanical, ballistic, both)
            max_rows: Maximum rows to process
            
        Returns:
            Batch processing response with batch ID
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            data = {
                'file_format': file_path.suffix[1:],  # Remove the dot
                'output_format': output_format,
                'include_uncertainty': include_uncertainty,
                'include_feature_importance': False,  # Disabled for batch
                'prediction_type': prediction_type,
                'max_rows': max_rows
            }
            
            # Remove Content-Type header for multipart upload
            headers = {k: v for k, v in self.session.headers.items() if k != 'Content-Type'}
            
            response = requests.post(
                f"{self.base_url}/api/v1/predict/batch",
                files=files,
                data=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch processing status.
        
        Args:
            batch_id: Batch processing ID
            
        Returns:
            Batch status information
        """
        response = self._make_request('GET', f'/api/v1/predict/batch/{batch_id}/status')
        return response.json()
    
    def download_batch_results(self, batch_id: str, output_path: str) -> None:
        """
        Download batch processing results.
        
        Args:
            batch_id: Batch processing ID
            output_path: Local path to save results
        """
        response = self._make_request('GET', f'/api/v1/predict/batch/{batch_id}/download')
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Results downloaded to: {output_path}")


# Example material compositions
EXAMPLE_MATERIALS = {
    "high_performance_sic": {
        "name": "High Performance SiC Armor",
        "composition": {
            "SiC": 0.85,
            "B4C": 0.10,
            "Al2O3": 0.05,
            "WC": 0.0,
            "TiC": 0.0
        },
        "processing": {
            "sintering_temperature": 2100,
            "pressure": 80,
            "grain_size": 5.0,
            "holding_time": 180,
            "heating_rate": 10,
            "atmosphere": "argon"
        },
        "microstructure": {
            "porosity": 0.01,
            "phase_distribution": "uniform",
            "interface_quality": "excellent",
            "pore_size": 0.5,
            "connectivity": 0.05
        }
    },
    "lightweight_b4c": {
        "name": "Lightweight B₄C Composite",
        "composition": {
            "SiC": 0.20,
            "B4C": 0.70,
            "Al2O3": 0.10,
            "WC": 0.0,
            "TiC": 0.0
        },
        "processing": {
            "sintering_temperature": 1950,
            "pressure": 60,
            "grain_size": 8.0,
            "holding_time": 120,
            "heating_rate": 15,
            "atmosphere": "nitrogen"
        },
        "microstructure": {
            "porosity": 0.03,
            "phase_distribution": "uniform",
            "interface_quality": "good",
            "pore_size": 1.2,
            "connectivity": 0.15
        }
    },
    "cost_effective_alumina": {
        "name": "Cost-Effective Al₂O₃ Armor",
        "composition": {
            "SiC": 0.15,
            "B4C": 0.05,
            "Al2O3": 0.75,
            "WC": 0.05,
            "TiC": 0.0
        },
        "processing": {
            "sintering_temperature": 1650,
            "pressure": 40,
            "grain_size": 15.0,
            "holding_time": 90,
            "heating_rate": 12,
            "atmosphere": "air"
        },
        "microstructure": {
            "porosity": 0.05,
            "phase_distribution": "gradient",
            "interface_quality": "fair",
            "pore_size": 2.0,
            "connectivity": 0.25
        }
    },
    "multi_hit_armor": {
        "name": "Multi-Hit Capable Armor",
        "composition": {
            "SiC": 0.60,
            "B4C": 0.25,
            "Al2O3": 0.10,
            "WC": 0.05,
            "TiC": 0.0
        },
        "processing": {
            "sintering_temperature": 2000,
            "pressure": 70,
            "grain_size": 6.0,
            "holding_time": 150,
            "heating_rate": 8,
            "atmosphere": "argon"
        },
        "microstructure": {
            "porosity": 0.02,
            "phase_distribution": "layered",
            "interface_quality": "excellent",
            "pore_size": 0.8,
            "connectivity": 0.08
        }
    }
}


def example_health_check():
    """Example: Check API health status."""
    print("=== Health Check Example ===")
    
    client = CeramicArmorAPIClient()
    
    try:
        health = client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime_seconds']} seconds")
        
    except Exception as e:
        print(f"Health check failed: {e}")


def example_system_status():
    """Example: Get detailed system status."""
    print("\n=== System Status Example ===")
    
    client = CeramicArmorAPIClient()
    
    try:
        status = client.get_system_status()
        print(f"Overall Status: {status['overall_status']}")
        print(f"Memory Usage: {status['system']['resources']['memory']['usage_percent']}%")
        print(f"CPU Usage: {status['system']['resources']['cpu']['usage_percent']}%")
        print(f"ML System Health: {status['ml_system']['overall_health']}")
        
    except Exception as e:
        print(f"System status check failed: {e}")


def example_mechanical_prediction():
    """Example: Predict mechanical properties."""
    print("\n=== Mechanical Prediction Example ===")
    
    client = CeramicArmorAPIClient()
    material = EXAMPLE_MATERIALS["high_performance_sic"]
    
    try:
        result = client.predict_mechanical_properties(
            composition=material["composition"],
            processing=material["processing"],
            microstructure=material["microstructure"],
            include_uncertainty=True,
            include_feature_importance=True
        )
        
        print(f"Material: {material['name']}")
        print(f"Request ID: {result['request_id']}")
        print(f"Status: {result['status']}")
        
        predictions = result['predictions']
        print(f"\nPredicted Properties:")
        print(f"  Fracture Toughness: {predictions['fracture_toughness']['value']:.2f} {predictions['fracture_toughness']['unit']}")
        print(f"  Vickers Hardness: {predictions['vickers_hardness']['value']:.0f} {predictions['vickers_hardness']['unit']}")
        print(f"  Density: {predictions['density']['value']:.2f} {predictions['density']['unit']}")
        print(f"  Elastic Modulus: {predictions['elastic_modulus']['value']:.0f} {predictions['elastic_modulus']['unit']}")
        
        if result.get('feature_importance'):
            print(f"\nTop 3 Important Features:")
            for i, feature in enumerate(result['feature_importance'][:3]):
                print(f"  {i+1}. {feature['name']}: {feature['importance']:.3f}")
        
    except Exception as e:
        print(f"Mechanical prediction failed: {e}")


def example_ballistic_prediction():
    """Example: Predict ballistic properties."""
    print("\n=== Ballistic Prediction Example ===")
    
    client = CeramicArmorAPIClient()
    material = EXAMPLE_MATERIALS["multi_hit_armor"]
    
    try:
        result = client.predict_ballistic_properties(
            composition=material["composition"],
            processing=material["processing"],
            microstructure=material["microstructure"],
            include_uncertainty=True,
            include_feature_importance=True
        )
        
        print(f"Material: {material['name']}")
        print(f"Request ID: {result['request_id']}")
        print(f"Status: {result['status']}")
        
        predictions = result['predictions']
        print(f"\nPredicted Properties:")
        print(f"  V50 Velocity: {predictions['v50_velocity']['value']:.0f} {predictions['v50_velocity']['unit']}")
        print(f"  Penetration Resistance: {predictions['penetration_resistance']['value']:.2f} {predictions['penetration_resistance']['unit']}")
        print(f"  Back Face Deformation: {predictions['back_face_deformation']['value']:.1f} {predictions['back_face_deformation']['unit']}")
        print(f"  Multi-Hit Capability: {predictions['multi_hit_capability']['value']:.2f} {predictions['multi_hit_capability']['unit']}")
        
        if result.get('processing_notes'):
            print(f"\nProcessing Notes:")
            for note in result['processing_notes']:
                print(f"  - {note}")
        
    except Exception as e:
        print(f"Ballistic prediction failed: {e}")


def example_batch_processing():
    """Example: Batch processing with CSV file."""
    print("\n=== Batch Processing Example ===")
    
    # Create sample CSV file
    sample_data = []
    for material_key, material in EXAMPLE_MATERIALS.items():
        row = {}
        row.update(material["composition"])
        row.update(material["processing"])
        row.update(material["microstructure"])
        row["material_name"] = material["name"]
        sample_data.append(row)
    
    df = pd.DataFrame(sample_data)
    csv_path = "sample_materials.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV file: {csv_path}")
    
    client = CeramicArmorAPIClient()
    
    try:
        # Upload batch file
        batch_response = client.upload_batch_file(
            file_path=csv_path,
            output_format="csv",
            include_uncertainty=True,
            prediction_type="both",
            max_rows=100
        )
        
        batch_id = batch_response["request_id"]
        print(f"Batch uploaded successfully. Batch ID: {batch_id}")
        
        # Poll for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status = client.get_batch_status(batch_id)
            print(f"Attempt {attempt + 1}: Status = {status['status']}")
            
            if status['status'] == 'success':
                print(f"Batch completed successfully!")
                print(f"Total processed: {status['total_processed']}")
                print(f"Successful: {status['successful_predictions']}")
                print(f"Failed: {status['failed_predictions']}")
                print(f"Processing time: {status['processing_time_seconds']:.2f} seconds")
                
                if status.get('download_url'):
                    # Download results
                    results_path = f"batch_results_{batch_id}.csv"
                    client.download_batch_results(batch_id, results_path)
                    print(f"Results saved to: {results_path}")
                
                break
            elif status['status'] == 'failed':
                print(f"Batch processing failed!")
                break
            else:
                time.sleep(2)  # Wait 2 seconds before next check
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
    
    finally:
        # Clean up sample file
        if Path(csv_path).exists():
            Path(csv_path).unlink()


def example_error_handling():
    """Example: Demonstrate error handling."""
    print("\n=== Error Handling Example ===")
    
    client = CeramicArmorAPIClient()
    
    # Example 1: Invalid composition (sum > 1.0)
    print("1. Testing invalid composition...")
    try:
        result = client.predict_mechanical_properties(
            composition={
                "SiC": 0.8,
                "B4C": 0.5,  # This makes total > 1.0
                "Al2O3": 0.2
            },
            processing={
                "sintering_temperature": 1800,
                "pressure": 50,
                "grain_size": 10
            },
            microstructure={
                "porosity": 0.02,
                "phase_distribution": "uniform"
            }
        )
    except requests.HTTPError as e:
        if e.response.status_code == 422:
            error_data = e.response.json()
            print(f"  Validation Error: {error_data['detail']}")
        else:
            print(f"  HTTP Error: {e}")
    
    # Example 2: Out of range temperature
    print("2. Testing out-of-range temperature...")
    try:
        result = client.predict_mechanical_properties(
            composition={
                "SiC": 0.6,
                "B4C": 0.3,
                "Al2O3": 0.1
            },
            processing={
                "sintering_temperature": 3000,  # Too high
                "pressure": 50,
                "grain_size": 10
            },
            microstructure={
                "porosity": 0.02,
                "phase_distribution": "uniform"
            }
        )
    except requests.HTTPError as e:
        if e.response.status_code == 422:
            error_data = e.response.json()
            print(f"  Validation Error: {error_data['detail']}")
        else:
            print(f"  HTTP Error: {e}")


def run_all_examples():
    """Run all API examples."""
    print("Ceramic Armor ML API - Comprehensive Examples")
    print("=" * 50)
    
    example_health_check()
    example_system_status()
    example_mechanical_prediction()
    example_ballistic_prediction()
    example_batch_processing()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()