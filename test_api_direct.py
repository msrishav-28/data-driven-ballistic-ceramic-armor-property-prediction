"""
Direct API test script to verify the mechanical prediction endpoint.
"""

import requests
import json

# Test data
test_data = {
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
    "include_feature_importance": True,
    "prediction_type": "mechanical"
}

def test_api():
    """Test the API endpoint directly."""
    try:
        # Test the endpoint
        response = requests.post(
            "http://localhost:8000/api/v1/predict/mechanical",
            json=test_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Request ID: {result.get('request_id')}")
            print(f"Status: {result.get('status')}")
            
            # Check predictions
            predictions = result.get('predictions', {})
            print("\nPredictions:")
            for prop, pred in predictions.items():
                if pred and isinstance(pred, dict) and 'value' in pred:
                    print(f"  {prop}: {pred['value']} {pred.get('unit', '')} (quality: {pred.get('prediction_quality', 'unknown')})")
            
            # Check feature importance
            feature_importance = result.get('feature_importance', [])
            if feature_importance:
                print(f"\nTop 3 Important Features:")
                for feat in feature_importance[:3]:
                    print(f"  {feat['name']}: {feat['importance']:.3f}")
            
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()