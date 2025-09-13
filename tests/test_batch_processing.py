"""
Tests for batch processing and file upload functionality.
"""

import pytest
import pandas as pd
import io
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.main import app
from src.api.routes.upload import validate_csv_structure, parse_uploaded_file, convert_dataframe_row_to_request


client = TestClient(app)


class TestCSVValidation:
    """Test CSV file validation functionality."""
    
    def test_validate_csv_structure_valid(self):
        """Test validation with valid CSV structure."""
        df = pd.DataFrame({
            'SiC': [0.6, 0.7],
            'B4C': [0.3, 0.2],
            'Al2O3': [0.1, 0.1],
            'sintering_temperature': [1800, 1900],
            'pressure': [50, 60],
            'grain_size': [10, 12],
            'porosity': [0.02, 0.03],
            'phase_distribution': ['uniform', 'gradient']
        })
        
        result = validate_csv_structure(df)
        
        assert result['valid'] is True
        assert result['row_count'] == 2
        assert result['column_count'] == 8
        assert len(result['errors']) == 0
    
    def test_validate_csv_structure_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'SiC': [0.6],
            'B4C': [0.3],
            # Missing Al2O3, processing, and microstructure columns
        })
        
        result = validate_csv_structure(df)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('Al2O3' in error for error in result['errors'])
    
    def test_validate_csv_structure_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        
        result = validate_csv_structure(df)
        
        assert result['valid'] is False
        assert any('empty' in error.lower() for error in result['errors'])
    
    def test_validate_csv_structure_warnings(self):
        """Test validation warnings for data quality issues."""
        df = pd.DataFrame({
            'SiC': [0.6, 1.5],  # Invalid composition sum in second row
            'B4C': [0.3, 0.2],
            'Al2O3': [0.1, 0.1],
            'sintering_temperature': [1800, 3000],  # Invalid temperature in second row
            'pressure': [50, 60],
            'grain_size': [10, 12],
            'porosity': [0.02, 0.5],  # Invalid porosity in second row
            'phase_distribution': ['uniform', 'gradient']
        })
        
        result = validate_csv_structure(df)
        
        assert result['valid'] is True  # Structure is valid
        assert len(result['warnings']) > 0
        assert any('composition' in warning.lower() for warning in result['warnings'])
        assert any('temperature' in warning.lower() for warning in result['warnings'])
        assert any('porosity' in warning.lower() for warning in result['warnings'])


class TestFileParser:
    """Test file parsing functionality."""
    
    def test_parse_csv_file(self):
        """Test parsing CSV file content."""
        csv_content = """SiC,B4C,Al2O3,sintering_temperature,pressure,grain_size,porosity,phase_distribution
0.6,0.3,0.1,1800,50,10,0.02,uniform
0.7,0.2,0.1,1900,60,12,0.03,gradient"""
        
        file_content = csv_content.encode('utf-8')
        df = parse_uploaded_file(file_content, "test.csv")
        
        assert len(df) == 2
        assert 'SiC' in df.columns
        assert df.iloc[0]['SiC'] == 0.6
        assert df.iloc[1]['sintering_temperature'] == 1900
    
    def test_parse_csv_with_column_mapping(self):
        """Test parsing CSV with alternative column names."""
        csv_content = """sic,b4c,al2o3,temperature,pressure,grain,pore,phase_distribution
0.6,0.3,0.1,1800,50,10,0.02,uniform"""
        
        file_content = csv_content.encode('utf-8')
        df = parse_uploaded_file(file_content, "test.csv")
        
        # Check that columns were mapped correctly
        assert 'SiC' in df.columns
        assert 'B4C' in df.columns
        assert 'Al2O3' in df.columns
        assert 'sintering_temperature' in df.columns
        assert 'grain_size' in df.columns
        assert 'porosity' in df.columns


class TestDataConversion:
    """Test data conversion functionality."""
    
    def test_convert_dataframe_row_to_request(self):
        """Test converting DataFrame row to request format."""
        row = pd.Series({
            'SiC': 0.6,
            'B4C': 0.3,
            'Al2O3': 0.1,
            'sintering_temperature': 1800,
            'pressure': 50,
            'grain_size': 10,
            'porosity': 0.02,
            'phase_distribution': 'uniform'
        })
        
        request_data = convert_dataframe_row_to_request(row)
        
        assert 'composition' in request_data
        assert 'processing' in request_data
        assert 'microstructure' in request_data
        
        assert request_data['composition']['SiC'] == 0.6
        assert request_data['composition']['B4C'] == 0.3
        assert request_data['processing']['sintering_temperature'] == 1800
        assert request_data['microstructure']['porosity'] == 0.02
    
    def test_convert_dataframe_row_with_defaults(self):
        """Test conversion with missing values using defaults."""
        row = pd.Series({
            'SiC': 0.6,
            'B4C': 0.3,
            'Al2O3': 0.1,
            # Missing some processing and microstructure data
        })
        
        request_data = convert_dataframe_row_to_request(row)
        
        # Check that defaults were applied
        assert request_data['processing']['sintering_temperature'] == 1800  # Default
        assert request_data['processing']['pressure'] == 50  # Default
        assert request_data['microstructure']['porosity'] == 0.02  # Default


class TestBatchEndpoints:
    """Test batch processing API endpoints."""
    
    def create_test_csv_file(self) -> io.BytesIO:
        """Create a test CSV file for upload."""
        csv_content = """SiC,B4C,Al2O3,sintering_temperature,pressure,grain_size,porosity,phase_distribution
0.6,0.3,0.1,1800,50,10,0.02,uniform
0.7,0.2,0.1,1900,60,12,0.03,gradient
0.5,0.4,0.1,1750,45,8,0.015,layered"""
        
        return io.BytesIO(csv_content.encode('utf-8'))
    
    @patch('src.api.routes.upload.get_predictor')
    @patch('src.api.routes.upload.CeramicFeatureExtractor')
    def test_upload_batch_file_success(self, mock_extractor, mock_predictor):
        """Test successful batch file upload."""
        # Mock the predictor and feature extractor
        mock_predictor_instance = Mock()
        mock_predictor.return_value = mock_predictor_instance
        
        mock_predictor_instance.predict_mechanical_properties.return_value = {
            'predictions': {
                'fracture_toughness': {'value': 4.5, 'unit': 'MPa·m^0.5', 'uncertainty': 0.1},
                'vickers_hardness': {'value': 2800, 'unit': 'HV', 'uncertainty': 0.1},
                'density': {'value': 3.2, 'unit': 'g/cm³', 'uncertainty': 0.05},
                'elastic_modulus': {'value': 400, 'unit': 'GPa', 'uncertainty': 0.1}
            }
        }
        
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_all_features.return_value = pd.DataFrame({
            'feature_1': [1.0], 'feature_2': [2.0], 'feature_3': [3.0]
        })
        
        # Create test file
        test_file = self.create_test_csv_file()
        
        # Make request
        response = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.csv", test_file, "text/csv")},
            data={
                "file_format": "csv",
                "output_format": "csv",
                "include_uncertainty": "true",
                "prediction_type": "mechanical",
                "max_rows": "1000"
            }
        )
        
        assert response.status_code == 202
        response_data = response.json()
        assert response_data["status"] == "success"
        assert "request_id" in response_data
        assert response_data["request_id"].startswith("batch_")
    
    def test_upload_invalid_file_format(self):
        """Test upload with invalid file format."""
        # Create a text file that's not CSV/Excel/JSON
        test_file = io.BytesIO(b"This is not a valid CSV file")
        
        response = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.txt", test_file, "text/plain")},
            data={
                "file_format": "csv",
                "output_format": "csv"
            }
        )
        
        assert response.status_code == 422
    
    def test_upload_missing_required_columns(self):
        """Test upload with missing required columns."""
        csv_content = """SiC,B4C
0.6,0.3
0.7,0.2"""
        
        test_file = io.BytesIO(csv_content.encode('utf-8'))
        
        response = client.post(
            "/api/v1/predict/batch",
            files={"file": ("test.csv", test_file, "text/csv")},
            data={
                "file_format": "csv",
                "output_format": "csv"
            }
        )
        
        assert response.status_code == 422
        response_data = response.json()
        # The response structure may vary, check for error information
        assert "detail" in response_data or "message" in response_data
    
    def test_get_batch_status_not_found(self):
        """Test getting status for non-existent batch job."""
        response = client.get("/api/v1/predict/batch/nonexistent_batch_id/status")
        
        assert response.status_code == 404
    
    def test_download_batch_results_not_found(self):
        """Test downloading results for non-existent batch job."""
        response = client.get("/api/v1/predict/batch/nonexistent_batch_id/download")
        
        assert response.status_code == 404
    
    def test_delete_batch_job_not_found(self):
        """Test deleting non-existent batch job."""
        response = client.delete("/api/v1/predict/batch/nonexistent_batch_id")
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__])