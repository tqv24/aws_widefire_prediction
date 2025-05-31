import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
import yaml
from unittest.mock import patch, MagicMock
import sys
from sklearn.tree import DecisionTreeRegressor

# Mock Streamlit before any imports that use it
class MockStreamlit:
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []
        self.info_calls = []
        self.selectbox = MagicMock()
        self.file_uploader = MagicMock()
        self.button = MagicMock()
        self.components = MagicMock()
        self.components.v1 = MagicMock()
    
    def error(self, message):
        self.error_calls.append(message)
    
    def warning(self, message):
        self.warning_calls.append(message)
    
    def info(self, message):
        self.info_calls.append(message)

# Create and install the mock
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st
sys.modules['streamlit.components'] = mock_st.components
sys.modules['streamlit.components.v1'] = mock_st.components.v1

# Import utility functions
from src.app.utils.model_manager import (
    normalize_path,
    upload_model_to_s3,
    download_model_from_s3,
    load_model,
    predict_fire
)
from src.app.utils.data_loader import (
    generate_synthetic_data,
    load_data_from_s3,
    ensure_date_column
)
from src.app.utils.config_loader import load_config

class TestModelManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")
        
        # Create a simple model for testing
        self.model = DecisionTreeRegressor(random_state=42)
        X = np.random.rand(10, 7)  # 7 features as expected by the model
        y = np.random.rand(10)
        self.model.fit(X, y)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'latitude': np.random.uniform(-44, -10, 5),
            'longitude': np.random.uniform(112, 154, 5),
            'scan': np.random.uniform(0.5, 2.0, 5),
            'track': np.random.uniform(0.5, 2.0, 5),
            'bright_t31': np.random.uniform(250, 350, 5),
            'confidence': np.random.randint(0, 100, 5),
            'frp': np.random.uniform(5, 100, 5)
        })

    def test_normalize_path(self):
        # Test path normalization
        test_paths = [
            ("path\\to\\file", "path/to/file"),
            ("path//to//file", "path/to/file"),
            ("path/to/file", "path/to/file")
        ]
        for input_path, expected in test_paths:
            self.assertEqual(normalize_path(input_path), expected)

    def test_load_model(self):
        # Save model to temporary file
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Test loading the model
        loaded_model = load_model(self.model_path)
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, DecisionTreeRegressor)

    def test_predict_fire(self):
        # Test prediction with valid data
        predictions = predict_fire(self.model, self.test_data)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))

        # Test prediction with missing features
        invalid_data = self.test_data.drop('latitude', axis=1)
        with self.assertRaises(ValueError):
            predict_fire(self.model, invalid_data)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        os.rmdir(self.test_dir)

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.n_samples = 100

    def test_generate_synthetic_data(self):
        # Test synthetic data generation
        df = generate_synthetic_data(self.n_samples)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.n_samples)
        
        # Check required columns
        required_columns = [
            'latitude', 'longitude', 'brightness', 'scan', 'track',
            'bright_t31', 'frp', 'confidence', 'acq_date'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data ranges
        self.assertTrue(all(df['latitude'].between(-44, -10)))
        self.assertTrue(all(df['longitude'].between(112, 154)))
        self.assertTrue(all(df['brightness'].between(300, 500)))

    def test_ensure_date_column(self):
        # Test with DataFrame missing date column
        df = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [4, 5, 6]
        })
        
        df_with_date = ensure_date_column(df)
        self.assertIn('acq_date', df_with_date.columns)
        self.assertEqual(len(df_with_date), len(df))
        
        # Test with DataFrame having acquisition_date
        df_with_acq = pd.DataFrame({
            'latitude': [1, 2, 3],
            'acquisition_date': ['2019-10-01', '2019-10-02', '2019-10-03']
        })
        
        df_converted = ensure_date_column(df_with_acq)
        self.assertIn('acq_date', df_converted.columns)
        self.assertEqual(len(df_converted), len(df_with_acq))

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test config files
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_dir, "test_config.yaml")
        self.test_fallback_path = os.path.join(self.test_dir, "test_fallback.yaml")
        
        # Create test config files
        test_config = {
            "aws": {
                "bucket_name": "test-bucket",
                "prefix": "test-prefix"
            },
            "data": {
                "primary_data_path": "test_data.csv",
                "fallback_paths": ["test_fallback.csv"]
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        with open(self.test_fallback_path, 'w') as f:
            yaml.dump({"aws": {"bucket_name": "fallback-bucket"}}, f)

    def test_load_config(self):
        # Test loading from primary config
        config = load_config(self.test_config_path, self.test_fallback_path)
        self.assertEqual(config["aws"]["bucket_name"], "test-bucket")
        
        # Test loading from fallback config
        os.remove(self.test_config_path)
        config = load_config(self.test_config_path, self.test_fallback_path)
        self.assertEqual(config["aws"]["bucket_name"], "fallback-bucket")
        
        # Test loading with no config files
        os.remove(self.test_fallback_path)
        config = load_config(self.test_config_path, self.test_fallback_path)
        self.assertIn("aws", config)
        self.assertIn("bucket_name", config["aws"])

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_fallback_path):
            os.remove(self.test_fallback_path)
        os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()