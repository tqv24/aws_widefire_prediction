import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add the project root to the path so we can import the src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.generate_features import generate_features


@pytest.fixture
def test_data():
    """Create test data for all tests"""
    return pd.DataFrame({
        'visible_entropy': [1.5, 2.3, 0.8, 3.1],
        'visible_contrast': [0.5, 0.7, 0.3, 0.9],
        'IR_max': [300, 320, 280, 350],
        'IR_min': [270, 290, 260, 310],
        'IR_mean': [285, 305, 270, 330],
        'class': [0, 1, 0, 1]
    })

@pytest.fixture
def log_transform_config():
    """Configuration for log transform tests"""
    return {
        'log_transform': {
            'log_entropy': 'visible_entropy'
        }
    }

@pytest.fixture
def multiply_config():
    """Configuration for multiplication tests"""
    return {
        'multiply': {
            'entropy_x_contrast': {
                'col_a': 'visible_contrast',
                'col_b': 'visible_entropy'
            }
        }
    }

@pytest.fixture
def subtract_config():
    """Configuration for subtraction tests"""
    return {
        'subtract': {
            'IR_range': {
                'col_a': 'IR_max',
                'col_b': 'IR_min'
            }
        }
    }

@pytest.fixture
def norm_range_config():
    """Configuration for normalized range tests"""
    return {
        'calculate_norm_range': {
            'IR_norm_range': {
                'min_col': 'IR_min',
                'max_col': 'IR_max',
                'mean_col': 'IR_mean'
            }
        }
    }




# Happy path tests

def test_log_transform_happy(test_data, log_transform_config):
    """Test log transformation works correctly"""
    result = generate_features(test_data, log_transform_config)
    
    # Check the new column exists
    assert 'log_entropy' in result.columns
    
    # Verify values are correctly calculated
    expected_values = test_data['visible_entropy'].apply(np.log)
    expected_values.name = 'log_entropy'
    pd.testing.assert_series_equal(result['log_entropy'], expected_values)


def test_multiply_happy(test_data, multiply_config):
    """Test multiplication works correctly"""
    result = generate_features(test_data, multiply_config)
    
    # Check the new column exists
    assert 'entropy_x_contrast' in result.columns
    
    # Verify values are correctly calculated
    expected_values = test_data['visible_contrast'].multiply(test_data['visible_entropy'])
    expected_values.name = 'entropy_x_contrast'
    pd.testing.assert_series_equal(result['entropy_x_contrast'], expected_values)


def test_subtract_happy(test_data, subtract_config):
    """Test subtraction works correctly"""
    result = generate_features(test_data, subtract_config)
    
    # Check the new column exists
    assert 'IR_range' in result.columns
    
    # Verify values are correctly calculated
    expected_values = test_data['IR_max'] - test_data['IR_min']
    expected_values.name = 'IR_range'
    pd.testing.assert_series_equal(result['IR_range'], expected_values)


def test_norm_range_happy(test_data, norm_range_config):
    """Test normalized range calculation works correctly"""
    result = generate_features(test_data, norm_range_config)
    
    # Check the new column exists
    assert 'IR_norm_range' in result.columns
    
    # Verify values are correctly calculated
    expected_values = (test_data['IR_max'] - test_data['IR_min']).divide(test_data['IR_mean'])
    expected_values.name = 'IR_norm_range'
    pd.testing.assert_series_equal(result['IR_norm_range'], expected_values)




# Unhappy path tests
def test_log_transform_unhappy(test_data):
    """Test log transformation with missing column"""
    bad_config = {
        'log_transform': {
            'log_entropy': 'nonexistent_column'
        }
    }
    
    with pytest.raises(KeyError):
        generate_features(test_data, bad_config)


def test_multiply_unhappy(test_data):
    """Test multiplication with missing column"""
    bad_config = {
        'multiply': {
            'bad_feature': {
                'col_a': 'visible_contrast',
                'col_b': 'nonexistent_column'
            }
        }
    }
    
    with pytest.raises(KeyError):
        generate_features(test_data, bad_config)


def test_subtract_unhappy(test_data):
    """Test subtraction with missing column"""
    bad_config = {
        'subtract': {
            'bad_range': {
                'col_a': 'nonexistent_column',
                'col_b': 'IR_min'
            }
        }
    }
    
    with pytest.raises(KeyError):
        generate_features(test_data, bad_config)


def test_norm_range_unhappy(test_data, norm_range_config):
    """Test normalized range with zero division"""
    # Create data that will cause division by zero
    zero_data = test_data.copy()
    zero_data['IR_mean'] = [285, 0, 270, 330] 
    
    with pytest.raises(ZeroDivisionError):
        generate_features(zero_data, norm_range_config)


