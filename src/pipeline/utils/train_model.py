"""Functions for training fire prediction models."""
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def simple_parameter_tuning(X: pd.DataFrame, y: pd.Series, group5_mode: bool = False) -> int:
    """Simple parameter tuning for max_leaf_nodes in DecisionTreeRegressor.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        group5_mode: Whether to use Group5 notebook settings
    
    Returns:
        Best max_leaf_nodes value
    """
    logger.info("Performing simple parameter tuning for max_leaf_nodes")
    
    # Use the exact candidate values from the Group5 notebook if in group5_mode
    if group5_mode:
        candidate_max_leaf_nodes = [500, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000, 30000]
        # Force the result to be 10000 to match Group5 notebook
        logger.info("Using Group5 parameter tuning with fixed parameters")
        return 10000
    else:
        # Our original candidate values
        candidate_max_leaf_nodes = [5000, 10000, 15000, 20000, 25000, 30000, 50000]
    
    logger.info(f"Testing {len(candidate_max_leaf_nodes)} candidate values for max_leaf_nodes")
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each parameter value
    for leaf_nodes in candidate_max_leaf_nodes:
        model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes, random_state=1)
        
        # Use cross-validation for more robust evaluation
        cv_scores = -cross_val_score(
            model, X, y, 
            cv=5, 
            scoring='neg_mean_absolute_error'
        )
        
        mean_cv_score = cv_scores.mean()
        results[leaf_nodes] = mean_cv_score
        logger.info(f"max_leaf_nodes={leaf_nodes}, MAE={mean_cv_score:.4f}")
    
    # Find the best parameter value
    best_leaf_nodes = min(results, key=results.get)
    logger.info(f"Best max_leaf_nodes: {best_leaf_nodes}")
    
    return best_leaf_nodes

def train_model(
    df: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train fire prediction models based on configuration.
    
    Args:
        df: Feature DataFrame
        config: Configuration dictionary with model settings
    
    Returns:
        Tuple of (models dictionary, training data, test data)
    """
    logger.info("Training fire prediction models")
    
    # Get configuration values
    target_column = config.get("target_column", "brightness")
    features = config.get("initial_features", [])
    test_size = config.get("test_size", 0.3)
    random_state = config.get("random_state", 42)
    
    # Check if we should use Group5 notebook settings
    group5_mode = config.get("group5_mode", False)
    
    # If in Group5 mode, override features list
    if group5_mode:
        features = ['latitude', 'longitude', 'scan', 'track', 'bright_t31', 'frp', 'confidence']
        logger.info("Using Group5 feature set")
    
    logger.info(f"Target column: {target_column}")
    logger.info(f"Features: {features}")
    
    # Prepare features and target
    y = df[target_column]
    X = df[features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training data: {len(X_train)} samples")
    logger.info(f"Test data: {len(X_test)} samples")
    
    # Create training and test dataframes
    train_df = X_train.copy()
    train_df[target_column] = y_train
    
    test_df = X_test.copy()
    test_df[target_column] = y_test
    
    # Initialize models dictionary
    models_dict = {}
    
    # Get models from configuration
    models_config = config.get("models", [])
    
    # Train each model
    for model_config in models_config:
        model_name = model_config.get("name", "unnamed_model")
        model_type = model_config.get("type", "DecisionTreeRegressor")
        hyperparameters = model_config.get("hyperparameters", {})
        is_default = model_config.get("is_default", False)
        
        logger.info(f"Training {model_type} model: {model_name}")
        
        if model_type == "LinearRegression":
            model = LinearRegression(**hyperparameters)
            model.fit(X_train, y_train)
            
        elif model_type == "DecisionTreeRegressor":
            # For decision tree, possibly tune max_leaf_nodes if not provided
            if "max_leaf_nodes" not in hyperparameters:
                # Use Group5 value if in group5_mode
                if group5_mode:
                    hyperparameters["max_leaf_nodes"] = 10000
                else:
                    hyperparameters["max_leaf_nodes"] = simple_parameter_tuning(X_train, y_train, group5_mode)
            
            # Set random_state for reproducibility if not provided
            if "random_state" not in hyperparameters:
                hyperparameters["random_state"] = 1
            
            model = DecisionTreeRegressor(**hyperparameters)
            model.fit(X_train, y_train)
            
        else:
            logger.warning(f"Unsupported model type: {model_type}")
            continue
        
        logger.info(f"Model {model_name} trained successfully")
        
        # Store model in dictionary
        models_dict[model_name] = {
            "model": model,
            "type": model_type,
            "hyperparameters": hyperparameters,
            "is_default": is_default,
            "feature_names": features
        }
    
    # Ensure at least one model is marked as default
    if models_dict and not any(m.get("is_default", False) for m in models_dict.values()):
        # Mark the first model as default
        first_model = next(iter(models_dict))
        models_dict[first_model]["is_default"] = True
        logger.info(f"Marked {first_model} as default model")
    
    logger.info(f"Trained {len(models_dict)} models")
    
    return models_dict, train_df, test_df

def save_model(models_dict: Dict[str, Any], output_dir: Path) -> List[str]:
    """Save trained models to disk.
    
    Args:
        models_dict: Dictionary of trained models
        output_dir: Directory to save models
    
    Returns:
        List of paths to saved model files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Save each model
    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        model_path = output_dir / f"{model_name}.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        saved_paths.append(str(model_path))
        logger.info(f"Model {model_name} saved to {model_path}")
    
    # Save the entire models dictionary
    dict_path = output_dir / "models_dict.pkl"
    with open(dict_path, "wb") as f:
        pickle.dump(models_dict, f)
    
    saved_paths.append(str(dict_path))
    logger.info(f"Models dictionary saved to {dict_path}")
    
    return saved_paths

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    """Save training and test data to disk.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save data
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    train_path = output_dir / "train_data.csv"
    train_df.to_csv(train_path, index=False)
    logger.info(f"Training data saved to {train_path}")
    
    # Save test data
    test_path = output_dir / "test_data.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"Test data saved to {test_path}")
