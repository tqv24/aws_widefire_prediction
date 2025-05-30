"""Functions for scoring and evaluating fire prediction models."""
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, group5_mode: bool = False) -> Dict[str, Any]:
    """Evaluate a model on test data and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        group5_mode: Whether to use Group5 notebook settings
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Only calculate explained variance if not in group5_mode
    if not group5_mode:
        explained_variance = explained_variance_score(y_test, y_pred)
    else:
        explained_variance = None
    
    # Log results
    logger.info(f"Model Evaluation Results:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²: {r2:.4f}")
    if explained_variance is not None:
        logger.info(f"Explained Variance: {explained_variance:.4f}")
    
    # Return metrics
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred
    }
    
    if explained_variance is not None:
        metrics["explained_variance"] = explained_variance
    
    return metrics

def score_model(
    test_df: pd.DataFrame, 
    models_dict: Dict[str, Any], 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Score models on test data and return metrics.
    
    Args:
        test_df: Test DataFrame
        models_dict: Dictionary of trained models
        config: Configuration dictionary
    
    Returns:
        Dictionary of model scores
    """
    logger.info("Scoring models on test data")
    
    # Get configuration values
    metrics_list = config.get("metrics", ["mae", "rmse", "r2"])
    target_column = config.get("target_column", "brightness")
    group5_mode = config.get("group5_mode", False)
    
    # Initialize scores dictionary
    scores = {
        "model_scores": {},
        "default_model": None
    }
    
    # Get target
    y_test = test_df[target_column]
    
    # Score each model
    for model_name, model_info in models_dict.items():
        model = model_info["model"]
        feature_names = model_info["feature_names"]
        is_default = model_info.get("is_default", False)
        
        # Get features
        X_test = test_df[feature_names]
        
        # Evaluate model
        logger.info(f"Evaluating model: {model_name}")
        metrics = evaluate_model(model, X_test, y_test, group5_mode)
        
        # Store metrics in scores dictionary
        scores["model_scores"][model_name] = {
            metric: metrics[metric] for metric in metrics_list if metric in metrics
        }
        
        # Add predictions if requested
        if "predictions" in metrics_list:
            scores["model_scores"][model_name]["predictions"] = metrics["predictions"]
        
        # If this is the default model, store it
        if is_default:
            scores["default_model"] = model_name
            logger.info(f"Model {model_name} is the default model")
    
    # Make sure we have a default model
    if not scores["default_model"] and models_dict:
        scores["default_model"] = next(iter(scores["model_scores"]))
        logger.info(f"No default model specified, using {scores['default_model']}")
    
    return scores

def save_scores(scores: Dict[str, Any], output_path: str) -> None:
    """Save model scores to CSV.
    
    Args:
        scores: Dictionary of model scores
        output_path: Path to save scores CSV
    """
    # Extract model scores
    model_scores = scores["model_scores"]
    default_model = scores["default_model"]
    
    # Convert to DataFrame
    rows = []
    for model_name, metrics in model_scores.items():
        row = {"model": model_name, "is_default": model_name == default_model}
        for metric, value in metrics.items():
            if metric != "predictions":  # Skip predictions array
                row[metric] = value
        rows.append(row)
    
    scores_df = pd.DataFrame(rows)
    
    # Save to CSV
    scores_df.to_csv(output_path, index=False)
    logger.info(f"Scores saved to {output_path}")
