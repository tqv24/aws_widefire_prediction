"""Functions for generating features from cleaned fire data."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_fire_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Generate features for fire prediction model.
    
    Args:
        df: Cleaned fire data DataFrame
        config: Configuration dictionary with feature settings
    
    Returns:
        DataFrame with generated features
    """
    logger.info("Generating features for fire prediction model")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Get configuration values
    feature_columns = config.get("feature_columns", [])
    derived_features = config.get("derived_features", [])
    
    # If the Group5 mode is enabled, only use the basic features
    if config.get("group5_mode", False):
        logger.info("Using Group5 feature mode - only base features")
        # Match exactly the features used in the Group5 notebook
        feature_list = ['latitude', 'longitude', 'scan', 'track', 'bright_t31', 'frp', 'confidence']
        
        # Select only these features plus the target
        target_column = config.get("target_column", "brightness")
        columns_to_keep = feature_list + [target_column]
        
        # Only keep the necessary columns
        df_features = df_features[columns_to_keep]
        
        logger.info(f"Using Group5 feature set with {len(feature_list)} features")
        return df_features
    
    # Otherwise, continue with the normal derived features
    logger.info("Adding derived features")
    
    # Calculate derived features
    if "frp_per_area" in derived_features:
        # Calculate approximate pixel area from scan and track
        df_features["frp_per_area"] = df_features["frp"] / (df_features["scan"] * df_features["track"])
        logger.info("Added frp_per_area feature")
    
    if "temperature_diff" in derived_features:
        # Calculate difference between brightness and bright_t31
        df_features["temperature_diff"] = df_features["brightness"] - df_features["bright_t31"]
        logger.info("Added temperature_diff feature")
    
    # Add any transformations from the config
    transformations = config.get("transformations", {})
    
    # Log transform
    if "log_transform" in transformations:
        log_features = transformations.get("log_transform", {})
        for new_col, source_col in log_features.items():
            if source_col in df_features.columns:
                # Add small constant to avoid log(0)
                df_features[new_col] = np.log1p(df_features[source_col])
                logger.info(f"Added log transform feature: {new_col}")
    
    # Normalize features
    normalize_config = transformations.get("normalize", {})
    if normalize_config.get("enabled", False):
        columns_to_normalize = normalize_config.get("columns", [])
        method = normalize_config.get("method", "min_max")
        
        for col in columns_to_normalize:
            if col in df_features.columns:
                if method == "min_max":
                    min_val = df_features[col].min()
                    max_val = df_features[col].max()
                    df_features[f"{col}_norm"] = (df_features[col] - min_val) / (max_val - min_val)
                elif method == "standard":
                    mean_val = df_features[col].mean()
                    std_val = df_features[col].std()
                    df_features[f"{col}_norm"] = (df_features[col] - mean_val) / std_val
                
                logger.info(f"Normalized feature: {col} using {method} method")
    
    logger.info(f"Feature generation complete. Total features: {len(df_features.columns)}")
    return df_features

def save_features(df: pd.DataFrame, output_path: str) -> None:
    """Save generated features to a CSV file.
    
    Args:
        df: DataFrame with generated features
        output_path: Path to save the features CSV
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Features saved to {output_path}")
