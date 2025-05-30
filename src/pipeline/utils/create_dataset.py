import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def create_dataset(data_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Create a structured dataset from raw fire data

    Args:
        data_path: Path to the raw data file
        config: Configuration for creating dataset

    Returns:
        Structured pandas DataFrame with fire prediction data
    """
    logger.info("Creating dataset from %s", data_path)
    
    try:
        # Read data from CSV file
        df = pd.read_csv(data_path)
        logger.info("Loaded data with %d rows and %d columns", df.shape[0], df.shape[1])
    except Exception as e:
        logger.error("Error reading data file: %s", e)
        raise
    
    # Get critical columns from config
    critical_columns = config.get("critical_columns", [])
    if critical_columns:
        # Drop rows with missing values in critical columns
        df = df.dropna(subset=critical_columns)
        logger.info("Dropped rows with missing values in critical columns. %d rows remaining", df.shape[0])
    
    # Convert date columns to datetime if they exist
    if 'acq_date' in df.columns:
        df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
        logger.info("Converted 'acq_date' column to datetime")
    
    # Apply region filter if specified
    region_filter = config.get("region_filter", {})
    if region_filter.get("enabled", False):
        # Filter by region (Australia)
        min_lat = region_filter.get("min_latitude", -44)
        max_lat = region_filter.get("max_latitude", -10)
        min_lon = region_filter.get("min_longitude", 112)
        max_lon = region_filter.get("max_longitude", 154)
        
        logger.info("Filtering data to region: lat [%f, %f], lon [%f, %f]", min_lat, max_lat, min_lon, max_lon)
        df = df[(df['latitude'] >= min_lat) & 
                (df['latitude'] <= max_lat) & 
                (df['longitude'] >= min_lon) & 
                (df['longitude'] <= max_lon)]
        logger.info("After region filtering: %d rows remaining", df.shape[0])
    
    # Remove outliers if specified
    outlier_config = config.get("outlier_removal", {})
    if outlier_config.get("enabled", False):
        column = outlier_config.get("column")
        std_threshold = outlier_config.get("std_threshold", 3)
        
        if column in df.columns:
            logger.info("Removing outliers in '%s' column (threshold: %f std)", column, std_threshold)
            mean = df[column].mean()
            std = df[column].std()
            df = df[(df[column] > mean - std_threshold*std) & 
                    (df[column] < mean + std_threshold*std)]
            logger.info("After outlier removal: %d rows remaining", df.shape[0])
    
    # Ensure target column exists
    target_column = config.get("target_column")
    if target_column and target_column not in df.columns:
        logger.error("Target column '%s' not found in data", target_column)
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    logger.info("Created dataset with %d rows and %d columns", df.shape[0], df.shape[1])
    return df


def save_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """Save dataset to disk

    Args:
        data: The dataset to save
        output_path: Path where dataset should be saved
    """
    logger.info("Saving dataset to %s", output_path)
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save to CSV
    data.to_csv(output_path, index=False)
    logger.info("Dataset successfully saved with %d rows and %d columns", 
                data.shape[0], data.shape[1])

def clean_fire_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the fire data
    
    Args:
        data: Raw fire data DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning fire data")
    
    # Create a copy to avoid modifying the original
    df_clean = data.copy()
    
    # Drop rows with missing values in critical columns
    critical_cols = ['latitude', 'longitude', 'brightness', 'scan', 'track']
    df_clean = df_clean.dropna(subset=critical_cols)
    logger.info(f"After dropping missing values: {len(df_clean)} rows")
    
    # Convert date columns to datetime if they exist
    if 'acq_date' in df_clean.columns:
        df_clean['acq_date'] = pd.to_datetime(df_clean['acq_date'], errors='coerce')
    
    # Filter by region (Australia) if coordinates are available
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        # Australia's approximate bounding box
        df_clean = df_clean[(df_clean['latitude'] >= -44) & 
                          (df_clean['latitude'] <= -10) & 
                          (df_clean['longitude'] >= 112) & 
                          (df_clean['longitude'] <= 154)]
        logger.info(f"After geographic filtering: {len(df_clean)} rows")
    
    # Remove outliers in brightness (if outside 3 std deviations)
    if 'brightness' in df_clean.columns:
        mean = df_clean['brightness'].mean()
        std = df_clean['brightness'].std()
        df_clean = df_clean[(df_clean['brightness'] > mean - 3*std) & 
                          (df_clean['brightness'] < mean + 3*std)]
        logger.info(f"After outlier removal: {len(df_clean)} rows")
    
    logger.info(f"Data cleaning complete. Rows remaining: {len(df_clean)}")
    return df_clean