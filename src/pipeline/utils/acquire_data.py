"""Module for acquiring data from various sources."""
import logging
import os
import glob
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import urllib.parse

import pandas as pd

logger = logging.getLogger(__name__)

def acquire_data(config: Dict[str, Any]) -> Optional[Union[str, Path]]:
    """Acquire data from the specified source based on configuration
    
    Args:
        config: Configuration dictionary with data acquisition settings
        
    Returns:
        Path to the local data file, or None if acquisition failed
    """
    logger.info("Starting data acquisition process")
    
    # First try: Check if a local sample file exists in the data directory
    try:
        local_data_dir = Path("data")
        if local_data_dir.exists():
            sample_file = get_sample_data(local_data_dir)
            if sample_file:
                logger.info(f"Using existing local sample data: {sample_file}")
                return sample_file
    except Exception as e:
        logger.warning(f"Error checking for local sample data: {e}")
    
    # Second try: Kaggle data source if configured
    if "data_acquisition" in config and config["data_acquisition"].get("source") == "kaggle":
        try:
            logger.info("Attempting to acquire data from Kaggle")
            kaggle_file = get_kaggle_data(config)
            if kaggle_file and os.path.exists(kaggle_file):
                logger.info(f"Successfully acquired data from Kaggle: {kaggle_file}")
                return kaggle_file
            else:
                logger.warning("Failed to acquire data from Kaggle")
        except Exception as e:
            logger.warning(f"Error acquiring data from Kaggle: {e}")
    
    # Third try: S3 data source from run_config
    run_config = config.get("run_config", {})
    data_source = run_config.get("data_source", "")
    
    if data_source.startswith("s3://"):
        try:
            logger.info(f"Attempting to acquire data from S3: {data_source}")
            s3_file = get_s3_data(data_source, config)
            if s3_file and os.path.exists(s3_file):
                logger.info(f"Successfully acquired data from S3: {s3_file}")
                return s3_file
            else:
                logger.warning(f"Failed to acquire data from S3: {data_source}")
        except Exception as e:
            logger.warning(f"Error acquiring data from S3: {e}")
    
    # Fourth try: Create and use synthetic data as a last resort
    try:
        logger.info("Attempting to create synthetic data as fallback")
        synthetic_file = create_synthetic_data()
        if synthetic_file and os.path.exists(synthetic_file):
            logger.info(f"Created synthetic data as fallback: {synthetic_file}")
            return synthetic_file
    except Exception as e:
        logger.warning(f"Error creating synthetic data: {e}")
    
    # If all attempts fail, log a detailed error and return None
    logger.error("All data acquisition methods failed.")
    logger.error(f"Checked for Kaggle data: {'data_acquisition' in config}")
    logger.error(f"Checked for S3 data: {data_source}")
    return None


def get_sample_data(data_dir: Path) -> Optional[str]:
    """Try to find sample data in the data directory
    
    Args:
        data_dir: Directory to search for sample data
        
    Returns:
        Path to sample data file if found, None otherwise
    """
    logger.info(f"Looking for sample data in {data_dir}")
    
    # Look for any CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        sample_file = str(csv_files[0])
        logger.info(f"Found sample data file: {sample_file}")
        return sample_file
    
    # Look in subdirectories
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            csv_files = list(subdir.glob("*.csv"))
            if csv_files:
                sample_file = str(csv_files[0])
                logger.info(f"Found sample data file in subdirectory: {sample_file}")
                return sample_file
    
    logger.warning("No sample data files found")
    return None


def get_kaggle_data(config: Dict[str, Any]) -> Optional[str]:
    """Download Kaggle dataset and optionally upload to S3
    
    Args:
        config: Configuration dictionary with data acquisition settings
        
    Returns:
        Path to the file used, or None if failed
    """
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub package not installed. Please install with 'pip install kagglehub'")
        return None
    
    # Extract configuration
    data_acq_config = config.get("data_acquisition", {})
    
    # Get Kaggle dataset details
    kaggle_dataset = data_acq_config.get("kaggle_dataset")
    if not kaggle_dataset:
        logger.error("Kaggle dataset name not specified in configuration")
        return None
    
    # Local data directory
    local_data_dir = data_acq_config.get("local_data_dir", "data/raw")
    os.makedirs(local_data_dir, exist_ok=True)
    
    # Download dataset
    logger.info(f"Downloading Kaggle dataset: {kaggle_dataset}")
    try:
        dataset_path = kagglehub.dataset_download(kaggle_dataset, path=local_data_dir)
        logger.info(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        logger.error(f"Error downloading dataset from Kaggle: {e}")
        logger.info("If you encounter authentication issues, make sure your Kaggle API credentials are set up.")
        return None
    
    # Find the target file using the pattern
    file_pattern = data_acq_config.get("dataset_file_pattern", "*.csv")
    csv_files = glob.glob(os.path.join(dataset_path, file_pattern))
    
    if not csv_files:
        logger.error(f"No files matching pattern '{file_pattern}' found in downloaded dataset")
        # Try to find any CSV file
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        if not csv_files:
            logger.error("No CSV files found in the dataset")
            return None
    
    # Use the first matching file or a specific one
    fire_data_file = None
    for file in csv_files:
        if 'fire_nrt' in os.path.basename(file).lower():
            fire_data_file = file
            break
    
    if not fire_data_file and csv_files:
        fire_data_file = csv_files[0]
        
    if not fire_data_file:
        logger.error("No suitable CSV file found in the downloaded dataset")
        return None
    
    logger.info(f"Using fire data file: {fire_data_file}")
    
    # Make a copy in the data directory for easier access
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    target_file = data_dir / os.path.basename(fire_data_file)
    shutil.copy(fire_data_file, target_file)
    logger.info(f"Copied fire data to {target_file}")
    
    return str(target_file)


def get_s3_data(data_source: str, config: Dict[str, Any]) -> Optional[str]:
    """Download data from S3
    
    Args:
        data_source: S3 URL (s3://bucket/key)
        config: Configuration dictionary
        
    Returns:
        Path to the local data file, or None if download failed
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error("boto3 package not installed. Please install with 'pip install boto3'")
        return None
    
    if not data_source.startswith("s3://"):
        logger.error(f"Invalid S3 URL: {data_source}")
        return None
    
    # Parse the S3 URL
    parsed_url = urllib.parse.urlparse(data_source)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    # Get AWS region from config
    aws_config = config.get("aws", {})
    region = aws_config.get("region") or os.environ.get("AWS_REGION", "us-west-2")
    
    # Create S3 client
    try:
        s3_client = boto3.client("s3", region_name=region)
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        return None
    
    # Define local path
    file_name = os.path.basename(key)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    local_path = data_dir / file_name
    
    # Download from S3
    try:
        logger.info(f"Downloading {data_source} to {local_path}")
        s3_client.download_file(bucket, key, str(local_path))
        logger.info(f"Successfully downloaded data to {local_path}")
        return str(local_path)
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return None


def create_synthetic_data() -> Optional[str]:
    """Create synthetic fire data as a fallback
    
    Returns:
        Path to the synthetic data file, or None if creation failed
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        logger.error("pandas or numpy not installed. Please install with 'pip install pandas numpy'")
        return None
    
    logger.info("Creating synthetic fire data")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 1000
    data = {
        'latitude': np.random.uniform(-44, -10, n_samples),
        'longitude': np.random.uniform(112, 154, n_samples),
        'brightness': np.random.uniform(300, 500, n_samples),
        'scan': np.random.uniform(0.5, 2.0, n_samples),
        'track': np.random.uniform(0.5, 2.0, n_samples),
        'acq_date': ['2022-01-01'] * n_samples,
        'acq_time': np.random.randint(0, 2400, n_samples),
        'satellite': ['Terra'] * n_samples,
        'instrument': ['MODIS'] * n_samples,
        'confidence': np.random.randint(0, 100, n_samples),
        'version': ['1.0'] * n_samples,
        'bright_t31': np.random.uniform(250, 350, n_samples),
        'frp': np.random.uniform(5, 100, n_samples),
        'daynight': np.random.choice(['D', 'N'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "synthetic_fire_data.csv"
    
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic data created and saved to {file_path}")
    
    return str(file_path)