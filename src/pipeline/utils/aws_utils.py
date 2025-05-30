"""Utilities for interacting with AWS S3."""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import urllib.parse
import glob

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def ensure_bucket_exists(bucket_name: str, region: str = None, create_if_missing: bool = True) -> bool:
    """Ensure the S3 bucket exists, creating it if necessary
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region to create bucket in if it doesn't exist
        create_if_missing: Whether to create the bucket if it doesn't exist
        
    Returns:
        True if the bucket exists or was created, False otherwise
    """
    logger.info(f"Checking if bucket '{bucket_name}' exists")
    
    # Use specified region or get from environment
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    
    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)
    
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' exists")
        return True
    except ClientError as e:
        error_code = int(e.response.get("Error", {}).get("Code", "0"))
        if error_code == 404 and create_if_missing:
            logger.info(f"Bucket '{bucket_name}' doesn't exist, creating it in region '{region}'")
            try:
                if region == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": region}
                    )
                logger.info(f"Successfully created bucket '{bucket_name}'")
                return True
            except ClientError as create_error:
                logger.error(f"Failed to create bucket: {create_error}")
                return False
        else:
            logger.error(f"Error checking bucket existence: {e}")
            return False


def ensure_s3_url_exists(s3_url: str, config: Dict[str, Any]) -> bool:
    """Ensure the S3 URL's bucket exists, creating it if necessary
    
    Args:
        s3_url: S3 URL (s3://bucket/key)
        config: AWS configuration dictionary
        
    Returns:
        True if the bucket exists or was created, False otherwise
    """
    if not s3_url.startswith("s3://"):
        logger.warning(f"URL '{s3_url}' is not an S3 URL")
        return False
    
    # Parse the URL to get the bucket name
    parsed = urllib.parse.urlparse(s3_url)
    bucket_name = parsed.netloc
    
    # Get configuration parameters
    region = config.get("region") or os.environ.get("AWS_REGION", "us-west-2")
    create_if_missing = config.get("create_bucket_if_missing", True)
    
    return ensure_bucket_exists(bucket_name, region, create_if_missing)


def upload_artifacts(artifacts: Path, config: Dict[str, Any]) -> List[str]:
    """Upload all the artifacts in the specified directory to S3

    Args:
        artifacts: Directory containing all the artifacts from a given experiment
        config: Config required to upload artifacts to S3; see example config file for structure

    Returns:
        List of S3 uri's for each file that was uploaded
    """
    logger.info("Starting upload of artifacts to S3")
    
    # Extract configuration parameters
    bucket_name = os.environ.get("AWS_S3_BUCKET") or config.get("bucket_name")
    prefix = config.get("prefix", "experiments")
    region = os.environ.get("AWS_REGION") or config.get("region", "us-west-2")
    create_if_missing = config.get("create_bucket_if_missing", True)  # Default to True
    
    if not bucket_name:
        raise ValueError("Bucket name not specified in AWS configuration")
    
    # Ensure the bucket exists before trying to upload
    if not ensure_bucket_exists(bucket_name, region, create_if_missing):
        raise ValueError(f"Bucket '{bucket_name}' does not exist and could not be created")
    
    # Create boto3 client
    try:
        s3_client = boto3.client("s3", region_name=region)
    except Exception as e:
        logger.error("Failed to create S3 client: %s", e)
        raise
    
    # Get list of all files in artifacts directory (recursively)
    artifact_files = list(artifacts.glob("**/*"))
    artifact_files = [f for f in artifact_files if f.is_file()]
    
    # Upload each file to S3
    uploaded_uris = []
    for file_path in artifact_files:
        # Create S3 key by joining prefix with relative path from artifacts directory
        relative_path = file_path.relative_to(artifacts)
        s3_key = f"{prefix}/{artifacts.name}/{relative_path}"
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info("Uploading %s to %s", file_path, s3_uri)
        try:
            s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket_name,
                Key=s3_key
            )
            uploaded_uris.append(s3_uri)
            logger.info("Successfully uploaded %s", s3_uri)
        except ClientError as e:
            logger.error("Error uploading %s: %s", file_path, e)
    
    logger.info("Completed upload of %d files to S3", len(uploaded_uris))
    return uploaded_uris


def upload_file_to_s3(file_path: Path, bucket_name: str, s3_key: str, region: str = None) -> bool:
    """Upload a single file to S3
    
    Args:
        file_path: Path to the local file
        bucket_name: S3 bucket name
        s3_key: S3 object key (path in the bucket)
        region: AWS region
        
    Returns:
        True if upload was successful, False otherwise
    """
    logger.info(f"Uploading file {file_path} to s3://{bucket_name}/{s3_key}")
    
    # Use specified region or get from environment
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    
    # Ensure file exists
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist")
        return False
    
    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)
    
    try:
        # Upload the file
        s3_client.upload_file(
            Filename=str(file_path),
            Bucket=bucket_name,
            Key=s3_key
        )
        logger.info(f"Successfully uploaded file to s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False


def download_from_s3(bucket_name: str, s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to a local path
    
    Args:
        bucket_name: S3 bucket name
        s3_key: Key (path) of the file in S3
        local_path: Local path to save the file
        
    Returns:
        True if download was successful, False otherwise
    """
    logger.info(f"Downloading {s3_key} from bucket {bucket_name} to {local_path}")
    
    # Ensure directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create S3 client
    s3_client = boto3.client("s3")
    
    try:
        # Download the file
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Successfully downloaded file to {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        return False


def upload_model_to_s3(model_path: Path, s3_filename: str, bucket_name: str = None, region: str = None) -> bool:
    """Upload a model file to S3
    
    Args:
        model_path: Path to the local model file
        s3_filename: Filename to use in S3
        bucket_name: S3 bucket name (optional, uses environment variable if not provided)
        region: AWS region (optional, uses environment variable if not provided)
        
    Returns:
        True if upload was successful, False otherwise
    """
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return False
        
    # S3 configuration
    bucket = bucket_name or os.environ.get("AWS_S3_BUCKET")
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    
    if not bucket:
        logger.error("Bucket name not provided and AWS_S3_BUCKET environment variable not set")
        return False
        
    s3_key = f'models/{s3_filename}'
    
    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)
    
    try:
        # Upload the model file
        s3_client.upload_file(str(model_path), bucket, s3_key)
        logger.info(f"Successfully uploaded model to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading model to S3: {e}")
        return False


def download_kaggle_dataset(dataset_name: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Download a dataset from Kaggle
    
    Args:
        dataset_name: Name of the Kaggle dataset (format: "username/dataset-name")
        output_dir: Directory to save the dataset (optional)
        
    Returns:
        Path to the downloaded dataset, or None if failed
    """
    try:
        import kagglehub
        
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Download the dataset
        dataset_path = kagglehub.dataset_download(dataset_name, path=output_dir)
        
        logger.info(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except ImportError:
        logger.error("kagglehub package not installed. Please install with 'pip install kagglehub'")
        return None
    except Exception as e:
        logger.error(f"Error downloading dataset from Kaggle: {e}")
        logger.info("If you encounter authentication issues, make sure your Kaggle API credentials are set up.")
        logger.info(f"You can download the dataset manually from: https://www.kaggle.com/datasets/{dataset_name}")
        return None

def upload_kaggle_dataset_to_s3(local_file_path: Union[str, Path], 
                              bucket_name: str, 
                              s3_key: str, 
                              region: Optional[str] = None) -> bool:
    """Upload a Kaggle dataset file to S3
    
    Args:
        local_file_path: Path to the downloaded Kaggle dataset file
        bucket_name: S3 bucket name
        s3_key: Key (path) to use in S3
        region: AWS region (optional)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Uploading Kaggle dataset {local_file_path} to s3://{bucket_name}/{s3_key}")
    
    # Use specified region or get from environment
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    
    # Ensure the file exists
    if not os.path.exists(local_file_path):
        logger.error(f"File not found: {local_file_path}")
        return False
    
    # Ensure the bucket exists
    if not ensure_bucket_exists(bucket_name, region, True):
        logger.error(f"Failed to ensure bucket {bucket_name} exists")
        return False
    
    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)
    
    try:
        # Upload the file
        s3_client.upload_file(str(local_file_path), bucket_name, s3_key)
        logger.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def get_kaggle_data(config: Dict[str, Any]) -> Optional[str]:
    """Download Kaggle dataset and optionally upload to S3 based on configuration
    
    Args:
        config: Configuration dictionary with data acquisition settings
        
    Returns:
        Path to the file used, or None if failed
    """
    # Extract configuration
    data_acq_config = config.get("data_acquisition", {})
    if data_acq_config.get("source") != "kaggle":
        logger.info("Data source is not Kaggle, skipping Kaggle data acquisition")
        return None
    
    # Get Kaggle dataset details
    kaggle_dataset = data_acq_config.get("kaggle_dataset")
    if not kaggle_dataset:
        logger.error("Kaggle dataset name not specified in configuration")
        return None
    
    # Local data directory
    local_data_dir = data_acq_config.get("local_data_dir", "data/raw")
    os.makedirs(local_data_dir, exist_ok=True)
    
    # Download dataset
    dataset_path = download_kaggle_dataset(kaggle_dataset, local_data_dir)
    if not dataset_path:
        return None
    
    # Find the target file using the pattern
    file_pattern = data_acq_config.get("dataset_file_pattern", "*.csv")
    csv_files = glob.glob(os.path.join(dataset_path, file_pattern))
    
    if not csv_files:
        logger.error(f"No files matching pattern '{file_pattern}' found in downloaded dataset")
        return None
    
    # Use the first matching file
    fire_data_file = csv_files[0]
    logger.info(f"Using fire data file: {fire_data_file}")
    
    # Upload to S3 if configured
    if data_acq_config.get("upload_to_s3", False):
        # Get AWS configuration
        aws_config = config.get("aws", {})
        bucket_name = aws_config.get("bucket_name") or os.environ.get("AWS_S3_BUCKET")
        region = aws_config.get("region") or os.environ.get("AWS_REGION", "us-west-2")
        
        if not bucket_name:
            logger.error("S3 bucket name not specified in configuration")
            return fire_data_file
        
        # Determine S3 key from the data_source in run_config
        run_config = config.get("run_config", {})
        data_source = run_config.get("data_source", "")
        
        if data_source.startswith("s3://"):
            s3_url = urllib.parse.urlparse(data_source)
            s3_key = s3_url.path.lstrip('/')
        else:
            # Default key if data_source not specified
            s3_key = f"data/{os.path.basename(fire_data_file)}"
        
        # Upload the file
        upload_success = upload_kaggle_dataset_to_s3(
            fire_data_file, 
            bucket_name, 
            s3_key, 
            region
        )
        
        if upload_success:
            logger.info(f"Kaggle data successfully uploaded to S3: s3://{bucket_name}/{s3_key}")
        else:
            logger.warning(f"Failed to upload Kaggle data to S3, but local file is available")
    
    return fire_data_file

def create_sample_data(s3_url: str, config: Dict[str, Any]) -> bool:
    """Create and upload sample fire data to S3 if it doesn't exist
    
    Args:
        s3_url: S3 URL where sample data should be stored (s3://bucket/key)
        config: AWS configuration dictionary
        
    Returns:
        True if data creation and upload was successful, False otherwise
    """
    import pandas as pd
    import numpy as np
    import tempfile
    
    if not s3_url.startswith("s3://"):
        logger.warning(f"URL '{s3_url}' is not an S3 URL")
        return False
    
    # Parse the URL to get the bucket name and key
    parsed = urllib.parse.urlparse(s3_url)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip('/')
    
    # Get configuration parameters
    region = config.get("region") or os.environ.get("AWS_REGION", "us-west-2")
    
    # Ensure bucket exists
    if not ensure_bucket_exists(bucket_name, region, True):
        logger.error(f"Bucket '{bucket_name}' does not exist and could not be created")
        return False
    
    # Create S3 client
    s3_client = boto3.client("s3", region_name=region)
    
    # Check if the data already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        logger.info(f"Sample data already exists at s3://{bucket_name}/{key}")
        return True
    except ClientError as e:
        error_code = int(e.response.get("Error", {}).get("Code", "0"))
        if error_code != 404:
            logger.error(f"Error checking if sample data exists: {e}")
            return False
    
    # Data doesn't exist, create sample data
    logger.info(f"Creating sample fire data and uploading to s3://{bucket_name}/{key}")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    sample_data = pd.DataFrame({
        'latitude': np.random.uniform(-44, -10, n_samples),
        'longitude': np.random.uniform(112, 154, n_samples),
        'brightness': np.random.uniform(300, 500, n_samples),
        'scan': np.random.uniform(0.5, 2.0, n_samples),
        'track': np.random.uniform(0.5, 2.0, n_samples),
        'bright_t31': np.random.uniform(250, 350, n_samples),
        'frp': np.random.uniform(5, 100, n_samples),
        'confidence': np.random.randint(0, 100, n_samples)
    })
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        temp_path = tmp.name
    
    # Save to the temporary file
    sample_data.to_csv(temp_path, index=False)
    
    # Upload to S3
    try:
        s3_client.upload_file(temp_path, bucket_name, key)
        logger.info(f"Successfully uploaded sample data to s3://{bucket_name}/{key}")
        
        # Clean up temporary file
        os.unlink(temp_path)
        return True
    except ClientError as e:
        logger.error(f"Error uploading sample data to S3: {e}")
        # Clean up temporary file
        os.unlink(temp_path)
        return False
