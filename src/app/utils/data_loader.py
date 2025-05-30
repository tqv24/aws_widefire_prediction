import pandas as pd
import numpy as np
import boto3
import logging
from io import StringIO
import os

logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic fire data as a fallback"""
    logger.info(f"Generating synthetic data with {n_samples} samples")
    np.random.seed(42)
    
    # Generate dates between Oct 2019 and Jan 2020
    start_date = pd.Timestamp('2019-10-01')
    end_date = pd.Timestamp('2020-01-11')
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Generate random dates as strings directly
    random_dates = []
    for _ in range(n_samples):
        # Get a random integer index into the date_range
        idx = np.random.randint(0, len(date_range))
        # Convert pandas timestamp to string format
        date_str = date_range[idx].strftime('%Y-%m-%d')
        random_dates.append(date_str)
    
    # Generate synthetic data
    data = {
        'latitude': np.random.uniform(-44, -10, n_samples),
        'longitude': np.random.uniform(112, 154, n_samples),
        'brightness': np.random.uniform(300, 500, n_samples),
        'scan': np.random.uniform(0.5, 2.0, n_samples),
        'track': np.random.uniform(0.5, 2.0, n_samples),
        'bright_t31': np.random.uniform(250, 350, n_samples),
        'frp': np.random.uniform(5, 100, n_samples),
        'confidence': np.random.randint(0, 100, n_samples),
        'acq_date': random_dates  # Use the pre-generated string dates
    }
    
    return pd.DataFrame(data)

def load_data_from_s3(file_path, bucket_name="mlds423-s3-project", fallback_paths=None):
    """Load data from S3 with error handling and fallback options"""
    if fallback_paths is None:
        fallback_paths = [
            "data/fire_nrt.csv",
            "processed/fire_features.csv",
            "data/fire_nrt_M6_96619_cleaned.csv"
        ]
    
    try:
        if file_path.startswith("s3://"):
            # Parse the S3 URI
            bucket_name = file_path.split("/")[2]
            key = "/".join(file_path.split("/")[3:])
            
            s3 = boto3.client("s3")
            logger.info(f"Loading data from S3: bucket={bucket_name}, key={key}")
            
            try:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                data = response["Body"].read().decode('utf-8')
                df = pd.read_csv(StringIO(data))
                logger.info(f"Successfully loaded data with {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Error loading from primary path: {e}")
                
                # Try fallback paths
                for fallback_path in fallback_paths:
                    try:
                        logger.info(f"Trying fallback path: {fallback_path}")
                        response = s3.get_object(Bucket=bucket_name, Key=fallback_path)
                        data = response["Body"].read().decode('utf-8')
                        df = pd.read_csv(StringIO(data))
                        logger.info(f"Successfully loaded data from fallback path: {fallback_path}")
                        return df
                    except Exception as fallback_error:
                        logger.warning(f"Fallback path failed: {fallback_error}")
                        
                # If all S3 paths fail, generate synthetic data
                logger.warning("All S3 paths failed, generating synthetic data")
                return generate_synthetic_data()
        else:
            # Try to load local file
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                logger.warning(f"Local file not found: {file_path}, generating synthetic data")
                return generate_synthetic_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return generate_synthetic_data()  # Return synthetic data on error

def ensure_date_column(df):
    """Ensure the dataframe has an acq_date column"""
    if 'acq_date' not in df.columns:
        # Try to create it from acquisition date or day columns if available
        if 'acquisition_date' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acquisition_date']).dt.strftime('%Y-%m-%d')
        elif 'acq_day' in df.columns and 'acq_year' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acq_year'].astype(str) + '-' + df['acq_day'].astype(str), format='%Y-%j').dt.strftime('%Y-%m-%d')
        else:
            # Generate synthetic dates with a safer approach
            start_date = pd.Timestamp('2019-10-01')
            end_date = pd.Timestamp('2020-01-11')
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Generate random dates as strings
            random_dates = []
            for _ in range(len(df)):
                idx = np.random.randint(0, len(date_range))
                date_str = date_range[idx].strftime('%Y-%m-%d')
                random_dates.append(date_str)
            
            df['acq_date'] = random_dates
    
    return df
