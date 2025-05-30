import boto3
import pickle
import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from .data_loader import generate_synthetic_data

logger = logging.getLogger(__name__)

def normalize_path(path):
    """Convert path separators to forward slashes for S3 compatibility"""
    # Make sure to normalize both types of slashes to forward slash
    return path.replace('\\', '/').replace('//', '/')

def upload_model_to_s3(model, model_name, bucket, prefix="experiments"):
    """Upload a trained model to S3"""
    try:
        # Create a local file first
        os.makedirs("models", exist_ok=True)
        local_path = os.path.join("models", model_name)
        
        with open(local_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Upload to S3
        s3 = boto3.client('s3')
        
        # Only upload to experiments directory for pipeline compatibility
        experiment_path = normalize_path(f"{prefix}/latest/models/{model_name}")
        logger.info(f"Uploading model to s3://{bucket}/{experiment_path}")
        s3.upload_file(local_path, bucket, experiment_path)
        
        logger.info(f"Successfully uploaded model to S3")
        return True
    except Exception as e:
        logger.error(f"Error uploading model to S3: {e}")
        return False

def download_model_from_s3(model_key, local_path, bucket, prefix="experiments", fallback_experiments=None, selected_experiment="1748638871"):
    """Download a model from S3 to local path with error handling"""
    if fallback_experiments is None:
        # Use known experiment IDs that work - removed "latest" since it's not available
        fallback_experiments = ["1748573816", "1748576650", "1748638871"]
    
    # Ensure local directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
    if not os.path.exists(local_path):
        try:
            # Configure S3 client - removed invalid parameter
            s3 = boto3.client("s3")
            
            # Build potential paths, starting with the selected experiment
            potential_paths = []
            
            # Normalize model_key to ensure forward slashes
            model_key_normalized = normalize_path(model_key)
            
            # Try first with backslash for the last separator (what seems to work based on logs)
            key_parts = model_key_normalized.split('/')
            model_filename = key_parts[-1]
            
            # Create both forward and backslash versions of the path
            forward_slash_path = f"{normalize_path(prefix)}/{selected_experiment}/models/{model_filename}"
            backslash_path = f"{normalize_path(prefix)}/{selected_experiment}/models\\{model_filename}"
            
            # Try both path formats
            potential_paths.append(forward_slash_path)
            potential_paths.append(backslash_path)
            
            # Then try fallback experiments if selected isn't already in fallbacks
            if selected_experiment not in fallback_experiments:
                for exp in fallback_experiments:
                    potential_paths.append(f"{normalize_path(prefix)}/{exp}/models/{model_filename}")
                    potential_paths.append(f"{normalize_path(prefix)}/{exp}/models\\{model_filename}")
            
            # Try to list objects to find the model
            logger.info(f"Searching for model {model_key} in S3 bucket {bucket}, experiment {selected_experiment}")
            
            # Try direct paths first as they're faster
            model_found = False
            for path in potential_paths:
                try:
                    logger.info(f"Trying to download from s3://{bucket}/{path}")
                    s3.download_file(bucket, path, local_path)
                    logger.info(f"Successfully downloaded model from {path}")
                    model_found = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to download from {path}: {e}")
            
            # If direct paths failed, use a more flexible search approach
            if not model_found:
                try:
                    # Look for any models in the experiments directory
                    prefix_path = normalize_path(f"{prefix}/")
                    logger.info(f"Listing objects in s3://{bucket}/{prefix_path}")
                    
                    response = s3.list_objects_v2(
                        Bucket=bucket,
                        Prefix=prefix_path
                    )
                    
                    if 'Contents' in response:
                        # Sort by LastModified to get the most recent first
                        objects = sorted(response['Contents'], 
                                        key=lambda x: x.get('LastModified', ''), 
                                        reverse=True)
                        
                        # Check for model keys containing the model filename (regardless of path separator)
                        for obj in objects:
                            s3_key = obj['Key']
                            if model_filename in s3_key and "models" in s3_key:
                                # Use the exact key from S3 as is (no normalization)
                                logger.info(f"Found model at s3://{bucket}/{s3_key}")
                                s3.download_file(bucket, s3_key, local_path)
                                logger.info(f"Successfully downloaded model to {local_path}")
                                model_found = True
                                break
                except Exception as e:
                    logger.warning(f"Error searching in experiments: {e}")
            
            if not model_found:
                # Last resort - create a dummy model
                logger.warning("Creating dummy model for demonstration purposes")
                
                if "decision_tree" in model_key.lower():
                    model = DecisionTreeRegressor(random_state=42)
                elif "random_forest" in model_key.lower():
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = LinearRegression()
                
                # Train on synthetic data
                synth_data = generate_synthetic_data(500)
                X = synth_data[['latitude', 'longitude', 'scan', 'track', 'bright_t31', 'confidence', 'frp']]
                y = synth_data['brightness']
                model.fit(X, y)
                
                # Save model locally
                with open(local_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"Created and saved dummy model to {local_path}")
                st.warning("⚠️ Using a dummy model for demonstration - predictions may not be accurate")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            st.error(f"Failed to download model: {e}")
            return None
    return local_path

def load_model(local_path):
    """Load a ML model from local path with error handling"""
    try:
        with open(local_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {local_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {e}")
        return None

def predict_fire(model, input_data):
    """Make predictions with error handling"""
    # Suppress all feature name related warnings globally
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")
        
        try:
            if model is None:
                raise ValueError("Model is not loaded")
            
            # Define the expected feature order that matches the model's training
            expected_features = ['latitude', 'longitude', 'scan', 'track', 'bright_t31', 'confidence', 'frp']
            
            # Verify all required features are present
            missing_features = [feat for feat in expected_features if feat not in input_data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Create a copy of the input data with features in the correct order
            input_ordered = input_data[expected_features].copy()
            
            # Log the feature order to help with debugging
            logger.info(f"Making prediction with features in order: {list(input_ordered.columns)}")
            
            try:
                # First attempt: Try prediction with ordered features
                predictions = model.predict(input_ordered)
            except Exception as feature_error:
                logger.info("Ordered DataFrame approach failed, trying numpy array...")
                
                # Second attempt: Try converting to numpy array to bypass feature name checks
                X_array = input_ordered.values
                logger.info(f"Making prediction with numpy array of shape {X_array.shape}")
                
                try:
                    predictions = model.predict(X_array)
                    logger.info("Successfully made prediction with numpy array")
                except Exception as array_error:
                    logger.warning(f"Error with numpy array: {array_error}")
                    
                    # Third attempt: Try manually overriding the model's feature names
                    if hasattr(model, 'feature_names_in_'):
                        logger.info(f"Model expects features: {model.feature_names_in_}")
                        # Create dataframe with exactly matching feature names
                        renamed_data = pd.DataFrame(
                            input_ordered.values,
                            columns=model.feature_names_in_
                        )
                        predictions = model.predict(renamed_data)
                        logger.info("Successfully made prediction with renamed features")
                    else:
                        # Last resort
                        raise ValueError("Cannot match feature names")
            
            logger.info(f"Made prediction: {predictions[0]}")
            return predictions
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            st.error(f"Failed to make prediction: {e}")
            return None