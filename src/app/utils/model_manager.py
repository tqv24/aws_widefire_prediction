import boto3
import pickle
import os
import logging
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from .data_loader import generate_synthetic_data

logger = logging.getLogger(__name__)

def upload_model_to_s3(model, model_name, bucket, prefix="experiments"):
    """Upload a trained model to S3"""
    try:
        # Create a local file first
        os.makedirs("models", exist_ok=True)
        local_path = os.path.join("models", model_name)
        
        with open(local_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Upload to S3 - try both paths to ensure compatibility
        s3 = boto3.client('s3')
        
        # Upload to root models directory
        logger.info(f"Uploading model {model_name} to s3://{bucket}/models/{model_name}")
        s3.upload_file(local_path, bucket, f"models/{model_name}")
        
        # Also upload to experiments directory for pipeline compatibility
        experiment_path = f"{prefix}/latest/models/{model_name}"
        logger.info(f"Also uploading model to s3://{bucket}/{experiment_path}")
        try:
            s3.upload_file(local_path, bucket, experiment_path)
        except Exception as e:
            logger.warning(f"Could not upload to experiments path: {e}")
        
        logger.info(f"Successfully uploaded model to S3")
        return True
    except Exception as e:
        logger.error(f"Error uploading model to S3: {e}")
        return False

def download_model_from_s3(model_key, local_path, bucket, prefix="experiments", fallback_experiments=None):
    """Download a model from S3 to local path with error handling"""
    if fallback_experiments is None:
        fallback_experiments = ["1748576650", "1748573816"]
        
    if not os.path.exists(local_path):
        try:
            s3 = boto3.client("s3")
            # Try several possible locations for the model in order of likelihood
            potential_paths = [
                f"models/{model_key}",  # Direct in models folder
                f"{prefix}/latest/models/{model_key}",  # In latest experiment
            ]
            
            # Add fallback experiment paths
            for exp in fallback_experiments:
                potential_paths.append(f"experiments/{exp}/models/{model_key}")
            
            # Try to list objects to find the model
            logger.info(f"Searching for model {model_key} in S3 bucket {bucket}")
            
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
            
            # If direct paths failed, try searching through objects
            if not model_found:
                try:
                    response = s3.list_objects_v2(
                        Bucket=bucket,
                        Prefix="models/"
                    )
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            if model_key in obj['Key']:
                                logger.info(f"Found model at s3://{bucket}/{obj['Key']}")
                                s3.download_file(bucket, obj['Key'], local_path)
                                logger.info(f"Successfully downloaded model to {local_path}")
                                model_found = True
                                break
                except Exception as e:
                    logger.warning(f"Error searching for model: {e}")
            
            # If still not found, try searching in experiments
            if not model_found:
                try:
                    response = s3.list_objects_v2(
                        Bucket=bucket,
                        Prefix=f"{prefix}/"
                    )
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            if model_key in obj['Key'] and "models" in obj['Key']:
                                logger.info(f"Found model at s3://{bucket}/{obj['Key']}")
                                s3.download_file(bucket, obj['Key'], local_path)
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
    try:
        if model is None:
            raise ValueError("Model is not loaded")
        
        # Ensure input data has all required features
        required_features = ['latitude', 'longitude', 'scan', 'track', 'bright_t31', 'confidence']
        if 'frp' in input_data.columns:
            required_features.append('frp')
            
        # Verify all required features are present
        missing_features = [feat for feat in required_features if feat not in input_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Make prediction
        predictions = model.predict(input_data[required_features])
        logger.info(f"Made prediction: {predictions[0]}")
        return predictions
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        st.error(f"Failed to make prediction: {e}")
        return None
