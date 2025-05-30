import argparse
import datetime
import logging.config
import traceback
from pathlib import Path
import os
import shutil
import sys

# Set matplotlib backend to non-interactive to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Updated imports to use the new module locations
from utils.acquire_data import acquire_data
from utils.analysis import save_figures, plot_feature_importance
from utils.aws_utils import ensure_s3_url_exists, create_sample_data, upload_artifacts
from utils.create_dataset import create_dataset, save_dataset
from utils.evaluate_performance import evaluate_performance, save_metrics
from utils.generate_features import generate_fire_features, save_features
from utils.score_model import score_model, save_scores
from utils.train_model import simple_parameter_tuning, train_model, save_model, save_data

# Configure logging
logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("firemodel")

def run_pipeline(config_path, group5_mode=False):
    """Run the full fire prediction pipeline with detailed logging and error handling
    
    Args:
        config_path: Path to the configuration YAML file
        group5_mode: Enable compatibility mode with Group5 notebook
    """
    try:
        logger.info("Starting fire prediction pipeline")
        logger.info(f"Using configuration from {config_path}")
        
        # Load configuration file for parameters and run config
        with open(config_path, "r") as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
                logger.info("Configuration file loaded successfully")
            except yaml.error.YAMLError as e:
                logger.error(f"Error while loading configuration from {config_path}: {e}")
                return
                
        logger.info("Parsed configuration:")
        for section in config.keys():
            logger.info(f"  - {section}")
        
        run_config = config.get("run_config", {})
        aws_config = config.get("aws", {})

        # Set up output directory for saving artifacts
        now = int(datetime.datetime.now().timestamp())
        artifacts = Path(run_config.get("output", "runs")) / str(now)
        artifacts.mkdir(parents=True)
        logger.info(f"Artifacts will be saved to {artifacts}")

        # Save config file to artifacts directory for traceability
        with (artifacts / "config.yaml").open("w") as f:
            yaml.dump(config, f)
        logger.info(f"Configuration saved to {artifacts / 'config.yaml'}")
        
        # Ensure the S3 bucket exists before trying to access it
        data_source = run_config.get("data_source", "")
        logger.info(f"Data source: {data_source}")
        
        if data_source.startswith("s3://"):
            # Ensure the bucket exists
            logger.info("Ensuring S3 bucket exists...")
            bucket_exists = ensure_s3_url_exists(data_source, aws_config)
            if not bucket_exists:
                logger.error(f"Failed to ensure S3 bucket exists for {data_source}")
                return
            
            # Create sample data if it doesn't exist yet
            logger.info("Checking for sample data...")
            data_created = create_sample_data(data_source, aws_config)
            if not data_created:
                logger.warning(f"Failed to ensure sample data exists at {data_source}")
            
            logger.info(f"S3 bucket for data source '{data_source}' is ready")

        # STEP 1: Acquire data from source and save to disk
        logger.info("STEP 1: Acquiring data")
        data_path = artifacts / "fire_data.csv"
        
        # Try to acquire data with extensive error handling
        try:
            data_file = acquire_data(config)
            
            if data_file and os.path.exists(data_file):
                # If the data was successfully acquired, copy it to the artifacts directory
                shutil.copy(data_file, data_path)
                logger.info(f"Data acquired from {data_file} and saved to {data_path}")
            else:
                logger.error(f"Failed to acquire data. No data file found or data_file is None: {data_file}")
                # Last resort - try to create sample data directly in the artifacts directory
                try:
                    logger.info("Attempting to create sample data directly in artifacts directory")
                    # Create synthetic data (similar to what's in the notebook)
                    np.random.seed(42)
                    n_samples = 1000
                    data = {
                        'latitude': np.random.uniform(-44, -10, n_samples),
                        'longitude': np.random.uniform(112, 154, n_samples),
                        'brightness': np.random.uniform(300, 500, n_samples),
                        'scan': np.random.uniform(0.5, 2.0, n_samples),
                        'track': np.random.uniform(0.5, 2.0, n_samples),
                        'bright_t31': np.random.uniform(250, 350, n_samples),
                        'frp': np.random.uniform(5, 100, n_samples),
                        'confidence': np.random.randint(0, 100, n_samples)
                    }
                    
                    df = pd.DataFrame(data)
                    df.to_csv(data_path, index=False)
                    logger.info(f"Created emergency sample data at {data_path}")
                except Exception as e:
                    logger.error(f"Failed to create emergency sample data: {e}")
                    return
        except Exception as e:
            logger.error(f"Exception during data acquisition: {e}")
            logger.error(traceback.format_exc())
            return
            
        # Verify that data_path exists before continuing
        if not data_path.exists():
            logger.error(f"Data file {data_path} does not exist after acquisition step")
            return
            
        # STEP 2: Create structured dataset from raw data; save to disk
        logger.info("STEP 2: Creating structured dataset")
        data = create_dataset(data_path, config["create_dataset"])
        processed_path = artifacts / "fire_data_processed.csv"
        save_dataset(data, processed_path)
        logger.info(f"Structured dataset created with {len(data)} rows and saved to {processed_path}")

        # STEP 3: Enrich dataset with features for model training; save to disk
        logger.info("STEP 3: Generating features")
        features = generate_fire_features(data, config["generate_features"])
        features_path = artifacts / "fire_features.csv"
        save_features(features, str(features_path))
        logger.info(f"Features generated with {len(features.columns)} columns and saved to {features_path}")

        # STEP 4: Generate statistics and visualizations for summarizing the data
        logger.info("STEP 4: Creating exploratory data analysis")
        figures = artifacts / "figures"
        figures.mkdir(exist_ok=True)
        figure_paths = save_figures(features, figures)
        logger.info(f"Generated {len(figure_paths)} EDA visualizations in {figures}")

        # STEP 5: Train models based on the notebook approach
        logger.info("STEP 5: Training models")
        
        # Get configuration
        train_config = config["train_model"]
        target_column = train_config.get("target_column", "brightness")
        selected_features = train_config.get("initial_features", [])
        test_size = train_config.get("test_size", 0.3)
        random_state = train_config.get("random_state", 42)
        
        # Extract features and target similar to the notebook
        logger.info(f"Using target column: {target_column}")
        logger.info(f"Using features: {selected_features}")
        
        # Prepare features and target
        y = features[target_column]
        X = features[selected_features]
        
        # Split data - exactly like in the notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")
        
        # Create training and test dataframes with features and target
        train_data = X_train.copy()
        train_data[target_column] = y_train
        
        test_data = X_test.copy()
        test_data[target_column] = y_test
        
        # Save train/test data
        train_path = artifacts / "train_data.csv"
        test_path = artifacts / "test_data.csv"
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info(f"Saved training data ({len(train_data)} rows) to {train_path}")
        logger.info(f"Saved test data ({len(test_data)} rows) to {test_path}")
        
        # Train models as defined in config - this will now match the notebook approach
        models_dict = {}
        models_config = train_config.get("models", [])
        
        # Train each model specified in the config
        for model_config in models_config:
            model_name = model_config.get("name", "unnamed_model")
            model_type = model_config.get("type", "DecisionTreeRegressor")
            hyperparameters = model_config.get("hyperparameters", {})
            is_default = model_config.get("is_default", False)
            
            logger.info(f"Training {model_type} model '{model_name}' with hyperparameters: {hyperparameters}")
            
            # Create and train model exactly as in the notebook
            if model_type == "LinearRegression":
                model = LinearRegression(**hyperparameters)
            elif model_type == "DecisionTreeRegressor":
                model = DecisionTreeRegressor(**hyperparameters)
            elif model_type == "RandomForestRegressor":
                model = RandomForestRegressor(**hyperparameters)
            else:
                logger.warning(f"Unsupported model type: {model_type}, using DecisionTreeRegressor")
                model = DecisionTreeRegressor(**hyperparameters)
            
            # Train model
            model.fit(X_train, y_train)
            logger.info(f"Model '{model_name}' training complete")
            
            # Store model in dictionary
            models_dict[model_name] = {
                "model": model,
                "type": model_type,
                "hyperparameters": hyperparameters,
                "is_default": is_default,
                "feature_names": selected_features
            }
        
        # Ensure default model is set
        default_model = None
        for model_name, model_info in models_dict.items():
            if model_info.get("is_default", False):
                default_model = model_name
                break
        
        if not default_model and models_dict:
            default_model = list(models_dict.keys())[0]
            models_dict[default_model]["is_default"] = True
            logger.info(f"Setting '{default_model}' as default model")
        
        # Save models
        models_path = artifacts / "models"
        models_path.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model_info in models_dict.items():
            model_obj = model_info["model"]
            model_filename = f"{model_name}.pkl"
            model_path = models_path / model_filename
            
            logger.info(f"Saving model '{model_name}' to {model_path}")
            
            # Save model using pickle
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f)
            
            logger.info(f"Model '{model_name}' saved successfully")
        
        # Also save the entire model dictionary
        models_dict_path = models_path / "models_dict.pkl"
        with open(models_dict_path, "wb") as f:
            pickle.dump(models_dict, f)
        
        logger.info("All models saved successfully")

        # STEP 6: Score models on test set; save scores to disk
        logger.info("STEP 6: Scoring models")
        scores = score_model(test_data, models_dict, config["score_model"])
        scores_path = artifacts / "scores.csv"
        save_scores(scores, scores_path)
        
        # Log scores for each model
        for model_name, model_scores in scores.get("model_scores", {}).items():
            logger.info(f"  - Model: {model_name}")
            for metric in ["mae", "rmse", "r2", "explained_variance"]:
                if metric in model_scores:
                    logger.info(f"    {metric.upper()}: {model_scores[metric]:.4f}")

        # STEP 7: Evaluate model performance metrics; save metrics to disk
        logger.info("STEP 7: Evaluating model performance")
        metrics = evaluate_performance(scores, config["evaluate_performance"])
        metrics_path = artifacts / "metrics.yaml"
        save_metrics(metrics, metrics_path)
        logger.info(f"Performance metrics and visualizations saved to {metrics_path} and {artifacts / 'figures'}")

        # STEP 8: Upload all artifacts to S3
        if aws_config.get("upload", False):
            logger.info("STEP 8: Uploading artifacts to S3")
            s3_uris = upload_artifacts(artifacts, aws_config)
            logger.info(f"Uploaded {len(s3_uris)} artifacts to S3")
            for uri in s3_uris[:5]:  # Log first 5 URIs
                logger.info(f"  - {uri}")
            if len(s3_uris) > 5:
                logger.info(f"  - ...and {len(s3_uris) - 5} more")
        else:
            logger.info("STEP 8: Skipping S3 upload (disabled in config)")

        logger.info("Pipeline completed successfully")
        return artifacts
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from fire data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--group5-mode", 
        action="store_true",
        help="Enable compatibility mode with Group5 notebook"
    )
    args = parser.parse_args()

    # Run the pipeline with the specified configg
    run_pipeline(args.config, args.group5_mode)

