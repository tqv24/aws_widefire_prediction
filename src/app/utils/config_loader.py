import yaml
import logging
import os

logger = logging.getLogger(__name__)

def load_config(config_path="config/app_config.yaml", fallback_path="config/default-config.yaml"):
    """Load configuration from YAML file with fallback"""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        
        # Try fallback
        if os.path.exists(fallback_path):
            with open(fallback_path, "r") as f:
                config = yaml.safe_load(f)
                logger.warning(f"Using fallback configuration from {fallback_path}")
                
                # Extract needed values from default-config
                aws_config = config.get("aws", {})
                app_config = {
                    "aws": aws_config,
                    "data": {
                        "primary_data_path": config.get("data_acquisition", {}).get("dataset_file_pattern", "fire_nrt*.csv"),
                        "fallback_paths": ["data/fire_nrt.csv"]
                    },
                    "models": {
                        "paths": {
                            "decision_tree": "decision_tree.pkl",
                            "random_forest": "random_forest.pkl",
                            "linear_regression": "linear_regression.pkl"
                        }
                    },
                    "app": {
                        "title": "🔥 Australian Fire Analysis App",
                        "version": "1.0.0"
                    }
                }
                return app_config
        
        # Default config if all else fails
        logger.warning(f"No configuration files found. Using default values.")
        return {
            "aws": {
                "bucket_name": "mlds423-s3-project",
                "prefix": "experiments"
            },
            "models": {
                "paths": {
                    "decision_tree": "decision_tree.pkl",
                    "random_forest": "random_forest.pkl",
                    "linear_regression": "linear_regression.pkl"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return minimal default configuration
        return {
            "aws": {
                "bucket_name": "mlds423-s3-project",
                "prefix": "experiments"
            }
        }
