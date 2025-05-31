import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

from src.utils.create_dataset import create_dataset, clean_fire_data
from src.utils.generate_features import generate_fire_features
from src.utils.train_model import train_model
from src.utils.score_model import evaluate_model, score_model
from src.utils.analysis import save_figures

@pytest.fixture
def sample_df():
    data = {
        "latitude": np.linspace(-20, -30, 10),
        "longitude": np.linspace(130, 150, 10),
        "brightness": np.linspace(320, 340, 10),
        "scan": np.random.uniform(1.5, 1.7, 10),
        "track": np.random.uniform(1.1, 1.3, 10),
        "bright_t31": np.linspace(300, 310, 10),
        "frp": np.linspace(10, 30, 10),
        "confidence": np.random.randint(80, 100, 10),
        "acq_date": pd.date_range("2019-10-01", periods=10).strftime("%Y-%m-%d")
    }
    df = pd.DataFrame(data)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    return df

def test_clean_fire_data(sample_df):
    cleaned = clean_fire_data(sample_df)
    assert not cleaned.isnull().any().any()
    assert all((cleaned['latitude'] >= -44) & (cleaned['latitude'] <= -10))

def test_create_dataset(sample_df, tmp_path):
    config = {
        "critical_columns": ["latitude", "longitude", "brightness"],
        "region_filter": {"enabled": True, "min_latitude": -44, "max_latitude": -10, "min_longitude": 112, "max_longitude": 154},
        "outlier_removal": {"enabled": True, "column": "brightness", "std_threshold": 3},
        "target_column": "brightness"
    }
    data_path = tmp_path / "sample.csv"
    sample_df.to_csv(data_path, index=False)
    df = create_dataset(data_path, config)
    assert isinstance(df, pd.DataFrame)
    assert "brightness" in df.columns

def test_generate_fire_features(sample_df):
    config = {"feature_columns": ["latitude", "longitude"], "derived_features": ["frp_per_area", "temperature_diff"]}
    features_df = generate_fire_features(sample_df, config)
    assert "frp_per_area" in features_df.columns
    assert "temperature_diff" in features_df.columns

def test_train_model(sample_df):
    config = {
        "target_column": "brightness",
        "initial_features": ["latitude", "longitude", "scan", "track", "bright_t31", "frp", "confidence"],
        "models": [{"name": "tree", "type": "DecisionTreeRegressor", "hyperparameters": {}}],
        "test_size": 0.2,
        "random_state": 1
    }
    models_dict, train_df, test_df = train_model(sample_df, config)
    assert "tree" in models_dict
    assert not train_df.empty
    assert not test_df.empty

def test_evaluate_model(sample_df):
    X = sample_df[["latitude", "longitude", "scan", "track", "bright_t31", "frp", "confidence"]]
    y = sample_df["brightness"]
    model = DecisionTreeRegressor().fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

def test_score_model(sample_df):
    X = sample_df[["latitude", "longitude", "scan", "track", "bright_t31", "frp", "confidence"]]
    y = sample_df["brightness"]
    model = DecisionTreeRegressor().fit(X, y)
    models_dict = {"tree": {"model": model, "feature_names": list(X.columns), "is_default": True}}
    config = {"metrics": ["mae", "rmse", "r2"], "target_column": "brightness"}
    scores = score_model(sample_df, models_dict, config)
    assert "model_scores" in scores
    assert "tree" in scores["model_scores"]

def test_save_figures(sample_df, tmp_path):
    paths = save_figures(sample_df, tmp_path)
    assert isinstance(paths, list)
    assert all(Path(p).exists() for p in paths)
