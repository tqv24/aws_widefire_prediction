# Cloud Classification Pipeline

This repository contains a pipeline for classifying clouds based on their features. The pipeline performs data acquisition, preprocessing, feature engineering, model training, and evaluation.

## Technical Requirements

- Python 3.10+
- Docker
- Required Python packages are listed in `requirements.txt`

## Docker Setup

### Building the Docker Image

To build the Docker image for the pipeline, run:

```bash
docker build -t cloud-pipeline -f dockerfiles/Dockerfile .
```

### Running the Pipeline

To run the pipeline using the Docker container:

```bash
docker run -v $(pwd)/artifacts:/app/artifacts cloud-pipeline
```

### Running the Fire Prediction App

To run the Streamlit app using Docker:

```bash
docker run -p 8501:8501 fire-app
```

> **Note**: If you see "Cannot connect to the server" errors, try accessing the app at http://localhost:8501

> **Security Note**: Never expose your AWS credentials in documentation, repositories, or public communications. The above example is for illustration only. Use environment variables, AWS profiles, or secrets management systems in production.

### Environment Variables

The following environment variables can be set to configure AWS S3 uploads:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_S3_BUCKET`: Override the S3 bucket name defined in the config
- `AWS_REGION`: AWS region to use (defaults to us-west-2 if not specified)

Example with AWS configuration:

```bash
docker run -v $(pwd)/artifacts:/app/artifacts \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_S3_BUCKET=your_bucket_name \
  -e AWS_REGION=us-west-2 \
  cloud-pipeline
```

### Troubleshooting Docker Issues

If you encounter the error "streamlit: executable file not found in $PATH", make sure you've built the Docker image with the latest Dockerfile that correctly installs streamlit:

```bash
docker build -t fire-app -f dockerfiles/Dockerfile .
```

## Running Tests

### Test Files Overview

The project includes two main test files:

1. `test/test_app.py`: Tests for utility functions in the Streamlit application
2. `test/test_pipeline.py`: Tests for utility functions in the data pipeline

### Running Tests Locally

To run the tests locally, you need to set the `PYTHONPATH` to include the `src` directory:

```bash
PYTHONPATH=src python3 -m unittest test/test_app.py
PYTHONPATH=src python3 -m unittest test/test_pipeline.py
```

### Test File Structure

#### test_app.py

This file tests the utility functions in `src/app/utils/`:

- `TestModelManager`: Tests model-related utilities
  - `test_normalize_path`: Tests path normalization
  - `test_load_model`: Tests model loading functionality
  - `test_predict_fire`: Tests fire prediction with valid and invalid data

- `TestDataLoader`: Tests data loading utilities
  - `test_generate_synthetic_data`: Tests synthetic data generation
  - `test_ensure_date_column`: Tests date column handling

- `TestConfigLoader`: Tests configuration loading
  - `test_load_config`: Tests loading from primary and fallback configs

#### test_pipeline.py

This file tests the utility functions in `src/pipeline/utils/`:

- Tests for data pipeline utilities
- Tests for feature generation
- Tests for model training and evaluation

### Test Results

When running the tests, you should see output similar to:

```
Using fallback configuration from /tmp/test_fallback.yaml
No configuration files found. Using default values.
......
----------------------------------------------------------------------
Ran 6 tests in 0.007s

OK
```

The dots (.) indicate passed tests, and "OK" means all tests completed successfully.

### Running Tests in Docker

The Docker container is configured to run tests using pytest. To run all tests:

```bash
docker run cloud-pipeline pytest -v
```

To run specific test files:

```bash
docker run cloud-pipeline pytest tests/test_generate_feature.py -v
```

For tests with more options:

```bash
docker run cloud-pipeline pytest -v -s --cov=src
```

## Local Development

### Setting Up the Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline Locally

```bash
python pipeline.py --config config/default-config.yaml
```

### Running Tests Locally

```bash
pytest tests/
```

### Code Style

The codebase follows PEP8 guidelines. To check for linting errors:
```bash
pylint src/
```

## Project Structure

- `config/`: Configuration files
- `src/`: Source code modules
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration
- `artifacts/`: Output directory for pipeline runs
- `dockerfiles/`: Docker configuration files
- `pipeline.py`: Main pipeline execution script
- `requirements.txt`: Project dependencies

### Project Tree
```{text}
cloud-pipeline/
│
├── config/
│   ├── default-config.yaml
│   └── custom-config.yaml (optional)
│
├── src/
│   ├── __init__.py
│   ├── acquire_data.py
│   ├── create_dataset.py
│   ├── analysis.py
│   ├── evaluate_performance.py
│   ├── generate_features.py
│   ├── score_model.py
│   ├── train_model.py
│   └── aws_utils.py
│
├── tests/
│   ├── __init__.py
│   └── test_generate_feature.py
│
├── notebooks/
│   └── cloud.ipynb
│
├── artifacts/
│   └── ...
│
├── dockerfiles/
│   └── Dockerfile
│
├── pipeline.py
└── requirements.txt
```

docker build -f dockerfiles/Dockerfile -t fire-app .


docker run -p 8501:8501 \
  -e AWS_ACCESS_KEY_ID=AKIAYLIWAPXRUH5QWTOD \
  -e AWS_SECRET_ACCESS_KEY=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 \
  -e AWS_DEFAULT_REGION=us-east-2 \
  fire-app


docker run -p 8501:8501 \
  -e AWS_ACCESS_KEY_ID=AKIAYLIWAPXR3NLVTAXZ \
  -e AWS_SECRET_ACCESS_KEY=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 \
  -e AWS_DEFAULT_REGION=us-east-2 \
  fire-app