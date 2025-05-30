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

You can override the default config file:

```bash
docker run -v $(pwd)/artifacts:/app/artifacts cloud-pipeline --config config/custom-config.yaml
```

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

## Running Tests

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