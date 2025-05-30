# Start with a Python base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies and AWS SDK
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3 awscli

# Copy the project files into the container
COPY config/ ./config/
# Uncomment if you need local model/data files
# COPY data/ ./data/
# COPY models/ ./models/
COPY src/ ./src/

# Make directory for artifacts
RUN mkdir -p runs

# Set environment variables
ENV PYTHONPATH=/app
ENV AWS_DEFAULT_REGION=us-east-1
# Set this to use forward slashes in paths (for S3 compatibility)
ENV PATH_SEP="/"

# Expose port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]