# Start with a Python base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Set environment variables to improve Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AWS_DEFAULT_REGION=us-east-1 \
    PATH_SEP="/"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies and AWS SDK
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3 awscli pytest streamlit

# Create directories for artifacts and data
RUN mkdir -p /app/artifacts /app/runs /app/data

# Copy the project files into the container
COPY config/ ./config/
# Uncomment if you need local model/data files
# COPY data/ ./data/
# COPY models/ ./models/
COPY src/ ./src/
COPY tests/ ./tests/

# Create entrypoint script for multi-mode operation
RUN echo '#!/bin/bash\n\
# Function to show usage\n\
show_usage() {\n\
    echo "Usage: entrypoint.sh [app|pipeline|pytest] [arguments]"\n\
    echo ""\n\
    echo "Modes:"\n\
    echo "  app       Run the Streamlit web application"\n\
    echo "  pipeline  Run the data processing pipeline"\n\
    echo "  pytest    Run tests"\n\
    echo ""\n\
    exit 1\n\
}\n\
\n\
# Check if mode is provided\n\
if [ $# -eq 0 ]; then\n\
    MODE="pipeline"\n\
else\n\
    MODE=$1\n\
    shift  # Remove the first argument (mode)\n\
fi\n\
\n\
case $MODE in\n\
    app)\n\
        echo "Starting web application..."\n\
        exec streamlit run src/app/app.py --server.port=8501 --server.address=0.0.0.0 "$@"\n\
        ;;\n\
    pipeline)\n\
        echo "Running pipeline..."\n\
        exec python -m src.pipeline.pipeline "$@"\n\
        ;;\n\
    pytest)\n\
        exec pytest "$@"\n\
        ;;\n\
    *)\n\
        echo "Error: Unknown mode '\''$MODE'\''"\n\
        show_usage\n\
        ;;\n\
esac' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create volume for artifacts - allows persistence between runs
VOLUME ["/app/artifacts", "/app/runs"]

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

# Expose port for Streamlit
EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]
# Default command is 'app' to maintain compatibility with your original Dockerfile
CMD ["app"]