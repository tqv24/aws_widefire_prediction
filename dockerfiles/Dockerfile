# FROM python:3.10-slim
# # Set working directory
# WORKDIR /app
# # Set environment variables to improve Python behavior in containers
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt pytest
# RUN mkdir -p /app/artifacts
# COPY src/ /app/src/
# COPY config/ /app/config/
# COPY tests/ /app/tests/
# COPY pipeline.py /app/
# # Create entrypoint script
# RUN echo '#!/bin/bash\nif [ "$1" = "pytest" ]; then\n    shift\n    exec pytest "$@"\nelse\n    exec python pipeline.py "$@"\nfi' > /app/entrypoint.sh && \
#     chmod +x /app/entrypoint.sh
# # Create volume for artifacts - allows persistence between runs
# VOLUME ["/app/artifacts"]
# # Make directories for data and output
# RUN mkdir -p /app/data
# # Add a healthcheck
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD python -c "import sys; sys.exit(0)"
# ENTRYPOINT ["/app/entrypoint.sh"]
# CMD ["--config", "config/default-config.yaml"]

# # Alternative command to run tests
# # Usage: docker run cloud-classfier pytest
# # Or: docker run hw2-pipeline pytest tests/test_specific_file.py


FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables to improve Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

# Create directory for artifacts
RUN mkdir -p /app/artifacts

# Copy application files
COPY src/ /app/src/
COPY config/ /app/config/
COPY tests/ /app/tests/
COPY pipeline.py /app/

# Create entrypoint script
RUN echo '#!/bin/bash\nif [ "$1" = "pytest" ]; then\n    shift\n    exec pytest "$@"\nelse\n    exec python pipeline.py "$@"\nfi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create volume for artifacts - allows persistence between runs
VOLUME ["/app/artifacts"]

# Make directories for data
RUN mkdir -p /app/data

# Add a healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "config/default-config.yaml"]
