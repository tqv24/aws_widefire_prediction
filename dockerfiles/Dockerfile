FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for better compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies with proper error handling
COPY requirements.txt .
# Install streamlit and dependencies first to ensure they're properly resolved
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit==1.11.0 altair==4.2.0 vega_datasets && \
    pip install --no-cache-dir -r requirements.txt || (echo "Requirements installation failed" && exit 1)

# Show installed packages for debugging
RUN pip list

# Create necessary directories to ensure proper structure
RUN mkdir -p /app/models /app/data

# Copy the application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables for AWS (these will be overridden by user-provided values)
ENV AWS_DEFAULT_REGION=us-east-2
ENV PYTHONPATH=/app

# Expose port for Streamlit
EXPOSE 8501

# Add a health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/ || exit 1

# Verify streamlit is available and check its location
RUN which streamlit || echo "Streamlit not found in PATH"

# Run streamlit with proper host binding
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]
