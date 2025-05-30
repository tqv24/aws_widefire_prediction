#!/bin/bash

# Function to show usage
show_usage() {
    echo "Usage: $0 [app|pipeline] [arguments]"
    echo ""
    echo "Modes:"
    echo "  app       Run the Streamlit web application"
    echo "  pipeline  Run the data processing pipeline"
    echo ""
    echo "Examples:"
    echo "  $0 app           # Run the web application"
    echo "  $0 pipeline      # Run the pipeline with default arguments"
    echo "  $0 pipeline --config=config/custom.yaml  # Run pipeline with custom config"
    exit 1
}

# Check if mode is provided
if [ $# -eq 0 ]; then
    show_usage
fi

MODE=$1
shift  # Remove the first argument (mode)

case $MODE in
    app)
        echo "Starting web application..."
        streamlit run src/app/app.py "$@"
        ;;
    pipeline)
        echo "Running pipeline..."
        python -m src.pipeline.pipeline "$@"
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        show_usage
        ;;
esac
