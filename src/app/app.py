import streamlit as st
import logging
import os
import sys
import boto3

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules using absolute imports
from utils.config_loader import load_config
from utils.prediction_tab import prediction_tab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = load_config("config/app_config.yaml")
    aws_config = config.get("aws", {})
    S3_BUCKET = aws_config.get("bucket_name", "mlds423-s3-project")
    S3_PREFIX = aws_config.get("prefix", "experiments")
    data_config = config.get("data", {})
    app_config = config.get("app", {})
    
    # Set up model keys from config
    model_paths = config.get("models", {}).get("paths", {})
    MODEL_KEYS = {
        "Decision Tree": model_paths.get("decision_tree", "decision_tree.pkl"),
        "Random Forest": model_paths.get("random_forest", "random_forest.pkl"),
        "Linear Regression": model_paths.get("linear_regression", "linear_regression.pkl")
    }
    
    # Set up page configuration
    st.set_page_config(
        page_title=app_config.get("title", "Australian Fire Analysis App"),
        layout=app_config.get("layout", "wide"),
        initial_sidebar_state=app_config.get("initial_sidebar_state", "expanded")
    )
    
    # Main app title
    st.title(app_config.get("title", "Australian Fire Analysis App"))
    st.markdown(app_config.get("description", "This application provides tools to analyze and predict fire brightness in Australia."))
    
    # Sidebar layout
    with st.sidebar:
        # Add model selection to sidebar (at the top)
        st.subheader("Model Selection")
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = list(MODEL_KEYS.keys())[0]
        
        st.session_state.selected_model = st.selectbox(
            "Select prediction model:",
            options=list(MODEL_KEYS.keys()),
            index=list(MODEL_KEYS.keys()).index(st.session_state.selected_model)
        )
        
        # Add experiment selection
        st.subheader("Experiment Selection")
        
        # Get available experiments from S3
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def get_available_experiments(bucket, prefix):
            try:
                s3 = boto3.client('s3')
                response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix + '/',
                    Delimiter='/'
                )
                
                experiments = []
                # Do not add 'latest' option since it's not available
                
                if 'CommonPrefixes' in response:
                    for obj in response['CommonPrefixes']:
                        # Extract experiment ID from path
                        exp_id = obj['Prefix'].split('/')[-2]
                        if exp_id != 'latest':  # Skip 'latest'
                            experiments.append(exp_id)
                
                # Add known working experiments if list is empty
                if not experiments:
                    experiments = ["1748573816", "1748576650", "1748638871"]
                    
                # Sort experiments numerically (most recent first assuming timestamps)
                experiments = sorted(experiments, reverse=True)
                return experiments
            except Exception as e:
                logging.error(f"Error fetching experiments: {e}")
                # Return default experiments if S3 fetch fails
                return ["1748638871", "1748573816", "1748576650"]
        
        experiments = get_available_experiments(S3_BUCKET, S3_PREFIX)
        
        # Set default experiment in session state to a known working one
        if 'selected_experiment' not in st.session_state:
            st.session_state.selected_experiment = "1748638871"  # Use a known working experiment ID
        
        st.session_state.selected_experiment = st.selectbox(
            "Select experiment run:",
            options=experiments,
            index=experiments.index(st.session_state.selected_experiment) if st.session_state.selected_experiment in experiments else 0
        )
        
        # Display model specifications
        st.subheader("Model Specifications")
        
        # Model specs dictionary - can be expanded with more details
        model_specs = {
            "Decision Tree": {
                "Type": "Classification/Regression Tree",
                "Strengths": "Simple to understand, handles non-linear data",
                "Limitations": "Can overfit, less accurate than ensemble methods"
            },
            "Random Forest": {
                "Type": "Ensemble of Decision Trees",
                "Strengths": "High accuracy, handles large datasets well",
                "Limitations": "Less interpretable, computationally intensive"
            },
            "Linear Regression": {
                "Type": "Linear Model",
                "Strengths": "Simple, interpretable, fast training",
                "Limitations": "Only captures linear relationships"
            }
        }
        
        # Display the specifications for the selected model
        selected_specs = model_specs.get(st.session_state.selected_model, {})
        for key, value in selected_specs.items():
            st.write(f"**{key}:** {value}")
        
        # Create a similar helper function here
        def safe_rerun():
            """Safely rerun the app regardless of Streamlit version"""
            try:
                st.rerun()
            except AttributeError:
                try:
                    st.experimental_rerun()
                except AttributeError:
                    # For very old versions, just show a message
                    st.warning("Please refresh the page to see changes")
        
        # Add a button to refresh the app
        if st.button("Refresh App"):
            safe_rerun()
        
        # Move Application Info to the bottom of the sidebar
        st.markdown("---")
        st.subheader("Application Info")
        st.info(f"""
        **S3 Bucket:** {S3_BUCKET}
        **Version:** {app_config.get("version", "1.0.0")}
        **Data:** Australian Fires 2019-2020
        **Experiment:** {st.session_state.selected_experiment}
        """)
        
        # Add a status indicator
        status = st.success("App is running normally")
        
        # Add GitHub link
        st.markdown(f"[GitHub Repository]({app_config.get('github_repo', 'https://github.com/yourusername/cloudengr-aus-fire-Group5')})")
    
    # Display the prediction tab and pass the selected experiment
    prediction_tab(MODEL_KEYS, S3_BUCKET, S3_PREFIX, st.session_state.selected_model, st.session_state.selected_experiment)
    
    # Add footer
    st.markdown("---")
    st.markdown("Developed by Group 5 | MLDS423 Cloud Engineering")

if __name__ == "__main__":
    main()