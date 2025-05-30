import streamlit as st
import logging
import os
import sys

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
        """)
        
        # Add a status indicator
        status = st.success("App is running normally")
        
        # Add GitHub link
        st.markdown(f"[GitHub Repository]({app_config.get('github_repo', 'https://github.com/yourusername/cloudengr-aus-fire-Group5')})")
    
    # Display only the prediction tab without tabs UI
    prediction_tab(MODEL_KEYS, S3_BUCKET, S3_PREFIX, st.session_state.selected_model)
    
    # Add footer
    st.markdown("---")
    st.markdown("Developed by Group 5 | MLDS423 Cloud Engineering")

if __name__ == "__main__":
    main()