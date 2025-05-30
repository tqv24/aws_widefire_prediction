import streamlit as st
import logging
import os
import sys

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules using absolute imports
from utils.config_loader import load_config
from utils.prediction_tab import prediction_tab
from utils.animated_map_tab import animated_map_tab
from utils.train_model_tab import train_model_tab
from utils.documentation_tab import documentation_tab

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
    
    # Add S3 configuration info in the sidebar
    with st.sidebar:
        st.subheader("Application Info")
        st.info(f"""
        **S3 Bucket:** {S3_BUCKET}
        **Version:** {app_config.get("version", "1.0.0")}
        **Data:** Australian Fires 2019-2020
        """)
        
        # Add a status indicator
        status = st.success("App is running normally")
        
        # Add a button to refresh the app
        if st.button("Refresh App"):
            st.rerun()
        
        # Add GitHub link
        st.markdown(f"[GitHub Repository]({app_config.get('github_repo', 'https://github.com/yourusername/cloudengr-aus-fire-Group5')})")
    
    # Add tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Animated Map", "Train Model", "Documentation"])
    
    with tab1:
        prediction_tab(MODEL_KEYS, S3_BUCKET, S3_PREFIX)
    with tab2:
        animated_map_tab(S3_BUCKET, data_config)
    with tab3:
        train_model_tab(S3_BUCKET, S3_PREFIX, data_config)
    with tab4:
        documentation_tab()
    
    # Add footer
    st.markdown("---")
    st.markdown("Developed by Group 5 | MLDS423 Cloud Engineering")

if __name__ == "__main__":
    main()