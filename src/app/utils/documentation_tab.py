import streamlit as st

def documentation_tab():
    st.header("Australian Fire Analysis App Documentation")
    
    st.markdown("""
    ## Overview
    
    This application analyzes fire data collected from satellite observations over Australia, focusing on
    the period from October 2019 to January 2020, which captures the devastating 2019-2020 bushfire season.
    
    The app provides several key features:
    
    1. **Fire Brightness Prediction**: Use machine learning models to predict fire brightness based on geographical and satellite data
    2. **Animated Fire Map**: Visualize how fires spread and intensified across Australia over time
    3. **Model Training**: Train your own ML models on the fire data and compare their performance
    4. **Documentation**: Learn how to use the app and understand the underlying data and models
    
    ## Data Description
    
    The application uses the MODIS (Moderate Resolution Imaging Spectroradiometer) fire detection data, which includes:
    
    | Feature | Description |
    |---------|-------------|
    | latitude | Geographical latitude coordinate of fire detection |
    | longitude | Geographical longitude coordinate of fire detection |
    | brightness | Brightness temperature of the fire pixel (target variable) |
    | scan | Along scan pixel size |
    | track | Along track pixel size |
    | bright_t31 | Brightness temperature in channel 31 |
    | frp | Fire Radiative Power in megawatts |
    | confidence | Confidence of fire detection (0-100%) |
    | acq_date | Date of fire acquisition |
    
    ## Machine Learning Models
    
    The app includes several pre-trained models:
    
    1. **Decision Tree**: A tree-based model that makes decisions based on feature thresholds
    2. **Random Forest**: An ensemble of decision trees for improved prediction accuracy
    3. **Linear Regression**: A basic linear model that establishes relationships between features
    
    You can also train your own models with custom parameters in the "Train Model" tab.
    
    ## Using the App
    
    ### Prediction Tab
    1. Select a model from the dropdown
    2. Enter input values for fire features
    3. Click "Predict Brightness" to get a prediction
    
    ### Animated Map Tab
    1. The map loads automatically with fire data
    2. Use the slider to select a date range
    3. Click "Play" to animate the fire progression
    4. Explore fire statistics in the expandable section
    
    ### Train Model Tab
    1. Select a model type
    2. Adjust the hyperparameters
    3. Select features to use
    4. Click "Train Model" to train and evaluate
    5. Optionally save your model to S3
    
    ## About the Project
    
    This application was developed by Group 5 for the MLDS423 Cloud Engineering course.
    The project analyzes wildfire data from the devastating 2019-2020 Australian bushfire season.
    """)
    
    # Add troubleshooting section
    st.subheader("Troubleshooting")
    
    with st.expander("Common Issues and Solutions"):
        st.markdown("""
        ### Model Loading Failures
        
        If you see errors about models not loading from S3:
        
        1. The app will automatically generate a temporary model for demonstration
        2. You can train your own model in the "Train Model" tab
        3. Check that you have proper AWS credentials configured
        
        ### Data Loading Issues
        
        If fire data doesn't load properly:
        
        1. The app will generate synthetic data for demonstration
        2. Ensure your AWS credentials have access to the S3 bucket
        3. Try refreshing the app using the sidebar button
        
        ### AWS Configuration
        
        For AWS access issues:
        
        1. Ensure you have valid AWS credentials in ~/.aws/credentials
        2. Verify that you have access to the bucket specified in the config file
        3. Check that the S3 paths in the configuration are correct
        """)
