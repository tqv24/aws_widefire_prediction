import streamlit as st
import pandas as pd
import os
import logging

# Use absolute imports instead of relative
from utils.model_manager import download_model_from_s3, load_model, predict_fire

logger = logging.getLogger(__name__)

def prediction_tab(model_keys, s3_bucket, s3_prefix):
    st.header("Australian Fire Brightness Prediction")
    st.write("Predict fire brightness using different models loaded from S3.")

    model_name = st.selectbox("Select Model", list(model_keys.keys()))
    model_key = model_keys[model_name]
    local_model_path = os.path.join("models", model_key)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Input form for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Latitude", value=-25.0, min_value=-44.0, max_value=-10.0)
            longitude = st.number_input("Longitude", value=135.0, min_value=112.0, max_value=154.0)
            scan = st.number_input("Scan", value=1.7, min_value=0.1, max_value=5.0)
        
        with col2:
            track = st.number_input("Track", value=1.3, min_value=0.1, max_value=5.0)
            bright_t31 = st.number_input("Bright T31", value=300.0, min_value=200.0, max_value=400.0)
            confidence = st.number_input("Confidence", value=70, min_value=0, max_value=100)
        
        frp = st.number_input("Fire Radiative Power (FRP)", value=20.0, min_value=0.0, max_value=200.0)
        
        submit_button = st.form_submit_button("Predict Brightness")

    if submit_button:
        with st.spinner("Loading model and making prediction..."):
            # Prepare input data
            input_df = pd.DataFrame([{
                "latitude": latitude,
                "longitude": longitude,
                "scan": scan,
                "track": track,
                "bright_t31": bright_t31,
                "confidence": confidence,
                "frp": frp
            }])
            
            # Display input data
            st.subheader("Input Data")
            st.dataframe(input_df)

            # Download and load model
            model_path = download_model_from_s3(model_key, local_model_path, s3_bucket, s3_prefix)
            if model_path:
                model = load_model(model_path)
                
                if model:
                    # Make prediction
                    prediction = predict_fire(model, input_df)
                    
                    if prediction is not None:
                        # Display prediction with some context
                        st.success(f"Predicted Fire Brightness: {prediction[0]:.2f}")
                        
                        # Add interpretation
                        if prediction[0] > 400:
                            st.warning("⚠️ This indicates a very high intensity fire.")
                        elif prediction[0] > 350:
                            st.info("ℹ️ This indicates a moderate to high intensity fire.")
                        else:
                            st.info("ℹ️ This indicates a lower intensity fire.")
