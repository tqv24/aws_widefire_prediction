import streamlit as st
import pandas as pd
import os
import logging
import numpy as np
import folium
from streamlit_folium import st_folium
import pydeck as pdk

# Use absolute imports instead of relative
from utils.model_manager import download_model_from_s3, load_model, predict_fire

logger = logging.getLogger(__name__)

# At the top of the file, let's add a helper function for rerunning
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

def prediction_tab(model_keys, s3_bucket, s3_prefix, selected_model=None):
    """
    Display the prediction tab content with an interactive map and prediction form
    """
    # Display model information
    if selected_model is None:
        model_name = st.selectbox("Select Model", list(model_keys.keys()))
        model_key = model_keys[model_name]
    else:
        model_name = selected_model
        model_key = model_keys[selected_model]
        st.info(f"Using model: {selected_model}")
    
    # Initialize session state variables
    if 'latitude' not in st.session_state:
        st.session_state.latitude = -25.0
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 135.0
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_value' not in st.session_state:
        st.session_state.prediction_value = None
    
    # Create local model path
    local_model_path = os.path.join("models", model_key)
    os.makedirs("models", exist_ok=True)
    
    # Display current location at the top
    st.subheader(f"Seclected Location on the Map")
    
    # Main layout with map and parameters side by side
    map_col, params_col = st.columns([2, 3])  # Map on left, parameters on right
    
    # Column 1: Interactive Map
    with map_col:
        # Use a simpler Folium map with click functionality
        m = folium.Map(
            location=[st.session_state.latitude, st.session_state.longitude],
            zoom_start=4,
            tiles="CartoDB positron"
        )
        
        # Add click handler for location selection
        folium.ClickForMarker(
            popup="Click to select this location"
        ).add_to(m)
        
        # Add current marker with color based on prediction
        if st.session_state.prediction_made and st.session_state.prediction_value is not None:
            pred_value = st.session_state.prediction_value
            if pred_value > 400:
                color = 'red'
                intensity = 'High'
            elif pred_value > 350:
                color = 'orange'
                intensity = 'Moderate'
            else:
                color = 'blue'
                intensity = 'Low'
                
            # Add marker with prediction info
            folium.Marker(
                [st.session_state.latitude, st.session_state.longitude],
                popup=f"<b>Prediction:</b> {pred_value:.2f}<br><b>Intensity:</b> {intensity}",
                tooltip=f"Fire Brightness: {pred_value:.2f}",
                icon=folium.Icon(color=color, icon="fire", prefix='fa')
            ).add_to(m)
            
            # Add a circle to represent fire intensity
            folium.Circle(
                location=[st.session_state.latitude, st.session_state.longitude],
                radius=pred_value * 50,  # Scale radius by prediction
                color=color,
                fill=True,
                fill_opacity=0.4,
                popup=f"Fire Brightness: {pred_value:.2f}"
            ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; 
                 padding: 10px; border-radius: 5px; border:2px solid grey;">
            <p><b>Fire Intensity</b></p>
            <p><i class="fa fa-circle" style="color:red"></i> High (>400)</p>
            <p><i class="fa fa-circle" style="color:orange"></i> Moderate (350-400)</p>
            <p><i class="fa fa-circle" style="color:blue"></i> Low (<350)</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
        else:
            # Regular marker for initial selection
            folium.Marker(
                [st.session_state.latitude, st.session_state.longitude],
                popup="Current Selection",
                tooltip="Current Location",
                icon=folium.Icon(color='green', icon='map-marker', prefix='fa')
            ).add_to(m)
        
        # Display the map with interaction - taller but narrower
        map_data = st_folium(
            m, 
            width=None,  # Let container width control this
            height=500,  # Make it taller
            returned_objects=["last_clicked", "last_object_clicked"],
            use_container_width=True
        )
        
        # Handle map clicks to update location
        if map_data:
            if "last_clicked" in map_data and map_data["last_clicked"]:
                clicked_lat = map_data["last_clicked"].get("lat")
                clicked_lng = map_data["last_clicked"].get("lng")
                
                if clicked_lat and clicked_lng:
                    logger.info(f"Map clicked at: {clicked_lat}, {clicked_lng}")
                    st.session_state.latitude = clicked_lat
                    st.session_state.longitude = clicked_lng
                    st.session_state.prediction_made = False
                    safe_rerun()  # Use the helper function
        
        # Display prediction results directly below the map
        if st.session_state.prediction_made and st.session_state.prediction_value is not None:
            pred_value = st.session_state.prediction_value
            
            # Create a clean prediction result display
            st.markdown("---")
            result_container = st.container()
            with result_container:
                # Use rows instead of columns for prediction results
                
                # Row 1: Prediction value
                st.markdown(f"#### Prediction: {pred_value:.2f}")
                
                # Row 2: Intensity with color coding
                if pred_value > 400:
                    st.error("⚠️ High Intensity Fire")
                elif pred_value > 350:
                    st.warning("⚠️ Moderate Intensity Fire")
                else:
                    st.info("ℹ️ Low Intensity Fire")
                

    
    # Column 2: All parameters shown at once
    with params_col:
        # Location coordinates section
        st.markdown("##### Location Coordinates")
        lat_col, lon_col = st.columns(2)
        with lat_col:
            new_lat = st.number_input(
                "Latitude", 
                value=st.session_state.latitude,
                min_value=-90.0,
                max_value=90.0,
                step=0.1,
                format="%.4f",
                key="lat_input"
            )
        
        with lon_col:
            new_lon = st.number_input(
                "Longitude", 
                value=st.session_state.longitude,
                min_value=-180.0,
                max_value=180.0,
                step=0.1,
                format="%.4f",
                key="lon_input"
            )
        
        # Update coordinates if changed
        if new_lat != st.session_state.latitude or new_lon != st.session_state.longitude:
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
            st.session_state.prediction_made = False
            
        update_button = st.button("Update Location", type="primary", key="update_loc")
        if update_button:
            safe_rerun()  # Use the helper function

        st.markdown("---")
        
        # Sensor parameters section
        st.markdown("##### Sensor Parameters")
        scan_col, track_col, bright_col = st.columns(3)
        with scan_col:
            scan = st.slider("Scan", min_value=0.1, max_value=5.0, value=1.7, step=0.1)
        with track_col:
            track = st.slider("Track", min_value=0.1, max_value=5.0, value=1.3, step=0.1)
        with bright_col:
            bright_t31 = st.slider("Bright T31", min_value=200.0, max_value=400.0, value=300.0, step=1.0)
        
        st.markdown("---")
        
        # Fire parameters section
        st.markdown("##### Fire Parameters")
        conf_col, frp_col = st.columns(2)
        with conf_col:
            confidence = st.slider("Confidence", min_value=0, max_value=100, value=70, step=1)
        with frp_col:
            frp = st.slider("Fire Radiative Power (FRP)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button("🔥 Predict Fire Brightness", type="primary", key="predict_btn", use_container_width=True)
    
    # Process prediction when button is clicked
    if predict_button:
        with st.spinner("Calculating fire brightness prediction..."):
            # Prepare input data
            input_df = pd.DataFrame([{
                "latitude": st.session_state.latitude,
                "longitude": st.session_state.longitude,
                "scan": scan,
                "track": track,
                "bright_t31": bright_t31,
                "confidence": confidence,
                "frp": frp
            }])
            
            # Download and load model
            model_path = download_model_from_s3(model_key, local_model_path, s3_bucket, s3_prefix)
            if model_path:
                model = load_model(model_path)
                
                if model:
                    # Make prediction
                    prediction = predict_fire(model, input_df)
                    
                    if prediction is not None:
                        # Store prediction in session state
                        st.session_state.prediction_made = True
                        st.session_state.prediction_value = prediction[0]
                        safe_rerun()  # Use the helper function
                    else:
                        st.error("Failed to make prediction. Please try again.")
                else:
                    st.error("Failed to load model. Please try again.")
            else:
                st.error("Failed to download model from S3. Please check your connection.")
