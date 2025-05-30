import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use absolute imports instead of relative
from utils.data_loader import load_data_from_s3, generate_synthetic_data
from utils.model_manager import upload_model_to_s3

logger = logging.getLogger(__name__)

def train_model_tab(s3_bucket, s3_prefix, data_config):
    st.header("Train Your Own Fire Prediction Model")
    st.write("Train a model on the Australian fire data and use it for predictions.")
    
    # Load data
    with st.spinner("Loading training data..."):
        data_path = data_config.get("primary_data_path", "s3://mlds423-s3-project/data/fire_nrt.csv")
        fallback_paths = data_config.get("fallback_paths", [])
        
        df = load_data_from_s3(data_path, s3_bucket, fallback_paths)
        
        if df.empty:
            st.error("Failed to load training data. Using synthetic data instead.")
            df = generate_synthetic_data(2000)
            
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show data stats
        st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
    
    # Model training form
    with st.form("model_training_form"):
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Decision Tree", "Random Forest", "Linear Regression"]
        )
        
        # Hyperparameters based on model type
        if model_type == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 50, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            hyperparams = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "random_state": 42
            }
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 1, 30, 10)
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42
            }
        else:  # Linear Regression
            hyperparams = {}
        
        # Feature selection
        feature_cols = ["latitude", "longitude", "scan", "track", "bright_t31", "confidence", "frp"]
        selected_features = st.multiselect(
            "Select Features", 
            feature_cols,
            default=feature_cols
        )
        
        # Target column is fixed
        target_column = "brightness"
        
        # Training options
        test_size = st.slider("Test Size", 0.1, 0.5, 0.25)
        
        # Save options
        save_to_s3 = st.checkbox("Save model to S3", value=True)
        model_save_name = st.text_input("Model filename", value=f"{model_type.lower().replace(' ', '_')}.pkl")
        
        # Train button
        train_button = st.form_submit_button("Train Model")
    
    if train_button:
        if not selected_features:
            st.error("Please select at least one feature for training.")
            return
            
        with st.spinner(f"Training {model_type} model..."):
            # Prepare data
            X = df[selected_features]
            y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize model
            if model_type == "Decision Tree":
                model = DecisionTreeRegressor(**hyperparams)
            elif model_type == "Random Forest":
                model = RandomForestRegressor(**hyperparams)
            else:  # Linear Regression
                model = LinearRegression()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Display results
            st.success(f"Model training complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Results")
                st.write(f"Training R² Score: {train_score:.4f}")
                st.write(f"Test R² Score: {test_score:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
                st.write(f"Root Mean Squared Error: {rmse:.4f}")
            
            with col2:
                st.subheader("Model Information")
                st.write(f"Model Type: {model_type}")
                st.write(f"Hyperparameters: {hyperparams}")
                st.write(f"Features: {selected_features}")
            
            # Save the model locally
            os.makedirs("models", exist_ok=True)
            local_path = os.path.join("models", model_save_name)
            with open(local_path, 'wb') as f:
                import pickle
                pickle.dump(model, f)
            
            st.info(f"Model saved locally to {local_path}")
            
            # Save to S3 if requested
            if save_to_s3:
                success = upload_model_to_s3(model, model_save_name, s3_bucket, s3_prefix)
                if success:
                    st.success(f"Model uploaded to S3: s3://{s3_bucket}/models/{model_save_name}")
                else:
                    st.error("Failed to upload model to S3. See logs for details.")
            
            # Show a plot of predictions vs actual
            st.subheader("Predictions vs Actual")
            
            pred_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            })
            
            fig, ax = plt.subplots()
            ax.scatter(pred_df['Actual'], pred_df['Predicted'], alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predictions vs Actual')
            st.pyplot(fig)
