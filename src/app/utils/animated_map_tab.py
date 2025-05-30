import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logging
import numpy as np

# Use absolute imports instead of relative
from utils.data_loader import load_data_from_s3, generate_synthetic_data

logger = logging.getLogger(__name__)

def ensure_date_column(df):
    """Ensure the dataframe has a properly formatted acq_date column"""
    if 'acq_date' not in df.columns:
        # Try to create it from acquisition date or day columns if available
        if 'acquisition_date' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acquisition_date'])
        elif 'acq_day' in df.columns and 'acq_year' in df.columns:
            df['acq_date'] = pd.to_datetime(df['acq_year'].astype(str) + '-' + df['acq_day'].astype(str), format='%Y-%j')
        else:
            # Generate synthetic dates
            logger.warning("No date column found, generating synthetic dates")
            start_date = pd.Timestamp('2019-10-01')
            end_date = pd.Timestamp('2020-01-11')
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # Generate random dates
            random_dates = []
            for _ in range(len(df)):
                idx = np.random.randint(0, len(date_range))
                random_dates.append(date_range[idx])
            
            df['acq_date'] = random_dates
    else:
        # Ensure acq_date is datetime format if it exists
        if not pd.api.types.is_datetime64_any_dtype(df['acq_date']):
            df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    return df

def animated_plot(df, start_date=None, end_date=None, map_style='stamen-terrain'):
    """
    Create an animated plot of fire data over time.
    This function is directly copied from the working notebook.
    
    Args:
        df: DataFrame with fire data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        map_style: Map style to use (default: stamen-terrain)
    
    Returns:
        Plotly figure object
    """
    # Sort the data
    df1 = df.sort_values(by=['acq_date'], ascending=True)
    
    # Filter by date if provided
    if start_date and end_date:
        df1 = df1[(df1['acq_date'] >= start_date) & (df1['acq_date'] <= end_date)]

    # Create frames for animation
    frames = []
    for date in df1['acq_date'].unique():
        df_date = df1[df1['acq_date'] == date]
        frame = go.Frame(
            data=[go.Densitymap(
                lat=df_date['latitude'],
                lon=df_date['longitude'],
                z=df_date['brightness'],
                radius=8,
                colorscale='Reds',
                showscale=True
            )],
            name=str(date)
        )
        frames.append(frame)

    # Create the figure
    fig = go.Figure(
        data=[go.Densitymap(
            lat=df1['latitude'],
            lon=df1['longitude'],
            z=df1['brightness'],
            radius=8,
            colorscale='Reds',
            showscale=True
        )],
        frames=frames
    )

    # Format dates for title
    start_str = start_date.strftime("%Y/%m/%d") if start_date else "2019/10/01"
    end_str = end_date.strftime("%Y/%m/%d") if end_date else "2020/01/11"

    fig.update_layout(
        title=f'Australian Fires: From {start_str} to {end_str}',
        title_font=dict(size=18, color='FireBrick'),
        title_x=0.5,
        mapbox=dict(
            center=dict(lon=134, lat=-25),
            zoom=2.4,
            style=map_style
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
                }
            ]
        }],
        sliders=[{
            "steps": [
                {
                    "args": [
                        [str(date)],
                        {"frame": {"duration": 500, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}}
                    ],
                    "label": str(date),
                    "method": "animate"
                }
                for date in df1['acq_date'].unique()
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "y": 0,
            "currentvalue": {"font": {"size": 16}, "prefix": "Date=", "visible": True, "xanchor": "center"},
            "len": 0.9
        }]
    )
    
    # Return the figure instead of showing it
    return fig

def animated_map_tab(s3_bucket, data_config):
    st.header("Australian Fire Animated Map")
    st.write("This animated map shows the spread and intensity of fires in Australia over time.")

    # Data loading with error handling
    with st.spinner("Loading fire data from S3..."):
        # First try the path from config
        data_path = data_config.get("primary_data_path", "s3://mlds423-s3-project/data/fire_nrt.csv")
        fallback_paths = data_config.get("fallback_paths", [])
        
        logger.info(f"Attempting to load data from primary source: {data_path}")
        
        df = load_data_from_s3(data_path, s3_bucket, fallback_paths)
        
        if df.empty:
            st.error("Failed to load fire data. Using synthetic data instead.")
            df = generate_synthetic_data(2000)
        
        # Ensure acq_date column exists and is in datetime format
        df = ensure_date_column(df)
    
    # Date filter using Streamlit widgets
    available_dates = sorted(df['acq_date'].unique())
    start_date, end_date = st.select_slider(
        "Select date range",
        options=available_dates,
        value=(available_dates[0], available_dates[-1])
    )
    
    # Map style selection
    map_style = st.selectbox(
        "Select Map Style", 
        ["stamen-terrain", "carto-positron", "open-street-map"],
        index=0  # Default to stamen-terrain
    )
    
    # Create and display a simple version first (non-animated)
    st.subheader("Static Fire Map")
    simple_fig = go.Figure(go.Densitymap(
        lat=df['latitude'], 
        lon=df['longitude'], 
        z=df['brightness'], 
        radius=5, 
        colorscale='Reds'))
    
    simple_fig.update_layout(
        mapbox_style=map_style,
        mapbox_center_lon=134,
        mapbox_center_lat=-25,
        mapbox_zoom=2.4,
        title='Australian Fires - Static View',
        title_font=dict(size=20, color='FireBrick'),
        title_x=0.5
    )
    
    st.plotly_chart(simple_fig, use_container_width=True)
    
    # Use the animated_plot function from the notebook
    st.subheader("Animated Fire Map")
    st.write("Use the Play button to animate the fire progression over time.")
    
    try:
        # Generate the animated figure using the notebook function
        fig = animated_plot(df, start_date, end_date, map_style)
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating animated map: {e}")
        st.info("The animated map encountered an error. Please try a different date range or map style.")
    
    # Add data stats
    with st.expander("Fire Data Statistics"):
        # Filter data for the selected date range
        df_filtered = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
        
        st.write("Summary statistics for fire brightness:")
        st.dataframe(df_filtered['brightness'].describe())
        
        st.write("Fire counts by date:")
        fire_counts = df_filtered.groupby('acq_date').size().reset_index(name='count')
        st.bar_chart(fire_counts.set_index('acq_date'))
