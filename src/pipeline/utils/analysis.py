"""Functions for analyzing and visualizing fire prediction results."""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set matplotlib backend to non-interactive to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def save_figures(df: pd.DataFrame, output_dir: Path, group5_mode: bool = False) -> List[str]:
    """Generate and save EDA figures.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save figures
        group5_mode: Whether to use Group5 notebook settings
    
    Returns:
        List of paths to saved figures
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figure_paths = []
    
    # In Group5 mode, generate only the figures they created
    if group5_mode:
        logger.info("Generating figures in Group5 mode")
        
        # Generate correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        
        # Save figure
        corr_path = output_dir / "correlation_heatmap.png"
        plt.savefig(corr_path)
        plt.close()
        figure_paths.append(str(corr_path))
        logger.info(f"Saved correlation heatmap to {corr_path}")
        
        # Generate brightness distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df["brightness"], kde=True, bins=30)
        plt.title("Distribution of Brightness Values")
        plt.xlabel("Brightness")
        plt.ylabel("Count")
        
        # Save figure
        bright_path = output_dir / "brightness_distribution.png"
        plt.savefig(bright_path)
        plt.close()
        figure_paths.append(str(bright_path))
        logger.info(f"Saved brightness distribution to {bright_path}")
        
        return figure_paths
    
    # Regular mode - generate more comprehensive visualizations
    logger.info("Generating EDA figures")
    
    # Generate histograms for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for i, col in enumerate(numerical_cols[:10]):  # Limit to first 10 columns
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        
        # Save figure
        hist_path = output_dir / f"{col}_histogram.png"
        plt.savefig(hist_path)
        plt.close()
        figure_paths.append(str(hist_path))
    
    logger.info(f"Generated {len(figure_paths)} histogram figures")
    
    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", 
               linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    
    # Save figure
    corr_path = output_dir / "correlation_heatmap.png"
    plt.savefig(corr_path)
    plt.close()
    figure_paths.append(str(corr_path))
    logger.info(f"Generated correlation heatmap")
    
    # Generate scatter plots for top correlated features with target
    if "brightness" in df.columns:
        target = "brightness"
        correlations = df.corr()[target].sort_values(ascending=False)
        top_features = correlations.index[1:6]  # Top 5 features excluding target itself
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature, y=target, data=df, alpha=0.5)
            plt.title(f"{feature} vs {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            
            # Save figure
            scatter_path = output_dir / f"{feature}_scatter.png"
            plt.savefig(scatter_path)
            plt.close()
            figure_paths.append(str(scatter_path))
        
        logger.info(f"Generated {len(top_features)} scatter plots")
    
    return figure_paths

def plot_feature_importance(model, feature_names, output_path, group5_mode=False):
    """Plot feature importance for a tree-based model.
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        output_path: Path to save the figure
        group5_mode: Whether to use Group5 notebook settings
    """
    # Make sure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=importance_df)
        plt.title("Feature Importance")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        return importance_df
    else:
        logger.warning(f"Model does not have feature_importances_ attribute")
        return None