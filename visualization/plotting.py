"""
Main visualization functions for KV cache analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import from our visualization modules
from visualization.layer_plots import plot_layer_statistics, plot_layer_pruning_potential
from visualization.head_plots import plot_head_sparsity, plot_head_pruning_potential, plot_head_layer_heatmap
from visualization.common import plot_weight_magnitude_distribution, ensure_graph_dir

def plot_all_visualizations(layer_df, head_df, all_keys=None, all_values=None, k_threshold=None, v_threshold=None):
    """
    Create all basic visualizations (backward compatibility function).
    Consider using the individual visualization functions directly for more control.
    
    Args:
        layer_df: DataFrame with layer statistics
        head_df: DataFrame with head statistics
        all_keys: List of key tensors
        all_values: List of value tensors
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
    """
    # Ensure graphs directory exists
    ensure_graph_dir("graphs")
    
    # Layer-level visualizations
    plot_layer_statistics(layer_df)
    plot_layer_pruning_potential(layer_df)
    
    # Head-level visualizations
    plot_head_sparsity(head_df)
    plot_head_pruning_potential(head_df)
    plot_head_layer_heatmap(head_df)
    
    # Weight magnitude distributions if provided
    if all_keys is not None and all_values is not None and k_threshold is not None and v_threshold is not None:
        plot_weight_magnitude_distribution(all_keys, all_values, k_threshold, v_threshold)