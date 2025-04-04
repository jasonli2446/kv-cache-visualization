"""
Visualization functions for layer-level KV cache analysis.
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
from visualization.common import ensure_graph_dir

def plot_layer_statistics(layer_df):
    """
    Plot layer-level statistics.
    
    Args:
        layer_df: DataFrame with layer statistics
    """
    plt.figure(figsize=config.DEFAULT_FIGSIZE)

    plt.subplot(2, 2, 1)
    plt.plot(layer_df["layer"], layer_df["k_sparsity"], "b-o", label="Key Sparsity")
    plt.plot(layer_df["layer"], layer_df["v_sparsity"], "r-o", label="Value Sparsity")
    plt.title("Sparsity Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Sparsity (ratio of near-zero values)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(layer_df["layer"], layer_df["kv_correlation"], "g-o")
    plt.title("Key-Value Correlation Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Correlation")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(layer_df["layer"], layer_df["k_std"], "b-o", label="Key Std")
    plt.plot(layer_df["layer"], layer_df["v_std"], "r-o", label="Value Std")
    plt.title("Standard Deviation Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(layer_df["layer"], layer_df["k_mean"], "b-o", label="Key Mean")
    plt.plot(layer_df["layer"], layer_df["v_mean"], "r-o", label="Value Mean")
    plt.title("Mean Absolute Value Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Mean")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs/layers")
    
    plt.savefig("graphs/layers/layer_statistics.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_layer_pruning_potential(layer_df):
    """
    Plot layer-wise pruning potential as a bar chart.
    
    Args:
        layer_df: DataFrame with layer statistics
    """
    # Calculate per-layer sparsity
    layer_sparsity_k = layer_df["k_sparsity"].tolist()
    layer_sparsity_v = layer_df["v_sparsity"].tolist()
    
    num_layers = len(layer_df)
    x = np.arange(num_layers)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=config.BAR_CHART_FIGSIZE)
    ax.bar(x - width/2, layer_sparsity_k, width, label='Key Sparsity')
    ax.bar(x + width/2, layer_sparsity_v, width, label='Value Sparsity')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Pruning Potential (Sparsity)')
    ax.set_title('Layer-wise Pruning Potential')
    ax.set_xticks(x)
    ax.set_xticklabels(range(num_layers))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line for average sparsity
    avg_k_sparsity = np.mean(layer_sparsity_k)
    avg_v_sparsity = np.mean(layer_sparsity_v)
    ax.axhline(y=avg_k_sparsity, color='blue', linestyle='--', alpha=0.7, 
               label=f'Avg K Sparsity: {avg_k_sparsity:.3f}')
    ax.axhline(y=avg_v_sparsity, color='orange', linestyle='--', alpha=0.7,
               label=f'Avg V Sparsity: {avg_v_sparsity:.3f}')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs/layers")
    
    plt.savefig("graphs/layers/layer_pruning_potential.png", dpi=config.FIGURE_DPI)
    plt.close()