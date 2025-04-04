"""
Visualization functions for head-level KV cache analysis.
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

def plot_head_sparsity(head_df):
    """
    Plot head-level sparsity heatmaps.
    
    Args:
        head_df: DataFrame with head statistics
    """
    # Get dimensions
    num_layers = head_df["layer"].nunique()
    num_heads = head_df["head"].nunique()
    
    plt.figure(figsize=config.HEATMAP_FIGSIZE)

    plt.subplot(2, 1, 1)
    head_k_sparsity = head_df.pivot(index="layer", columns="head", values="k_sparsity")
    sns.heatmap(head_k_sparsity, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Key Sparsity by Head and Layer")
    plt.xlabel("Head Index")
    plt.ylabel("Layer")

    plt.subplot(2, 1, 2)
    head_v_sparsity = head_df.pivot(index="layer", columns="head", values="v_sparsity")
    sns.heatmap(head_v_sparsity, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Value Sparsity by Head and Layer")
    plt.xlabel("Head Index")
    plt.ylabel("Layer")

    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs/heads")
    
    plt.savefig("graphs/heads/head_sparsity.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_head_pruning_potential(head_df):
    """
    Plot head-wise pruning potential as a bar chart.
    
    Args:
        head_df: DataFrame with head statistics
    """
    # Calculate average sparsity for each head across all layers
    head_avg_df = head_df.groupby("head").agg({
        "k_sparsity": ["mean", "std"],
        "v_sparsity": ["mean"]
    }).reset_index()
    head_avg_df.columns = ["head", "avg_k_sparsity", "std_k_sparsity", "avg_v_sparsity"]
    
    num_heads = len(head_avg_df)
    x = np.arange(num_heads)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=config.BAR_CHART_FIGSIZE)
    ax.bar(x - width/2, head_avg_df["avg_k_sparsity"], width, label='Key Sparsity', 
           yerr=head_avg_df["std_k_sparsity"], capsize=5)
    ax.bar(x + width/2, head_avg_df["avg_v_sparsity"], width, label='Value Sparsity')
    
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Average Pruning Potential (Sparsity)')
    ax.set_title('Head-wise Pruning Potential (Averaged Across All Layers)')
    ax.set_xticks(x)
    ax.set_xticklabels(range(num_heads))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs/heads")
    
    plt.savefig("graphs/heads/head_pruning_potential.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_head_layer_heatmap(head_df):
    """
    Plot detailed heatmap of pruning potential by layer and head.
    
    Args:
        head_df: DataFrame with head statistics
    """
    # Get dimensions
    num_layers = head_df["layer"].nunique()
    num_heads = head_df["head"].nunique()
    
    # Create a pivot table to organize data by layer and head
    head_k_sparsity = head_df.pivot(index="layer", columns="head", values="k_sparsity")
    
    # Plot the heatmap
    plt.figure(figsize=config.HEATMAP_FIGSIZE)
    sns.heatmap(head_k_sparsity, 
                cmap="Blues", 
                annot=True, 
                fmt=".2f", 
                xticklabels=range(num_heads),
                yticklabels=range(num_layers))
    plt.title("Pruning Potential: Sparsity by Layer and Head")
    plt.xlabel("Attention Head")
    plt.ylabel("Model Layer")
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs/heads")
    
    plt.savefig("graphs/heads/pruning_heatmap_by_layer_head.png", dpi=config.FIGURE_DPI)
    plt.close()