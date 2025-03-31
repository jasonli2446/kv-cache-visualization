"""
Visualization functions for KV cache analysis.
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
    plt.savefig("layer_statistics.png", dpi=config.FIGURE_DPI)
    plt.close()

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
    plt.savefig("head_sparsity.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_enhanced_visualizations(layer_stats, head_stats, all_keys, all_values, k_threshold, v_threshold):
    """
    Create enhanced visualizations for pruning analysis.
    
    Args:
        layer_stats: DataFrame with layer statistics
        head_stats: DataFrame with head statistics
        all_keys: List of key tensors
        all_values: List of value tensors
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
    """
    num_layers = len(all_keys)
    num_heads = all_keys[0].shape[1]
    
    # 1. HEATMAP: Sparsity Across Layers and Heads
    plt.figure(figsize=config.HEATMAP_FIGSIZE)
    
    # Calculate head-level sparsity across all layers
    head_sparsity_matrix = np.zeros((num_layers, num_heads))
    for layer_idx in range(num_layers):
        keys = all_keys[layer_idx].squeeze(0)  # [num_heads, seq_len, head_dim]
        for head_idx in range(num_heads):
            head_keys = keys[head_idx]  # [seq_len, head_dim]
            head_sparsity_matrix[layer_idx, head_idx] = (head_keys.abs() < k_threshold).float().mean().item()
    
    # Plot the heatmap
    sns.heatmap(head_sparsity_matrix, 
                cmap="Blues", 
                annot=True, 
                fmt=".2f", 
                xticklabels=range(num_heads),
                yticklabels=range(num_layers))
    plt.title("Pruning Potential: Sparsity by Layer and Head")
    plt.xlabel("Attention Head")
    plt.ylabel("Model Layer")
    plt.tight_layout()
    plt.savefig("pruning_heatmap_by_layer_head.png", dpi=config.FIGURE_DPI)
    plt.close()
    
    # 2. BAR CHART: Layer-wise Pruning Potential
    plt.figure(figsize=config.BAR_CHART_FIGSIZE)
    
    # Calculate per-layer sparsity
    layer_sparsity_k = layer_stats["k_sparsity"].tolist()
    layer_sparsity_v = layer_stats["v_sparsity"].tolist()
    
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
    plt.savefig("layer_pruning_potential.png", dpi=config.FIGURE_DPI)
    plt.close()
    
    # 3. GROUPED BAR CHART: Head-wise Pruning Potential
    # Calculate average sparsity for each head across all layers
    head_avg_df = head_stats.groupby("head").agg({
        "k_sparsity": ["mean", "std"],
        "v_sparsity": ["mean"]
    }).reset_index()
    head_avg_df.columns = ["head", "avg_k_sparsity", "std_k_sparsity", "avg_v_sparsity"]
    
    plt.figure(figsize=config.BAR_CHART_FIGSIZE)
    
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
    plt.savefig("head_pruning_potential.png", dpi=config.FIGURE_DPI)
    plt.close()
    
    # 4. HISTOGRAM: Distribution of Weight Magnitudes
    plt.figure(figsize=config.DEFAULT_FIGSIZE)
    
    # FIX: Move tensors to CPU before converting to NumPy
    all_key_weights = np.concatenate([k.reshape(-1).cpu().numpy() for k in all_keys])
    all_value_weights = np.concatenate([v.reshape(-1).cpu().numpy() for v in all_values])
    
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(all_key_weights), bins=50, alpha=0.7, color='blue')
    plt.axvline(x=k_threshold, color='red', linestyle='--', 
                label=f'Pruning Threshold: {k_threshold:.6f}')
    plt.title('Key Weight Magnitude Distribution')
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(all_value_weights), bins=50, alpha=0.7, color='orange')
    plt.axvline(x=v_threshold, color='red', linestyle='--', 
                label=f'Pruning Threshold: {v_threshold:.6f}')
    plt.title('Value Weight Magnitude Distribution')
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("weight_magnitude_distribution.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_all_visualizations(layer_df, head_df, all_keys=None, all_values=None, k_threshold=None, v_threshold=None):
    """
    Create all visualizations.
    
    Args:
        layer_df: DataFrame with layer statistics
        head_df: DataFrame with head statistics
        all_keys: List of key tensors
        all_values: List of value tensors
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
    """
    # Basic visualizations
    plot_layer_statistics(layer_df)
    plot_head_sparsity(head_df)
    
    # Enhanced visualizations if all data provided
    if all_keys is not None and all_values is not None and k_threshold is not None and v_threshold is not None:
        plot_enhanced_visualizations(layer_df, head_df, all_keys, all_values, k_threshold, v_threshold)