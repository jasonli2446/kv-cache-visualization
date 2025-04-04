"""
Visualization functions for embedding-level KV cache analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def plot_embedding_consistency(embedding_df):
    """
    Plot consistency of embedding dimensions.
    
    Args:
        embedding_df: DataFrame with embedding dimension statistics
    """
    plt.figure(figsize=(14, 10))
    
    # Sort embeddings by descending key sparsity
    sorted_df = embedding_df.sort_values("k_sparsity", ascending=False).reset_index(drop=True)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot key sparsity across embedding dimensions
    ax1.bar(range(len(sorted_df)), sorted_df["k_sparsity"], color="blue", alpha=0.7)
    ax1.set_title("Key Sparsity Across Embedding Dimensions")
    ax1.set_ylabel("Sparsity Ratio")
    ax1.set_xlabel("Embedding Dimensions (Sorted by Sparsity)")
    
    # Add reference lines
    ax1.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, 
              label="Consistently Sparse (>80%)")
    ax1.axhline(y=0.2, color="green", linestyle="--", alpha=0.7, 
              label="Consistently Dense (<20%)")
    ax1.legend()
    
    # Add dimension numbers for highly sparse/dense dimensions
    for i, row in sorted_df.iterrows():
        if row["k_sparsity"] > 0.8 or row["k_sparsity"] < 0.2:
            ax1.annotate(f"Dim {int(row['dimension'])}", 
                      (i, row["k_sparsity"]),
                      textcoords="offset points",
                      xytext=(0, 10),
                      ha='center',
                      fontsize=8,
                      rotation=90)
    
    # Plot value sparsity across embedding dimensions
    v_sorted_df = embedding_df.sort_values("v_sparsity", ascending=False).reset_index(drop=True)
    ax2.bar(range(len(v_sorted_df)), v_sorted_df["v_sparsity"], color="orange", alpha=0.7)
    ax2.set_title("Value Sparsity Across Embedding Dimensions")
    ax2.set_ylabel("Sparsity Ratio")
    ax2.set_xlabel("Embedding Dimensions (Sorted by Sparsity)")
    
    # Add reference lines
    ax2.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, 
              label="Consistently Sparse (>80%)")
    ax2.axhline(y=0.2, color="green", linestyle="--", alpha=0.7, 
              label="Consistently Dense (<20%)")
    ax2.legend()
    
    # Add dimension numbers for highly sparse/dense dimensions
    for i, row in v_sorted_df.iterrows():
        if row["v_sparsity"] > 0.8 or row["v_sparsity"] < 0.2:
            ax2.annotate(f"Dim {int(row['dimension'])}", 
                      (i, row["v_sparsity"]),
                      textcoords="offset points",
                      xytext=(0, 10),
                      ha='center',
                      fontsize=8,
                      rotation=90)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs("graphs/embeddings", exist_ok=True)
    
    plt.savefig("graphs/embeddings/embedding_consistency.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_embedding_importance_heatmap(embedding_importance_results):
    """
    Plot heatmap of embedding importance across layers.
    
    Args:
        embedding_importance_results: Dict with embedding importance analysis
    """
    k_importance_matrix = embedding_importance_results["k_importance_matrix"]
    v_importance_matrix = embedding_importance_results["v_importance_matrix"]
    dim_summary = embedding_importance_results["dim_summary"]
    
    num_layers, head_dim = k_importance_matrix.shape
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    
    # Sort dimensions by overall importance for better visualization
    top_dims = dim_summary["dimension"].values
    
    # Reorder matrices by dimension importance
    k_reordered = np.zeros_like(k_importance_matrix)
    v_reordered = np.zeros_like(v_importance_matrix)
    
    for i, dim in enumerate(top_dims):
        k_reordered[:, i] = k_importance_matrix[:, dim]
        v_reordered[:, i] = v_importance_matrix[:, dim]
    
    # Plot key importance heatmap
    sns.heatmap(k_reordered, cmap="viridis", ax=ax1, cbar_kws={"label": "Importance (1-Sparsity)"})
    ax1.set_title("Key Dimension Importance Across Layers")
    ax1.set_ylabel("Layer")
    ax1.set_xlabel("Embedding Dimensions (Ordered by Overall Importance)")
    
    # Plot value importance heatmap
    sns.heatmap(v_reordered, cmap="viridis", ax=ax2, cbar_kws={"label": "Importance (1-Sparsity)"})
    ax2.set_title("Value Dimension Importance Across Layers")
    ax2.set_ylabel("Layer")
    ax2.set_xlabel("Embedding Dimensions (Ordered by Overall Importance)")
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs("graphs/embeddings", exist_ok=True)
    
    plt.savefig("graphs/embeddings/embedding_importance_heatmap.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_sparse_dense_embedding_patterns(embedding_pattern_results, top_k=5):
    """
    Plot patterns of sparse and dense embedding dimensions across token positions.
    
    Args:
        embedding_pattern_results: Dict with embedding pattern analysis
        top_k: Number of top dimensions to highlight
    """
    k_patterns_normalized = embedding_pattern_results["k_patterns_normalized"]
    v_patterns_normalized = embedding_pattern_results["v_patterns_normalized"]
    pattern_summary = embedding_pattern_results["pattern_summary"]
    
    head_dim, seq_len = k_patterns_normalized.shape
    
    # Get top-k dimensions with highest variance (most position-sensitive)
    top_varying_dims = pattern_summary.head(top_k)["dimension"].values
    
    # Get bottom-k dimensions with lowest variance (most consistent)
    bottom_varying_dims = pattern_summary.tail(top_k)["dimension"].values
    
    # Set up the figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot for top varying key dimensions
    for dim in top_varying_dims:
        axs[0, 0].plot(range(seq_len), k_patterns_normalized[dim], 
                     label=f"Dim {dim}", alpha=0.7)
    axs[0, 0].set_title("Key Patterns: Most Position-Sensitive Dimensions")
    axs[0, 0].set_xlabel("Token Position")
    axs[0, 0].set_ylabel("Normalized Activation")
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Plot for top varying value dimensions  
    for dim in top_varying_dims:
        axs[0, 1].plot(range(seq_len), v_patterns_normalized[dim], 
                     label=f"Dim {dim}", alpha=0.7)
    axs[0, 1].set_title("Value Patterns: Most Position-Sensitive Dimensions")
    axs[0, 1].set_xlabel("Token Position")
    axs[0, 1].set_ylabel("Normalized Activation")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # Plot for least varying key dimensions
    for dim in bottom_varying_dims:
        axs[1, 0].plot(range(seq_len), k_patterns_normalized[dim], 
                     label=f"Dim {dim}", alpha=0.7)
    axs[1, 0].set_title("Key Patterns: Most Position-Invariant Dimensions")
    axs[1, 0].set_xlabel("Token Position")
    axs[1, 0].set_ylabel("Normalized Activation")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Plot for least varying value dimensions
    for dim in bottom_varying_dims:
        axs[1, 1].plot(range(seq_len), v_patterns_normalized[dim], 
                     label=f"Dim {dim}", alpha=0.7)
    axs[1, 1].set_title("Value Patterns: Most Position-Invariant Dimensions")
    axs[1, 1].set_xlabel("Token Position")
    axs[1, 1].set_ylabel("Normalized Activation")
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs("graphs/embeddings", exist_ok=True)
    
    plt.savefig("graphs/embeddings/sparse_dense_embedding_patterns.png", dpi=config.FIGURE_DPI)
    plt.close()