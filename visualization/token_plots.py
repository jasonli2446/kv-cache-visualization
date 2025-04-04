"""
Visualization functions for token-level KV cache analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import sys
import os
import matplotlib.ticker as ticker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def plot_token_position_importance(token_importance_df, input_text=None, tokenizer=None):
    """
    Plot importance scores for each token position.
    
    Args:
        token_importance_df: DataFrame with token importance scores
        input_text: Optional input text for token labels
        tokenizer: Optional tokenizer for decoding tokens
    """
    plt.figure(figsize=(14, 8))
    
    # Sort by token position for sequential display
    sorted_df = token_importance_df.sort_values("token_position")
    
    # Calculate position as percentage
    total_tokens = len(sorted_df)
    position_percentages = [(pos / total_tokens) * 100 for pos in range(total_tokens)]
    
    # Plot importance scores
    plt.bar(position_percentages, sorted_df["importance_score"], color="teal", alpha=0.7)
    plt.xlabel("Token Position (% of sequence)")
    plt.ylabel("Importance Score")
    plt.title("Relative Importance of Each Token Position\n(Based on key/value norms and attention energy)")
    
    # Add horizontal grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add component contributions as line plots
    plt.plot(position_percentages, sorted_df["avg_k_norm_normalized"], "b--", 
            label="Key Norm (normalized)", alpha=0.5)
    plt.plot(position_percentages, sorted_df["avg_v_norm_normalized"], "r--", 
            label="Value Norm (normalized)", alpha=0.5)
    
    plt.legend()
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs("graphs/tokens", exist_ok=True)
    
    plt.savefig("graphs/tokens/token_position_importance.png", dpi=config.FIGURE_DPI)
    plt.close()

def plot_generation_stages_comparison(comparison_df, generated_text=None):
    """
    Plot comparison of sparsity across different generation stages.
    
    Args:
        comparison_df: DataFrame with generation stage comparison
        generated_text: Optional generated text to display
    """
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot key sparsity across stages
    stages = comparison_df["stage"].tolist()
    x = range(len(stages))
    
    ax1.bar(x, comparison_df["avg_k_sparsity"], color="blue", alpha=0.7)
    ax1.set_title("Key Sparsity Across Generation Stages")
    ax1.set_ylabel("Average Sparsity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add new tokens sparsity if available
    if "new_tokens_k_sparsity" in comparison_df.columns:
        new_token_values = comparison_df["new_tokens_k_sparsity"].tolist()
        # Filter out None values
        valid_indices = [i for i, val in enumerate(new_token_values) if val is not None]
        if valid_indices:
            ax1.plot([i for i in valid_indices], 
                     [new_token_values[i] for i in valid_indices], 
                     "ro-", label="New Tokens Only")
            ax1.legend()
    
    # Plot value sparsity across stages
    ax2.bar(x, comparison_df["avg_v_sparsity"], color="orange", alpha=0.7)
    ax2.set_title("Value Sparsity Across Generation Stages")
    ax2.set_xlabel("Generation Stage")
    ax2.set_ylabel("Average Sparsity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=45)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Add new tokens sparsity if available
    if "new_tokens_v_sparsity" in comparison_df.columns:
        new_token_values = comparison_df["new_tokens_v_sparsity"].tolist()
        # Filter out None values
        valid_indices = [i for i, val in enumerate(new_token_values) if val is not None]
        if valid_indices:
            ax2.plot([i for i in valid_indices], 
                     [new_token_values[i] for i in valid_indices], 
                     "ro-", label="New Tokens Only")
            ax2.legend()
    
    # Add generated text as a caption if provided
    if generated_text:
        plt.figtext(0.5, 0.01, f"Generated: {generated_text[:100]}...", 
                   wrap=True, horizontalalignment='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Create directory if it doesn't exist
    os.makedirs("graphs/tokens", exist_ok=True)
    
    plt.savefig("graphs/tokens/generation_stages_comparison.png", dpi=config.FIGURE_DPI)
    plt.close()