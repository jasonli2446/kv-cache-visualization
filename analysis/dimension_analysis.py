"""
Analysis of KV cache at the dimension level within attention heads.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze_dimensions(kv_cache, k_threshold=None, v_threshold=None, max_layers_to_analyze=5):
    """
    Analyze KV cache at the dimension level within attention heads.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        max_layers_to_analyze: Maximum number of layers to analyze (to avoid memory issues)
        
    Returns:
        DataFrame with dimension statistics
    """
    # Calculate thresholds if not provided
    if k_threshold is None or v_threshold is None:
        global_k_max = max([layer_kv[0].abs().max().item() for layer_kv in kv_cache])
        global_v_max = max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
        
        k_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_k_max
        v_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_v_max
    
    # Extract dimensions
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # We'll analyze a subset of layers if the model is very large
    layers_to_analyze = min(num_layers, max_layers_to_analyze)
    selected_layers = list(range(0, num_layers, max(1, num_layers // layers_to_analyze)))[:layers_to_analyze]
    
    # Store statistics for each dimension in each head
    dim_stats = []

    for layer_idx in selected_layers:
        keys = kv_cache[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
        values = kv_cache[layer_idx][1]
        
        for head_idx in range(num_heads):
            # Extract data for this head
            head_keys = keys[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
            head_values = values[:, head_idx, :, :]
            
            # Analyze each dimension
            for dim in range(head_dim):
                # Extract this specific dimension across all sequence positions
                dim_k = head_keys[:, :, dim]  # [batch_size, seq_len]
                dim_v = head_values[:, :, dim]
                
                # Calculate statistics
                k_sparsity = (dim_k.abs() < k_threshold).float().mean().item()
                v_sparsity = (dim_v.abs() < v_threshold).float().mean().item()
                k_mean = dim_k.abs().mean().item()
                v_mean = dim_v.abs().mean().item()
                k_std = dim_k.std().item()
                v_std = dim_v.std().item()
                
                # Add to dimension statistics
                dim_stats.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "dimension": dim,
                    "k_sparsity": k_sparsity,
                    "v_sparsity": v_sparsity,
                    "k_mean": k_mean,
                    "v_mean": v_mean,
                    "k_std": k_std,
                    "v_std": v_std,
                })
    
    # Convert to dataframe
    dim_df = pd.DataFrame(dim_stats)
    
    return dim_df, selected_layers

def find_prunable_dimensions(dim_df, sparsity_threshold=config.PRUNABLE_DIMS_THRESHOLD):
    """
    Identify dimensions that could be pruned based on sparsity.
    
    Args:
        dim_df: DataFrame with dimension statistics
        sparsity_threshold: Minimum sparsity for a dimension to be considered prunable
        
    Returns:
        DataFrame with prunable dimensions
    """
    prunable_dims = dim_df[
        (dim_df["k_sparsity"] > sparsity_threshold) | 
        (dim_df["v_sparsity"] > sparsity_threshold)
    ].sort_values(by=["k_sparsity", "v_sparsity"], ascending=False)
    
    return prunable_dims

def get_dimension_importance(dim_df):
    """
    Calculate importance score for each dimension based on statistics.
    
    Args:
        dim_df: DataFrame with dimension statistics
    
    Returns:
        DataFrame with importance scores added
    """
    # Create a copy to avoid modifying the original
    result_df = dim_df.copy()
    
    # Calculate importance score (inverse of sparsity, higher is more important)
    result_df["k_importance"] = 1.0 - result_df["k_sparsity"]
    result_df["v_importance"] = 1.0 - result_df["v_sparsity"]
    
    # Combined importance score (weighted average)
    result_df["importance_score"] = 0.5 * result_df["k_importance"] + 0.5 * result_df["v_importance"]
    
    return result_df.sort_values("importance_score", ascending=False)

def count_prunable_parameters(prunable_dims, kv_cache):
    """
    Count how many parameters could be pruned.
    
    Args:
        prunable_dims: DataFrame with prunable dimensions
        kv_cache: KV cache from model output
    
    Returns:
        Dict with pruning statistics
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Total parameters in KV cache
    total_params = num_layers * num_heads * head_dim * 2  # *2 for both keys and values
    
    # Number of prunable parameters
    prunable_params = len(prunable_dims)
    
    return {
        "total_params": total_params,
        "prunable_params": prunable_params,
        "prunable_percentage": prunable_params / total_params * 100 if total_params > 0 else 0
    }