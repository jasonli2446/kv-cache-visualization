"""
Analysis of KV cache at the attention head level.
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

def analyze_heads(kv_cache, k_threshold=None, v_threshold=None):
    """
    Analyze KV cache at the attention head level.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        DataFrame with head statistics
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
    
    # Store head-wise statistics
    head_stats = []
    
    for layer_idx in range(num_layers):
        keys = kv_cache[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
        values = kv_cache[layer_idx][1]
        
        for head_idx in range(num_heads):
            # Extract data for this head
            head_keys = keys[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
            head_values = values[:, head_idx, :, :]
            
            # Calculate statistics
            k_sparsity = (head_keys.abs() < k_threshold).float().mean().item()
            v_sparsity = (head_values.abs() < v_threshold).float().mean().item()
            k_mean = head_keys.abs().mean().item()
            v_mean = head_values.abs().mean().item()
            k_std = head_keys.std().item()
            v_std = head_values.std().item()
            
            # Add to head statistics
            head_stats.append({
                "layer": layer_idx,
                "head": head_idx,
                "k_sparsity": k_sparsity,
                "v_sparsity": v_sparsity,
                "k_mean": k_mean,
                "v_mean": v_mean,
                "k_std": k_std,
                "v_std": v_std,
            })
    
    # Convert to dataframe
    head_df = pd.DataFrame(head_stats)
    
    return head_df

def find_prunable_heads(head_df, sparsity_threshold=config.PRUNABLE_HEADS_THRESHOLD):
    """
    Identify heads that could be pruned based on sparsity.
    
    Args:
        head_df: DataFrame with head statistics
        sparsity_threshold: Minimum sparsity for a head to be considered prunable
        
    Returns:
        DataFrame with prunable heads
    """
    prunable_heads = head_df[
        (head_df["k_sparsity"] > sparsity_threshold) | 
        (head_df["v_sparsity"] > sparsity_threshold)
    ].sort_values(by=["k_sparsity", "v_sparsity"], ascending=False)
    
    return prunable_heads

def analyze_head_consistency(head_df):
    """
    Analyze consistency of head behavior across layers.
    
    Args:
        head_df: DataFrame with head statistics
        
    Returns:
        DataFrame with head consistency metrics
    """
    # Calculate the consistency of head sparsity across layers
    head_consistency = []
    num_heads = head_df["head"].nunique()

    for head_idx in range(num_heads):
        # Get all layers for this head
        head_layers = head_df[head_df["head"] == head_idx]
        
        # Calculate statistics
        k_sparsity_mean = head_layers["k_sparsity"].mean()
        k_sparsity_std = head_layers["k_sparsity"].std()
        v_sparsity_mean = head_layers["v_sparsity"].mean()
        v_sparsity_std = head_layers["v_sparsity"].std()
        
        # Get most and least sparse layers for this head
        max_k_layer = head_layers.loc[head_layers["k_sparsity"].idxmax()]["layer"]
        min_k_layer = head_layers.loc[head_layers["k_sparsity"].idxmin()]["layer"]
        
        head_consistency.append({
            "head": head_idx,
            "k_sparsity_mean": k_sparsity_mean,
            "k_sparsity_std": k_sparsity_std,
            "v_sparsity_mean": v_sparsity_mean,
            "v_sparsity_std": v_sparsity_std,
            "max_k_sparse_layer": max_k_layer,
            "min_k_sparse_layer": min_k_layer
        })

    # Create a dataframe
    consistency_df = pd.DataFrame(head_consistency)
    
    return consistency_df.sort_values(by="k_sparsity_std")