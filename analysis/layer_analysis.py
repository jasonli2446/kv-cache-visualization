"""
Analysis of KV cache at the layer level.
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

def analyze_layers(kv_cache, k_threshold=None, v_threshold=None):
    """
    Analyze KV cache at the layer level.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        DataFrame with layer statistics
    """
    # Calculate thresholds if not provided
    if k_threshold is None or v_threshold is None:
        global_k_max = max([layer_kv[0].abs().max().item() for layer_kv in kv_cache])
        global_v_max = max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
        
        k_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_k_max
        v_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_v_max
    
    # Store layer-wise statistics
    layer_stats = []
    
    all_keys = []
    all_values = []

    for layer_idx, layer_kv in enumerate(kv_cache):
        keys, values = layer_kv
        all_keys.append(keys)
        all_values.append(values)
        
        # Calculate statistics
        k_sparsity = (keys.abs() < k_threshold).float().mean().item()
        v_sparsity = (values.abs() < v_threshold).float().mean().item()
        k_mean = keys.abs().mean().item()
        v_mean = values.abs().mean().item()
        k_std = keys.std().item()
        v_std = values.std().item()
        
        # Calculate correlation between keys and values
        # Reshape to 2D for correlation calculation
        k_flat = keys.reshape(-1)
        v_flat = values.reshape(-1)
        
        # Subsample if too large to calculate correlation efficiently
        max_samples = 10000
        if k_flat.shape[0] > max_samples:
            indices = torch.randperm(k_flat.shape[0])[:max_samples]
            k_flat = k_flat[indices]
            v_flat = v_flat[indices]
        
        # Calculate correlation
        k_flat = k_flat - k_flat.mean()
        v_flat = v_flat - v_flat.mean()
        kv_correlation = (k_flat * v_flat).mean() / (k_flat.std() * v_flat.std() + 1e-8)
        
        # Store statistics
        layer_stats.append({
            "layer": layer_idx,
            "k_sparsity": k_sparsity,
            "v_sparsity": v_sparsity,
            "k_mean": k_mean,
            "v_mean": v_mean,
            "k_std": k_std,
            "v_std": v_std,
            "kv_correlation": kv_correlation.item()
        })

    # Convert to dataframe
    layer_df = pd.DataFrame(layer_stats)
    
    return layer_df

def find_prunable_layers(layer_df, sparsity_threshold=config.PRUNABLE_HEADS_THRESHOLD):
    """
    Identify layers that could be pruned based on sparsity.
    
    Args:
        layer_df: DataFrame with layer statistics
        sparsity_threshold: Minimum sparsity for a layer to be considered prunable
        
    Returns:
        List of prunable layer indices
    """
    prunable_layers = layer_df[
        (layer_df["k_sparsity"] > sparsity_threshold) | 
        (layer_df["v_sparsity"] > sparsity_threshold)
    ].sort_values(by=["k_sparsity", "v_sparsity"], ascending=False)
    
    return prunable_layers["layer"].astype(int).tolist()

def analyze_layer_importance(kv_cache):
    """
    Analyze importance of each layer in the KV cache.
    
    Args:
        kv_cache: KV cache from model output
        
    Returns:
        DataFrame with layer importance metrics
    """
    layer_importance = []
    num_layers = len(kv_cache)
    
    # Calculate metrics that indicate layer importance
    for layer_idx, layer_kv in enumerate(kv_cache):
        keys, values = layer_kv
        
        # Key-Value norms
        k_norm = torch.norm(keys).item()
        v_norm = torch.norm(values).item()
        
        # Calculate attention energy
        attn_energy = torch.sum(keys * values).item()
        
        # Relative position in model
        relative_position = layer_idx / (num_layers - 1)
        
        layer_importance.append({
            "layer": layer_idx,
            "k_norm": k_norm,
            "v_norm": v_norm,
            "attn_energy": attn_energy,
            "relative_position": relative_position
        })
    
    # Convert to dataframe
    importance_df = pd.DataFrame(layer_importance)
    
    # Normalize metrics
    for col in ["k_norm", "v_norm", "attn_energy"]:
        max_val = importance_df[col].max()
        importance_df[f"{col}_normalized"] = importance_df[col] / max_val
    
    # Calculate overall importance score
    importance_df["importance_score"] = (
        importance_df["k_norm_normalized"] * 0.4 + 
        importance_df["v_norm_normalized"] * 0.4 +
        importance_df["relative_position"] * 0.2
    )
    
    return importance_df.sort_values(by="importance_score", ascending=False)