"""
Analysis of KV cache patterns across token positions.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze_token_positions(kv_cache, k_threshold=None, v_threshold=None):
    """
    Analyze KV cache patterns across token positions.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        DataFrame with token position statistics
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
    
    # Store token-wise statistics
    token_stats = []
    
    for token_idx in range(seq_len):
        # Metrics across all layers and heads for this token position
        k_sparsity_values = []
        v_sparsity_values = []
        k_magnitude_values = []
        v_magnitude_values = []
        
        for layer_idx in range(num_layers):
            keys = kv_cache[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
            values = kv_cache[layer_idx][1]
            
            # Extract data for this token position across all heads
            # Shape will be [batch_size, num_heads, head_dim]
            token_keys = keys[:, :, token_idx, :]
            token_values = values[:, :, token_idx, :]
            
            # Calculate sparsity for this token position across all heads in this layer
            k_sparsity = (token_keys.abs() < k_threshold).float().mean().item()
            v_sparsity = (token_values.abs() < v_threshold).float().mean().item()
            
            # Calculate average magnitude
            k_magnitude = token_keys.abs().mean().item()
            v_magnitude = token_values.abs().mean().item()
            
            k_sparsity_values.append(k_sparsity)
            v_sparsity_values.append(v_sparsity)
            k_magnitude_values.append(k_magnitude)
            v_magnitude_values.append(v_magnitude)
        
        # Calculate statistics across all layers for this token
        avg_k_sparsity = np.mean(k_sparsity_values)
        avg_v_sparsity = np.mean(v_sparsity_values)
        std_k_sparsity = np.std(k_sparsity_values)
        std_v_sparsity = np.std(v_sparsity_values)
        
        avg_k_magnitude = np.mean(k_magnitude_values)
        avg_v_magnitude = np.mean(v_magnitude_values)
        
        # Add to token statistics
        token_stats.append({
            "token_position": token_idx,
            "avg_k_sparsity": avg_k_sparsity,
            "avg_v_sparsity": avg_v_sparsity,
            "std_k_sparsity": std_k_sparsity,
            "std_v_sparsity": std_v_sparsity,
            "avg_k_magnitude": avg_k_magnitude,
            "avg_v_magnitude": avg_v_magnitude,
        })
    
    # Convert to dataframe
    token_df = pd.DataFrame(token_stats)
    
    return token_df

def analyze_token_layer_patterns(kv_cache, k_threshold=None, v_threshold=None):
    """
    Create a detailed heatmap of token position vs layer sparsity.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        Tuple of (key_sparsity_matrix, value_sparsity_matrix)
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
    
    # Create matrices to store sparsity values
    # Dimensions: [num_layers, seq_len] - for each layer and token position
    k_sparsity_matrix = np.zeros((num_layers, seq_len))
    v_sparsity_matrix = np.zeros((num_layers, seq_len))
    
    for layer_idx in range(num_layers):
        keys = kv_cache[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
        values = kv_cache[layer_idx][1]
        
        for token_idx in range(seq_len):
            # Extract values for this token position across all heads
            token_keys = keys[:, :, token_idx, :]  # [batch_size, num_heads, head_dim]
            token_values = values[:, :, token_idx, :]
            
            # Calculate sparsity
            k_sparsity = (token_keys.abs() < k_threshold).float().mean().item()
            v_sparsity = (token_values.abs() < v_threshold).float().mean().item()
            
            k_sparsity_matrix[layer_idx, token_idx] = k_sparsity
            v_sparsity_matrix[layer_idx, token_idx] = v_sparsity
    
    return k_sparsity_matrix, v_sparsity_matrix

def compare_generation_stages(prefill_kv_cache, decoding_kv_cache_list, k_threshold=None, v_threshold=None):
    """
    Compare KV cache patterns between prefill and different stages of decoding.
    
    Args:
        prefill_kv_cache: KV cache from prefill stage
        decoding_kv_cache_list: List of KV caches from different decoding stages
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        DataFrame with comparison metrics
    """
    # Calculate combined threshold if not provided
    if k_threshold is None or v_threshold is None:
        # Get max values across all caches
        all_k_maxes = [max([layer_kv[0].abs().max().item() for layer_kv in kv_cache]) 
                      for kv_cache in [prefill_kv_cache] + decoding_kv_cache_list]
        all_v_maxes = [max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
                      for kv_cache in [prefill_kv_cache] + decoding_kv_cache_list]
        
        global_k_max = max(all_k_maxes)
        global_v_max = max(all_v_maxes)
        
        k_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_k_max
        v_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_v_max
    
    # Store comparison statistics
    comparison_stats = []
    
    # Analyze prefill stage
    prefill_token_df = analyze_token_positions(prefill_kv_cache, k_threshold, v_threshold)
    prefill_avg_k_sparsity = prefill_token_df['avg_k_sparsity'].mean()
    prefill_avg_v_sparsity = prefill_token_df['avg_v_sparsity'].mean()
    
    comparison_stats.append({
        "stage": "prefill",
        "avg_k_sparsity": prefill_avg_k_sparsity,
        "avg_v_sparsity": prefill_avg_v_sparsity,
    })
    
    # Analyze decoding stages
    stage_names = ["early_decoding", "mid_decoding", "late_decoding"]
    if len(decoding_kv_cache_list) != len(stage_names):
        stage_names = [f"decoding_{i+1}" for i in range(len(decoding_kv_cache_list))]
        
    for i, (stage_name, decoding_kv_cache) in enumerate(zip(stage_names, decoding_kv_cache_list)):
        decoding_token_df = analyze_token_positions(decoding_kv_cache, k_threshold, v_threshold)
        decoding_avg_k_sparsity = decoding_token_df['avg_k_sparsity'].mean()
        decoding_avg_v_sparsity = decoding_token_df['avg_v_sparsity'].mean()
        
        # Get statistics for newly generated tokens only
        prefill_len = prefill_token_df.shape[0]
        if decoding_token_df.shape[0] > prefill_len:
            new_tokens_df = decoding_token_df.iloc[prefill_len:]
            new_tokens_k_sparsity = new_tokens_df['avg_k_sparsity'].mean()
            new_tokens_v_sparsity = new_tokens_df['avg_v_sparsity'].mean()
        else:
            new_tokens_k_sparsity = None
            new_tokens_v_sparsity = None
        
        comparison_stats.append({
            "stage": stage_name,
            "avg_k_sparsity": decoding_avg_k_sparsity,
            "avg_v_sparsity": decoding_avg_v_sparsity,
            "new_tokens_k_sparsity": new_tokens_k_sparsity,
            "new_tokens_v_sparsity": new_tokens_v_sparsity
        })
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison_stats)
    
    return comparison_df

def calculate_token_importance(kv_cache):
    """
    Calculate relative importance of each token position.
    
    Args:
        kv_cache: KV cache from model output
        
    Returns:
        DataFrame with token importance scores
    """
    # Extract dimensions
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Store token importance metrics
    token_importance = []
    
    for token_idx in range(seq_len):
        # Metrics across all layers and heads for this token position
        k_norms = []
        v_norms = []
        
        for layer_idx in range(num_layers):
            keys = kv_cache[layer_idx][0]
            values = kv_cache[layer_idx][1]
            
            # Extract data for this token position
            token_keys = keys[:, :, token_idx, :]
            token_values = values[:, :, token_idx, :]
            
            # Calculate norms (L2)
            k_norm = torch.norm(token_keys).item()
            v_norm = torch.norm(token_values).item()
            
            k_norms.append(k_norm)
            v_norms.append(v_norm)
        
        # Calculate average norms across layers
        avg_k_norm = np.mean(k_norms)
        avg_v_norm = np.mean(v_norms)
        
        # Calculate attention energy for this token
        attention_energy = avg_k_norm * avg_v_norm
        
        # Add to token importance
        token_importance.append({
            "token_position": token_idx,
            "avg_k_norm": avg_k_norm,
            "avg_v_norm": avg_v_norm,
            "attention_energy": attention_energy
        })
    
    # Convert to dataframe
    importance_df = pd.DataFrame(token_importance)
    
    # Normalize metrics
    for col in ["avg_k_norm", "avg_v_norm", "attention_energy"]:
        max_val = importance_df[col].max()
        if max_val > 0:  # Avoid division by zero
            importance_df[f"{col}_normalized"] = importance_df[col] / max_val
        else:
            importance_df[f"{col}_normalized"] = 0
    
    # Calculate overall importance score
    importance_df["importance_score"] = (
        importance_df["avg_k_norm_normalized"] * 0.35 + 
        importance_df["avg_v_norm_normalized"] * 0.35 +
        importance_df["attention_energy_normalized"] * 0.3
    )
    
    return importance_df.sort_values(by="importance_score", ascending=False)