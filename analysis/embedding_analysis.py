"""
Analysis of KV cache at the embedding dimension level.
This module combines dimension-level analysis (individual feature dimensions within 
heads) and embedding-level analysis (patterns across layers/heads/tokens).
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

def analyze_dimensions(kv_cache, k_threshold=None, v_threshold=None, max_layers_to_analyze=5):
    """
    Analyze KV cache at the dimension level within attention heads.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        max_layers_to_analyze: Maximum number of layers to analyze (to avoid memory issues)
        
    Returns:
        DataFrame with dimension statistics and list of analyzed layers
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

def analyze_embedding_dimensions(kv_cache, k_threshold=None, v_threshold=None):
    """
    Analyze consistency of embedding dimensions across different heads and layers.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity (if None, calculated from data)
        v_threshold: Threshold for value sparsity (if None, calculated from data)
        
    Returns:
        DataFrame with embedding dimension statistics
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
    
    # Create matrices to track sparsity for each embedding dimension
    # For each dimension, we'll track sparsity across all layers, heads, and tokens
    k_embedding_sparsity = torch.zeros(head_dim)
    v_embedding_sparsity = torch.zeros(head_dim)
    
    # Count total cells for each dimension
    total_cells = num_layers * num_heads * seq_len
    
    # Track how many times each dimension is sparse
    sparse_k_counts = torch.zeros(head_dim)
    sparse_v_counts = torch.zeros(head_dim)
    
    for layer_idx in range(num_layers):
        keys = kv_cache[layer_idx][0]  # [batch_size, num_heads, seq_len, head_dim]
        values = kv_cache[layer_idx][1]
        
        # Calculate sparsity per dimension across all heads and tokens
        for dim_idx in range(head_dim):
            # Extract this dimension across all heads and tokens
            k_dim = keys[:, :, :, dim_idx]  # [batch_size, num_heads, seq_len]
            v_dim = values[:, :, :, dim_idx]
            
            # Count sparse elements
            sparse_k_counts[dim_idx] += (k_dim.abs() < k_threshold).sum().item()
            sparse_v_counts[dim_idx] += (v_dim.abs() < v_threshold).sum().item()
    
    # Calculate sparsity for each dimension
    k_embedding_sparsity = sparse_k_counts / total_cells
    v_embedding_sparsity = sparse_v_counts / total_cells
    
    # Create DataFrame with statistics
    embedding_stats = []
    
    for dim_idx in range(head_dim):
        embedding_stats.append({
            "dimension": dim_idx,
            "k_sparsity": k_embedding_sparsity[dim_idx].item(),
            "v_sparsity": v_embedding_sparsity[dim_idx].item(),
            "overall_sparsity": (k_embedding_sparsity[dim_idx].item() + v_embedding_sparsity[dim_idx].item()) / 2,
            "k_dense_ratio": 1.0 - k_embedding_sparsity[dim_idx].item(),
            "v_dense_ratio": 1.0 - v_embedding_sparsity[dim_idx].item(),
        })
    
    embedding_df = pd.DataFrame(embedding_stats)
    
    return embedding_df

def find_consistent_embeddings(kv_cache, k_threshold=None, v_threshold=None, consistency_threshold=0.8):
    """
    Find embedding dimensions that are consistently sparse or dense across contexts.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
        consistency_threshold: Threshold for determining consistency (0.8 = 80% consistent)
        
    Returns:
        Dict with consistently sparse and dense dimensions
    """
    # Get basic embedding dimension analysis
    embedding_df = analyze_embedding_dimensions(kv_cache, k_threshold, v_threshold)
    
    # Find consistently sparse dimensions (sparse in >= consistency_threshold of contexts)
    consistently_sparse_k = embedding_df[embedding_df["k_sparsity"] >= consistency_threshold]["dimension"].tolist()
    consistently_sparse_v = embedding_df[embedding_df["v_sparsity"] >= consistency_threshold]["dimension"].tolist()
    
    # Find consistently dense dimensions (dense in >= consistency_threshold of contexts)
    consistently_dense_k = embedding_df[embedding_df["k_sparsity"] <= (1 - consistency_threshold)]["dimension"].tolist()
    consistently_dense_v = embedding_df[embedding_df["v_sparsity"] <= (1 - consistency_threshold)]["dimension"].tolist()
    
    return {
        "consistently_sparse_k": consistently_sparse_k,
        "consistently_sparse_v": consistently_sparse_v,
        "consistently_dense_k": consistently_dense_k,
        "consistently_dense_v": consistently_dense_v,
        "embedding_df": embedding_df
    }

def analyze_embedding_importance_by_layer(kv_cache, k_threshold=None, v_threshold=None):
    """
    Analyze how embedding dimension importance changes across layers.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
        
    Returns:
        Dict with embedding importance analysis
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
    
    # Create a matrix to store embedding importance by layer
    # Dimensions: [num_layers, head_dim]
    k_importance_matrix = np.zeros((num_layers, head_dim))
    v_importance_matrix = np.zeros((num_layers, head_dim))
    
    for layer_idx in range(num_layers):
        keys = kv_cache[layer_idx][0]
        values = kv_cache[layer_idx][1]
        
        for dim_idx in range(head_dim):
            # Extract this dimension across all heads and tokens
            k_dim = keys[:, :, :, dim_idx]  # [batch_size, num_heads, seq_len]
            v_dim = values[:, :, :, dim_idx]
            
            # Calculate metrics (using inverse of sparsity as importance)
            k_sparsity = (k_dim.abs() < k_threshold).float().mean().item()
            v_sparsity = (v_dim.abs() < v_threshold).float().mean().item()
            
            # Store as importance (1 - sparsity)
            k_importance_matrix[layer_idx, dim_idx] = 1 - k_sparsity
            v_importance_matrix[layer_idx, dim_idx] = 1 - v_sparsity
    
    # Calculate overall importance of each dimension
    k_dim_importance = k_importance_matrix.mean(axis=0)  # Average across layers
    v_dim_importance = v_importance_matrix.mean(axis=0)
    
    # Store in DataFrames for easy processing
    k_importance_df = pd.DataFrame(k_importance_matrix, columns=[f"dim_{i}" for i in range(head_dim)])
    v_importance_df = pd.DataFrame(v_importance_matrix, columns=[f"dim_{i}" for i in range(head_dim)])
    
    # Add layer column
    k_importance_df["layer"] = range(num_layers)
    v_importance_df["layer"] = range(num_layers)
    
    # Create a summary DataFrame with overall dimension importance
    dim_summary = pd.DataFrame({
        "dimension": range(head_dim),
        "k_importance": k_dim_importance,
        "v_importance": v_dim_importance,
        "combined_importance": (k_dim_importance + v_dim_importance) / 2
    })
    
    return {
        "k_importance_by_layer": k_importance_df,
        "v_importance_by_layer": v_importance_df,
        "dim_summary": dim_summary.sort_values("combined_importance", ascending=False),
        "k_importance_matrix": k_importance_matrix,
        "v_importance_matrix": v_importance_matrix
    }

def identify_embedding_patterns(kv_cache, k_threshold=None, v_threshold=None):
    """
    Identify patterns in embedding activations across token positions.
    
    Args:
        kv_cache: KV cache from model output
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
        
    Returns:
        Dict with embedding pattern analysis
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
    
    # Create matrices to store patterns
    # For each embedding dimension, we'll look at its pattern across token positions
    # We'll average across layers and heads
    k_patterns = np.zeros((head_dim, seq_len))
    v_patterns = np.zeros((head_dim, seq_len))
    
    for dim_idx in range(head_dim):
        for token_idx in range(seq_len):
            # Collect values for this dimension and token across all layers and heads
            k_values = []
            v_values = []
            
            for layer_idx in range(num_layers):
                keys = kv_cache[layer_idx][0]
                values = kv_cache[layer_idx][1]
                
                # Extract values for this dimension and token across all heads
                k_dim_token = keys[:, :, token_idx, dim_idx]  # [batch_size, num_heads]
                v_dim_token = values[:, :, token_idx, dim_idx]
                
                # Calculate average magnitude across heads
                k_mag = k_dim_token.abs().mean().item()
                v_mag = v_dim_token.abs().mean().item()
                
                k_values.append(k_mag)
                v_values.append(v_mag)
            
            # Store average across layers
            k_patterns[dim_idx, token_idx] = np.mean(k_values)
            v_patterns[dim_idx, token_idx] = np.mean(v_values)
    
    # Normalize patterns for easier comparison
    k_patterns_normalized = k_patterns / np.max(k_patterns) if np.max(k_patterns) > 0 else k_patterns
    v_patterns_normalized = v_patterns / np.max(v_patterns) if np.max(v_patterns) > 0 else v_patterns
    
    # Calculate variance across token positions for each dimension
    # Higher variance means the dimension is more sensitive to token position
    k_variance = np.var(k_patterns, axis=1)
    v_variance = np.var(v_patterns, axis=1)
    
    # Create summary DataFrame
    pattern_summary = pd.DataFrame({
        "dimension": range(head_dim),
        "k_variance": k_variance,
        "v_variance": v_variance,
        "combined_variance": (k_variance + v_variance) / 2
    })
    
    return {
        "k_patterns": k_patterns,
        "v_patterns": v_patterns,
        "k_patterns_normalized": k_patterns_normalized,
        "v_patterns_normalized": v_patterns_normalized,
        "pattern_summary": pattern_summary.sort_values("combined_variance", ascending=False)
    }

def identify_similar_embedding_dimensions(kv_cache, similarity_threshold=0.9):
    """
    Identify embedding dimensions that have similar behavior and could potentially be grouped.
    
    Args:
        kv_cache: KV cache from model output
        similarity_threshold: Threshold for determining similarity (0.9 = 90% similar)
        
    Returns:
        Dict with grouped dimensions
    """
    # Extract dimensions
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Flatten each dimension into a feature vector
    # For each dimension, we'll create a vector of all its values across all layers, heads, tokens
    k_feature_vectors = []
    v_feature_vectors = []
    
    for dim_idx in range(head_dim):
        k_dim_values = []
        v_dim_values = []
        
        for layer_idx in range(num_layers):
            keys = kv_cache[layer_idx][0]
            values = kv_cache[layer_idx][1]
            
            # Extract values for this dimension
            k_dim = keys[:, :, :, dim_idx].flatten().cpu().numpy()
            v_dim = values[:, :, :, dim_idx].flatten().cpu().numpy()
            
            k_dim_values.append(k_dim)
            v_dim_values.append(v_dim)
        
        # Flatten and concatenate
        k_feature_vector = np.concatenate(k_dim_values)
        v_feature_vector = np.concatenate(v_dim_values)
        
        k_feature_vectors.append(k_feature_vector)
        v_feature_vectors.append(v_feature_vector)
    
    # Calculate pairwise cosine similarity between dimensions
    k_similarity_matrix = np.zeros((head_dim, head_dim))
    v_similarity_matrix = np.zeros((head_dim, head_dim))
    
    for i in range(head_dim):
        for j in range(head_dim):
            if i == j:
                k_similarity_matrix[i, j] = 1.0
                v_similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity
                k_sim = np.dot(k_feature_vectors[i], k_feature_vectors[j]) / (
                    np.linalg.norm(k_feature_vectors[i]) * np.linalg.norm(k_feature_vectors[j]) + 1e-8)
                v_sim = np.dot(v_feature_vectors[i], v_feature_vectors[j]) / (
                    np.linalg.norm(v_feature_vectors[i]) * np.linalg.norm(v_feature_vectors[j]) + 1e-8)
                
                k_similarity_matrix[i, j] = k_sim
                v_similarity_matrix[i, j] = v_sim
    
    # Find groups of similar dimensions
    k_groups = []
    v_groups = []
    
    k_visited = set()
    v_visited = set()
    
    for dim_idx in range(head_dim):
        if dim_idx not in k_visited:
            # Find similar dimensions for keys
            k_similar = [dim_idx]
            for j in range(dim_idx + 1, head_dim):
                if k_similarity_matrix[dim_idx, j] >= similarity_threshold:
                    k_similar.append(j)
                    k_visited.add(j)
            
            if len(k_similar) > 1:  # Only add groups with more than one dimension
                k_groups.append(k_similar)
            k_visited.add(dim_idx)
        
        if dim_idx not in v_visited:
            # Find similar dimensions for values
            v_similar = [dim_idx]
            for j in range(dim_idx + 1, head_dim):
                if v_similarity_matrix[dim_idx, j] >= similarity_threshold:
                    v_similar.append(j)
                    v_visited.add(j)
            
            if len(v_similar) > 1:  # Only add groups with more than one dimension
                v_groups.append(v_similar)
            v_visited.add(dim_idx)
    
    return {
        "k_similarity_matrix": k_similarity_matrix,
        "v_similarity_matrix": v_similarity_matrix,
        "k_similar_groups": k_groups,
        "v_similar_groups": v_groups
    }