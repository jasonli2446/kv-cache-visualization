"""
Analysis of KV cache for similarity across different dimensions.
Identifies groups of similar values that could potentially be clustered or compressed.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import sys
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze_layer_similarity(kv_cache, similarity_threshold=0.8):
    """
    Analyze similarity between different layers in the KV cache.
    
    Args:
        kv_cache: KV cache from model output
        similarity_threshold: Threshold for determining similarity (0.95 = 95% similar)
        
    Returns:
        Dict with similarity matrices and layer groups
    """
    num_layers = len(kv_cache)
    
    # Create feature vectors for each layer by flattening both keys and values
    layer_vectors = []
    
    for layer_idx in range(num_layers):
        keys, values = kv_cache[layer_idx]
        
        # Flatten and combine keys and values to create a single feature vector for this layer
        # This captures the full layer activation pattern across all heads and tokens
        combined = torch.cat([keys.flatten(), values.flatten()]).cpu().numpy()
        
        # Sample if extremely large
        max_samples = 50000
        if len(combined) > max_samples:
            indices = np.random.choice(len(combined), max_samples, replace=False)
            combined = combined[indices]
            
        layer_vectors.append(combined)
    
    # Calculate pairwise cosine similarity between all layers
    similarity_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity
                sim = np.dot(layer_vectors[i], layer_vectors[j]) / (
                    np.linalg.norm(layer_vectors[i]) * np.linalg.norm(layer_vectors[j]) + 1e-8)
                similarity_matrix[i, j] = sim
    
    # Find groups of similar layers
    similar_groups = []
    visited = set()
    
    for i in range(num_layers):
        if i not in visited:
            group = [i]
            for j in range(i+1, num_layers):
                if similarity_matrix[i, j] >= similarity_threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:  # Only add groups with more than one layer
                similar_groups.append(group)
            visited.add(i)
    
    # Split into key and value groups for consistency with previous code
    k_similar_groups = similar_groups.copy()
    v_similar_groups = similar_groups.copy()
    
    return {
        "similarity_matrix": similarity_matrix,
        "k_similar_groups": k_similar_groups,
        "v_similar_groups": v_similar_groups
    }

def analyze_head_similarity(kv_cache, similarity_threshold=0.75):
    """
    Analyze similarity between attention heads across the model.
    
    Args:
        kv_cache: KV cache from model output
        similarity_threshold: Threshold for determining similarity
        
    Returns:
        Dict with similarity matrices, head groups, and hierarchical clustering
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Create feature vectors for each head in each layer
    head_vectors = []  # Will be a list of (layer_idx, head_idx, feature_vector)
    head_indices = []  # Will be a list of (layer_idx, head_idx) tuples
    
    for layer_idx in range(num_layers):
        keys, values = kv_cache[layer_idx]
        
        for head_idx in range(num_heads):
            # Extract this head
            head_keys = keys[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
            head_values = values[:, head_idx, :, :]
            
            # Flatten and combine to create feature vector
            combined = torch.cat([head_keys.flatten(), head_values.flatten()]).cpu().numpy()
            
            # Sample if too large
            max_samples = 10000
            if len(combined) > max_samples:
                indices = np.random.choice(len(combined), max_samples, replace=False)
                combined = combined[indices]
            
            head_vectors.append(combined)
            head_indices.append((layer_idx, head_idx))
    
    total_heads = len(head_vectors)
    
    # Calculate pairwise distance matrix (1 - cosine similarity)
    distance_matrix = np.zeros((total_heads, total_heads))
    
    for i in range(total_heads):
        for j in range(total_heads):
            if i == j:
                distance_matrix[i, j] = 0.0  # Zero distance to self
            else:
                # Calculate cosine similarity
                sim = np.dot(head_vectors[i], head_vectors[j]) / (
                    np.linalg.norm(head_vectors[i]) * np.linalg.norm(head_vectors[j]) + 1e-8)
                # Convert to distance: 1 - cosine similarity
                distance_matrix[i, j] = 1.0 - sim
    
    # Perform hierarchical clustering using the distance matrix
    condensed_dist = squareform(distance_matrix)  # Convert to condensed form
    linkage_matrix = hierarchy.linkage(condensed_dist, method='average')
    
    # Find groups at similarity threshold (distance = 1 - similarity)
    clusters = hierarchy.fcluster(linkage_matrix, 1.0 - similarity_threshold, criterion='distance')
    
    # Group heads by cluster
    cluster_map = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(head_indices[idx])
    
    # Convert to the format expected by previous code
    k_similar_groups = []
    v_similar_groups = []
    for cluster_id, heads in cluster_map.items():
        if len(heads) > 1:  # Only include clusters with multiple heads
            k_similar_groups.append(heads)
            v_similar_groups.append(heads)
    
    # Create dataframes for the groups
    k_groups_df = pd.DataFrame([(group_idx, layer_idx, head_idx) 
                             for group_idx, group in enumerate(k_similar_groups)
                             for layer_idx, head_idx in group],
                            columns=['group_id', 'layer', 'head'])
    
    v_groups_df = pd.DataFrame([(group_idx, layer_idx, head_idx) 
                             for group_idx, group in enumerate(v_similar_groups)
                             for layer_idx, head_idx in group],
                            columns=['group_id', 'layer', 'head'])
    
    return {
        "distance_matrix": distance_matrix,
        "linkage_matrix": linkage_matrix,
        "head_indices": head_indices,
        "k_similar_groups": k_similar_groups,
        "v_similar_groups": v_similar_groups,
        "k_groups_df": k_groups_df,
        "v_groups_df": v_groups_df,
        "clusters": clusters
    }

def analyze_embedding_dimension_correlation(kv_cache):
    """
    Analyze correlation between embedding dimensions.
    
    Args:
        kv_cache: KV cache from model output
        
    Returns:
        Dict with correlation matrices
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Collect activations for each dimension across all tokens, heads and layers
    k_dim_activations = [[] for _ in range(head_dim)]
    v_dim_activations = [[] for _ in range(head_dim)]
    
    # Collect samples across layers, heads, and tokens
    for layer_idx in range(num_layers):
        keys, values = kv_cache[layer_idx]
        
        # For each embedding dimension, collect values across batch, heads, seq_len
        for dim_idx in range(head_dim):
            # Reshape to get all activations for this dimension
            k_dim_vals = keys[:, :, :, dim_idx].flatten().cpu().numpy()
            v_dim_vals = values[:, :, :, dim_idx].flatten().cpu().numpy()
            
            # Sample if very large
            max_samples = 10000
            if len(k_dim_vals) > max_samples:
                indices = np.random.choice(len(k_dim_vals), max_samples, replace=False)
                k_dim_vals = k_dim_vals[indices]
                v_dim_vals = v_dim_vals[indices]
            
            k_dim_activations[dim_idx].extend(k_dim_vals)
            v_dim_activations[dim_idx].extend(v_dim_vals)
    
    # Convert lists to arrays
    k_dim_activations = [np.array(vals) for vals in k_dim_activations]
    v_dim_activations = [np.array(vals) for vals in v_dim_activations]
    
    # Calculate Pearson correlation matrices
    k_corr_matrix = np.zeros((head_dim, head_dim))
    v_corr_matrix = np.zeros((head_dim, head_dim))
    combined_corr_matrix = np.zeros((head_dim, head_dim))
    
    # Calculate correlation for each pair of dimensions
    for i in range(head_dim):
        for j in range(head_dim):
            if i == j:
                k_corr_matrix[i, j] = 1.0
                v_corr_matrix[i, j] = 1.0
                combined_corr_matrix[i, j] = 1.0
            else:
                # Calculate Pearson correlation
                k_corr = np.corrcoef(k_dim_activations[i], k_dim_activations[j])[0, 1]
                v_corr = np.corrcoef(v_dim_activations[i], v_dim_activations[j])[0, 1]
                
                # Handle NaN values
                k_corr = 0.0 if np.isnan(k_corr) else k_corr
                v_corr = 0.0 if np.isnan(v_corr) else v_corr
                
                k_corr_matrix[i, j] = k_corr
                v_corr_matrix[i, j] = v_corr
                combined_corr_matrix[i, j] = (k_corr + v_corr) / 2
    
    # Find groups of highly correlated dimensions
    threshold = 0.75
    k_similar_groups = []
    v_similar_groups = []
    
    # For key dimensions
    visited = set()
    for i in range(head_dim):
        if i not in visited:
            group = [i]
            for j in range(i+1, head_dim):
                if abs(k_corr_matrix[i, j]) >= threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:  # Only add groups with more than one dimension
                k_similar_groups.append(group)
            visited.add(i)
    
    # For value dimensions
    visited = set()
    for i in range(head_dim):
        if i not in visited:
            group = [i]
            for j in range(i+1, head_dim):
                if abs(v_corr_matrix[i, j]) >= threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:  # Only add groups with more than one dimension
                v_similar_groups.append(group)
            visited.add(i)
    
    return {
        "k_correlation": k_corr_matrix,
        "v_correlation": v_corr_matrix,
        "combined_correlation": combined_corr_matrix,
        "k_similar_groups": k_similar_groups,
        "v_similar_groups": v_similar_groups
    }

def analyze_token_similarity(kv_cache, max_clusters=8, sample_tokens=None):
    """
    Analyze similarity between token positions using clustering.
    
    Args:
        kv_cache: KV cache from model output
        max_clusters: Maximum number of clusters to try
        sample_tokens: Optional subset of token positions to analyze
        
    Returns:
        Dict with token clustering results
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Limit token analysis if sequence is very long
    if sample_tokens is None and seq_len > 50:
        sample_tokens = np.random.choice(seq_len, 50, replace=False)
    elif sample_tokens is None:
        sample_tokens = range(seq_len)
    
    # Create feature vectors for each token position
    k_token_vectors = []
    v_token_vectors = []
    
    for token_idx in sample_tokens:
        # Collect features across layers and heads
        k_features = []
        v_features = []
        
        for layer_idx in range(num_layers):
            keys, values = kv_cache[layer_idx]
            
            # Extract data for this token position across all heads
            token_keys = keys[:, :, token_idx, :]  # [batch_size, num_heads, head_dim]
            token_values = values[:, :, token_idx, :]
            
            # Get average representation across heads
            k_avg = token_keys.mean(dim=1).cpu().numpy().flatten()  # Average across heads
            v_avg = token_values.mean(dim=1).cpu().numpy().flatten()
            
            k_features.append(k_avg)
            v_features.append(v_avg)
        
        # Concatenate features across layers
        k_token_vector = np.concatenate(k_features)
        v_token_vector = np.concatenate(v_features)
        
        k_token_vectors.append((token_idx, k_token_vector))
        v_token_vectors.append((token_idx, v_token_vector))
    
    # Extract just the feature vectors for clustering
    k_features_array = np.array([vec for _, vec in k_token_vectors])
    v_features_array = np.array([vec for _, vec in v_token_vectors])
    
    # Find optimal number of clusters using silhouette analysis
    from sklearn.metrics import silhouette_score
    
    k_silhouette_scores = []
    v_silhouette_scores = []
    
    # Try different numbers of clusters if we have enough tokens
    min_clusters = min(2, len(sample_tokens) - 1)
    max_clusters = min(max_clusters, len(sample_tokens) - 1)
    
    if max_clusters >= min_clusters and len(sample_tokens) >= 3:
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            # Keys clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            k_clusters = kmeans.fit_predict(k_features_array)
            try:
                k_score = silhouette_score(k_features_array, k_clusters)
                k_silhouette_scores.append((n_clusters, k_score))
            except:
                pass
            
            # Values clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            v_clusters = kmeans.fit_predict(v_features_array)
            try:
                v_score = silhouette_score(v_features_array, v_clusters)
                v_silhouette_scores.append((n_clusters, v_score))
            except:
                pass
    
    # Find best number of clusters
    k_best_n_clusters = max(min_clusters, len(sample_tokens) // 4)  # Default
    v_best_n_clusters = max(min_clusters, len(sample_tokens) // 4)  # Default
    
    if k_silhouette_scores:
        k_best_n_clusters = max(k_silhouette_scores, key=lambda x: x[1])[0]
    if v_silhouette_scores:
        v_best_n_clusters = max(v_silhouette_scores, key=lambda x: x[1])[0]
    
    # Final clustering
    k_kmeans = KMeans(n_clusters=k_best_n_clusters, random_state=42)
    v_kmeans = KMeans(n_clusters=v_best_n_clusters, random_state=42)
    
    k_clusters = k_kmeans.fit_predict(k_features_array)
    v_clusters = v_kmeans.fit_predict(v_features_array)
    
    # Create dataframes for token clusters
    token_positions = [pos for pos, _ in k_token_vectors]
    
    k_clusters_df = pd.DataFrame({
        'token_position': token_positions,
        'cluster': k_clusters
    })
    
    v_clusters_df = pd.DataFrame({
        'token_position': token_positions,
        'cluster': v_clusters
    })
    
    # Calculate cluster statistics
    k_cluster_stats = []
    for cluster_id in range(k_best_n_clusters):
        cluster_tokens = k_clusters_df[k_clusters_df['cluster'] == cluster_id]['token_position'].tolist()
        cluster_size = len(cluster_tokens)
        
        # Get cluster centroid
        centroid = k_kmeans.cluster_centers_[cluster_id]
        
        # Calculate average distance to centroid
        cluster_vectors = k_features_array[k_clusters == cluster_id]
        avg_distance = np.mean([np.linalg.norm(vec - centroid) for vec in cluster_vectors]) if cluster_size > 0 else 0
        
        k_cluster_stats.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'token_positions': cluster_tokens,
            'avg_distance': avg_distance
        })
    
    v_cluster_stats = []
    for cluster_id in range(v_best_n_clusters):
        cluster_tokens = v_clusters_df[v_clusters_df['cluster'] == cluster_id]['token_position'].tolist()
        cluster_size = len(cluster_tokens)
        
        # Get cluster centroid
        centroid = v_kmeans.cluster_centers_[cluster_id]
        
        # Calculate average distance to centroid
        cluster_vectors = v_features_array[v_clusters == cluster_id]
        avg_distance = np.mean([np.linalg.norm(vec - centroid) for vec in cluster_vectors]) if cluster_size > 0 else 0
        
        v_cluster_stats.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'token_positions': cluster_tokens,
            'avg_distance': avg_distance
        })
    
    return {
        "k_clusters_df": k_clusters_df,
        "v_clusters_df": v_clusters_df,
        "k_cluster_stats": pd.DataFrame(k_cluster_stats),
        "v_cluster_stats": pd.DataFrame(v_cluster_stats),
        "k_best_n_clusters": k_best_n_clusters,
        "v_best_n_clusters": v_best_n_clusters,
        "k_silhouette_scores": k_silhouette_scores,
        "v_silhouette_scores": v_silhouette_scores
    }

def compute_token_similarity_matrix(kv_cache, max_tokens=100):
    """
    Compute pairwise similarity between tokens.
    
    Args:
        kv_cache: KV cache from model output
        max_tokens: Maximum number of tokens to analyze (for performance)
        
    Returns:
        Dictionary with token similarity matrices
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Limit token count if sequence is very long
    if seq_len > max_tokens:
        token_indices = np.linspace(0, seq_len-1, max_tokens, dtype=int)
    else:
        token_indices = range(seq_len)
    
    token_count = len(token_indices)
    
    # Create feature vectors for each token by averaging across layers and heads
    k_token_vectors = []
    v_token_vectors = []
    
    for token_idx in token_indices:
        # Collect features for this token across layers
        k_features = torch.zeros(num_layers, num_heads, head_dim)
        v_features = torch.zeros(num_layers, num_heads, head_dim)
        
        for layer_idx in range(num_layers):
            keys, values = kv_cache[layer_idx]
            
            # Extract data for this token
            k_features[layer_idx] = keys[0, :, token_idx, :]  # [num_heads, head_dim]
            v_features[layer_idx] = values[0, :, token_idx, :]
        
        # Average across layers and heads to get token representation
        k_avg = k_features.mean(dim=(0, 1)).cpu().numpy()  # [head_dim]
        v_avg = v_features.mean(dim=(0, 1)).cpu().numpy()
        
        # Combined representation (keys and values)
        combined = np.concatenate([k_avg, v_avg])
        
        k_token_vectors.append(k_avg)
        v_token_vectors.append(v_avg)
    
    # Compute similarity matrices
    k_similarity = np.zeros((token_count, token_count))
    v_similarity = np.zeros((token_count, token_count))
    combined_similarity = np.zeros((token_count, token_count))
    
    for i in range(token_count):
        for j in range(token_count):
            if i == j:
                k_similarity[i, j] = 1.0
                v_similarity[i, j] = 1.0
                combined_similarity[i, j] = 1.0
            else:
                # Compute cosine similarity
                k_sim = np.dot(k_token_vectors[i], k_token_vectors[j]) / (
                    np.linalg.norm(k_token_vectors[i]) * np.linalg.norm(k_token_vectors[j]) + 1e-8)
                v_sim = np.dot(v_token_vectors[i], v_token_vectors[j]) / (
                    np.linalg.norm(v_token_vectors[i]) * np.linalg.norm(v_token_vectors[j]) + 1e-8)
                
                k_similarity[i, j] = k_sim
                v_similarity[i, j] = v_sim
                combined_similarity[i, j] = (k_sim + v_sim) / 2
    
    return {
        "token_indices": token_indices,
        "k_similarity": k_similarity,
        "v_similarity": v_similarity,
        "combined_similarity": combined_similarity
    }

def find_compressible_patterns(kv_cache):
    """
    Find patterns in KV cache that could be compressed or clustered.
    """
    layer_similarity = analyze_layer_similarity(kv_cache)
    head_similarity = analyze_head_similarity(kv_cache)
    token_similarity = analyze_token_similarity(kv_cache)
    token_similarity_matrix = compute_token_similarity_matrix(kv_cache)
    embedding_similarity = analyze_embedding_dimension_correlation(kv_cache)
    token_embedding_similarity = analyze_token_embedding_similarity(kv_cache)
    token_embedding_compression = identify_token_embedding_compression(token_embedding_similarity)
    
    # Calculate compression benefits
    compression_benefits = {}
    
    # Layer-level compression
    k_layer_groups, v_layer_groups = layer_similarity['k_similar_groups'], layer_similarity['v_similar_groups']
    compression_benefits['k_layer_savings'] = sum([len(group) - 1 for group in k_layer_groups]) if k_layer_groups else 0
    compression_benefits['v_layer_savings'] = sum([len(group) - 1 for group in v_layer_groups]) if v_layer_groups else 0
    
    # Head-level compression
    k_head_groups, v_head_groups = head_similarity['k_similar_groups'], head_similarity['v_similar_groups']
    compression_benefits['k_head_savings'] = sum([len(group) - 1 for group in k_head_groups]) if k_head_groups else 0
    compression_benefits['v_head_savings'] = sum([len(group) - 1 for group in v_head_groups]) if v_head_groups else 0
    
    # Token-level compression
    k_token_clusters = token_similarity["k_best_n_clusters"]
    v_token_clusters = token_similarity["v_best_n_clusters"]
    total_tokens = len(token_similarity["k_clusters_df"])
    compression_benefits['k_token_savings'] = total_tokens - k_token_clusters if total_tokens > 0 else 0
    compression_benefits['v_token_savings'] = total_tokens - v_token_clusters if total_tokens > 0 else 0
    
    # Embedding dimension compression
    k_dim_groups, v_dim_groups = embedding_similarity['k_similar_groups'], embedding_similarity['v_similar_groups']
    compression_benefits['k_dim_savings'] = sum([len(group) - 1 for group in k_dim_groups]) if k_dim_groups else 0
    compression_benefits['v_dim_savings'] = sum([len(group) - 1 for group in v_dim_groups]) if v_dim_groups else 0
    
    # Add token-embedding compression benefits
    compression_benefits['token_emb_potential'] = token_embedding_compression['compression']['compression_potential']
    
    return {
        "layer_similarity": layer_similarity,
        "head_similarity": head_similarity,
        "token_similarity": token_similarity,
        "token_similarity_matrix": token_similarity_matrix,
        "embedding_similarity": embedding_similarity,
        "compression_benefits": compression_benefits,
        "token_embedding_similarity": token_embedding_similarity,
        "token_embedding_compression": token_embedding_compression
    }

def analyze_token_embedding_similarity(kv_cache, max_tokens=100, sample_dims=None):
    """
    Analyze similarity patterns between token positions and embedding dimensions.
    
    Args:
        kv_cache: KV cache from model output
        max_tokens: Maximum number of tokens to analyze
        sample_dims: Number of embedding dimensions to sample (if None, use all)
        
    Returns:
        Dict with token-embedding similarity analysis
    """
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    # Limit tokens if sequence is very long
    if seq_len > max_tokens:
        token_indices = np.linspace(0, seq_len-1, max_tokens, dtype=int)
    else:
        token_indices = range(seq_len)
    
    # Optionally sample embedding dimensions
    if sample_dims and sample_dims < head_dim:
        emb_indices = np.random.choice(head_dim, sample_dims, replace=False)
    else:
        emb_indices = range(head_dim)
    
    # Create token-embedding activation matrices [tokens × embedding_dims]
    k_token_emb = np.zeros((len(token_indices), len(emb_indices)))
    v_token_emb = np.zeros((len(token_indices), len(emb_indices)))
    
    # Fill matrices with average activation values
    for t_idx, token_idx in enumerate(token_indices):
        for e_idx, emb_idx in enumerate(emb_indices):
            # Collect values across layers and heads
            k_values = []
            v_values = []
            
            for layer_idx in range(num_layers):
                keys, values = kv_cache[layer_idx]
                
                # Extract values for this token and embedding dimension
                k_layer_values = keys[0, :, token_idx, emb_idx].cpu().numpy()
                v_layer_values = values[0, :, token_idx, emb_idx].cpu().numpy()
                
                k_values.append(np.mean(np.abs(k_layer_values)))
                v_values.append(np.mean(np.abs(v_layer_values)))
            
            # Store average values across layers
            k_token_emb[t_idx, e_idx] = np.mean(k_values)
            v_token_emb[t_idx, e_idx] = np.mean(v_values)
    
    # Normalize matrices
    k_norm = k_token_emb / np.max(k_token_emb) if np.max(k_token_emb) > 0 else k_token_emb
    v_norm = v_token_emb / np.max(v_token_emb) if np.max(v_token_emb) > 0 else v_token_emb
    
    # Find patterns through clustering
    from sklearn.cluster import KMeans
    
    # Cluster tokens based on embedding patterns
    k_token_clusters = KMeans(n_clusters=min(5, len(token_indices)), random_state=42).fit_predict(k_norm)
    v_token_clusters = KMeans(n_clusters=min(5, len(token_indices)), random_state=42).fit_predict(v_norm)
    
    # Cluster embeddings based on token patterns
    k_emb_clusters = KMeans(n_clusters=min(5, len(emb_indices)), random_state=42).fit_predict(k_norm.T)
    v_emb_clusters = KMeans(n_clusters=min(5, len(emb_indices)), random_state=42).fit_predict(v_norm.T)
    
    # Find high activation token-embedding pairs
    k_high = []
    v_high = []
    k_threshold = np.percentile(k_norm, 90)
    v_threshold = np.percentile(v_norm, 90)
    
    for t_idx, token_idx in enumerate(token_indices):
        for e_idx, emb_idx in enumerate(emb_indices):
            if k_norm[t_idx, e_idx] > k_threshold:
                k_high.append((token_idx, emb_idx, k_norm[t_idx, e_idx]))
            if v_norm[t_idx, e_idx] > v_threshold:
                v_high.append((token_idx, emb_idx, v_norm[t_idx, e_idx]))
    
    # Sort by activation strength
    k_high.sort(key=lambda x: x[2], reverse=True)
    v_high.sort(key=lambda x: x[2], reverse=True)
    
    return {
        "token_indices": token_indices,
        "emb_indices": emb_indices,
        "k_token_emb": k_norm,
        "v_token_emb": v_norm,
        "k_token_clusters": k_token_clusters,
        "v_token_clusters": v_token_clusters,
        "k_emb_clusters": k_emb_clusters,
        "v_emb_clusters": v_emb_clusters,
        "k_high_activations": k_high,
        "v_high_activations": v_high
    }

def identify_token_embedding_compression(results):
    """
    Identify compression opportunities from token-embedding similarity analysis.
    
    Args:
        results: Results from analyze_token_embedding_similarity
        
    Returns:
        Dict with compression opportunities
    """
    k_norm = results["k_token_emb"]
    v_norm = results["v_token_emb"]
    token_indices = results["token_indices"]
    emb_indices = results["emb_indices"]
    
    # Find token-specific embeddings (high variance across tokens)
    token_specific = []
    for e_idx, emb_idx in enumerate(emb_indices):
        k_var = np.var(k_norm[:, e_idx])
        v_var = np.var(v_norm[:, e_idx])
        
        if k_var > np.percentile(np.var(k_norm, axis=0), 75) or \
           v_var > np.percentile(np.var(v_norm, axis=0), 75):
            # Find tokens where this embedding is most active
            k_top = np.argsort(k_norm[:, e_idx])[-3:]  # Top 3 tokens
            v_top = np.argsort(v_norm[:, e_idx])[-3:]
            
            token_specific.append({
                "embedding": emb_idx,
                "k_variance": k_var,
                "v_variance": v_var,
                "k_top_tokens": [token_indices[i] for i in k_top],
                "v_top_tokens": [token_indices[i] for i in v_top]
            })
    
    # Calculate similarities between tokens based on embedding patterns
    token_sim = np.zeros((len(token_indices), len(token_indices)))
    for i in range(len(token_indices)):
        for j in range(len(token_indices)):
            if i == j:
                token_sim[i, j] = 1.0
            else:
                # Cosine similarity between token embedding patterns
                k_sim = np.dot(k_norm[i], k_norm[j]) / (
                    np.linalg.norm(k_norm[i]) * np.linalg.norm(k_norm[j]) + 1e-8)
                v_sim = np.dot(v_norm[i], v_norm[j]) / (
                    np.linalg.norm(v_norm[i]) * np.linalg.norm(v_norm[j]) + 1e-8)
                token_sim[i, j] = (k_sim + v_sim) / 2
    
    # Find groups of similar tokens
    similar_tokens = []
    threshold = 0.9
    visited = set()
    
    for i in range(len(token_indices)):
        if i not in visited:
            group = [i]
            for j in range(i+1, len(token_indices)):
                if token_sim[i, j] >= threshold:
                    group.append(j)
                    visited.add(j)
            
            if len(group) > 1:  # Only add groups with more than one token
                similar_tokens.append({
                    "tokens": [token_indices[idx] for idx in group],
                    "similarity": np.mean([token_sim[i, j] for j in group])
                })
            visited.add(i)
    
    # Calculate compression potential
    compression = {
        "token_specific_count": len(token_specific),
        "similar_token_groups": len(similar_tokens),
        "similar_tokens_total": sum(len(g["tokens"]) for g in similar_tokens),
        "compression_potential": sum(len(g["tokens"]) - 1 for g in similar_tokens)
    }
    
    return {
        "token_specific_embeddings": token_specific,
        "similar_token_groups": similar_tokens,
        "compression": compression
    }

def identify_groupable_embedding_dimensions(kv_cache, adaptive_threshold=True, min_similarity=0.6):
    """
    Identify embedding dimensions that can be grouped based on activation patterns.
    Uses adaptive clustering to find natural groups in the data.
    
    Args:
        kv_cache: KV cache from model output
        adaptive_threshold: Whether to adapt threshold based on data distribution
        min_similarity: Minimum baseline similarity threshold
        
    Returns:
        Dict with groups of similar dimensions that can be compressed
    """
    # First run token-embedding analysis
    token_emb = analyze_token_embedding_similarity(kv_cache, sample_dims=None)
    
    # Extract normalized key patterns (dimensions as columns)
    k_norm = token_emb["k_token_emb"]
    v_norm = token_emb["v_token_emb"]
    emb_indices = token_emb["emb_indices"]
    head_dim = len(emb_indices)
    
    # Calculate column-wise similarity matrices
    k_dimension_similarity = np.zeros((k_norm.shape[1], k_norm.shape[1]))
    v_dimension_similarity = np.zeros((v_norm.shape[1], v_norm.shape[1]))
    
    for i in range(k_norm.shape[1]):
        for j in range(k_norm.shape[1]):
            if i == j:
                k_dimension_similarity[i, j] = 1.0
                v_dimension_similarity[i, j] = 1.0
            else:
                # Cosine similarity between dimension patterns
                k_sim = np.dot(k_norm[:, i], k_norm[:, j]) / (
                    np.linalg.norm(k_norm[:, i]) * np.linalg.norm(k_norm[:, j]) + 1e-8)
                v_sim = np.dot(v_norm[:, i], v_norm[:, j]) / (
                    np.linalg.norm(v_norm[:, i]) * np.linalg.norm(v_norm[:, j]) + 1e-8)
                
                k_dimension_similarity[i, j] = k_sim
                v_dimension_similarity[i, j] = v_sim
    
    # Convert similarity matrices to distance matrices
    k_distance = 1.0 - k_dimension_similarity
    v_distance = 1.0 - v_dimension_similarity
    
    # Use hierarchical clustering to find natural groupings
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    # Convert to condensed distance matrices
    k_condensed = squareform(k_distance)
    v_condensed = squareform(v_distance)
    
    # Perform hierarchical clustering
    k_linkage = hierarchy.linkage(k_condensed, method='average')
    v_linkage = hierarchy.linkage(v_condensed, method='average')
    
    # Determine optimal clustering thresholds based on data
    if adaptive_threshold:
        # Find elbow in the distance curve using acceleration
        k_distances = k_linkage[:, 2]
        k_acceleration = np.diff(np.diff(k_distances))
        k_elbow_idx = np.argmax(k_acceleration) + 2  # +2 due to double diff
        k_threshold = 1.0 - k_linkage[k_elbow_idx, 2]
        k_threshold = max(min_similarity, min(0.95, k_threshold))  # Keep within reasonable range
        
        v_distances = v_linkage[:, 2]
        v_acceleration = np.diff(np.diff(v_distances))
        v_elbow_idx = np.argmax(v_acceleration) + 2
        v_threshold = 1.0 - v_linkage[v_elbow_idx, 2]
        v_threshold = max(min_similarity, min(0.95, v_threshold))
    else:
        k_threshold = min_similarity
        v_threshold = min_similarity
    
    # Cut the hierarchical tree to form clusters
    k_clusters = hierarchy.fcluster(k_linkage, 1.0 - k_threshold, criterion='distance')
    v_clusters = hierarchy.fcluster(v_linkage, 1.0 - v_threshold, criterion='distance')
    
    # Group dimensions by cluster
    k_groups = []
    v_groups = []
    
    # Process key clusters
    k_cluster_dict = {}
    for i, cluster_id in enumerate(k_clusters):
        if cluster_id not in k_cluster_dict:
            k_cluster_dict[cluster_id] = []
        k_cluster_dict[cluster_id].append(emb_indices[i])
    
    # Add clusters with multiple dimensions
    for cluster_dims in k_cluster_dict.values():
        if len(cluster_dims) > 1:
            # Sort dimensions by importance (optional)
            k_groups.append(sorted(cluster_dims))
    
    # Process value clusters
    v_cluster_dict = {}
    for i, cluster_id in enumerate(v_clusters):
        if cluster_id not in v_cluster_dict:
            v_cluster_dict[cluster_id] = []
        v_cluster_dict[cluster_id].append(emb_indices[i])
    
    # Add clusters with multiple dimensions
    for cluster_dims in v_cluster_dict.values():
        if len(cluster_dims) > 1:
            v_groups.append(sorted(cluster_dims))
    
    # Calculate compression benefits
    k_compression = sum([len(group) - 1 for group in k_groups])
    v_compression = sum([len(group) - 1 for group in v_groups])
    
    return {
        "k_similarity": k_dimension_similarity,
        "v_similarity": v_dimension_similarity,
        "k_groups": k_groups,
        "v_groups": v_groups,
        "k_compression": k_compression,
        "v_compression": v_compression,
        "total_compression": k_compression + v_compression,
        "k_threshold_used": k_threshold,
        "v_threshold_used": v_threshold
    }