"""
Visualization functions for KV cache similarity analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import sys
import os
from scipy.cluster import hierarchy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from visualization.common import ensure_graph_dir

def plot_layer_similarity_heatmap(layer_similarity_results):
    """
    Plot heatmap of layer similarity for keys and values.
    
    Args:
        layer_similarity_results: Results from analyze_layer_similarity
    """
    ensure_graph_dir("similarity")
    
    # Get the combined similarity matrix (keys + values)
    similarity_matrix = layer_similarity_results["similarity_matrix"]
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(similarity_matrix, annot=False, cmap="viridis", 
                   vmin=0, vmax=1, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Layer Similarity Matrix")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    
    # Add grid lines to make the matrix more readable
    ax.set_xticks(np.arange(similarity_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(similarity_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/layer_similarity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_head_similarity_matrix(head_similarity_results):
    """
    Plot correlation matrix between heads (instead of dendrogram).
    
    Args:
        head_similarity_results: Results from analyze_head_similarity
    """
    ensure_graph_dir("similarity")
    
    # Convert distance matrix to similarity matrix
    distance_matrix = head_similarity_results["distance_matrix"]
    similarity_matrix = 1.0 - distance_matrix
    head_indices = head_similarity_results["head_indices"]
    
    # Create labels for heads
    labels = [f"L{layer},H{head}" for layer, head in head_indices]
    
    # If the number of heads is too large, create a smaller matrix with averages
    if len(head_indices) > 30:
        # Group by layer and average across heads in each layer
        unique_layers = sorted(set(layer for layer, _ in head_indices))
        num_layers = len(unique_layers)
        layer_similarity = np.zeros((num_layers, num_layers))
        
        for i, layer1 in enumerate(unique_layers):
            for j, layer2 in enumerate(unique_layers):
                # Find all head pairs between these layers
                head_pairs = []
                for idx1, (l1, _) in enumerate(head_indices):
                    if l1 == layer1:
                        for idx2, (l2, _) in enumerate(head_indices):
                            if l2 == layer2:
                                head_pairs.append(similarity_matrix[idx1, idx2])
                
                # Average similarity across all head pairs
                layer_similarity[i, j] = np.mean(head_pairs) if head_pairs else 0.0
        
        # Plot the layer-wise head similarity
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(layer_similarity, annot=False, cmap="viridis", 
                       vmin=0, vmax=1, cbar_kws={"label": "Average Cosine Similarity"})
        ax.set_title("Average Head Similarity Between Layers")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Layer Index")
        ax.set_xticklabels(unique_layers)
        ax.set_yticklabels(unique_layers)
    else:
        # Plot the full head similarity matrix
        plt.figure(figsize=(16, 14))
        ax = sns.heatmap(similarity_matrix, annot=False, cmap="viridis", 
                       vmin=0, vmax=1, cbar_kws={"label": "Cosine Similarity"})
        ax.set_title("Head Similarity Matrix")
        ax.set_xlabel("Head")
        ax.set_ylabel("Head")
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/head_similarity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_embedding_dimension_correlation(embedding_correlation_results):
    """
    Plot combined correlation heatmap between embedding dimensions.
    
    Args:
        embedding_correlation_results: Results from analyze_embedding_dimension_correlation
    """
    ensure_graph_dir("similarity")
    
    # Get correlation matrices
    k_corr = embedding_correlation_results["k_correlation"]
    v_corr = embedding_correlation_results["v_correlation"]
    
    # Create a combined figure showing both correlations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot key correlation
    im1 = ax1.imshow(k_corr, cmap="viridis", vmin=-1, vmax=1)
    ax1.set_title("Key Embedding Dimension Correlation")
    ax1.set_xlabel("Dimension Index")
    ax1.set_ylabel("Dimension Index")
    fig.colorbar(im1, ax=ax1, label="Pearson Correlation")
    
    # Plot value correlation
    im2 = ax2.imshow(v_corr, cmap="viridis", vmin=-1, vmax=1)
    ax2.set_title("Value Embedding Dimension Correlation")
    ax2.set_xlabel("Dimension Index")
    fig.colorbar(im2, ax=ax2, label="Pearson Correlation")
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/embedding_dimension_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_token_clusters(token_similarity_results, prompt=None, tokenizer=None):
    """
    Visualize token clusters.
    
    Args:
        token_similarity_results: Results from analyze_token_similarity
        prompt: Original prompt text (optional)
        tokenizer: Tokenizer for decoding token IDs (optional)
    """
    ensure_graph_dir("similarity")
    
    k_clusters_df = token_similarity_results["k_clusters_df"]
    v_clusters_df = token_similarity_results["v_clusters_df"]
    
    # Plot cluster distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(data=k_clusters_df, x="cluster", discrete=True, ax=ax1, color="teal")
    ax1.set_title(f"Key Token Clusters (K={token_similarity_results['k_best_n_clusters']})")
    ax1.set_xlabel("Cluster ID")
    ax1.set_ylabel("Number of Tokens")
    
    sns.histplot(data=v_clusters_df, x="cluster", discrete=True, ax=ax2, color="teal")
    ax2.set_title(f"Value Token Clusters (K={token_similarity_results['v_best_n_clusters']})")
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Number of Tokens")
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/token_cluster_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot token positions by cluster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # For keys
    token_clusters = []
    for idx, row in k_clusters_df.iterrows():
        token_clusters.append((row["token_position"], row["cluster"]))
    token_clusters.sort(key=lambda x: x[0])  # Sort by token position
    
    positions, clusters = zip(*token_clusters) if token_clusters else ([], [])
    ax1.scatter(positions, clusters, marker='o', s=50, color="teal")
    ax1.set_title(f"Key Cluster Assignments by Token Position")
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Cluster")
    
    # For values
    token_clusters = []
    for idx, row in v_clusters_df.iterrows():
        token_clusters.append((row["token_position"], row["cluster"]))
    token_clusters.sort(key=lambda x: x[0])  # Sort by token position
    
    positions, clusters = zip(*token_clusters) if token_clusters else ([], [])
    ax2.scatter(positions, clusters, marker='o', s=50, color="teal")
    ax2.set_title(f"Value Cluster Assignments by Token Position")
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Cluster")
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/token_cluster_positions.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_token_similarity_matrix(token_similarity_matrix_results, prompt=None, tokenizer=None):
    """
    Plot token-to-token similarity matrix.
    
    Args:
        token_similarity_matrix_results: Results from compute_token_similarity_matrix
        prompt: Original prompt text (optional)
        tokenizer: Tokenizer for decoding token IDs (optional)
    """
    ensure_graph_dir("similarity")
    
    # Get token indices and similarity matrices
    token_indices = token_similarity_matrix_results["token_indices"]
    k_similarity = token_similarity_matrix_results["k_similarity"]
    v_similarity = token_similarity_matrix_results["v_similarity"]
    combined_similarity = token_similarity_matrix_results["combined_similarity"]
    
    # Create token labels if tokenizer and prompt are available
    if prompt is not None and tokenizer is not None:
        try:
            # Tokenize the prompt
            token_ids = tokenizer.encode(prompt)
            
            # Create token labels
            token_labels = []
            for idx in token_indices:
                if idx < len(token_ids):
                    token_text = tokenizer.decode([token_ids[idx]])
                    token_labels.append(f"{idx}:{token_text}")
                else:
                    token_labels.append(f"{idx}")
                    
            use_token_labels = True
        except:
            # Fall back to indices if tokenization fails
            token_labels = [str(idx) for idx in token_indices]
            use_token_labels = False
    else:
        token_labels = [str(idx) for idx in token_indices]
        use_token_labels = False
    
    # Plot combined similarity matrix (keys + values)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(combined_similarity, annot=False, cmap="viridis",
                   vmin=0, vmax=1, cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Token Similarity Matrix (Combined Key+Value)")
    
    # If we have a reasonable number of tokens, show token labels
    if len(token_indices) <= 30 and use_token_labels:
        ax.set_xticks(np.arange(len(token_indices)))
        ax.set_yticks(np.arange(len(token_indices)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right")
        ax.set_yticklabels(token_labels)
    else:
        # Otherwise just show token positions
        if len(token_indices) <= 50:
            tick_positions = np.arange(len(token_indices))
            tick_labels = [str(idx) for idx in token_indices]
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticklabels(tick_labels)
        else:
            # For very large matrices, use fewer ticks
            step = max(1, len(token_indices) // 10)
            tick_positions = np.arange(0, len(token_indices), step)
            tick_labels = [str(token_indices[i]) for i in range(0, len(token_indices), step)]
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Token Position")
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/token_similarity_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot separate key and value similarity matrices side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Key similarity
    im1 = ax1.imshow(k_similarity, cmap="viridis", vmin=0, vmax=1)
    ax1.set_title("Key Token Similarity Matrix")
    fig.colorbar(im1, ax=ax1, label="Cosine Similarity")
    
    # Value similarity
    im2 = ax2.imshow(v_similarity, cmap="viridis", vmin=0, vmax=1)
    ax2.set_title("Value Token Similarity Matrix")
    fig.colorbar(im2, ax=ax2, label="Cosine Similarity")
    
    # Set labels
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Token Position")
    ax2.set_xlabel("Token Position")
    
    # If we have few tokens, add token labels
    if len(token_indices) <= 20 and use_token_labels:
        for ax in [ax1, ax2]:
            ax.set_xticks(np.arange(len(token_indices)))
            ax.set_yticks(np.arange(len(token_indices)))
            ax.set_xticklabels(token_labels, rotation=45, ha="right")
            ax.set_yticklabels(token_labels)
    
    plt.tight_layout()
    plt.savefig("graphs/similarity/token_kv_similarity_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_similarity_visualizations(similarity_results, prompt=None, tokenizer=None):
    """
    Plot all similarity visualizations.
    
    Args:
        similarity_results: Results from find_compressible_patterns
        prompt: Optional prompt text for token visualizations
        tokenizer: Optional tokenizer for token visualizations
    """
    print("Plotting layer similarity matrix...")
    plot_layer_similarity_heatmap(similarity_results["layer_similarity"])
    
    print("Plotting head similarity matrix...")
    plot_head_similarity_matrix(similarity_results["head_similarity"])
    
    print("Plotting embedding dimension correlations...")
    plot_embedding_dimension_correlation(similarity_results["embedding_similarity"])
    
    print("Plotting token similarity matrix...")
    plot_token_similarity_matrix(similarity_results["token_similarity_matrix"], prompt, tokenizer)