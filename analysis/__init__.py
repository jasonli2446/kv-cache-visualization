"""
Analysis modules for KV cache patterns across different dimensions.
"""

# Layer-level analysis
from analysis.layer_analysis import analyze_layers, find_prunable_layers

# Head-level analysis
from analysis.head_analysis import analyze_heads, find_prunable_heads, analyze_head_consistency

# Token-level analysis
from analysis.token_analysis import analyze_token_positions, analyze_token_layer_patterns, calculate_token_importance 

# Embedding-level analysis
from analysis.embedding_analysis import (
    analyze_dimensions, 
    find_prunable_dimensions,
    analyze_embedding_dimensions, 
    analyze_embedding_importance_by_layer,
    identify_embedding_patterns, 
    identify_similar_embedding_dimensions,
    find_consistent_embeddings
)

# Similarity and compression analysis
from analysis.similarity_analysis import (
    analyze_layer_similarity,
    analyze_head_similarity,
    analyze_token_similarity,
    find_compressible_patterns
)