"""
Visualization package for KV cache analysis.
"""

from visualization.plotting import plot_all_visualizations
from visualization.layer_plots import plot_layer_statistics, plot_layer_pruning_potential
from visualization.head_plots import plot_head_sparsity, plot_head_pruning_potential, plot_head_layer_heatmap
from visualization.token_plots import plot_token_position_importance, plot_generation_stages_comparison
from visualization.embedding_plots import plot_embedding_consistency, plot_sparse_dense_embedding_patterns
from visualization.common import plot_weight_magnitude_distribution
from visualization.similarity_plots import (
    plot_layer_similarity_heatmap,
    plot_head_similarity_matrix,
    plot_token_similarity_matrix,
    plot_embedding_dimension_correlation,
    plot_token_embedding_patterns,
    plot_similarity_visualizations
)