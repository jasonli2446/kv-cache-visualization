# KV Cache Visualization

This project provides tools to analyze and visualize the Key-Value (KV) cache in transformer-based language models. The visualizations and metrics help understand the internal workings of the model and identify optimization opportunities.

## Features

1. **Layer-wise Heatmaps** (`kv_cache_layers.png`):

   - Visualize the key and value tensors for each layer.
   - Identify patterns such as sparsity, vertical/horizontal stripes, and changes across layers.

2. **Layer Statistics Plots** (`kv_cache_statistics.png`):

   - Analyze sparsity, key-value correlation, standard deviation, and mean absolute values across layers.

3. **Value Distribution Histograms** (`kv_value_distribution.png`):

   - Explore the distribution of values in the key and value tensors.

4. **PCA of Keys and Values by Position** (`kv_pca_by_position.png`):

   - Reduce high-dimensional KV cache to 2D to understand token position relationships.

5. **Head Specialization** (`head_specialization.png`):

   - Visualize how attention heads specialize within a specific layer.

6. **Quantization Error Analysis** (`kv_quantization_error.png`):

   - Evaluate the impact of reducing precision on representation accuracy.

7. **Low Importance Tokens** (`low_importance_tokens.png`):

   - Identify tokens with the lowest magnitudes in the KV cache.
   - Highlight candidates for aggressive pruning or compression.

8. **Token Importance Comparison** (`token_importance_comparison.png`):

   - Compare highest and lowest importance tokens in a unified visualization.
   - Identify which specific tokens contribute most and least to model performance.

9. **Token Magnitude Distributions** (`token_magnitude_distributions.png`):

   - Show distributions of key and value magnitudes across different token types.
   - Reveal which categories of tokens could be treated differently in optimizations.

10. **Token Magnitudes by Position** (`token_magnitudes.png`):

    - Visualize how token importance varies by sequence position.
    - Identify position-based patterns for progressive KV cache pruning.

11. **Token Key-Value Scatter** (`token_key_value_scatter.png`):
    - Map tokens in 2D space based on their key and value magnitudes.
    - Identify clusters and outliers among tokens requiring special handling.

## How to Use

1. **Install Dependencies**:

   - Ensure you have Python installed.
   - Install the required libraries:
     ```bash
     pip install torch transformers matplotlib seaborn scikit-learn pandas
     ```

2. **Run the Script**:

   - Execute `main.py` to generate the visualizations and metrics:
     ```bash
     python main.py
     ```

3. **View Results**:

   - The generated visualizations will be saved as PNG files in the project directory.

4. **Interpret Results**:
   - Refer to [graph-explanations.md](graph-explanations.md) for detailed explanations of each visualization.

## Files

- `main.py`: The main script for generating KV cache visualizations and metrics.
- `graph-explanations.md`: Detailed descriptions of the generated visualizations.
- `README.md`: This file, providing an overview of the project.

## Requirements

- Python 3.7+
- GPU (optional but recommended for faster processing)
