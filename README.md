# KV Cache Visualization

This project provides tools to analyze and visualize the Key-Value (KV) cache in transformer-based language models. The visualizations and metrics help understand the internal workings of the model and identify optimization opportunities.

## Features

- **Multi-level Analysis**: Examine KV cache at layer, head, token, and embedding dimension levels
- **Visualization Suite**: Generate informative heatmaps, charts, and distribution plots
- **Pruning Simulation**: Test the impact of pruning on model quality and performance
- **Generation Analysis**: Analyze how KV cache patterns evolve during text generation
- **Similarity Analysis**: Identify compressible patterns across layers, heads, tokens, and embedding dimensions
- **Dataset Integration**: Use WikiText samples for realistic analysis scenarios
- **Compression Opportunities**: Quantify potential KV cache size reduction through similarity detection

## How to Use

1. **Install Dependencies**:

   - Ensure you have Python installed.
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Analysis Mode**:

   - Execute `main.py` to generate visualizations and analyze the KV cache:

     ```bash
     python main.py --mode analyze
     ```

   - Focus on specific aspects of analysis:
     ```bash
     python main.py --mode analyze --analysis_focus layers|heads|tokens|embeddings
     ```

3. **Generation Analysis**:

   - Analyze how KV cache changes during the generation process:
     ```bash
     python main.py --mode analyze_generation
     ```

4. **Similarity Analysis**:

   - Identify compressible patterns across the KV cache:
     ```bash
     python main.py --mode analyze_similarity
     ```
   
   - This mode detects redundant information that could be merged or compressed

5. **Pruning Simulation**:

   - Simulate pruning specific layers:

     ```bash
     python main.py --mode prune --prune_layers 0,1,2
     ```

   - Simulate pruning specific attention heads:

     ```bash
     python main.py --mode prune --prune_layers 0 --prune_heads 2,3
     ```

   - Use custom prompts and continuations:
     ```bash
     python main.py --mode prune --prune_layers 0 --prompt "Your custom prompt text" --continuation "Text to evaluate model quality"
     ```

6. **Use Different Datasets**:

   - List available sample texts:

     ```bash
     python main.py --mode list_samples
     ```

   - Use specific WikiText sample:
     ```bash
     python main.py --use_wikitext --wikitext_index 2
     ```

## Generated Visualizations

The tool produces various visualizations saved in the `graphs/` directory:

### Layer-level Analysis

- `graphs/layers/layer_statistics.png`: Metrics across model layers
- `graphs/layers/layer_pruning_potential.png`: Pruning potential by layer

### Head-level Analysis

- `graphs/heads/head_sparsity.png`: Sparsity heatmaps by layer and head
- `graphs/heads/head_pruning_potential.png`: Pruning potential by attention head
- `graphs/heads/pruning_heatmap_by_layer_head.png`: Detailed pruning heatmap

### Token-level Analysis

- `graphs/tokens/token_position_importance.png`: Importance scores for token positions
- `graphs/tokens/generation_stages_comparison.png`: Sparsity patterns during generation

### Embedding-level Analysis

- `graphs/embeddings/embedding_consistency.png`: Sparsity across embedding dimensions
- `graphs/embeddings/sparse_dense_embedding_patterns.png`: Dimension activation patterns

### Weight Analysis

- `graphs/weight_magnitude_distribution.png`: Distribution of weight magnitudes

### Similarity Analysis

- `graphs/similarity/layer_similarity_matrix.png`: Similarity between model layers
- `graphs/similarity/head_similarity_matrix.png`: Similarity between attention heads
- `graphs/similarity/embedding_dimension_correlations.png`: Correlation between embedding dimensions
- `graphs/similarity/token_similarity_matrix.png`: Similarity between token positions
- `graphs/similarity/token_kv_similarity_matrices.png`: Separate key and value token similarity

## Project Structure

- `main.py`: Main entry point with command-line interface
- `config.py`: Configuration parameters for analysis and visualization
- `analysis/`: Modules for analyzing different aspects of KV cache
  - `layer_analysis.py`: Layer-level analysis functions
  - `head_analysis.py`: Head-level analysis functions
  - `token_analysis.py`: Token position analysis functions
  - `embedding_analysis.py`: Embedding dimension analysis functions
  - `similarity_analysis.py`: Functions to find compressible patterns
- `visualization/`: Plotting functions for different visualization types
  - `layer_plots.py`: Layer-level visualizations
  - `head_plots.py`: Head-level visualizations
  - `token_plots.py`: Token-level visualizations
  - `embedding_plots.py`: Embedding-level visualizations
  - `similarity_plots.py`: Similarity and compression visualizations
  - `common.py`: Common visualization utilities
- `pruning/`: KV cache pruning simulation
  - `pruner.py`: Core pruning functionality
  - `evaluation.py`: Metrics for evaluating pruning impact
- `utils/`: Utility functions
  - `data_collection.py`: Functions for collecting KV cache data
  - `dataset_loaders.py`: Functions for loading sample texts
  - `generation_stages.py`: Utilities for analyzing generation process
- `graph-explanations.md`: Detailed explanations of visualizations

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (optional but recommended for faster processing)
