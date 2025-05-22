# KV Cache Visualization and Analysis

This project provides tools for analyzing and visualizing the Key-Value (KV) cache in transformer-based language models, with a focus on understanding cache patterns and exploring compression techniques.

## Features

- Layer-level analysis of KV cache sparsity
- Head-level analysis of attention patterns
- Token-level analysis of cache usage
- Embedding-level analysis of dimension importance
- Generation stage analysis
- Similarity analysis for compression opportunities
- Tucker decomposition for KV cache compression

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kv-cache-visualization.git
cd kv-cache-visualization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis

Run the main analysis pipeline:

```bash
python main.py --mode analyze
```

### Tucker Decomposition Analysis

The Tucker decomposition analysis allows you to compress the KV cache along the embedding and token dimensions. This can help reduce memory usage while maintaining model performance.

```bash
python compression/run_tucker_analysis.py --sequence_length 512 --embedding_rank 32 --token_rank 32
```

Parameters:
- `--sequence_length`: Target sequence length (default: 2048)
- `--embedding_rank`: Rank for embedding dimension (default: 32)
- `--token_rank`: Rank for token dimension (default: 32)
- `--no_sensitivity`: Skip layer sensitivity analysis

The analysis will:
1. Load the model and extract the KV cache
2. Perform Tucker decomposition with specified ranks
3. Calculate compression ratio and reconstruction error
4. Generate visualizations of the decomposition components
5. Optionally run layer sensitivity analysis

Outputs:
- Compression statistics (ratio, memory usage, reconstruction error)
- Tucker component visualizations in `graphs/tucker/`
- Layer sensitivity analysis plots (if enabled)

### Other Analysis Modes

- Layer analysis:
```bash
python main.py --mode analyze --analysis_focus layers
```

- Head analysis:
```bash
python main.py --mode analyze --analysis_focus heads
```

- Token analysis:
```bash
python main.py --mode analyze --analysis_focus tokens
```

- Embedding analysis:
```bash
python main.py --mode analyze --analysis_focus embeddings
```

### Generation Stage Analysis

Analyze how the KV cache evolves during text generation:

```bash
python main.py --mode analyze_generation
```

### Similarity Analysis

Find compressible patterns in the KV cache:

```bash
python main.py --mode analyze_similarity
```

### Compression Evaluation

Evaluate embedding dimension compression:

```bash
python main.py --mode compress
```

### Pruning Simulation

Simulate pruning specific layers or heads:

```bash
python main.py --mode prune --prune_layers 0,1,2
```

### Dataset Options

- List available WikiText samples:
```bash
python main.py --mode list_samples
```

- Use a specific WikiText sample:
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

### Tucker Decomposition Analysis
- `graphs/tucker/tucker_components.png`: Visualization of Tucker decomposition components
- `graphs/tucker/layer_sensitivity.png`: Layer-wise reconstruction error analysis

## Configuration

Key parameters can be adjusted in `config.py`:
- Model selection
- Analysis thresholds
- Visualization settings
- Dataset options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
