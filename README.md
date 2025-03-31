# KV Cache Visualization

This project provides tools to analyze and visualize the Key-Value (KV) cache in transformer-based language models. The visualizations and metrics help understand the internal workings of the model and identify optimization opportunities.

## How to Use

1. **Install Dependencies**:

   - Ensure you have Python installed.
   - Install the required libraries:
     ```bash
     pip install torch transformers matplotlib seaborn scikit-learn pandas
     ```

2. **Analysis Mode**:

   - Execute `main.py` to generate visualizations and analyze the KV cache:
     ```bash
     python main.py --mode analyze
     ```

3. **Pruning Simulation**:

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
   
   - The results will show perplexity and latency changes comparing the baseline to the pruned model.

4. **View Results**:

   - The generated visualizations will be saved as PNG files in the project directory:
     - `layer_statistics.png`: Comparison of sparsity, correlation, standard deviation, and mean values across layers
     - `head_sparsity.png`: Heatmap of key/value sparsity by head and layer
     - `pruning_heatmap_by_layer_head.png`: Detailed heatmap showing pruning potential by layer and head
     - `layer_pruning_potential.png`: Bar chart of pruning potential per layer
     - `head_pruning_potential.png`: Bar chart of pruning potential per attention head
     - `weight_magnitude_distribution.png`: Histograms of key and value weight magnitudes

5. **Interpret Results**:
   - Refer to [graph-explanations.md](graph-explanations.md) for detailed explanations of each visualization.
   - For pruning simulations, lower perplexity change percentages indicate better model preservation.
   - Negative latency change percentages indicate performance improvements.

## Files

- `main.py`: The main script for generating KV cache visualizations and metrics.
- `graph-explanations.md`: Detailed descriptions of the generated visualizations.
- `README.md`: This file, providing an overview of the project.
- `analysis/`: Contains modules for analyzing layers, heads, and dimensions.
- `pruning/`: Contains modules for KV cache pruning and evaluation.
- `utils/`: Contains utility functions for data collection and processing.
- `visualization/`: Contains plotting functions for visualizations.

## Requirements

- Python 3.7+
- GPU (optional but recommended for faster processing)
