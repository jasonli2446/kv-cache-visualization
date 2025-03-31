"""
Configuration parameters for KV cache visualization and pruning.
"""

# Model configuration
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DEVICE = "cuda"  # Will fall back to "cpu" if CUDA not available

# Analysis thresholds
SPARSITY_THRESHOLD_PERCENTAGE = 1.0  # Use 1% of max weight as threshold
PRUNABLE_HEADS_THRESHOLD = 0.5  # Threshold to identify prunable heads
PRUNABLE_DIMS_THRESHOLD = 0.2  # Threshold to identify prunable dimensions

# Visualization settings
FIGURE_DPI = 300
DEFAULT_FIGSIZE = (12, 8)
HEATMAP_FIGSIZE = (14, 10)
BAR_CHART_FIGSIZE = (14, 6)

# Sample text for generation and evaluation
SAMPLE_PROMPT = "The history of artificial intelligence began in ancient times with myths and stories about artificial beings."
SAMPLE_CONTINUATION = "The seeds of modern AI were planted by classical philosophers who attempted to describe human thinking as a symbolic system."

# Pruning configuration
PRUNE_METHODS = ["layer", "head", "dimension", "threshold"]
DEFAULT_PRUNE_METHOD = "layer"