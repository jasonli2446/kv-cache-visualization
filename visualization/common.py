"""
Common utilities for visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def ensure_graph_dir(dir_path):
    """
    Ensure that a directory exists for saving graphs.
    If the path doesn't include 'graphs/' prefix, add it.
    
    Args:
        dir_path: Directory path
    """
    # Ensure the path starts with 'graphs/'
    if not dir_path.startswith('graphs/'):
        dir_path = f'graphs/{dir_path}'
    
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

def plot_weight_magnitude_distribution(all_keys, all_values, k_threshold, v_threshold):
    """
    Plot histograms of weight magnitude distributions.
    
    Args:
        all_keys: List of key tensors
        all_values: List of value tensors
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
    """
    plt.figure(figsize=config.DEFAULT_FIGSIZE)
    
    # Move tensors to CPU before converting to NumPy
    all_key_weights = np.concatenate([k.reshape(-1).cpu().numpy() for k in all_keys])
    all_value_weights = np.concatenate([v.reshape(-1).cpu().numpy() for v in all_values])
    
    plt.subplot(1, 2, 1)
    plt.hist(np.abs(all_key_weights), bins=50, alpha=0.7, color='blue')
    plt.axvline(x=k_threshold, color='red', linestyle='--', 
                label=f'Pruning Threshold: {k_threshold:.6f}')
    plt.title('Key Weight Magnitude Distribution')
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(all_value_weights), bins=50, alpha=0.7, color='orange')
    plt.axvline(x=v_threshold, color='red', linestyle='--', 
                label=f'Pruning Threshold: {v_threshold:.6f}')
    plt.title('Value Weight Magnitude Distribution')
    plt.xlabel('Absolute Magnitude')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    ensure_graph_dir("graphs")
    
    plt.savefig("graphs/weight_magnitude_distribution.png", dpi=config.FIGURE_DPI)
    plt.close()