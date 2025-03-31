from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import gc
from scipy.stats import entropy
import pandas as pd

# Memory optimization
torch.cuda.empty_cache()
gc.collect()

# ====================
# Generate KV cache
# ====================

# Use a more accessible model that doesn't require special permissions
model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Process longer text to see more patterns
prompt = "The history of artificial intelligence began in ancient times with myths and stories about artificial beings. The seeds of modern AI were planted by classical philosophers who attempted to describe human thinking as a symbolic system. This work culminated in the invention of the programmable digital computer, a machine based on the abstract essence of mathematical reasoning."

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model(**inputs, output_hidden_states=True, use_cache=True)
kv_cache = outputs.past_key_values

# Extract basic info about the model
num_layers = len(kv_cache)
batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
print(f"KV cache structure: {num_layers} layers, {num_heads} heads, {seq_len} sequence length, {head_dim} head dimensions")

# ====================
# Layer-wise Analysis
# ====================

# Store layer-wise statistics
layer_stats = []

# Process each layer
all_keys = []
all_values = []

# Calculate global max values for relative thresholding
global_k_max = max([layer_kv[0].abs().max().item() for layer_kv in kv_cache])
global_v_max = max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
k_threshold = 0.01 * global_k_max
v_threshold = 0.01 * global_v_max
print(f"Using thresholds - Keys: {k_threshold:.6f}, Values: {v_threshold:.6f} (1% of max weights)")

for layer_idx, layer_kv in enumerate(kv_cache):
    # Extract keys and values
    keys = layer_kv[0].detach().cpu()  # [batch_size, num_heads, seq_len, head_dim]
    values = layer_kv[1].detach().cpu()  # [batch_size, num_heads, seq_len, head_dim]

    # Store for global analysis
    all_keys.append(keys)
    all_values.append(values)

    # Calculate statistics for this layer using relative threshold
    k_sparsity = (keys.abs() < k_threshold).float().mean().item()
    v_sparsity = (values.abs() < v_threshold).float().mean().item()
    k_mean = keys.abs().mean().item()
    v_mean = values.abs().mean().item()
    k_std = keys.std().item()
    v_std = values.std().item()

    # Correlation between keys and values
    k_flat = keys.reshape(-1).numpy()
    v_flat = values.reshape(-1).numpy()
    correlation = np.corrcoef(k_flat, v_flat)[0, 1]

    # Add to statistics
    layer_stats.append({
        "layer": layer_idx,
        "k_sparsity": k_sparsity,
        "v_sparsity": v_sparsity,
        "k_mean": k_mean,
        "v_mean": v_mean,
        "k_std": k_std,
        "v_std": v_std,
        "kv_correlation": correlation,
    })

# Create a dataframe with layer statistics
layer_df = pd.DataFrame(layer_stats)
print("Layer-wise Statistics:")
print(layer_df)

# Plot layer statistics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(layer_df["layer"], layer_df["k_sparsity"], "b-o", label="Key Sparsity")
plt.plot(layer_df["layer"], layer_df["v_sparsity"], "r-o", label="Value Sparsity")
plt.title("Sparsity Across Layers")
plt.xlabel("Layer")
plt.ylabel("Sparsity (ratio of near-zero values)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(layer_df["layer"], layer_df["kv_correlation"], "g-o")
plt.title("Key-Value Correlation Across Layers")
plt.xlabel("Layer")
plt.ylabel("Correlation")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(layer_df["layer"], layer_df["k_std"], "b-o", label="Key Std")
plt.plot(layer_df["layer"], layer_df["v_std"], "r-o", label="Value Std")
plt.title("Standard Deviation Across Layers")
plt.xlabel("Layer")
plt.ylabel("Standard Deviation")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(layer_df["layer"], layer_df["k_mean"], "b-o", label="Key Mean")
plt.plot(layer_df["layer"], layer_df["v_mean"], "r-o", label="Value Mean")
plt.title("Mean Absolute Value Across Layers")
plt.xlabel("Layer")
plt.ylabel("Mean")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("layer_statistics.png", dpi=300)

# ====================
# Head-level Analysis
# ====================

# Store head-wise statistics across all layers
head_stats = []

for layer_idx in range(num_layers):
    keys = all_keys[layer_idx]  # [batch_size, num_heads, seq_len, head_dim]
    values = all_values[layer_idx]
    
    for head_idx in range(num_heads):
        # Extract data for this head
        head_keys = keys[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
        head_values = values[:, head_idx, :, :]
        
        # Calculate sparsity for this head
        k_sparsity = (head_keys.abs() < k_threshold).float().mean().item()
        v_sparsity = (head_values.abs() < v_threshold).float().mean().item()
        k_mean = head_keys.abs().mean().item()
        v_mean = head_values.abs().mean().item()
        k_std = head_keys.std().item()
        v_std = head_values.std().item()
        
        # Add to head statistics
        head_stats.append({
            "layer": layer_idx,
            "head": head_idx,
            "k_sparsity": k_sparsity,
            "v_sparsity": v_sparsity,
            "k_mean": k_mean,
            "v_mean": v_mean,
            "k_std": k_std,
            "v_std": v_std,
        })

# Create a dataframe with head statistics
head_df = pd.DataFrame(head_stats)

# Create a heatmap of head sparsity across layers
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
head_k_sparsity = head_df.pivot(index="layer", columns="head", values="k_sparsity")
sns.heatmap(head_k_sparsity, cmap="Blues", annot=True, fmt=".2f")
plt.title("Key Sparsity by Head and Layer")
plt.xlabel("Head Index")
plt.ylabel("Layer")

plt.subplot(2, 1, 2)
head_v_sparsity = head_df.pivot(index="layer", columns="head", values="v_sparsity")
sns.heatmap(head_v_sparsity, cmap="Blues", annot=True, fmt=".2f")
plt.title("Value Sparsity by Head and Layer")
plt.xlabel("Head Index")
plt.ylabel("Layer")

plt.tight_layout()
plt.savefig("head_sparsity.png", dpi=300)

# Identify prunable heads (high sparsity)
prunable_heads_threshold = 0.5  # Adjust based on your sparsity criteria
prunable_heads = head_df[(head_df["k_sparsity"] > prunable_heads_threshold) | 
                        (head_df["v_sparsity"] > prunable_heads_threshold)]

if not prunable_heads.empty:
    print("\nPotentially Prunable Heads (high sparsity):")
    print(prunable_heads[["layer", "head", "k_sparsity", "v_sparsity"]].sort_values(
        by=["k_sparsity", "v_sparsity"], ascending=False))

# ====================
# Head Dimension Analysis
# ====================

# Store statistics for each dimension in each head
dim_stats = []

# We'll analyze a subset of layers if the model is very large
layers_to_analyze = min(num_layers, 5)  # Analyze up to 5 layers
selected_layers = list(range(0, num_layers, max(1, num_layers // layers_to_analyze)))[:layers_to_analyze]

for layer_idx in selected_layers:
    keys = all_keys[layer_idx]  # [batch_size, num_heads, seq_len, head_dim]
    values = all_values[layer_idx]
    
    for head_idx in range(num_heads):
        # Extract data for this head
        head_keys = keys[:, head_idx, :, :]  # [batch_size, seq_len, head_dim]
        head_values = values[:, head_idx, :, :]
        
        # Analyze each dimension
        for dim in range(head_dim):
            # Extract this specific dimension across all sequence positions
            dim_k = head_keys[:, :, dim]  # [batch_size, seq_len]
            dim_v = head_values[:, :, dim]
            
            # Calculate statistics
            k_sparsity = (dim_k.abs() < k_threshold).float().mean().item()
            v_sparsity = (dim_v.abs() < v_threshold).float().mean().item()
            k_mean = dim_k.abs().mean().item()
            v_mean = dim_v.abs().mean().item()
            k_std = dim_k.std().item()
            v_std = dim_v.std().item()
            
            # Add to dimension statistics
            dim_stats.append({
                "layer": layer_idx,
                "head": head_idx,
                "dimension": dim,
                "k_sparsity": k_sparsity,
                "v_sparsity": v_sparsity,
                "k_mean": k_mean,
                "v_mean": v_mean,
                "k_std": k_std,
                "v_std": v_std,
            })

# Create a dataframe with dimension statistics
dim_df = pd.DataFrame(dim_stats)

# 1. Remove layer dimension sparsity graphs from head dimension analysis section
for layer_idx in selected_layers:
    layer_dim_df = dim_df[dim_df["layer"] == layer_idx]

# Identify prunable dimensions (high sparsity)
prunable_dims_threshold = 0.2  # Try 20% instead of 70%
prunable_dims = dim_df[(dim_df["k_sparsity"] > prunable_dims_threshold) | 
                       (dim_df["v_sparsity"] > prunable_dims_threshold)]

if not prunable_dims.empty:
    prunable_dims = prunable_dims.sort_values(by=["k_sparsity", "v_sparsity"], ascending=False)
    print("\nTop 20 Potentially Prunable Dimensions (high sparsity):")
    print(prunable_dims[["layer", "head", "dimension", "k_sparsity", "v_sparsity"]].head(20))

# ====================
# Across-Layer Head Consistency
# ====================

# Calculate the consistency of head sparsity across layers
head_consistency = []

for head_idx in range(num_heads):
    # Get all layers for this head
    head_layers = head_df[head_df["head"] == head_idx]
    
    # Calculate statistics
    k_sparsity_mean = head_layers["k_sparsity"].mean()
    k_sparsity_std = head_layers["k_sparsity"].std()
    v_sparsity_mean = head_layers["v_sparsity"].mean()
    v_sparsity_std = head_layers["v_sparsity"].std()
    
    # Get most and least sparse layers for this head
    max_k_layer = head_layers.loc[head_layers["k_sparsity"].idxmax()]["layer"]
    min_k_layer = head_layers.loc[head_layers["k_sparsity"].idxmin()]["layer"]
    
    head_consistency.append({
        "head": head_idx,
        "k_sparsity_mean": k_sparsity_mean,
        "k_sparsity_std": k_sparsity_std,
        "v_sparsity_mean": v_sparsity_mean,
        "v_sparsity_std": v_sparsity_std,
        "max_k_sparse_layer": max_k_layer,
        "min_k_sparse_layer": min_k_layer
    })

# Create a dataframe for head consistency
consistency_df = pd.DataFrame(head_consistency)
print("\nHead Consistency Across Layers:")
print(consistency_df.sort_values(by="k_sparsity_std"))

# ====================
# Enhanced KV Cache Visualizations
# ====================

def create_enhanced_visualizations():
    print("Creating enhanced visualizations for dimension-based pruning analysis...")
    
    # 1. HEATMAP: Sparsity Across Layers and Heads
    plt.figure(figsize=(12, 10))
    
    # Calculate head-level sparsity across all layers
    head_sparsity_matrix = np.zeros((num_layers, num_heads))
    for layer_idx in range(num_layers):
        keys = all_keys[layer_idx].squeeze(0)  # [num_heads, seq_len, head_dim]
        for head_idx in range(num_heads):
            head_keys = keys[head_idx]  # [seq_len, head_dim]
            head_sparsity_matrix[layer_idx, head_idx] = (head_keys.abs() < k_threshold).float().mean().item()
    
    # Plot the heatmap
    sns.heatmap(head_sparsity_matrix, 
                cmap="Blues", 
                annot=True, 
                fmt=".2f", 
                xticklabels=range(num_heads),
                yticklabels=range(num_layers))
    plt.title("Pruning Potential: Sparsity by Layer and Head")
    plt.xlabel("Attention Head")
    plt.ylabel("Model Layer")
    plt.tight_layout()
    plt.savefig("pruning_heatmap_by_layer_head.png", dpi=300)
    
    # 2. BAR CHART: Layer-wise Pruning Potential
    plt.figure(figsize=(12, 6))
    
    # Calculate per-layer sparsity
    layer_sparsity_k = [layer["k_sparsity"] for layer in layer_stats]
    layer_sparsity_v = [layer["v_sparsity"] for layer in layer_stats]
    
    x = np.arange(num_layers)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, layer_sparsity_k, width, label='Key Sparsity')
    ax.bar(x + width/2, layer_sparsity_v, width, label='Value Sparsity')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Pruning Potential (Sparsity)')
    ax.set_title('Layer-wise Pruning Potential')
    ax.set_xticks(x)
    ax.set_xticklabels(range(num_layers))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line for average sparsity
    avg_k_sparsity = np.mean(layer_sparsity_k)
    avg_v_sparsity = np.mean(layer_sparsity_v)
    ax.axhline(y=avg_k_sparsity, color='blue', linestyle='--', alpha=0.7, 
               label=f'Avg K Sparsity: {avg_k_sparsity:.3f}')
    ax.axhline(y=avg_v_sparsity, color='orange', linestyle='--', alpha=0.7,
               label=f'Avg V Sparsity: {avg_v_sparsity:.3f}')
    
    plt.tight_layout()
    plt.savefig("layer_pruning_potential.png", dpi=300)
    
    # 3. GROUPED BAR CHART: Head-wise Pruning Potential
    # Calculate average sparsity for each head across all layers
    head_avg_sparsity = []
    for head_idx in range(num_heads):
        # Filter head stats for this head
        head_data = [stats for stats in head_stats if stats["head"] == head_idx]
        avg_k_sparsity = np.mean([h["k_sparsity"] for h in head_data])
        avg_v_sparsity = np.mean([h["v_sparsity"] for h in head_data])
        std_k_sparsity = np.std([h["k_sparsity"] for h in head_data])
        
        head_avg_sparsity.append({
            "head": head_idx,
            "avg_k_sparsity": avg_k_sparsity,
            "avg_v_sparsity": avg_v_sparsity,
            "std_k_sparsity": std_k_sparsity
        })
    
    # Create the chart
    plt.figure(figsize=(12, 6))
    
    x = np.arange(num_heads)
    width = 0.35
    
    k_sparsity = [h["avg_k_sparsity"] for h in head_avg_sparsity]
    v_sparsity = [h["avg_v_sparsity"] for h in head_avg_sparsity]
    k_std = [h["std_k_sparsity"] for h in head_avg_sparsity]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, k_sparsity, width, label='Key Sparsity', yerr=k_std, capsize=5)
    ax.bar(x + width/2, v_sparsity, width, label='Value Sparsity')
    
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Average Pruning Potential (Sparsity)')
    ax.set_title('Head-wise Pruning Potential (Averaged Across All Layers)')
    ax.set_xticks(x)
    ax.set_xticklabels(range(num_heads))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("head_pruning_potential.png", dpi=300)
    
    # 4. HISTOGRAM: Distribution of Weight Magnitudes
    plt.figure(figsize=(12, 6))
    
    # Collect all key and value weights
    all_key_weights = np.concatenate([k.reshape(-1).numpy() for k in all_keys])
    all_value_weights = np.concatenate([v.reshape(-1).numpy() for v in all_values])
    
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
    plt.savefig("weight_magnitude_distribution.png", dpi=300)
    
    # 5. REMOVE: Pruning Efficiency Curve
    # This section has been removed
    
    print(f"\nEnhanced visualizations created with model dimensions:")
    print(f"- {num_layers} layers")
    print(f"- {num_heads} heads per layer")
    print(f"- {head_dim} dimensions per head")
    print(f"- {seq_len} sequence length")
    print(f"- Pruning threshold: {k_threshold:.6f} (1% of max weight)")
    print("\nGraphs saved:")
    print("1. pruning_heatmap_by_layer_head.png - Sparsity across layers and heads")
    print("2. layer_pruning_potential.png - Pruning potential by layer")
    print("3. head_pruning_potential.png - Pruning potential by attention head")
    print("4. weight_magnitude_distribution.png - Distribution of weight magnitudes")
    # Removed reference to pruning_efficiency_curve.png

# Call the function to create enhanced visualizations
create_enhanced_visualizations()

# ====================
# Summary and Recommendations
# ====================

# Find the most prunable layers, heads, and dimensions
most_prunable_layers = layer_df.sort_values(by=["k_sparsity", "v_sparsity"], ascending=False).head(3)
most_consistent_heads = consistency_df.sort_values(by=["k_sparsity_mean"], ascending=False).head(3)

# Calculate overall sparsity
overall_k_sparsity = np.mean([layer["k_sparsity"] for layer in layer_stats])
overall_v_sparsity = np.mean([layer["v_sparsity"] for layer in layer_stats])

print("\n=== KV Cache Optimization Summary ===")
print(f"Overall Key Sparsity: {overall_k_sparsity:.4f}")
print(f"Overall Value Sparsity: {overall_v_sparsity:.4f}")

print("\n=== Dimension-wise Pruning Recommendations ===")
print("Most Prunable Layers:")
for _, row in most_prunable_layers.iterrows():
    print(f"  Layer {int(row['layer'])}: Key Sparsity={row['k_sparsity']:.4f}, Value Sparsity={row['v_sparsity']:.4f}")

print("\nMost Consistently Sparse Heads:")
for _, row in most_consistent_heads.iterrows():
    print(f"  Head {int(row['head'])}: Mean Key Sparsity={row['k_sparsity_mean']:.4f} ± {row['k_sparsity_std']:.4f}")

if not prunable_dims.empty:
    print("\nSpecific Pruning Targets:")
    for _, row in prunable_dims.head(5).iterrows():
        print(f"  Layer {int(row['layer'])}, Head {int(row['head'])}, Dimension {int(row['dimension'])}: " 
              f"K-Sparsity={row['k_sparsity']:.4f}, V-Sparsity={row['v_sparsity']:.4f}")

# Calculate total pruning potential
total_params = num_layers * num_heads * head_dim * 2  # *2 for both keys and values
prunable_params = len(prunable_dims) if not prunable_dims.empty else 0
print(f"\nPruning Potential: {prunable_params}/{total_params} parameters ({prunable_params/total_params*100:.2f}%)")