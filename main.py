from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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
print(
    f"KV cache structure: {num_layers} layers, {num_heads} heads, {seq_len} sequence length, {head_dim} head dimensions"
)

# ====================
# Layer-wise Analysis
# ====================

# Store layer-wise statistics
layer_stats = []

# Process each layer
all_keys = []
all_values = []
fig, axes = plt.subplots(num_layers, 2, figsize=(15, 3 * num_layers))

for layer_idx, layer_kv in enumerate(kv_cache):
    # Extract keys and values
    keys = layer_kv[0].detach().cpu()  # [batch_size, num_heads, seq_len, head_dim]
    values = layer_kv[1].detach().cpu()  # [batch_size, num_heads, seq_len, head_dim]

    # Reshape for analysis: [batch_size, num_heads, seq_len, head_dim] -> [seq_len, num_heads*head_dim]
    k_reshaped = (
        keys.view(batch_size, num_heads, seq_len, head_dim)
        .squeeze(0)
        .permute(1, 0, 2)
        .reshape(seq_len, -1)
    )
    v_reshaped = (
        values.view(batch_size, num_heads, seq_len, head_dim)
        .squeeze(0)
        .permute(1, 0, 2)
        .reshape(seq_len, -1)
    )

    # Store for global analysis
    all_keys.append(keys.reshape(-1))
    all_values.append(values.reshape(-1))

    # Calculate statistics for this layer
    k_sparsity = (keys.abs() < 1e-5).float().mean().item()
    v_sparsity = (values.abs() < 1e-5).float().mean().item()
    k_mean = keys.abs().mean().item()
    v_mean = values.abs().mean().item()
    k_std = keys.std().item()
    v_std = values.std().item()

    # Correlation between keys and values
    k_flat = keys.reshape(-1).numpy()
    v_flat = values.reshape(-1).numpy()
    correlation = np.corrcoef(k_flat, v_flat)[0, 1]

    # Visualize this layer's KV patterns
    sns.heatmap(k_reshaped, cmap="viridis", ax=axes[layer_idx, 0])
    axes[layer_idx, 0].set_title(f"Layer {layer_idx} Keys")
    axes[layer_idx, 0].set_xlabel("Head Dimension × Heads")
    axes[layer_idx, 0].set_ylabel("Position")

    sns.heatmap(v_reshaped, cmap="magma", ax=axes[layer_idx, 1])
    axes[layer_idx, 1].set_title(f"Layer {layer_idx} Values")
    axes[layer_idx, 1].set_xlabel("Head Dimension × Heads")
    axes[layer_idx, 1].set_ylabel("Position")

    # Add to statistics
    layer_stats.append(
        {
            "layer": layer_idx,
            "k_sparsity": k_sparsity,
            "v_sparsity": v_sparsity,
            "k_mean": k_mean,
            "v_mean": v_mean,
            "k_std": k_std,
            "v_std": v_std,
            "kv_correlation": correlation,
        }
    )

plt.tight_layout()
plt.savefig("kv_cache_layers.png", dpi=300)

# ====================
# Global Statistics
# ====================

# Concatenate all keys and values for global analysis
k_matrix = torch.cat(all_keys, dim=0)
v_matrix = torch.cat(all_values, dim=0)

# Create a dataframe with layer statistics
stats_df = pd.DataFrame(layer_stats)
print("Layer-wise Statistics:")
print(stats_df)

# Plot layer statistics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(stats_df["layer"], stats_df["k_sparsity"], "b-o", label="Key Sparsity")
plt.plot(stats_df["layer"], stats_df["v_sparsity"], "r-o", label="Value Sparsity")
plt.title("Sparsity Across Layers")
plt.xlabel("Layer")
plt.ylabel("Sparsity (ratio of near-zero values)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(stats_df["layer"], stats_df["kv_correlation"], "g-o")
plt.title("Key-Value Correlation Across Layers")
plt.xlabel("Layer")
plt.ylabel("Correlation")

plt.subplot(2, 2, 3)
plt.plot(stats_df["layer"], stats_df["k_std"], "b-o", label="Key Std")
plt.plot(stats_df["layer"], stats_df["v_std"], "r-o", label="Value Std")
plt.title("Standard Deviation Across Layers")
plt.xlabel("Layer")
plt.ylabel("Standard Deviation")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(stats_df["layer"], stats_df["k_mean"], "b-o", label="Key Mean")
plt.plot(stats_df["layer"], stats_df["v_mean"], "r-o", label="Value Mean")
plt.title("Mean Absolute Value Across Layers")
plt.xlabel("Layer")
plt.ylabel("Mean")
plt.legend()

plt.tight_layout()
plt.savefig("kv_cache_statistics.png", dpi=300)

# ====================
# Token-level Analysis
# ====================

# Decode tokens for analysis
input_ids = inputs["input_ids"][0].cpu().numpy()
tokens = tokenizer.convert_ids_to_tokens(input_ids)
token_strings = [tokenizer.decode([id]) for id in input_ids]

# Create arrays to store token-wise statistics
token_stats = []

# Calculate key and value magnitudes for each token
for pos in range(seq_len):
    # Get token identity
    token_id = input_ids[pos]
    token = tokens[pos]
    token_str = token_strings[pos].strip()

    # Calculate average magnitudes across all layers and heads
    k_magnitudes = []
    v_magnitudes = []

    for layer_idx, layer_kv in enumerate(kv_cache):
        keys = layer_kv[0].detach().cpu()[0, :, pos, :]  # [num_heads, head_dim]
        values = layer_kv[1].detach().cpu()[0, :, pos, :]  # [num_heads, head_dim]

        k_mag = keys.abs().mean().item()  # Average magnitude across all heads
        v_mag = values.abs().mean().item()

        k_magnitudes.append(k_mag)
        v_magnitudes.append(v_mag)

    # Average across all layers
    avg_k_mag = np.mean(k_magnitudes)
    avg_v_mag = np.mean(v_magnitudes)

    # Add to token statistics
    token_stats.append(
        {
            "position": pos,
            "token_id": token_id,
            "token": token,
            "token_str": token_str,
            "key_magnitude": avg_k_mag,
            "value_magnitude": avg_v_mag,
        }
    )

# Convert to DataFrame for easier analysis
token_df = pd.DataFrame(token_stats)

# Categorize tokens into types before using them
token_types = []
for token in token_df["token_str"]:
    # Check for numbers
    if any(char.isdigit() for char in token):
        token_types.append("number")
    # Check for punctuation
    elif any(char in ".,;:!?()[]{}\"'`" for char in token):
        token_types.append("punctuation")
    # Check for common stopwords (you can expand this)
    elif token.lower().strip() in ["the", "of", "and", "a", "to", "in", "that", "with"]:
        token_types.append("stopword")
    # Check for capitalized words (potential proper nouns)
    elif token and token[0].isupper():
        token_types.append("proper_noun")
    # Everything else
    else:
        token_types.append("other")

# Add the token types to the dataframe
token_df["token_type"] = token_types

print("\nToken-wise Statistics (Top 10 by Key Magnitude):")
print(token_df.sort_values(by="key_magnitude", ascending=False).head(10))
print("\nToken-wise Statistics (Top 10 by Value Magnitude):")
print(token_df.sort_values(by="value_magnitude", ascending=False).head(10))

# Print lowest importance tokens
print("\nToken-wise Statistics (Bottom 10 by Key Magnitude):")
print(token_df.sort_values(by="key_magnitude", ascending=True).head(10))
print("\nToken-wise Statistics (Bottom 10 by Value Magnitude):")
print(token_df.sort_values(by="value_magnitude", ascending=True).head(10))

# Create a visualization for low-importance tokens
bottom_tokens = pd.concat(
    [
        token_df.sort_values(by="key_magnitude", ascending=True).head(10),
        token_df.sort_values(by="value_magnitude", ascending=True).head(10),
    ]
).drop_duplicates()

plt.figure(figsize=(15, 10))

# Create a scatter plot with all tokens, highlighting low importance ones
plt.scatter(
    token_df["key_magnitude"],
    token_df["value_magnitude"],
    alpha=0.4,
    c=token_df["position"],
    cmap="viridis",
    label="All tokens",
)

# Add special markers for lowest importance tokens
plt.scatter(
    bottom_tokens["key_magnitude"],
    bottom_tokens["value_magnitude"],
    color="red",
    s=100,
    marker="x",
    label="Lowest importance tokens",
)

# Label bottom tokens
for _, row in bottom_tokens.iterrows():
    plt.annotate(
        row["token_str"],
        (row["key_magnitude"], row["value_magnitude"]),
        xytext=(5, 5),
        textcoords="offset points",
        color="red",
    )

plt.colorbar(label="Position in sequence")
plt.title("Low Importance Tokens in KV Space")
plt.xlabel("Key Magnitude")
plt.ylabel("Value Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("low_importance_tokens.png", dpi=300)

# Add a combined visualization with both high and low importance tokens
plt.figure(figsize=(15, 10))

# Define top_tokens based on the highest key and value magnitudes
top_tokens = pd.concat(
    [
        token_df.sort_values(by="key_magnitude", ascending=False).head(10),
        token_df.sort_values(by="value_magnitude", ascending=False).head(10),
    ]
).drop_duplicates()

# Plot all tokens first
plt.scatter(
    token_df["key_magnitude"],
    token_df["value_magnitude"],
    alpha=0.3,
    c="gray",
    label="Regular tokens",
)

# Add high importance tokens
plt.scatter(
    top_tokens["key_magnitude"],
    top_tokens["value_magnitude"],
    color="blue",
    marker="o",
    s=80,
    label="Highest importance tokens",
)

# Add low importance tokens
plt.scatter(
    bottom_tokens["key_magnitude"],
    bottom_tokens["value_magnitude"],
    color="red",
    marker="x",
    s=80,
    label="Lowest importance tokens",
)

# Label tokens
for _, row in pd.concat([top_tokens, bottom_tokens]).iterrows():
    color = (
        "blue" if row["key_magnitude"] in top_tokens["key_magnitude"].values else "red"
    )
    plt.annotate(
        row["token_str"],
        (row["key_magnitude"], row["value_magnitude"]),
        xytext=(5, 5),
        textcoords="offset points",
        color=color,
        fontweight="bold",
    )

plt.title("High vs Low Importance Tokens in KV Space")
plt.xlabel("Key Magnitude")
plt.ylabel("Value Magnitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("token_importance_comparison.png", dpi=300)

# Add boxplots to compare magnitude distribution between token types
plt.figure(figsize=(15, 10))

# Prepare data for boxplot
token_types_unique = sorted(token_df["token_type"].unique())
key_data = [
    token_df[token_df["token_type"] == t]["key_magnitude"] for t in token_types_unique
]
value_data = [
    token_df[token_df["token_type"] == t]["value_magnitude"] for t in token_types_unique
]

# Create box plots
plt.subplot(2, 1, 1)
plt.boxplot(key_data, labels=token_types_unique)
plt.title("Distribution of Key Magnitudes by Token Type")
plt.ylabel("Key Magnitude")
plt.grid(True, axis="y")
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.boxplot(value_data, labels=token_types_unique)
plt.title("Distribution of Value Magnitudes by Token Type")
plt.ylabel("Value Magnitude")
plt.grid(True, axis="y")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("token_magnitude_distributions.png", dpi=300)

# Visualize token magnitudes
plt.figure(figsize=(15, 8))

# Plot magnitude by position
plt.subplot(2, 1, 1)
plt.plot(token_df["position"], token_df["key_magnitude"], "b-o", label="Key Magnitude")
plt.plot(
    token_df["position"], token_df["value_magnitude"], "r-o", label="Value Magnitude"
)
plt.title("KV Magnitudes by Token Position")
plt.xlabel("Token Position")
plt.ylabel("Average Magnitude")
plt.xticks(
    token_df["position"][::5], token_df["token_str"][::5], rotation=45, ha="right"
)
plt.legend()
plt.grid(True)

# Categorize tokens into types (this is a simple example - you can make it more sophisticated)
token_types = []
for token in token_df["token_str"]:
    # Check for numbers
    if any(char.isdigit() for char in token):
        token_types.append("number")
    # Check for punctuation
    elif any(char in ".,;:!?()[]{}\"'`" for char in token):
        token_types.append("punctuation")
    # Check for common stopwords (you can expand this)
    elif token.lower().strip() in ["the", "of", "and", "a", "to", "in", "that", "with"]:
        token_types.append("stopword")
    # Check for capitalized words (potential proper nouns)
    elif token and token[0].isupper():
        token_types.append("proper_noun")
    # Everything else
    else:
        token_types.append("other")

token_df["token_type"] = token_types

# Boxplot of magnitudes by token type
plt.subplot(2, 1, 2)
token_type_data = []
for token_type in set(token_types):
    type_df = token_df[token_df["token_type"] == token_type]
    token_type_data.append(
        {
            "token_type": token_type,
            "avg_key_mag": type_df["key_magnitude"].mean(),
            "avg_value_mag": type_df["value_magnitude"].mean(),
            "key_values": type_df["key_magnitude"].values,
            "value_values": type_df["value_magnitude"].values,
        }
    )

type_df = pd.DataFrame(token_type_data).sort_values(by="avg_key_mag", ascending=False)

# Create side-by-side bars for key and value magnitudes
x = np.arange(len(type_df))
width = 0.35
plt.bar(
    x - width / 2, type_df["avg_key_mag"], width, label="Key Magnitude", color="blue"
)
plt.bar(
    x + width / 2, type_df["avg_value_mag"], width, label="Value Magnitude", color="red"
)
plt.title("Average KV Magnitudes by Token Type")
plt.xlabel("Token Type")
plt.ylabel("Average Magnitude")
plt.xticks(x, type_df["token_type"], rotation=45, ha="right")
plt.legend()
plt.grid(True, axis="y")

plt.tight_layout()
plt.savefig("token_magnitudes.png", dpi=300)

# Create a more detailed token analysis for the top 20 tokens
top_tokens = pd.concat(
    [
        token_df.sort_values(by="key_magnitude", ascending=False).head(10),
        token_df.sort_values(by="value_magnitude", ascending=False).head(10),
    ]
).drop_duplicates()

plt.figure(figsize=(15, 10))

# Create a scatter plot of tokens with key vs value magnitudes
plt.scatter(
    token_df["key_magnitude"],
    token_df["value_magnitude"],
    alpha=0.7,
    c=token_df["position"],
    cmap="viridis",
)

# Label top tokens
for _, row in top_tokens.iterrows():
    plt.annotate(
        row["token_str"],
        (row["key_magnitude"], row["value_magnitude"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.colorbar(label="Position in sequence")
plt.title("Key vs Value Magnitudes for Input Tokens")
plt.xlabel("Key Magnitude")
plt.ylabel("Value Magnitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("token_key_value_scatter.png", dpi=300)

# ====================
# Value Distribution
# ====================

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.hist(k_matrix.numpy(), bins=100, alpha=0.7, color="blue")
plt.title("Distribution of Key Values")
plt.xlabel("Value")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(v_matrix.numpy(), bins=100, alpha=0.7, color="red")
plt.title("Distribution of Value Values")
plt.xlabel("Value")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("kv_value_distribution.png", dpi=300)

# ====================
# Dimensionality Reduction
# ====================

# Reshape the data for meaningful PCA analysis
# We'll analyze how keys/values vary across different positions in the sequence

# Initialize position-based arrays
position_keys = []
position_values = []

# For each position in the sequence
for pos in range(seq_len):
    # Extract features for this position across all layers and heads
    pos_k_features = []
    pos_v_features = []

    for layer_idx, layer_kv in enumerate(kv_cache):
        keys = layer_kv[0].detach().cpu()  # [batch_size, num_heads, seq_len, head_dim]
        values = (
            layer_kv[1].detach().cpu()
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Extract all head dimensions for this position
        for head in range(num_heads):
            pos_k_features.extend(keys[0, head, pos].numpy())
            pos_v_features.extend(values[0, head, pos].numpy())

    # Add this position's features
    position_keys.append(pos_k_features)
    position_values.append(pos_v_features)

# Convert to numpy arrays - shape becomes [seq_len, num_layers*num_heads*head_dim]
position_keys = np.array(position_keys)
position_values = np.array(position_values)

# PCA Analysis
pca = PCA(n_components=2)
k_pca = pca.fit_transform(position_keys)
v_pca = pca.fit_transform(position_values)

# Plot PCA with positions highlighted
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(
    k_pca[:, 0], k_pca[:, 1], c=np.arange(seq_len), cmap="viridis", alpha=0.8, s=30
)
plt.colorbar(scatter, label="Position in sequence")
plt.title("PCA of Keys by Position")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.subplot(1, 2, 2)
scatter = plt.scatter(
    v_pca[:, 0], v_pca[:, 1], c=np.arange(seq_len), cmap="magma", alpha=0.8, s=30
)
plt.colorbar(scatter, label="Position in sequence")
plt.title("PCA of Values by Position")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("kv_pca_by_position.png", dpi=300)

# ====================
# Head-level Analysis (Optional)
# ====================

# Initialize head-based arrays for one selected layer
layer_to_analyze = 0  # Change this to analyze different layers
head_keys = []
head_values = []

# Extract the selected layer
layer_keys = kv_cache[layer_to_analyze][0].detach().cpu()  # [batch, heads, seq, dim]
layer_values = kv_cache[layer_to_analyze][1].detach().cpu()

# For each head
for head in range(num_heads):
    # Get all positions for this head
    head_k = layer_keys[0, head].reshape(-1).numpy()  # Flatten seq_len × head_dim
    head_v = layer_values[0, head].reshape(-1).numpy()

    head_keys.append(head_k)
    head_values.append(head_v)

# Convert to numpy arrays
head_keys = np.array(head_keys)
head_values = np.array(head_values)

# Visualize attention head patterns
plt.figure(figsize=(15, 6))

# Only proceed if we have enough heads for meaningful PCA
if num_heads > 2:
    pca = PCA(n_components=2)
    head_k_pca = pca.fit_transform(head_keys)
    head_v_pca = pca.fit_transform(head_values)

    plt.subplot(1, 2, 1)
    plt.scatter(
        head_k_pca[:, 0],
        head_k_pca[:, 1],
        c=np.arange(num_heads),
        cmap="viridis",
        alpha=0.8,
        s=50,
    )
    plt.title(f"Head Specialization - Layer {layer_to_analyze} Keys")
    for i in range(num_heads):
        plt.annotate(f"H{i}", (head_k_pca[i, 0], head_k_pca[i, 1]))

    plt.subplot(1, 2, 2)
    plt.scatter(
        head_v_pca[:, 0],
        head_v_pca[:, 1],
        c=np.arange(num_heads),
        cmap="magma",
        alpha=0.8,
        s=50,
    )
    plt.title(f"Head Specialization - Layer {layer_to_analyze} Values")
    for i in range(num_heads):
        plt.annotate(f"H{i}", (head_v_pca[i, 0], head_v_pca[i, 1]))
else:
    plt.text(
        0.5,
        0.5,
        "Not enough heads for PCA visualization",
        ha="center",
        va="center",
        fontsize=14,
    )

plt.tight_layout()
plt.savefig("head_specialization.png", dpi=300)

# ====================
# Compressibility Analysis
# ====================

# Test simple quantization schemes
bits_to_test = [16, 8, 4, 2]
k_errors = []
v_errors = []

plt.figure(figsize=(15, 6))
for i, bits in enumerate(bits_to_test):
    # Calculate max and min for scaling
    k_min, k_max = k_matrix.min(), k_matrix.max()
    v_min, v_max = v_matrix.min(), v_matrix.max()

    # Number of quantization levels
    levels = 2**bits

    # Quantize and dequantize
    k_quantized = (
        torch.round((k_matrix - k_min) / (k_max - k_min) * (levels - 1))
        / (levels - 1)
        * (k_max - k_min)
        + k_min
    )
    v_quantized = (
        torch.round((v_matrix - v_min) / (v_max - v_min) * (levels - 1))
        / (levels - 1)
        * (v_max - v_min)
        + v_min
    )

    # Calculate error
    k_error = torch.mean(torch.abs(k_matrix - k_quantized)).item()
    v_error = torch.mean(torch.abs(v_matrix - v_quantized)).item()

    k_errors.append(k_error)
    v_errors.append(v_error)

# Plot quantization results
plt.plot(bits_to_test, k_errors, "b-o", label="Key Error")
plt.plot(bits_to_test, v_errors, "r-o", label="Value Error")
plt.title("Quantization Error vs Bit Precision")
plt.xlabel("Bits")
plt.ylabel("Mean Absolute Error")
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.savefig("kv_quantization_error.png", dpi=300)

# ====================
# Summary of Findings
# ====================
print("\n=== KV Cache Analysis Summary ===")
print(f"Total parameters analyzed: {k_matrix.shape[0]}")
print(f"Overall Key Sparsity: {(k_matrix.abs() < 1e-5).float().mean():.4f}")
print(f"Overall Value Sparsity: {(v_matrix.abs() < 1e-5).float().mean():.4f}")
print(
    f"Highest layer-wise key sparsity: {stats_df['k_sparsity'].max():.4f} at layer {stats_df['k_sparsity'].idxmax()}"
)
print(
    f"Highest layer-wise value sparsity: {stats_df['v_sparsity'].max():.4f} at layer {stats_df['v_sparsity'].idxmax()}"
)
print(
    f"Highest key-value correlation: {stats_df['kv_correlation'].max():.4f} at layer {stats_df['kv_correlation'].idxmax()}"
)

print("\n=== Optimization Opportunities ===")
print(
    f"1. Pruning potential: {(k_matrix.abs() < 1e-3).float().mean():.4f} of keys and {(v_matrix.abs() < 1e-3).float().mean():.4f} of values are near-zero"
)
print(
    f"2. Quantization: {bits_to_test[np.argmin(k_errors)]}-bit precision for keys and {bits_to_test[np.argmin(v_errors)]}-bit for values"
)
print(f"3. Layer-specific optimizations: Focus on layers with highest sparsity")
print(f"4. Position-based pruning: Later positions show more distinct patterns")
