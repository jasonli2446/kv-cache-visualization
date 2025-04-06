# KV Cache Visualization Explanations

## 1. Layer-wise Statistics (`layer_statistics.png`)

This figure contains four subplots showing critical metrics across layers:

### a) Sparsity Across Layers

- **What it shows**: The percentage of near-zero values (< 1% of max weight) in keys and values for each layer.
- **Interpretation**:
  - Higher values mean more potential for pruning.
  - Layers with 30%+ sparsity are prime candidates for optimization.
  - If sparsity increases in later layers, these layers may be less critical.
- **Implementation details**: Threshold used is 1% of the global maximum weight value across all layers.

### b) Key-Value Correlation Across Layers

- **What it shows**: The correlation coefficient between key and value tensors for each layer.
- **Interpretation**:
  - Higher correlation suggests redundancy between keys and values.
  - Layers with correlation > 0.5 might benefit from shared representations.
  - Spikes in correlation indicate layers where keys and values capture similar information.
- **Implementation details**: Calculated using Pearson correlation between flattened key and value tensors.

### c) Standard Deviation Across Layers

- **What it shows**: How widely the values are spread in each layer.
- **Interpretation**:
  - Higher std dev indicates more diverse representations.
  - Decreasing std dev in later layers might indicate information consolidation.
  - Large differences between key and value std dev suggest different roles.

### d) Mean Absolute Value Across Layers

- **What it shows**: The average magnitude of values in each layer.
- **Interpretation**:
  - Indicates overall activation strength.
  - Declining means in later layers suggest diminishing importance.
  - Consistent means suggest evenly distributed importance across layers.

---

## 2. Layer-wise Pruning Potential (`layer_pruning_potential.png`)

This bar chart compares pruning potential across layers:

- **What you're seeing**: Bars showing key and value sparsity for each layer, with horizontal lines indicating average sparsity.
- **Interpretation**:
  - **Tall bars**: Layers with highest pruning potential.
  - **Key vs value differences**: Some layers may have imbalanced sparsity between keys and values.
  - **Comparison to average**: Layers significantly above the average line are best pruning candidates.
- **Implementation details**: Horizontal dashed lines indicate the mean sparsity across all layers, providing a reference point for identifying above-average pruning candidates.

---

## 3. Head Sparsity Heatmaps (`head_sparsity.png`)

This visualization shows the sparsity distribution across heads and layers:

- **What you're seeing**: Two heatmaps showing key and value sparsity for each combination of layer (rows) and attention head (columns).
- **Color intensity**: Darker blue indicates higher sparsity (more pruning potential).
- **Interpretation**:
  - **Hot spots**: Head-layer combinations with high sparsity are prime pruning targets.
  - **Patterns across layers**: Some layers may consistently show high sparsity across heads.
  - **Patterns across heads**: Some heads may be consistently sparse across layers.
- **Implementation details**: Values represent the proportion of weights below 1% of the global maximum weight.

---

## 4. Head-wise Pruning Potential (`head_pruning_potential.png`)

This visualization compares pruning potential across attention heads:

- **What you're seeing**: Grouped bars showing average key and value sparsity for each head across all layers, with error bars showing variability.
- **Interpretation**:
  - **Tall bars**: Heads with highest pruning potential.
  - **Large error bars**: Heads with inconsistent behavior across layers.
  - **Consistently tall bars with small error**: Most reliable pruning targets.
- **Implementation details**: Error bars represent the standard deviation of key sparsity across layers, indicating how consistent the pruning potential is for each head.

---

## 5. Pruning Heatmap by Layer and Head (`pruning_heatmap_by_layer_head.png`)

This detailed heatmap visualizes pruning potential across the model architecture:

- **What you're seeing**: A heatmap where each cell represents an attention head at a specific layer, with color intensity indicating sparsity.
- **Interpretation**:
  - **Layer patterns**: Whether early, middle, or late layers have more pruning potential.
  - **Head patterns**: Whether specific head positions have consistent sparsity.
  - **Diagonal/checkerboard patterns**: May indicate systematic redundancy in the model architecture.
- **Implementation details**: Focuses specifically on key sparsity as the primary indicator of pruning potential.

---

## 6. Weight Magnitude Distribution (`weight_magnitude_distribution.png`)

These histograms show the distribution of absolute weight values:

- **What you're seeing**: Frequency distribution of key and value weight magnitudes, with pruning threshold marked.
- **Interpretation**:
  - **Left-skewed distribution**: Many small weights that could be pruned.
  - **Long tails**: Some very large values that are critical to preserve.
  - **Area to left of threshold**: Proportion of weights that would be pruned at current threshold.
- **Implementation details**: Y-axis uses logarithmic scale to better visualize the distribution across magnitudes. Red vertical line indicates the 1% threshold used for sparsity calculations.

---

## 7. Token Position Importance (`token_position_importance.png`)

This visualization shows the relative importance of each token position in the sequence:

- **What you're seeing**: Bar chart showing importance score for each token position, with overlaid line plots showing normalized key and value norms.
- **Interpretation**:
  - **Taller bars**: Token positions that have greater influence on the model's predictions.
  - **Position trends**: Whether earlier or later tokens in the sequence are more important.
  - **Key vs value contributions**: Differences between key and value norms can reveal which aspect dominates importance.
- **Implementation details**: Importance score combines normalized key norms, value norms, and attention energy metrics.

---

## 8. Generation Stages Comparison (`generation_stages_comparison.png`)

This analysis compares KV cache characteristics across different stages of text generation:

- **What you're seeing**: Two bar charts showing key and value sparsity across different generation stages (prefill, early, mid, and late decoding).
- **Interpretation**:
  - **Stage differences**: How sparsity changes as generation progresses.
  - **New tokens vs. context**: The red line (when present) shows sparsity for newly generated tokens only.
  - **Prefill vs. decoding**: Comparing initial context processing with subsequent token generation.
- **Implementation details**: Compares the same model at different stages of the generation process to identify temporal patterns.

---

## 9. Embedding Consistency (`embedding_consistency.png`)

This visualization shows sparsity patterns across embedding dimensions:

- **What you're seeing**: Two bar charts showing key and value sparsity for each embedding dimension across the model.
- **Interpretation**:
  - **Consistently sparse dimensions**: Dimensions with sparsity above 80% (red line) may be candidates for dimension reduction.
  - **Consistently dense dimensions**: Dimensions with sparsity below 20% (green line) are likely critical to preserve.
  - **Dimension-level patterns**: Whether certain dimensions show similar behavior in both keys and values.
- **Implementation details**: Analyzes sparsity at the finest granularity (individual embedding dimensions) across all layers, heads, and tokens.

---

## 10. Sparse-Dense Embedding Patterns (`sparse_dense_embedding_patterns.png`)

This multi-plot visualization shows how different embedding dimensions behave across token positions:

- **What you're seeing**: Four plots showing patterns of the most position-sensitive and position-invariant dimensions for keys and values.
- **Interpretation**:
  - **Position-sensitive dimensions**: Dimensions whose activations vary significantly based on token position (top row).
  - **Position-invariant dimensions**: Dimensions that maintain consistent behavior regardless of token position (bottom row).
  - **Pattern groupings**: Similar line patterns may indicate dimensions that could be grouped together.
- **Implementation details**: Variances are calculated across token positions to identify dimensions with highest and lowest sensitivity to position.

---

## 11. Overall Pruning Recommendations

Based on the visualizations, here are key takeaways for optimizing KV cache:

- **Layer-level pruning**: Focus on layers showing consistently high sparsity across multiple heads.
- **Head-level pruning**: Consider removing or compressing attention heads with high mean sparsity and low standard deviation.
- **Token-level pruning**: Lower importance token positions could be represented with reduced precision.
- **Embedding-level optimization**: Target specific embedding dimensions that are consistently sparse across contexts.
- **Generation-aware pruning**: Apply different pruning strategies at different stages of text generation.

The combination of these visualizations provides a comprehensive picture of pruning opportunities throughout the model, allowing for targeted optimizations that preserve model quality while reducing computational requirements.
