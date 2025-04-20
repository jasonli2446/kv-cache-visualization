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

## 12. Layer Similarity Matrix (`layer_similarity_matrix.png`)

This visualization shows the similarity relationships between different layers:

- **What you're seeing**: A heatmap where each cell represents the cosine similarity between two layers.
- **Interpretation**:
  - **Bright clusters**: Groups of layers with highly similar representations, suggesting redundancy.
  - **Diagonal patterns**: Adjacent layers with high similarity may be candidates for merging.
  - **Distinct blocks**: May indicate functional groups of layers that serve similar purposes.
- **Implementation details**: Calculated using cosine similarity between flattened layer representations, with values ranging from 0 (completely different) to 1 (identical).

---

## 13. Head Similarity Matrix (`head_similarity_matrix.png`)

This visualization shows similarity relationships between attention heads:

- **What you're seeing**: A heatmap where each cell represents the cosine similarity between two attention heads, potentially across different layers.
- **Interpretation**:
  - **Bright clusters**: Groups of heads with nearly identical behavior that could be merged.
  - **Cross-layer similarities**: Heads from different layers that capture similar patterns.
  - **Isolated heads**: Unique heads with no clear similarity to others may be performing specialized functions.
- **Implementation details**: Uses cosine similarity between the activation patterns of each head across all token positions.

---

## 14. Embedding Dimension Correlations (`embedding_dimension_correlations.png`)

This visualization shows correlations between embedding dimensions:

- **What you're seeing**: Two correlation matrices showing Pearson correlation coefficients between embedding dimensions for keys and values.
- **Interpretation**:
  - **Strong positive correlations**: Dimensions that tend to activate together (red/yellow).
  - **Strong negative correlations**: Dimensions with opposing activation patterns (blue).
  - **Blocks of correlation**: Suggest groups of dimensions that could be compressed together.
- **Implementation details**: Uses Pearson correlation, with values ranging from -1 (perfectly anti-correlated) to 1 (perfectly correlated).

---

## 15. Token Similarity Matrix (`token_similarity_matrix.png`)

This visualization shows similarity between different token positions:

- **What you're seeing**: A heatmap where each cell represents the cosine similarity between KV cache entries at two token positions.
- **Interpretation**:
  - **Bright diagonal**: Self-similarity (always highest).
  - **Bright off-diagonal regions**: Token positions with similar representations that could share cache entries.
  - **Block patterns**: May indicate phrases or semantic units that have coherent representations.
- **Implementation details**: Combines both key and value similarities into a single metric, using cosine similarity between token representations.

---

## 16. Token KV Similarity Matrices (`token_kv_similarity_matrices.png`)

This visualization provides separate views of key and value similarities between tokens:

- **What you're seeing**: Side-by-side heatmaps showing key similarity (left) and value similarity (right) between token positions.
- **Interpretation**:
  - **Differences between keys and values**: May reveal which aspect is more important for certain tokens.
  - **Clustering patterns**: Groups of tokens that could share key or value representations.
  - **Sequential patterns**: How similarity changes with token distance in the sequence.
- **Implementation details**: Uses cosine similarity with separate calculations for keys and values.

---

## 17. Token-Embedding Patterns (`token_embedding_patterns.png`)

This visualization reveals cross-dimensional relationships between token positions and embedding dimensions:

- **What you're seeing**: Side-by-side heatmaps showing activation patterns where rows represent token positions and columns represent embedding dimensions, for both keys (left) and values (right).
- **Interpretation**:
  - **Bright horizontal lines**: Token positions that strongly activate across many dimensions.
  - **Bright vertical lines**: Embedding dimensions that respond to many different tokens.
  - **Bright clusters**: Specific token-dimension pairs with strong associations.
  - **Pattern similarities**: Comparing key and value patterns reveals which aspect dominates in different regions.
- **Implementation details**: Activation strengths are averaged across layers and heads, then normalized for better visualization. This analysis identifies opportunities for cross-dimensional compression that aren't visible when analyzing tokens or embeddings separately.

---

## 18. Embedding Dimension Groups (`embedding_dimension_groups.png`)

This visualization shows groups of similar embedding dimensions that can be compressed:

- **What you're seeing**: Two bar charts showing key dimension groups (top) and value dimension groups (bottom), with annotations listing the specific dimensions in each group.
- **Interpretation**:
  - **Taller bars**: Larger groups of similar dimensions that provide greater compression potential.
  - **Group count**: More groups indicates fine-grained clustering of similar dimensions.
  - **Dimension patterns**: Which specific dimensions tend to behave similarly across contexts.
- **Implementation details**: Dimensions are grouped based on cosine similarity of their activation patterns across token positions, with a maximum group size limit to prevent over-clustering.

---