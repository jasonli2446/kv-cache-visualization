"""
Main entry point for KV cache analysis and pruning.
"""

import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from our modules
from analysis.layer_analysis import analyze_layers, find_prunable_layers
from analysis.head_analysis import analyze_heads, find_prunable_heads, analyze_head_consistency
from analysis.token_analysis import analyze_token_positions, analyze_token_layer_patterns, calculate_token_importance
from analysis.embedding_analysis import (
    analyze_embedding_dimensions, analyze_embedding_importance_by_layer, 
    identify_embedding_patterns, analyze_dimensions, find_prunable_dimensions
)
from pruning.pruner import KVCachePruner
from pruning.evaluation import KVCacheEvaluator
from utils.data_collection import extract_kv_cache, extract_model_info, prepare_kv_cache_data
from utils.generation_stages import capture_generation_stages, compare_generation_stages
# Import visualization functions directly
from visualization.layer_plots import plot_layer_statistics, plot_layer_pruning_potential
from visualization.head_plots import plot_head_sparsity, plot_head_pruning_potential, plot_head_layer_heatmap
from visualization.token_plots import plot_token_sparsity_heatmap, plot_token_position_importance, plot_generation_stages_comparison
from visualization.embedding_plots import plot_embedding_consistency, plot_embedding_importance_heatmap, plot_sparse_dense_embedding_patterns
from visualization.common import plot_weight_magnitude_distribution
import config

def run_analysis(model, tokenizer, prompt, device):
    """Run analysis pipeline on the model."""
    print(f"Running analysis on prompt: '{prompt[:50]}...'")
    
    # Extract KV cache
    kv_cache, outputs = extract_kv_cache(model, tokenizer, prompt, device)
    
    # Get model info
    model_info = extract_model_info(model)
    print("\nModel architecture:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Prepare data
    data = prepare_kv_cache_data(kv_cache)
    
    # Run standard analyses
    print("\nRunning layer analysis...")
    layer_df = analyze_layers(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running head analysis...")
    head_df = analyze_heads(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running dimension analysis...")
    dim_df, analyzed_layers = analyze_dimensions(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running head consistency analysis...")
    consistency_df = analyze_head_consistency(head_df)
    
    # Run new token-level analyses
    print("\nRunning token position analysis...")
    token_df = analyze_token_positions(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running token-layer pattern analysis...")
    k_sparsity_matrix, v_sparsity_matrix = analyze_token_layer_patterns(kv_cache, 
                                                                       data["k_threshold"], 
                                                                       data["v_threshold"])
    
    print("Calculating token importance...")
    token_importance_df = calculate_token_importance(kv_cache)
    
    # Run new embedding-level analyses
    print("\nRunning embedding dimension analysis...")
    embedding_df = analyze_embedding_dimensions(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running embedding importance analysis...")
    embedding_importance_results = analyze_embedding_importance_by_layer(kv_cache, 
                                                                       data["k_threshold"], 
                                                                       data["v_threshold"])
    
    print("Identifying embedding patterns...")
    embedding_pattern_results = identify_embedding_patterns(kv_cache, 
                                                          data["k_threshold"], 
                                                          data["v_threshold"])
    
    # Run analysis across generation stages
    print("\nAnalyzing generation stages...")
    gen_results = capture_generation_stages(model, tokenizer, prompt[:50], gen_tokens=20, device=device)
    gen_stages = gen_results["kv_caches"]
    gen_comparison = compare_generation_stages(gen_stages["prefill"], 
                                             [gen_stages["early_decoding"], 
                                              gen_stages["mid_decoding"], 
                                              gen_stages["late_decoding"]])
    
    # Find prunable components
    prunable_layers = find_prunable_layers(layer_df)
    prunable_heads = find_prunable_heads(head_df)
    prunable_dims = find_prunable_dimensions(dim_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Layer-level visualizations
    print("Generating layer-level visualizations...")
    plot_layer_statistics(layer_df)
    plot_layer_pruning_potential(layer_df)
    
    # Head-level visualizations
    print("Generating head-level visualizations...")
    plot_head_sparsity(head_df)
    plot_head_pruning_potential(head_df)
    plot_head_layer_heatmap(head_df)
    
    # Weight magnitude distributions
    print("Generating weight magnitude distributions...")
    plot_weight_magnitude_distribution(
        data["all_keys"], 
        data["all_values"],
        data["k_threshold"],
        data["v_threshold"]
    )
    
    # Token-level visualizations
    print("Generating token-level visualizations...")
    plot_token_sparsity_heatmap(k_sparsity_matrix, v_sparsity_matrix)
    plot_token_position_importance(token_importance_df, prompt, tokenizer)
    plot_generation_stages_comparison(gen_comparison, gen_results["generated_text"])
    
    # Embedding-level visualizations
    print("Generating embedding-level visualizations...")
    plot_embedding_consistency(embedding_df)
    plot_embedding_importance_heatmap(embedding_importance_results)
    plot_sparse_dense_embedding_patterns(embedding_pattern_results)
    
    # Print key findings
    print("\n=== KV Cache Analysis Summary ===")
    print(f"Overall Key Sparsity: {layer_df['k_sparsity'].mean():.4f}")
    print(f"Overall Value Sparsity: {layer_df['v_sparsity'].mean():.4f}")
    
    print("\nMost Prunable Layers:")
    top_layers = layer_df.sort_values(by=["k_sparsity", "v_sparsity"], ascending=False).head(3)
    for _, row in top_layers.iterrows():
        print(f"  Layer {int(row['layer'])}: Key Sparsity={row['k_sparsity']:.4f}, Value Sparsity={row['v_sparsity']:.4f}")
    
    print("\nMost Consistently Sparse Heads:")
    top_heads = consistency_df.sort_values(by=["k_sparsity_mean"], ascending=False).head(3)
    for _, row in top_heads.iterrows():
        print(f"  Head {int(row['head'])}: Mean Key Sparsity={row['k_sparsity_mean']:.4f} ± {row['k_sparsity_std']:.4f}")
    
    # Print token-level findings
    print("\nMost Important Token Positions:")
    for _, row in token_importance_df.head(3).iterrows():
        token_text = ""
        try:
            token_text = f" ('{tokenizer.decode([tokenizer.encode(prompt)[row['token_position']]])}')"
        except:
            pass
        print(f"  Position {int(row['token_position'])}{token_text}: Importance={row['importance_score']:.4f}")
    
    # Print embedding-level findings
    print("\nConsistently Sparse Embedding Dimensions:")
    top_sparse_dims = embedding_df.sort_values(by="k_sparsity", ascending=False).head(3)
    for _, row in top_sparse_dims.iterrows():
        print(f"  Dimension {int(row['dimension'])}: K-Sparsity={row['k_sparsity']:.4f}, V-Sparsity={row['v_sparsity']:.4f}")
    
    # Print generation stage comparison
    print("\nSparsity Across Generation Stages:")
    for _, row in gen_comparison.iterrows():
        print(f"  {row['stage']}: K-Sparsity={row['avg_k_sparsity']:.4f}, V-Sparsity={row['avg_v_sparsity']:.4f}")
    
    # Calculate pruning potential
    num_layers = data["num_layers"]
    num_heads = data["num_heads"]
    head_dim = data["head_dim"]
    total_params = num_layers * num_heads * head_dim * 2  # *2 for both keys and values
    prunable_params = len(prunable_dims) if not prunable_dims.empty else 0
    print(f"\nPruning Potential: {prunable_params}/{total_params} parameters ({prunable_params/total_params*100:.2f}%)")
    
    return {
        "kv_cache": kv_cache,
        "layer_df": layer_df,
        "head_df": head_df,
        "dim_df": dim_df,
        "consistency_df": consistency_df,
        "token_df": token_df,
        "token_importance_df": token_importance_df,
        "embedding_df": embedding_df,
        "embedding_importance_results": embedding_importance_results,
        "embedding_pattern_results": embedding_pattern_results,
        "generation_stages": gen_stages,
        "generation_comparison": gen_comparison,
        "prunable_layers": prunable_layers,
        "prunable_heads": prunable_heads,
        "prunable_dims": prunable_dims,
        "data": data
    }

def run_generation_stage_analysis(model, tokenizer, prompt, device):
    """Run generation stage analysis pipeline."""
    print(f"Running generation stage analysis on prompt: '{prompt[:50]}...'")
    
    # Capture KV cache at different generation stages
    gen_results = capture_generation_stages(
        model, 
        tokenizer, 
        prompt, 
        gen_tokens=30, 
        device=device
    )
    
    # Extract KV caches from different stages
    prefill_kv = gen_results["kv_caches"]["prefill"]
    early_kv = gen_results["kv_caches"]["early_decoding"]
    mid_kv = gen_results["kv_caches"]["mid_decoding"]
    late_kv = gen_results["kv_caches"]["late_decoding"]
    
    # Calculate thresholds for consistency across stages
    all_kv_caches = [prefill_kv, early_kv, mid_kv, late_kv]
    all_k_maxes = [max([layer_kv[0].abs().max().item() for layer_kv in kv]) 
                  for kv in all_kv_caches]
    all_v_maxes = [max([layer_kv[1].abs().max().item() for layer_kv in kv])
                  for kv in all_kv_caches]
    
    global_k_max = max(all_k_maxes)
    global_v_max = max(all_v_maxes)
    
    k_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_k_max
    v_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_v_max
    
    # Compare sparsity patterns across stages
    stage_comparison = compare_generation_stages(
        prefill_kv,
        [early_kv, mid_kv, late_kv],
        k_threshold,
        v_threshold
    )
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_generation_stages_comparison(
        stage_comparison, 
        gen_results["generated_text"]
    )
    
    # Print findings
    print("\n=== Generation Stages Analysis ===")
    print(f"Generated text: {gen_results['generated_text']}")
    print(f"Original input length: {gen_results['input_length']}")
    
    print("\nSparsity Across Generation Stages:")
    for _, row in stage_comparison.iterrows():
        new_tokens_info = ""
        if not pd.isna(row.get("new_tokens_k_sparsity", pd.NA)):
            new_tokens_info = f" (New tokens only: K={row['new_tokens_k_sparsity']:.4f}, V={row['new_tokens_v_sparsity']:.4f})"
        print(f"  {row['stage']}: K-Sparsity={row['avg_k_sparsity']:.4f}, V-Sparsity={row['avg_v_sparsity']:.4f}{new_tokens_info}")
    
    return {
        "generation_stages": gen_results["kv_caches"],
        "stage_comparison": stage_comparison,
        "generated_text": gen_results["generated_text"],
        "input_length": gen_results["input_length"],
        "generated_tokens": gen_results["generated_tokens"]
    }

def main():
    parser = argparse.ArgumentParser(description="KV Cache Analysis and Pruning")
    parser.add_argument("--mode", choices=["analyze", "prune", "analyze_generation"], default="analyze",
                        help="Run analysis, pruning simulation, or generation analysis")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, 
                        help="Model to analyze")
    parser.add_argument("--prune_layers", type=str, default="",
                        help="Comma-separated list of layers to prune")
    parser.add_argument("--prune_heads", type=str, default="",
                        help="Comma-separated list of heads to prune")
    parser.add_argument("--prune_method", choices=config.PRUNE_METHODS, 
                        default=config.DEFAULT_PRUNE_METHOD,
                        help="Method for pruning")
    parser.add_argument("--prompt", type=str, default=config.SAMPLE_PROMPT,
                        help="Prompt text for analysis/pruning")
    parser.add_argument("--continuation", type=str, default=config.SAMPLE_CONTINUATION,
                        help="Continuation text for evaluation")
    parser.add_argument("--analysis_focus", 
                        choices=["all", "layers", "heads", "tokens", "embeddings"], 
                        default="all",
                        help="Focus analysis on specific aspects")
    args = parser.parse_args()
    
    # Check for CUDA
    device_name = config.DEFAULT_DEVICE
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_name = "cpu"
    
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Memory optimization
    if device_name == "cuda":
        torch.cuda.empty_cache()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.mode == "analyze":
        # Run analysis pipeline
        analysis_results = run_analysis(model, tokenizer, args.prompt, device)
        
    elif args.mode == "analyze_generation":
        # Run generation stage analysis
        gen_analysis = run_generation_stage_analysis(model, tokenizer, args.prompt, device)
    '''

         elif args.mode == "prune":
        # Parse pruning parameters
        layer_indices = [int(i) for i in args.prune_layers.split(",")] if args.prune_layers else None
        head_indices = [int(i) for i in args.prune_heads.split(",")] if args.prune_heads else None
        
        # Run pruning simulation
        pruning_results = run_pruning_simulation(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            continuation=args.continuation,
            device=device,
            layer_indices=layer_indices,
            head_indices=head_indices
        )
    '''   

    
    print("\nDone!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")