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
from analysis.dimension_analysis import analyze_dimensions, find_prunable_dimensions
from pruning.pruner import KVCachePruner
from pruning.evaluation import KVCacheEvaluator
from utils.data_collection import extract_kv_cache, extract_model_info, prepare_kv_cache_data
from visualization.plotting import plot_all_visualizations
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
    
    # Run analyses
    print("\nRunning layer analysis...")
    layer_df = analyze_layers(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running head analysis...")
    head_df = analyze_heads(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running dimension analysis...")
    dim_df, analyzed_layers = analyze_dimensions(kv_cache, data["k_threshold"], data["v_threshold"])
    
    print("Running head consistency analysis...")
    consistency_df = analyze_head_consistency(head_df)
    
    # Find prunable components
    prunable_layers = find_prunable_layers(layer_df)
    prunable_heads = find_prunable_heads(head_df)
    prunable_dims = find_prunable_dimensions(dim_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_all_visualizations(
        layer_df, 
        head_df, 
        data["all_keys"], 
        data["all_values"],
        data["k_threshold"],
        data["v_threshold"]
    )
    
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
    
    if not prunable_dims.empty:
        print("\nTop 5 Specific Pruning Targets:")
        for _, row in prunable_dims.head(5).iterrows():
            print(f"  Layer {int(row['layer'])}, Head {int(row['head'])}, Dimension {int(row['dimension'])}: " 
                f"K-Sparsity={row['k_sparsity']:.4f}, V-Sparsity={row['v_sparsity']:.4f}")
    
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
        "prunable_layers": prunable_layers,
        "prunable_heads": prunable_heads,
        "prunable_dims": prunable_dims,
        "data": data
    }

def run_pruning_simulation(model, tokenizer, prompt, continuation, device, 
                          layer_indices=None, head_indices=None, dim_indices=None):
    """Run pruning simulation on the model."""
    print(f"\n=== Running Pruning Simulation ===")
    
    # Initialize pruner
    pruner = KVCachePruner(model, tokenizer, device)
    evaluator = KVCacheEvaluator(model, tokenizer, device)
    
    # Generate KV cache from prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
    
    # Create pruning masks
    pruning_masks = pruner.create_pruning_mask(layer_indices, head_indices, dim_indices)
    
    # Apply pruning masks
    pruned_kv_cache = pruner.apply_pruning_masks(past_key_values, pruning_masks)
    
    # Calculate sparsity
    original_k_sparsity, original_v_sparsity = pruner.calculate_sparsity(past_key_values)
    pruned_k_sparsity, pruned_v_sparsity = pruner.calculate_sparsity(pruned_kv_cache)
    
    print(f"Original KV cache - Key sparsity: {original_k_sparsity:.2f}%, Value sparsity: {original_v_sparsity:.2f}%")
    print(f"Pruned KV cache - Key sparsity: {pruned_k_sparsity:.2f}%, Value sparsity: {pruned_v_sparsity:.2f}%")
    
    # Evaluate performance
    pruning_info = {
        "pruned_layers": layer_indices,
        "pruned_heads": head_indices,
        "pruned_dims": dim_indices
    }
    
    results = evaluator.evaluate_pruning(
        prompt=prompt,
        continuation=continuation,
        past_key_values=pruned_kv_cache,
        pruning_info=pruning_info
    )
    
    # Print results
    print("\n=== Pruning Simulation Results ===")
    print(f"Baseline Perplexity: {results['baseline_perplexity']:.4f}")
    print(f"Pruned Perplexity: {results['pruned_perplexity']:.4f}")
    print(f"Perplexity Change: {results['perplexity_change_percent']:.2f}%")
    print(f"Baseline Latency: {results['baseline_latency_ms']:.2f} ms")
    print(f"Pruned Latency: {results['pruned_latency_ms']:.2f} ms")
    print(f"Latency Change: {results['latency_change_percent']:.2f}%")
    
    # Print pruning details
    if layer_indices:
        print(f"Pruned layers: {layer_indices}")
    if head_indices:
        print(f"Pruned heads: {head_indices}")
    if dim_indices:
        print(f"Pruned dimensions: {len(dim_indices) if dim_indices else 0}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="KV Cache Analysis and Pruning")
    parser.add_argument("--mode", choices=["analyze", "prune"], default="analyze",
                        help="Run analysis or pruning simulation")
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
    
    print("\nDone!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")