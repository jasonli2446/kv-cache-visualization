"""
Script to run Tucker decomposition analysis on KV cache.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_kv_cache, prepare_input_for_model
from utils.dataset_loaders import get_wikitext_prompt
from compression.tucker_decomposition import TuckerDecomposition, plot_tucker_components
import config

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def analyze_rank_sensitivity(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    rank_range: Tuple[int, int] = (8, 128),
    num_ranks: int = 8,
    save_dir: str = "graphs/tucker"
) -> Dict:
    """
    Analyze how different ranks affect compression and reconstruction quality.
    
    Args:
        kv_cache: List of (key, value) tensor pairs
        rank_range: Range of ranks to test (min, max)
        num_ranks: Number of rank values to test
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing analysis results
    """
    ranks = np.linspace(rank_range[0], rank_range[1], num_ranks, dtype=int)
    compression_ratios = []
    key_errors = []
    value_errors = []
    
    print("\n=== Rank Sensitivity Analysis ===")
    for rank in ranks:
        print(f"\nTesting rank {rank}...")
        tucker = TuckerDecomposition(ranks=(rank, rank))
        results = tucker.decompose(kv_cache)
        
        compression_ratios.append(tucker.get_compression_ratio())
        key_errors.append(results['reconstruction_error']['keys'])
        value_errors.append(results['reconstruction_error']['values'])
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(ranks, compression_ratios, 'b-o')
    plt.xlabel('Rank')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio vs Rank')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(ranks, key_errors, 'r-o')
    plt.xlabel('Rank')
    plt.ylabel('Reconstruction Error')
    plt.title('Key Reconstruction Error vs Rank')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(ranks, value_errors, 'g-o')
    plt.xlabel('Rank')
    plt.ylabel('Reconstruction Error')
    plt.title('Value Reconstruction Error vs Rank')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rank_sensitivity.png"))
    plt.close()
    
    return {
        'ranks': ranks.tolist(),
        'compression_ratios': compression_ratios,
        'key_errors': key_errors,
        'value_errors': value_errors
    }

def analyze_layer_sensitivity(
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
    rank: int = 32,
    save_dir: str = "graphs/tucker"
) -> Dict:
    """
    Analyze how reconstruction quality varies across layers.
    
    Args:
        kv_cache: List of (key, value) tensor pairs
        rank: Rank to use for decomposition
        save_dir: Directory to save results
        
    Returns:
        Dictionary containing analysis results
    """
    num_layers = len(kv_cache)
    key_errors = []
    value_errors = []
    
    print("\n=== Layer Sensitivity Analysis ===")
    for layer_idx in range(num_layers):
        print(f"\nAnalyzing layer {layer_idx + 1}/{num_layers}...")
        tucker = TuckerDecomposition(ranks=(rank, rank))
        results = tucker.decompose([kv_cache[layer_idx]])
        
        key_errors.append(results['reconstruction_error']['keys'])
        value_errors.append(results['reconstruction_error']['values'])
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(num_layers), key_errors, 'r-o')
    plt.xlabel('Layer')
    plt.ylabel('Reconstruction Error')
    plt.title('Key Reconstruction Error by Layer')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(num_layers), value_errors, 'g-o')
    plt.xlabel('Layer')
    plt.ylabel('Reconstruction Error')
    plt.title('Value Reconstruction Error by Layer')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layer_sensitivity.png"))
    plt.close()
    
    return {
        'layers': list(range(num_layers)),
        'key_errors': key_errors,
        'value_errors': value_errors
    }

def run_tucker_analysis(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    prompt: str = None,
    embedding_rank: int = 32,
    token_rank: int = 32,
    save_dir: str = "graphs/tucker",
    sequence_length: int = 2048,  # Increased to max context length for Llama-3B
    run_sensitivity_analysis: bool = True
) -> Dict:
    """
    Run Tucker decomposition analysis on the KV cache.
    
    Args:
        model_name: Name of the model to analyze
        prompt: Optional prompt to use for analysis
        embedding_rank: Rank for embedding dimension
        token_rank: Rank for token dimension
        save_dir: Directory to save visualizations
        sequence_length: Target sequence length
        run_sensitivity_analysis: Whether to run layer sensitivity analysis
        
    Returns:
        Dictionary containing analysis results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and get KV cache
    print(f"Loading model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(model_name)
    
    if prompt is None:
        # Get a longer prompt from WikiText
        prompt = get_wikitext_prompt(config.WIKITEXT_INDEX)
        prompt = prepare_input_for_model(prompt, tokenizer, model_name, max_tokens=sequence_length)
    
    print(f"Using prompt of length {len(tokenizer.encode(prompt))} tokens")
    
    print("Extracting KV cache...")
    kv_cache, _ = extract_kv_cache(model, tokenizer, prompt, device)  # Unpack the tuple
    
    # Print KV cache structure
    num_layers = len(kv_cache)
    # Get shape from the key tensor in the first layer
    key_tensor = kv_cache[0][0]  # First layer, key tensor
    batch_size, num_heads, seq_len, head_dim = key_tensor.shape
    print(f"KV cache structure: {num_layers} layers, {num_heads} heads, {seq_len} sequence length, {head_dim} head dimensions")
    print(f"Total KV cache size: {num_layers * batch_size * num_heads * seq_len * head_dim * 2 * 4 / (1024*1024):.2f} MB")
    
    # Perform Tucker decomposition
    print("Performing Tucker decomposition...")
    tucker = TuckerDecomposition(ranks=(embedding_rank, token_rank))
    decomposition_results = tucker.decompose(kv_cache)
    
    # Calculate compression ratio
    compression_ratio = tucker.get_compression_ratio()
    
    # Print results
    print("\n=== Tucker Decomposition Analysis Results ===")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Key Reconstruction Error: {decomposition_results['reconstruction_error']['keys']:.4f}")
    print(f"Value Reconstruction Error: {decomposition_results['reconstruction_error']['values']:.4f}")
    
    # Calculate memory usage
    original_size = num_layers * batch_size * num_heads * seq_len * head_dim * 2  # *2 for both keys and values
    compressed_size = original_size / compression_ratio
    print(f"\nMemory Usage:")
    print(f"Original KV Cache: {original_size * 4 / (1024*1024):.2f} MB")
    print(f"Compressed Size: {compressed_size * 4 / (1024*1024):.2f} MB")
    
    # Visualize components
    print("\nGenerating visualizations...")
    plot_tucker_components(tucker, os.path.join(save_dir, "tucker_components.png"))
    
    # Run layer sensitivity analysis if requested
    sensitivity_results = {}
    if run_sensitivity_analysis:
        print("\nRunning layer sensitivity analysis...")
        sensitivity_results['layer'] = analyze_layer_sensitivity(kv_cache, rank=embedding_rank, save_dir=save_dir)
    
    # Return results
    return {
        'compression_ratio': compression_ratio,
        'reconstruction_error': decomposition_results['reconstruction_error'],
        'memory_usage': {
            'original': original_size * 4 / (1024*1024),  # Convert to MB
            'compressed': compressed_size * 4 / (1024*1024)
        },
        'structure': {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'head_dim': head_dim
        },
        'sensitivity_analysis': sensitivity_results
    }

def main():
    """Main function to run Tucker decomposition analysis."""
    parser = argparse.ArgumentParser(description="Run Tucker decomposition analysis")
    parser.add_argument("--sequence_length", type=int, default=2048,
                      help="Target sequence length (default: 2048)")
    parser.add_argument("--embedding_rank", type=int, default=32,
                      help="Rank for embedding dimension")
    parser.add_argument("--token_rank", type=int, default=32,
                      help="Rank for token dimension")
    parser.add_argument("--no_sensitivity", action="store_true",
                      help="Skip layer sensitivity analysis")
    
    args = parser.parse_args()
    
    results = run_tucker_analysis(
        embedding_rank=args.embedding_rank,
        token_rank=args.token_rank,
        sequence_length=args.sequence_length,
        run_sensitivity_analysis=not args.no_sensitivity
    )
    
    # Print detailed results
    print("\n=== Detailed Analysis ===")
    print(f"Model Structure:")
    print(f"- Layers: {results['structure']['num_layers']}")
    print(f"- Heads per layer: {results['structure']['num_heads']}")
    print(f"- Sequence length: {results['structure']['seq_len']}")
    print(f"- Head dimension: {results['structure']['head_dim']}")
    
    print(f"\nMemory Analysis:")
    print(f"- Original size: {results['memory_usage']['original']:.2f} MB")
    print(f"- Compressed size: {results['memory_usage']['compressed']:.2f} MB")
    print(f"- Compression ratio: {results['compression_ratio']:.2f}x")
    
    print(f"\nReconstruction Quality:")
    print(f"- Key error: {results['reconstruction_error']['keys']:.4f}")
    print(f"- Value error: {results['reconstruction_error']['values']:.4f}")
    
    if results['sensitivity_analysis']:
        print("\nLayer Sensitivity Analysis Results:")
        print("Check the generated plots in graphs/tucker/ for detailed analysis")

if __name__ == "__main__":
    main() 