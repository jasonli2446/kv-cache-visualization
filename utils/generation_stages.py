"""
Utilities for capturing KV cache at different generation stages.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def capture_generation_stages(model, tokenizer, prompt, gen_tokens=20, device="cuda"):
    """
    Capture KV cache at different stages of text generation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt text
        gen_tokens: Number of tokens to generate
        device: Device to run on
    
    Returns:
        Dict with KV caches from different stages
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    
    # Initialize storage for KV caches
    kv_caches = {
        "prefill": None,
        "early_decoding": None, 
        "mid_decoding": None,
        "late_decoding": None
    }
    
    # Define sampling points
    early_point = max(1, gen_tokens // 5)  # ~20% of generation
    mid_point = gen_tokens // 2            # 50% of generation
    late_point = gen_tokens - 1            # End of generation
    
    # Prepare generation parameters
    gen_tokens_tensor = torch.ones((1, 1), dtype=torch.long, device=device) * tokenizer.bos_token_id
    past_key_values = None
    
    # Capture prefill stage
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        kv_caches["prefill"] = outputs.past_key_values
        past_key_values = outputs.past_key_values
        
        # Get the next token prediction
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Begin decoding loop
        generated_tokens = []
        for i in range(gen_tokens):
            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            # Capture at sampling points
            if i == early_point - 1:
                kv_caches["early_decoding"] = past_key_values
            elif i == mid_point - 1:
                kv_caches["mid_decoding"] = past_key_values
            elif i == late_point - 1:
                kv_caches["late_decoding"] = past_key_values
    
    # Generate the decoded text
    decoded_output = tokenizer.decode(generated_tokens)
    
    return {
        "kv_caches": kv_caches,
        "generated_text": decoded_output,
        "input_length": input_len,
        "generated_tokens": generated_tokens
    }

def extract_kv_difference(kv_cache1, kv_cache2):
    """
    Calculate differences between two KV caches.
    
    Args:
        kv_cache1: First KV cache
        kv_cache2: Second KV cache (should be larger/later than kv_cache1)
        
    Returns:
        Dict with difference metrics
    """
    # Ensure the second cache is larger/later than the first
    if len(kv_cache1[0][0][0][0]) > len(kv_cache2[0][0][0][0]):
        kv_cache1, kv_cache2 = kv_cache2, kv_cache1
    
    num_layers = len(kv_cache1)
    
    # Calculate differences for overlapping parts
    k_differences = []
    v_differences = []
    
    for layer_idx in range(num_layers):
        keys1, values1 = kv_cache1[layer_idx]
        keys2, values2 = kv_cache2[layer_idx]
        
        # Extract dimensions
        seq_len1 = keys1.shape[2]
        
        # Calculate differences for overlapping tokens
        k_diff = torch.norm(keys2[:,:,:seq_len1,:] - keys1).item()
        v_diff = torch.norm(values2[:,:,:seq_len1,:] - values1).item()
        
        k_differences.append(k_diff)
        v_differences.append(v_diff)
    
    return {
        "k_differences": k_differences,
        "v_differences": v_differences,
        "avg_k_difference": np.mean(k_differences),
        "avg_v_difference": np.mean(v_differences),
        "max_k_difference": max(k_differences),
        "max_v_difference": max(v_differences)
    }

def compare_generation_stages(prefill_kv_cache, decoding_kv_cache_list, k_threshold=None, v_threshold=None):
    """
    Compare statistics between different generation stages.
    
    Args:
        prefill_kv_cache: KV cache from prefill stage
        decoding_kv_cache_list: List of KV caches from different decoding stages
        k_threshold: Threshold for key sparsity
        v_threshold: Threshold for value sparsity
        
    Returns:
        DataFrame with comparative statistics
    """
    # Calculate thresholds if not provided
    if k_threshold is None or v_threshold is None:
        # Get max values across all caches
        all_k_maxes = [max([layer_kv[0].abs().max().item() for layer_kv in kv_cache]) 
                      for kv_cache in [prefill_kv_cache] + decoding_kv_cache_list]
        all_v_maxes = [max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
                      for kv_cache in [prefill_kv_cache] + decoding_kv_cache_list]
        
        global_k_max = max(all_k_maxes)
        global_v_max = max(all_v_maxes)
        
        k_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_k_max
        v_threshold = config.SPARSITY_THRESHOLD_PERCENTAGE / 100.0 * global_v_max
    
    # Need to import analysis functions here to avoid circular imports
    from analysis.token_analysis import analyze_token_positions
    
    # Store comparison statistics
    comparison_stats = []
    
    # Analyze prefill stage
    prefill_token_df = analyze_token_positions(prefill_kv_cache, k_threshold, v_threshold)
    prefill_avg_k_sparsity = prefill_token_df['avg_k_sparsity'].mean()
    prefill_avg_v_sparsity = prefill_token_df['avg_v_sparsity'].mean()
    
    comparison_stats.append({
        "stage": "prefill",
        "avg_k_sparsity": prefill_avg_k_sparsity,
        "avg_v_sparsity": prefill_avg_v_sparsity,
        "new_tokens_k_sparsity": None,  # No new tokens in prefill
        "new_tokens_v_sparsity": None
    })
    
    # Analyze decoding stages
    stage_names = ["early_decoding", "mid_decoding", "late_decoding"]
    if len(decoding_kv_cache_list) != len(stage_names):
        stage_names = [f"decoding_{i+1}" for i in range(len(decoding_kv_cache_list))]
        
    for i, (stage_name, decoding_kv_cache) in enumerate(zip(stage_names, decoding_kv_cache_list)):
        decoding_token_df = analyze_token_positions(decoding_kv_cache, k_threshold, v_threshold)
        decoding_avg_k_sparsity = decoding_token_df['avg_k_sparsity'].mean()
        decoding_avg_v_sparsity = decoding_token_df['avg_v_sparsity'].mean()
        
        # Get statistics for newly generated tokens only
        prefill_len = len(prefill_token_df)
        if len(decoding_token_df) > prefill_len:
            new_tokens_df = decoding_token_df.iloc[prefill_len:]
            new_tokens_k_sparsity = new_tokens_df['avg_k_sparsity'].mean()
            new_tokens_v_sparsity = new_tokens_df['avg_v_sparsity'].mean()
        else:
            new_tokens_k_sparsity = None
            new_tokens_v_sparsity = None
        
        comparison_stats.append({
            "stage": stage_name,
            "avg_k_sparsity": decoding_avg_k_sparsity,
            "avg_v_sparsity": decoding_avg_v_sparsity,
            "new_tokens_k_sparsity": new_tokens_k_sparsity,
            "new_tokens_v_sparsity": new_tokens_v_sparsity
        })
    
    # Convert to dataframe
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_stats)
    
    return comparison_df