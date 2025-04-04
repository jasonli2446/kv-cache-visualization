"""
Utilities for collecting and preparing data from models.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def extract_kv_cache(model, tokenizer, prompt, device):
    """
    Extract KV cache from a model for a given prompt.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Associated tokenizer
        prompt: Input text
        device: Device to run on (cuda/cpu)
        
    Returns:
        Tuple of (kv_cache, model_outputs)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)
        kv_cache = outputs.past_key_values
        
    # Extract basic info about the model
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    print(f"KV cache structure: {num_layers} layers, {num_heads} heads, {seq_len} sequence length, {head_dim} head dimensions")
    
    return kv_cache, outputs

def extract_model_info(model):
    """
    Extract model architecture information.
    
    Args:
        model: HuggingFace transformer model
        
    Returns:
        Dict with model information
    """
    # Try to extract architecture details
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        # Try different attribute names used by some models
        num_layers = getattr(model.config, "n_layer", None)
    
    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        # Try different attribute names
        num_heads = getattr(model.config, "n_head", None)
        
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        # Try different attribute names
        hidden_size = getattr(model.config, "n_embd", None)
    
    head_dim = hidden_size // num_heads if hidden_size and num_heads else None
    
    return {
        "model_type": model.config.model_type,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "vocab_size": model.config.vocab_size,
        "total_params": sum(p.numel() for p in model.parameters())
    }

def prepare_kv_cache_data(kv_cache, threshold_pct=config.SPARSITY_THRESHOLD_PERCENTAGE):
    """
    Process KV cache for analysis.
    
    Args:
        kv_cache: KV cache from model output
        threshold_pct: Percentage of max value to use as threshold
        
    Returns:
        Dict with processed data: keys, values, thresholds, dimensions
    """
    # Extract all keys and values
    all_keys = []
    all_values = []
    
    for layer_idx, layer_kv in enumerate(kv_cache):
        keys, values = layer_kv
        all_keys.append(keys)
        all_values.append(values)
    
    # Calculate global max values for thresholding
    global_k_max = max([layer_kv[0].abs().max().item() for layer_kv in kv_cache])
    global_v_max = max([layer_kv[1].abs().max().item() for layer_kv in kv_cache])
    
    # Calculate thresholds
    k_threshold = threshold_pct / 100.0 * global_k_max
    v_threshold = threshold_pct / 100.0 * global_v_max
    
    # Extract dimensions
    num_layers = len(kv_cache)
    batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
    
    return {
        "all_keys": all_keys,
        "all_values": all_values,
        "k_threshold": k_threshold,
        "v_threshold": v_threshold,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "global_k_max": global_k_max,
        "global_v_max": global_v_max
    }

def get_memory_usage(kv_cache):
    """
    Calculate memory usage of KV cache.
    
    Args:
        kv_cache: KV cache from model output
        
    Returns:
        Memory usage in MB
    """
    total_bytes = 0
    
    for layer_kv in kv_cache:
        k, v = layer_kv
        total_bytes += k.element_size() * k.nelement() 
        total_bytes += v.element_size() * v.nelement()
    
    return total_bytes / (1024 * 1024)  # Convert to MB