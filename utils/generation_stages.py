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