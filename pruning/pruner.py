"""
Core functionality for KV cache pruning simulation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KVCachePruner:
    """Class for pruning KV cache in transformer models."""
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the pruner with a model and tokenizer.
        
        Args:
            model: HuggingFace transformer model
            tokenizer: Associated tokenizer
            device: Device to run on (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Extract model config info
        self.num_layers = getattr(model.config, "num_hidden_layers", None)
        if self.num_layers is None:
            # Try different attribute names used by some models
            self.num_layers = getattr(model.config, "n_layer", None)
        
        # Get the number of attention heads from config
        self.num_heads = getattr(model.config, "num_attention_heads", None)
        if self.num_heads is None:
            # Try different attribute names
            self.num_heads = getattr(model.config, "n_head", None)
        
        # For models with grouped-query attention (like Llama), get the actual number of KV heads
        self.num_kv_heads = getattr(model.config, "num_key_value_heads", None)
        if self.num_kv_heads is None:
            # Fall back to regular num_heads if not specified
            self.num_kv_heads = self.num_heads
            
        self.head_dim = getattr(model.config, "hidden_size", 0) // self.num_heads
        print(f"Model configuration: {self.num_layers} layers, {self.num_heads} attention heads, {self.num_kv_heads} KV heads, {self.head_dim} dims per head")
        
        # These will be set when we first see the KV cache
        self.actual_kv_heads = None
        self.actual_seq_len = None
        self.actual_head_dim = None
    
    def create_pruning_mask(self, 
                           layer_indices: List[int] = None,
                           head_indices: List[int] = None,
                           dim_indices: List[int] = None,
                           threshold: float = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create pruning masks for specified layers, heads, and dimensions.
        
        Args:
            layer_indices: List of layer indices to prune
            head_indices: List of head indices to prune
            dim_indices: List of dimension indices to prune
            threshold: Value threshold for pruning (if None, prune entire heads/layers)
            
        Returns:
            List of (key_mask, value_mask) tuples for each layer
        """
        # Initialize all masks to 1 (keep everything)
        pruning_masks = []
        for _ in range(self.num_layers):
            # Create masks for this layer (1s mean keep, 0s mean prune)
            k_mask = torch.ones(1, self.num_kv_heads, 1, self.head_dim).to(self.device)
            v_mask = torch.ones(1, self.num_kv_heads, 1, self.head_dim).to(self.device)
            pruning_masks.append((k_mask, v_mask))
        
        # Apply pruning to specified layers
        if layer_indices is not None:
            for layer_idx in layer_indices:
                if layer_idx >= self.num_layers:
                    continue
                    
                # If head_indices specified, only prune those heads
                if head_indices is not None:
                    for head_idx in head_indices:
                        # Make sure we don't go out of bounds for the actual KV heads
                        if head_idx < self.num_kv_heads:
                            # If dim_indices specified, only prune those dimensions
                            if dim_indices is not None:
                                for dim_idx in dim_indices:
                                    if dim_idx < self.head_dim:
                                        pruning_masks[layer_idx][0][:, head_idx, :, dim_idx] = 0  # Prune keys
                                        pruning_masks[layer_idx][1][:, head_idx, :, dim_idx] = 0  # Prune values
                            else:
                                pruning_masks[layer_idx][0][:, head_idx, :, :] = 0  # Prune keys
                                pruning_masks[layer_idx][1][:, head_idx, :, :] = 0  # Prune values
                # Otherwise prune entire layer
                else:
                    pruning_masks[layer_idx] = (
                        torch.zeros(1, self.num_kv_heads, 1, self.head_dim).to(self.device),
                        torch.zeros(1, self.num_kv_heads, 1, self.head_dim).to(self.device)
                    )
        
        return pruning_masks
    
    def apply_pruning_masks(self, past_key_values, pruning_masks):
        """
        Apply pruning masks to the KV cache.
        
        Args:
            past_key_values: Original KV cache from model
            pruning_masks: List of (key_mask, value_mask) tuples
            
        Returns:
            Pruned KV cache
        """
        # Update actual KV cache dimensions based on the first layer we see
        if self.actual_kv_heads is None:
            k, v = past_key_values[0]
            batch_size, actual_kv_heads, seq_len, head_dim = k.shape
            self.actual_kv_heads = actual_kv_heads
            self.actual_seq_len = seq_len
            self.actual_head_dim = head_dim
            
            # If this doesn't match our stored num_kv_heads, recreate the masks
            if actual_kv_heads != self.num_kv_heads:
                print(f"Warning: KV cache has {actual_kv_heads} KV heads, but model config shows {self.num_kv_heads}.")
                print(f"Adjusting pruning masks to match actual KV cache dimensions.")
                
                # Recreate pruning masks with correct dimensions
                self.num_kv_heads = actual_kv_heads
                pruning_masks = []
                for _ in range(self.num_layers):
                    k_mask = torch.ones(1, self.num_kv_heads, 1, self.head_dim).to(self.device)
                    v_mask = torch.ones(1, self.num_kv_heads, 1, self.head_dim).to(self.device)
                    pruning_masks.append((k_mask, v_mask))
        
        pruned_kv = []
        for layer_idx, (layer_past, (k_mask, v_mask)) in enumerate(zip(past_key_values, pruning_masks)):
            k, v = layer_past
            
            # Ensure mask dimensions match tensor dimensions
            if k.shape[1] != k_mask.shape[1]:
                # Adjust the mask to match the KV cache dimensions
                k_mask = torch.ones(1, k.shape[1], 1, k.shape[3]).to(self.device) 
                v_mask = torch.ones(1, v.shape[1], 1, v.shape[3]).to(self.device)
                
            pruned_k = k * k_mask
            pruned_v = v * v_mask
            pruned_kv.append((pruned_k, pruned_v))
            
        return pruned_kv
    
    def prune_by_threshold(self, past_key_values, threshold_pct=1.0):
        """
        Prune KV cache based on value magnitudes.
        
        Args:
            past_key_values: Original KV cache from model
            threshold_pct: Percentage of max value to use as threshold
            
        Returns:
            Pruned KV cache
        """
        pruned_kv = []
        
        # Calculate global thresholds
        global_k_max = max([layer_kv[0].abs().max().item() for layer_kv in past_key_values])
        global_v_max = max([layer_kv[1].abs().max().item() for layer_kv in past_key_values])
        k_threshold = threshold_pct / 100.0 * global_k_max
        v_threshold = threshold_pct / 100.0 * global_v_max
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Create masks based on threshold
            k_mask = (k.abs() >= k_threshold).float()
            v_mask = (v.abs() >= v_threshold).float()
            
            # Apply masks
            pruned_k = k * k_mask
            pruned_v = v * v_mask
            pruned_kv.append((pruned_k, pruned_v))
            
        return pruned_kv
    
    def calculate_sparsity(self, kv_cache):
        """
        Calculate sparsity in a KV cache.
        
        Args:
            kv_cache: KV cache from model
            
        Returns:
            Tuple of (key_sparsity, value_sparsity) as percentages
        """
        total_k_elements = 0
        total_v_elements = 0
        zero_k_elements = 0
        zero_v_elements = 0
        
        for k, v in kv_cache:
            total_k_elements += k.numel()
            total_v_elements += v.numel()
            zero_k_elements += (k == 0).sum().item()
            zero_v_elements += (v == 0).sum().item()
        
        k_sparsity = zero_k_elements / total_k_elements * 100
        v_sparsity = zero_v_elements / total_v_elements * 100
        
        return k_sparsity, v_sparsity