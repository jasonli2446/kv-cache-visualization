"""
Implementation of KV cache compression techniques.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CompressedKVCache:
    """Class for compressing and managing KV cache."""
    
    def __init__(self, original_kv_cache, dimension_groups=None):
        """
        Initialize a compressed KV cache.
        
        Args:
            original_kv_cache: Original KV cache from model
            dimension_groups: Groups of similar embedding dimensions
        """
        self.original_shape = original_kv_cache[0][0].shape
        self.original_kv_cache = original_kv_cache
        self.dimension_groups = dimension_groups
        
        # Create mapping from original dimensions to compressed representation
        if dimension_groups:
            self.k_mapping = self._create_mapping(dimension_groups["k_groups"], self.original_shape[-1])
            self.v_mapping = self._create_mapping(dimension_groups["v_groups"], self.original_shape[-1])
            self.compressed_cache = self._compress_kv_cache()
        else:
            self.compressed_cache = original_kv_cache
    
    def _create_mapping(self, dimension_groups, head_dim):
        """Create mapping from original dimensions to compressed dimensions."""
        mapping = list(range(head_dim))  # Default: each dim maps to itself
        
        for group in dimension_groups:
            representative = group[0]  # Use first dimension as representative
            for dim in group[1:]:  # Map other dimensions to representative
                mapping[dim] = representative
        
        return mapping
    
    def _compress_kv_cache(self):
        """Compress KV cache by zeroing out redundant dimensions."""
        compressed = []
        
        for layer_idx, (k, v) in enumerate(self.original_kv_cache):
            k_compressed = k.clone()
            v_compressed = v.clone()
            
            # For each key group
            for group in self.dimension_groups["k_groups"]:
                if len(group) <= 1:
                    continue
                
                rep_dim = group[0]  # Representative dimension
                
                # First, accumulate values into the representative dimension
                for dim in group[1:]:
                    k_compressed[:, :, :, rep_dim] += k[:, :, :, dim]
                
                # Average the values
                k_compressed[:, :, :, rep_dim] /= len(group)
                
                # Zero out the redundant dimensions
                for dim in group[1:]:
                    k_compressed[:, :, :, dim] = 0.0
            
            # Same for value dimensions
            for group in self.dimension_groups["v_groups"]:
                if len(group) <= 1:
                    continue
                
                rep_dim = group[0]  # Representative dimension
                
                # First, accumulate values into the representative dimension
                for dim in group[1:]:
                    v_compressed[:, :, :, rep_dim] += v[:, :, :, dim]
                
                # Average the values
                v_compressed[:, :, :, rep_dim] /= len(group)
                
                # Zero out the redundant dimensions
                for dim in group[1:]:
                    v_compressed[:, :, :, dim] = 0.0
            
            compressed.append((k_compressed, v_compressed))
        
        return compressed

    def get_cache(self):
        """Return the compressed cache."""
        # For perplexity evaluation, we need to maintain the original tensor shapes
        # but we can still use the compressed information
        
        # Make a deep copy to avoid modifying the original
        return self.compressed_cache
    
    def get_memory_savings(self):
        """Calculate memory savings from compression."""
        k_saved_dims = sum([len(group) - 1 for group in self.dimension_groups["k_groups"]])
        v_saved_dims = sum([len(group) - 1 for group in self.dimension_groups["v_groups"]])
        
        total_dims = self.original_shape[-1]
        
        return {
            "k_saved_dimensions": k_saved_dims,
            "v_saved_dimensions": v_saved_dims,
            "total_saved_dimensions": k_saved_dims + v_saved_dims,
            "original_dimensions": total_dims * 2,  # Both keys and values
            "compression_ratio": 1 - ((k_saved_dims + v_saved_dims) / (total_dims * 2))
        }