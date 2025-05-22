"""
Tucker decomposition for KV cache compression.
"""

import torch
import tensorly as tl
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class TuckerDecomposition:
    def __init__(self, ranks: Tuple[int, int]):
        """
        Initialize Tucker decomposition with specified ranks.
        
        Args:
            ranks: Tuple of (embedding_rank, token_rank) for Tucker decomposition
        """
        self.ranks = ranks
        self.core_tensor = None
        self.factors = None
        
    def decompose(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """
        Perform Tucker decomposition on the KV cache, processing one layer at a time.
        
        Args:
            kv_cache: List of (key, value) tensor pairs for each layer
            
        Returns:
            Dictionary containing the decomposed components and reconstruction error
        """
        num_layers = len(kv_cache)
        num_tokens = kv_cache[0][0].shape[2]  # sequence length
        embedding_dim = kv_cache[0][0].shape[3]  # head dimension
        batch_size, num_heads = kv_cache[0][0].shape[:2]
        
        # Initialize lists to store results for each layer
        keys_cores = []
        values_cores = []
        keys_factors = []
        values_factors = []
        keys_errors = []
        values_errors = []
        
        print("Processing layers one at a time...")
        for layer_idx, (keys, values) in enumerate(kv_cache):
            print(f"Processing layer {layer_idx + 1}/{num_layers}")
            
            # Reshape to combine batch and head dimensions
            keys_reshaped = keys.reshape(batch_size * num_heads, num_tokens, embedding_dim)
            values_reshaped = values.reshape(batch_size * num_heads, num_tokens, embedding_dim)
            
            # Move to CPU and convert to numpy
            keys_np = keys_reshaped.cpu().numpy()
            values_np = values_reshaped.cpu().numpy()
            
            # Perform Tucker decomposition for this layer
            keys_core, keys_factor = tl.decomposition.tucker(keys_np, rank=(self.ranks[0], self.ranks[1], batch_size * num_heads))
            values_core, values_factor = tl.decomposition.tucker(values_np, rank=(self.ranks[0], self.ranks[1], batch_size * num_heads))
            
            # Store results
            keys_cores.append(keys_core)
            values_cores.append(values_core)
            keys_factors.append(keys_factor)
            values_factors.append(values_factor)
            
            # Calculate reconstruction error for this layer
            keys_reconstructed = tl.tucker_to_tensor((keys_core, keys_factor))
            values_reconstructed = tl.tucker_to_tensor((values_core, values_factor))
            
            keys_error = np.linalg.norm(keys_np - keys_reconstructed) / np.linalg.norm(keys_np)
            values_error = np.linalg.norm(values_np - values_reconstructed) / np.linalg.norm(values_np)
            
            keys_errors.append(keys_error)
            values_errors.append(values_error)
            
            # Clear memory
            del keys_np, values_np, keys_reconstructed, values_reconstructed
            torch.cuda.empty_cache()
        
        # Store results
        self.core_tensor = {
            'keys': [torch.from_numpy(core) for core in keys_cores],
            'values': [torch.from_numpy(core) for core in values_cores]
        }
        self.factors = {
            'keys': [[torch.from_numpy(f) for f in factors] for factors in keys_factors],
            'values': [[torch.from_numpy(f) for f in factors] for factors in values_factors]
        }
        
        return {
            'core_tensor': self.core_tensor,
            'factors': self.factors,
            'reconstruction_error': {
                'keys': float(np.mean(keys_errors)),
                'values': float(np.mean(values_errors))
            }
        }
    
    def reconstruct(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct the KV cache from the Tucker decomposition.
        
        Returns:
            List of (key, value) tensor pairs for each layer
        """
        if self.core_tensor is None or self.factors is None:
            raise ValueError("Must perform decomposition before reconstruction")
        
        kv_cache = []
        num_layers = len(self.core_tensor['keys'])
        
        for layer_idx in range(num_layers):
            # Convert to numpy for tensorly
            keys_core = self.core_tensor['keys'][layer_idx].numpy()
            values_core = self.core_tensor['values'][layer_idx].numpy()
            keys_factors = [f.numpy() for f in self.factors['keys'][layer_idx]]
            values_factors = [f.numpy() for f in self.factors['values'][layer_idx]]
            
            # Reconstruct tensors
            keys_reconstructed = tl.tucker_to_tensor((keys_core, keys_factors))
            values_reconstructed = tl.tucker_to_tensor((values_core, values_factors))
            
            # Reshape back to original format
            batch_size = 1  # Assuming batch size of 1
            num_heads = keys_reconstructed.shape[0] // batch_size
            keys_reconstructed = keys_reconstructed.reshape(batch_size, num_heads, -1, keys_reconstructed.shape[2])
            values_reconstructed = values_reconstructed.reshape(batch_size, num_heads, -1, values_reconstructed.shape[2])
            
            # Convert back to torch tensors
            kv_cache.append((
                torch.from_numpy(keys_reconstructed),
                torch.from_numpy(values_reconstructed)
            ))
            
        return kv_cache
    
    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio achieved by Tucker decomposition.
        
        Returns:
            Compression ratio (original size / compressed size)
        """
        if self.core_tensor is None or self.factors is None:
            raise ValueError("Must perform decomposition before calculating compression ratio")
        
        # Calculate original size
        original_size = 0
        compressed_size = 0
        
        for layer_idx in range(len(self.core_tensor['keys'])):
            # Original size for this layer
            original_size += (
                self.core_tensor['keys'][layer_idx].shape[0] * 
                self.core_tensor['keys'][layer_idx].shape[1] * 
                self.core_tensor['keys'][layer_idx].shape[2] * 2  # *2 for both keys and values
            )
            
            # Compressed size for this layer
            compressed_size += (
                self.core_tensor['keys'][layer_idx].numel() +  # Core tensor size
                sum(f.numel() for f in self.factors['keys'][layer_idx]) +  # Key factors size
                self.core_tensor['values'][layer_idx].numel() +  # Core tensor size
                sum(f.numel() for f in self.factors['values'][layer_idx])  # Value factors size
            )
        
        return original_size / compressed_size

def plot_tucker_components(decomposition: TuckerDecomposition, save_path: Optional[str] = None):
    """
    Visualize the Tucker decomposition components.
    
    Args:
        decomposition: TuckerDecomposition object containing the decomposed components
        save_path: Optional path to save the visualization
    """
    if decomposition.core_tensor is None or decomposition.factors is None:
        raise ValueError("Must perform decomposition before visualization")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot core tensors for the first layer
    sns.heatmap(decomposition.core_tensor['keys'][0].numpy().reshape(-1, decomposition.core_tensor['keys'][0].shape[2]), 
                ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title('Keys Core Tensor (Layer 0)')
    
    sns.heatmap(decomposition.core_tensor['values'][0].numpy().reshape(-1, decomposition.core_tensor['values'][0].shape[2]), 
                ax=axes[0,1], cmap='viridis')
    axes[0,1].set_title('Values Core Tensor (Layer 0)')
    
    # Plot embedding factors for the first layer
    sns.heatmap(decomposition.factors['keys'][0][0].numpy(), ax=axes[1,0], cmap='viridis')
    axes[1,0].set_title('Keys Embedding Factors (Layer 0)')
    
    sns.heatmap(decomposition.factors['values'][0][0].numpy(), ax=axes[1,1], cmap='viridis')
    axes[1,1].set_title('Values Embedding Factors (Layer 0)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 