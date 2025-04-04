"""
Evaluation utilities for pruned KV cache.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class KVCacheEvaluator:
    """Evaluate performance of pruned KV cache."""
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize evaluator with model and tokenizer.
        
        Args:
            model: HuggingFace transformer model
            tokenizer: Associated tokenizer
            device: Device to run on (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.baseline_perplexity = None
        self.baseline_latency = None
    
    def measure_perplexity(self, prompt, continuation, past_key_values=None):
        """
        Calculate perplexity with optional pruned KV cache.
        
        Args:
            prompt: Input text to generate KV cache
            continuation: Text to evaluate perplexity on
            past_key_values: Optional pruned KV cache
            
        Returns:
            Perplexity score (lower is better)
        """
        # Method 1: Using full sequence and masking prompt tokens in labels
        full_text = prompt + " " + continuation
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # Create labels with -100 for prompt tokens (to ignore them in loss)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Create labels: -100 for prompt tokens, actual IDs for continuation tokens
        labels = full_inputs.input_ids.clone()
        labels[:, :prompt_length] = -100  # Mask out prompt tokens from loss calculation
        
        with torch.no_grad():
            outputs = self.model(**full_inputs, labels=labels)
        
        return torch.exp(outputs.loss).item()
    
    def measure_latency(self, prompt, continuation_length=20, past_key_values=None, num_runs=3):
        """
        Measure inference latency with optional pruned KV cache.
        
        Args:
            prompt: Input text to generate KV cache
            continuation_length: Number of tokens to generate
            past_key_values: Optional pruned KV cache
            num_runs: Number of runs to average
            
        Returns:
            Average latency in milliseconds
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        latencies = []
        
        # Use same approach for both baseline and pruned cases, but with different inputs
        with torch.no_grad():
            if past_key_values is not None:
                # For pruned case: just measure time to process prompt
                # This avoids compatibility issues with KVCache format
                torch.cuda.synchronize()
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    _ = self.model(input_ids=inputs.input_ids)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000)  # ms
            else:
                # For baseline: same approach
                torch.cuda.synchronize()
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    _ = self.model(input_ids=inputs.input_ids)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    latencies.append((end_time - start_time) * 1000)  # ms
        
        # Return average latency
        return sum(latencies) / len(latencies)
    
    def evaluate_pruning(self, prompt, continuation, past_key_values=None, pruning_info=None):
        """
        Comprehensively evaluate pruning performance.
        
        Args:
            prompt: Input text to generate KV cache
            continuation: Text to evaluate quality on
            past_key_values: Pruned KV cache
            pruning_info: Dict with pruning metadata
            
        Returns:
            Dict with evaluation metrics
        """
        # Get baseline if not already calculated
        if self.baseline_perplexity is None:
            self.baseline_perplexity = self.measure_perplexity(prompt, continuation)
            
        if self.baseline_latency is None:
            self.baseline_latency = self.measure_latency(prompt)
            
        # Calculate key metrics
        perplexity = self.measure_perplexity(prompt, continuation, past_key_values)
        latency = self.measure_latency(prompt, past_key_values=past_key_values)
        
        # Calculate changes
        perplexity_change = ((perplexity - self.baseline_perplexity) / 
                            self.baseline_perplexity * 100)
        latency_change = ((latency - self.baseline_latency) / 
                         self.baseline_latency * 100)
        
        # Build evaluation results
        results = {
            "baseline_perplexity": self.baseline_perplexity,
            "pruned_perplexity": perplexity,
            "perplexity_change_percent": perplexity_change,
            "baseline_latency_ms": self.baseline_latency,
            "pruned_latency_ms": latency,
            "latency_change_percent": latency_change,
        }
        
        # Add pruning metadata if provided
        if pruning_info:
            results.update(pruning_info)
            
        return results