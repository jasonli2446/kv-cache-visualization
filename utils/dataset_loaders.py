"""
Dataset loading utilities for KV cache analysis.
"""

from datasets import load_dataset
import random
import config
from transformers import AutoTokenizer

def load_wikitext_sample(split="test", num_samples=5, min_length=150):
    """
    Load a sample from WikiText dataset.
    
    Args:
        split: Dataset split to use ('train', 'validation', or 'test')
        num_samples: Number of samples to choose from
        min_length: Minimum text length to consider
        
    Returns:
        List of text samples from WikiText
    """
    # Load WikiText-103 dataset (much larger than WikiText-2)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    
    # Filter to get only substantial paragraphs
    filtered_texts = [
        text for text in dataset["text"] 
        if len(text) >= min_length and text.strip() and not text.startswith("=")
    ]
    
    # Choose a sample of texts
    if len(filtered_texts) > num_samples:
        samples = random.sample(filtered_texts, num_samples)
    else:
        samples = filtered_texts
    
    return samples

def concatenate_wikitext_samples(num_samples=5, target_tokens=512):
    """
    Concatenate multiple WikiText samples to reach target token length.
    
    Args:
        num_samples: Number of samples to concatenate
        target_tokens: Target number of tokens to reach
        
    Returns:
        Concatenated text from multiple WikiText samples
    """
    samples = load_wikitext_sample(num_samples=num_samples)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    # Concatenate samples with newlines
    concatenated = "\n\n".join(samples)
    tokens = tokenizer.encode(concatenated)
    
    # If we have more tokens than needed, truncate
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        print(f"⚠️ Concatenated text was truncated to {len(tokens)} tokens")
    
    return tokenizer.decode(tokens)

def get_wikitext_prompt(index=0, concatenate=True, target_tokens=512):
    """
    Get a WikiText prompt for analysis.
    
    Args:
        index: Index of the sample to use (0-4) if not concatenating
        concatenate: Whether to concatenate multiple samples
        target_tokens: Target number of tokens when concatenating
        
    Returns:
        A text prompt from WikiText
    """
    if concatenate:
        return concatenate_wikitext_samples(num_samples=5, target_tokens=target_tokens)
    else:
        # Use a higher minimum length for longer samples
        samples = load_wikitext_sample(min_length=500)  # Increased from 150
        index = max(0, min(len(samples)-1, index))  # Ensure valid index
        return samples[index]

def prepare_input_for_model(text, tokenizer, model_name, max_tokens=None):
    """
    Prepare input text for the model.
    
    Args:
        text: Input text
        tokenizer: The model tokenizer
        model_name: Name of the model
        max_tokens: Maximum number of tokens to keep (None for no limit)
        
    Returns:
        Text that fits within model context window
    """
    # Get model's max sequence length
    model_max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048
    
    # Use the smaller of the two max lengths if max_tokens is specified
    effective_max_length = min(max_tokens, model_max_length) if max_tokens is not None else model_max_length
    
    # First encode without truncation to check length
    tokens = tokenizer.encode(text, truncation=False)
    
    # If we need to truncate
    if len(tokens) > effective_max_length:
        tokens = tokens[:effective_max_length]
        print(f"⚠️ Input was truncated to {len(tokens)} tokens to fit model context window")
    
    # Decode back to text
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def print_wikitext_samples():
    """
    Print available WikiText samples to choose from.
    """
    samples = load_wikitext_sample(num_samples=10)
    
    print("\n=== Available WikiText Samples ===")
    for i, sample in enumerate(samples):
        preview = sample[:100].replace('\n', ' ')
        # Get tokenizer to count tokens
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        num_tokens = len(tokenizer.encode(sample))
        print(f"Sample {i}: {preview}...")
        print(f"  Length: {len(sample)} chars, {num_tokens} tokens")
        print()
    
    return samples