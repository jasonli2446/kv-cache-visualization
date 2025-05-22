"""
Dataset loading utilities for KV cache analysis.
"""

from datasets import load_dataset
import random
import config

def load_wikitext_sample(split="test", num_samples=5, min_length=150, max_length=None):
    """
    Load a sample from WikiText dataset.
    
    Args:
        split: Dataset split to use ('train', 'validation', or 'test')
        num_samples: Number of samples to choose from
        min_length: Minimum text length to consider
        max_length: Maximum text length to consider
        
    Returns:
        List of text samples from WikiText
    """
    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
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
    
    # Truncate to max_length if needed
    samples = [text[:max_length] for text in samples]
    
    return samples

def get_wikitext_prompt(index=0):
    """
    Get a single WikiText prompt for analysis.
    
    Args:
        index: Index of the sample to use (0-4)
        
    Returns:
        A text prompt from WikiText
    """
    samples = load_wikitext_sample()
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
    
    # Tokenize and truncate if needed
    tokens = tokenizer.encode(text, truncation=True, max_length=effective_max_length)
    
    # Decode back to text
    truncated_text = tokenizer.decode(tokens)
    
    if len(tokens) < len(tokenizer.encode(text)):
        print(f"⚠️ Input was truncated to {len(tokens)} tokens to fit model context window")
    
    return truncated_text

def print_wikitext_samples():
    """
    Print available WikiText samples to choose from.
    """
    samples = load_wikitext_sample(num_samples=10)
    
    print("\n=== Available WikiText Samples ===")
    for i, sample in enumerate(samples):
        preview = sample[:100].replace('\n', ' ')
        print(f"Sample {i}: {preview}...")
    
    return samples