"""
Utility functions for data collection and preprocessing.
"""

# Data collection utilities
from utils.data_collection import extract_kv_cache, extract_model_info, prepare_kv_cache_data, get_memory_usage

# Generation utilities
from utils.generation_stages import capture_generation_stages, compare_generation_stages, extract_kv_difference

# Dataset utilities
from utils.dataset_loaders import get_wikitext_prompt, prepare_input_for_model, print_wikitext_samples, load_wikitext_sample