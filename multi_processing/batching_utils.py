# llm_processor/batching_utils.py

from typing import List, Any, Callable
import tiktoken

def default_token_counter(text: Any) -> int:
    """Default token counting function using tiktoken"""
    if not isinstance(text, str):
        text = str(text)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def dynamic_token_batch(
    items: List[Any],
    max_tokens: int,
    token_counter: Callable[[Any], int]
) -> List[List[Any]]:
    """Split items into batches based on token counts"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for item in items:
        item_tokens = token_counter(item)
        if item_tokens > max_tokens:
            # Item too large, needs to be processed alone
            if current_batch:
                batches.append(current_batch)
            batches.append([item])
            current_batch = []
            current_tokens = 0
        elif current_tokens + item_tokens > max_tokens:
            # Current batch full, start new batch
            batches.append(current_batch)
            current_batch = [item]
            current_tokens = item_tokens
        else:
            # Add to current batch
            current_batch.append(item)
            current_tokens += item_tokens
    
    if current_batch:
        batches.append(current_batch)
        
    return batches
