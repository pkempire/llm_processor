#!/usr/bin/env python
import os
import time
import random
import string
import json
from typing import Dict, Any, List
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

########################################
# 1) Real LLMClient to Make Actual API Calls
########################################

from multi_processing.llm_client import DeepSeekClient

# Set your real API key here or via the environment.
API_KEY = "sk-cd405682db094b6781f9f815840163d8"

########################################
# 2) Generate a Synthetic Dataset of Review Prompts
########################################

def generate_review_prompts(num_items: int = 100) -> List[Dict[str, Any]]:
    """
    Generate a dataset of review-generation prompts.
    Each item has:
      - "id": a unique identifier.
      - "text": a prompt that instructs the LLM to generate a product review.
    
    The prompt instructs the LLM to return a JSON array (in batch mode) or a JSON object (in item mode)
    with the following fields: "id", "product", "sentiment", "review".
    """
    dataset = []
    for i in range(num_items):
        prompt = (
            "Generate a product review in JSON format. "
            "Return an object with the keys: id, product, sentiment, review. "
            "The review should be 2-3 sentences long, natural, and specific."
        )
        dataset.append({
            "id": i,
            "text": prompt
        })
    return dataset

########################################
# 3) Processing Functions for the Real LLM
########################################

def process_item_with_real_llm(item: Dict[str, Any], client: DeepSeekClient) -> Dict[str, Any]:
    """
    Process a single item using the real LLM.
    The system prompt provides additional instructions.
    """
    # For item-level processing, we simply pass the prompt (item["text"])
    response = client.call_api(
        prompt=item["text"],
        system_prompt="You are a customer writing product reviews. Write naturally and return valid JSON."
    )
    try:
        parsed = json.loads(response.get("content", ""))
    except Exception:
        parsed = {"error": "Invalid JSON"}
    return {
        "id": item["id"],
        "input": item["text"],
        "output": parsed,
        "success": response.get("success", False)
    }

def create_subbatch_prompt(items: List[Dict[str, Any]]) -> str:
    """
    Combine multiple items into a single prompt.
    The prompt instructs the LLM to return a JSON array of responses.
    Each response object should include the original id.
    """
    combined_text = ""
    for itm in items:
        combined_text += f"(ID={itm['id']}) {itm['text']}\n"
    prompt = (
        f"Generate product reviews for the following {len(items)} prompts. "
        "For each prompt, return a JSON object with the keys: id, product, sentiment, review. "
        "Return the results as a JSON array.\n" + combined_text
    )
    return prompt

def process_subbatch(items: List[Dict[str, Any]], client: DeepSeekClient) -> Dict[int, Any]:
    """
    Process a sub-batch of items by calling the LLM once.
    A response parser attempts to split the combined JSON array into individual responses.
    """
    prompt = create_subbatch_prompt(items)
    response = client.call_api(
        prompt=prompt,
        system_prompt="You are a customer writing product reviews. Write naturally and return valid JSON."
    )
    result_map = {}
    try:
        # Expecting a JSON array of objects, each with an "id" field.
        parsed = json.loads(response.get("content", ""))
        if isinstance(parsed, list):
            for obj in parsed:
                if "id" in obj:
                    result_map[obj["id"]] = obj
        else:
            # If not a list, fallback to assign the same response to each.
            for itm in items:
                result_map[itm["id"]] = {"error": "Expected JSON array", "raw": response.get("content", "")}
    except Exception as e:
        # On parsing error, assign the error to all items.
        for itm in items:
            result_map[itm["id"]] = {"error": f"Parsing failed: {str(e)}", "raw": response.get("content", "")}
    return result_map

########################################
# 4) Experimental Modes
########################################

# (A) Sequential processing (control)
def run_sequential_no_concurrency(dataset: List[Dict[str, Any]], client: DeepSeekClient):
    results = []
    start_time = time.perf_counter()
    for item in tqdm(dataset, desc="Sequential (control)"):
        res = process_item_with_real_llm(item, client)
        results.append(res)
    elapsed = time.perf_counter() - start_time
    print(f"[Control] Processed {len(results)} items sequentially in {elapsed:.2f} seconds.")
    return results, elapsed

# (B) Library item-level concurrency
from multi_processing.processor import LLMProcessor
from multi_processing.processor_config import ProcessorConfig

def run_with_library(dataset: List[Dict[str, Any]], max_workers: int = 10):
    config = ProcessorConfig(
        cache_enabled=False,
        max_workers=max_workers,
        rate_limit=0.0,
        max_retries=1,
        batch_size=1,         # item-level processing
        fail_fast=False,
        enable_batch_prompts=False  # force individual item processing
    )
    client = DeepSeekClient(api_key=API_KEY, model="deepseek-chat", temperature=0.1)
    processor = LLMProcessor(llm_client=client, config=config)
    
    def process_fn(item):
        return process_item_with_real_llm(item, client)
    
    start_time = time.perf_counter()
    results = processor.process_batch(
        items=dataset,
        process_fn=process_fn,
        cache_prefix="item_level",
        use_cache=False
    )
    elapsed = time.perf_counter() - start_time
    print(f"[Library Item-Level] Processed {len(results)} items in {elapsed:.2f} seconds.")
    return results, elapsed, processor.metrics

# (C) Library batch-mode concurrency
def run_with_library_batch_mode(dataset: List[Dict[str, Any]], max_workers: int = 10, batch_size: int = 10):
    config = ProcessorConfig(
        cache_enabled=False,
        max_workers=max_workers,
        rate_limit=0.0,
        max_retries=1,
        batch_size=batch_size,  # sub-batch size
        fail_fast=False,
        enable_batch_prompts=True  # enable batch mode
    )
    client = DeepSeekClient(api_key=API_KEY, model="deepseek-chat", temperature=0.1)
    processor = LLMProcessor(llm_client=client, config=config)
    
    def process_fn(subbatch: List[Dict[str, Any]]) -> Dict[int, Any]:
        return process_subbatch(subbatch, client)
    
    start_time = time.perf_counter()
    dict_list = processor.process_batch(
        items=dataset,
        process_fn=process_fn,
        cache_prefix="batch_mode",
        use_cache=False,
        desc="Library Batch Mode"
    )
    elapsed = time.perf_counter() - start_time
    # Merge all sub-dictionaries into one combined dictionary.
    combined = {}
    for subdict in dict_list:
        combined.update(subdict)
    print(f"[Library Batch Mode] Processed {len(combined)} items in {elapsed:.2f} seconds (sub-batch size={batch_size}).")
    return combined, elapsed, processor.metrics

# (D) Raw concurrency using ThreadPoolExecutor directly
def run_with_raw_concurrency(dataset: List[Dict[str, Any]], max_workers: int = 10):
    client = DeepSeekClient(api_key=API_KEY, model="deepseek-chat", temperature=0.1)
    results = []
    start_time = time.perf_counter()
    with tqdm(total=len(dataset), desc="Raw Concurrency") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item_with_real_llm, item, client) for item in dataset]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    elapsed = time.perf_counter() - start_time
    print(f"[Raw Concurrency] Processed {len(results)} items in {elapsed:.2f} seconds.")
    return results, elapsed

########################################
# 5) Graphing and Reporting Functions
########################################

def plot_performance(results: Dict[str, Dict[str, float]]):
    """
    Plot bar charts comparing throughput (items/sec) and elapsed time.
    """
    modes = list(results.keys())
    throughputs = [results[mode]['throughput'] for mode in modes]
    elapsed_times = [results[mode]['elapsed'] for mode in modes]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].bar(modes, throughputs, color='skyblue')
    axs[0].set_title('Throughput (Items per Second)')
    axs[0].set_ylabel('Items/sec')
    
    axs[1].bar(modes, elapsed_times, color='salmon')
    axs[1].set_title('Elapsed Time (Seconds)')
    axs[1].set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.show()

########################################
# 6) Main Comparison Experiment
########################################

if __name__ == "__main__":
    # Generate a dataset of review prompts.
    data_size = 100  # adjust as needed; keep small for testing real API calls
    dataset = generate_review_prompts(num_items=data_size)
    
    # Run experiments in different modes.
    control_client = DeepSeekClient(api_key=API_KEY, model="deepseek-chat", temperature=0.1)
    control_results, control_elapsed = run_sequential_no_concurrency(dataset, control_client)
    
    lib_item_results, lib_item_elapsed, item_metrics = run_with_library(dataset, max_workers=50)
    
    lib_batch_results, lib_batch_elapsed, batch_metrics = run_with_library_batch_mode(dataset, max_workers=50, batch_size=5)
    
    raw_results, raw_elapsed = run_with_raw_concurrency(dataset, max_workers=50)
    
    # Compile performance metrics for comparison.
    performance = {
        "Sequential": {
            "elapsed": control_elapsed,
            "throughput": len(dataset) / control_elapsed
        },
        "Library Item-Level": {
            "elapsed": lib_item_elapsed,
            "throughput": len(dataset) / lib_item_elapsed,
            "tokens_sec": item_metrics.get("token_usage", {}).get("total_tokens", 0) / lib_item_elapsed
        },
        "Library Batch Mode": {
            "elapsed": lib_batch_elapsed,
            "throughput": len(dataset) / lib_batch_elapsed,
            "tokens_sec": batch_metrics.get("token_usage", {}).get("total_tokens", 0) / lib_batch_elapsed
        },
        "Raw Concurrency": {
            "elapsed": raw_elapsed,
            "throughput": len(dataset) / raw_elapsed
        }
    }
    
    # Print performance summary.
    print("\nPerformance Summary:")
    for mode, metrics in performance.items():
        print(f"{mode}: {metrics}")
    
    # Plot performance comparisons.
    plot_performance(performance)
