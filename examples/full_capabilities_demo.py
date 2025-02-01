"""
LLM Processor Full Capabilities Demo
------------------------------------
This script demonstrates the full range of features in the LLMProcessor library,
including advanced scenarios and stress testing.
"""
import os
import time
import random
from datetime import datetime
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from multi_processing.llm_client import DeepSeekClient
from multi_processing.processor import LLMProcessor
from multi_processing.processor_config import ProcessorConfig
from multi_processing.batching_utils import dynamic_token_batch

# --------------------------------------------------
# 1. Basic Setup & Configuration
# --------------------------------------------------
def create_high_performance_config():
    """Configuration for maximum throughput stress test"""
    return ProcessorConfig(
        max_workers=32,
        enable_dynamic_token_batching=True,
        max_tokens_per_batch=4096,
        cache_enabled=True,
        cache_dir="llm_cache/stress_test",
        dynamic_rate_limit=True,
        requests_per_minute=500,
        max_retries=3,
        checkpoint_dir="checkpoints",
        checkpoint_interval=250,
        metrics_output_path="stress_test_metrics.json",
        pricing_models={
            'deepseek-chat': {
                'input': 0.75,  # Updated pricing
                'output': 2.25
            }
        }
    )

def create_safe_config():
    """Configuration for reliable processing with error handling"""
    return ProcessorConfig(
        max_workers=8,
        enable_dynamic_token_batching=False,
        cache_enabled=True,
        max_retries=5,
        fail_fast=False,
        error_output_path="error_logs.json",
        jitter=True,
        min_retry_delay=2.0,
        max_retry_delay=60.0
    )

# --------------------------------------------------
# 2. Demo Scenarios
# --------------------------------------------------
def stress_test_demo():
    """Demonstrate maximum throughput capabilities"""
    print("\n=== Running Stress Test (1000 prompts) ===")
    
    # Generate test prompts with varying complexity
    prompts = [
        f"Analyze this technical paper abstract: {'AI safety ' * random.randint(1, 50)}"
        if i % 3 == 0 else
        f"Customer review: {'Excellent' * random.randint(1, 20)} product experience!"
        for i in range(1000)
    ]
    
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
    processor = LLMProcessor(client, create_high_performance_config())
    
    # Custom processing function with error simulation
    def process_fn(prompt):
        # Simulate 5% error rate
        if random.random() < 0.05:
            raise Exception("Simulated API error")
        return client.call_api(
            prompt,
            system_prompt="You are an expert analyst. Provide detailed responses."
        )
    
    # Execute processing with real-time metrics
    start_time = time.time()
    results = processor.process_batch(
        prompts,
        process_fn,
        desc="Stress Test Processing"
    )
    
    # Calculate final metrics
    duration = time.time() - start_time
    print(f"\nStress Test Complete ({duration:.2f}s)")
    print(f"Processed {len(results)} items")
    print(f"Throughput: {len(prompts)/duration:.2f} items/sec")
    print(f"Total Tokens: {processor.metrics['token_usage']['total_tokens']}")
    print(f"Estimated Cost: ${processor._calculate_estimated_cost()['total_cost']:.2f}")

def error_handling_demo():
    """Demonstrate robust error recovery capabilities"""
    print("\n=== Running Error Handling Demo (50 error-prone prompts) ===")
    
    # Create intentionally problematic prompts
    prompts = [
        "INVALID_INPUT" * 100 if i % 5 == 0 else 
        "Normal prompt about machine learning applications"
        for i in range(50)
    ]
    
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
    processor = LLMProcessor(client, create_safe_config())
    
    results = processor.process_batch(
        prompts,
        lambda p: client.call_api(p),
        desc="Error Handling Test"
    )
    
    print(f"\nSuccessfully processed {len([r for r in results if r['success']])}/50 items")
    print(f"Error details saved to {processor.config.error_output_path}")

def dynamic_batching_demo():
    """Showcase token-aware batch processing"""
    print("\n=== Running Dynamic Batching Demo ===")
    
    # Generate text samples with varying lengths
    documents = [
        "Short text" + " with some content" * random.randint(1, 5)
        for _ in range(100)
    ] + [
        "Very long document: " + "AI safety discussion " * random.randint(100, 200)
        for _ in range(10)
    ]
    
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
    config = create_high_performance_config()
    config.enable_dynamic_token_batching = True
    processor = LLMProcessor(client, config)
    
    results = processor.process_batch(
        documents,
        lambda doc: client.call_api(
            f"Summarize this document: {doc}",
            system_prompt="You are a technical summarization expert."
        ),
        desc="Dynamic Batching"
    )
    
    print(f"\nProcessed {len(documents)} documents in {len(results)} token-aware batches")

def cache_demo():
    """Demonstrate caching functionality"""
    print("\n=== Running Cache Demonstration ===")
    
    duplicate_prompts = ["Explain quantum computing basics"] * 20
    
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
    config = create_safe_config()
    config.cache_enabled = True
    processor = LLMProcessor(client, config)
    
    # First run - populate cache
    print("First run (populating cache)...")
    processor.process_batch(duplicate_prompts, lambda p: client.call_api(p))
    
    # Second run - use cache
    print("\nSecond run (using cache)...")
    start_time = time.time()
    processor.process_batch(duplicate_prompts, lambda p: client.call_api(p))
    duration = time.time() - start_time
    
    print(f"\nCached processing completed in {duration:.2f}s")
    print(f"Cache hits: {processor.metrics['cache_hits']}")

def checkpoint_demo():
    """Demonstrate checkpoint recovery capabilities"""
    print("\n=== Running Checkpoint Demo ===")
    
    prompts = [f"Prompt {i}" for i in range(100)]
    client = DeepSeekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))
    config = create_safe_config()
    config.checkpoint_dir = "checkpoints/demo"
    config.checkpoint_interval = 20
    processor = LLMProcessor(client, config)
    
    # Simulate failure after 3 seconds
    try:
        for i, result in enumerate(processor.process_batch(prompts, lambda p: client.call_api(p))):
            if i == 40:  # Simulate failure at 40%
                raise Exception("Simulated system crash")
    except:
        print("\n--- Simulation: System crash occurred ---")
    
    # Resume from checkpoint
    print("\nResuming from checkpoint...")
    processor = LLMProcessor(client, config)
    results = processor.process_batch(prompts, lambda p: client.call_api(p))
    
    print(f"\nRecovered {len(results)} items after failure")

# --------------------------------------------------
# 3. Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    print("LLM Processor Capabilities Demonstration\n")
    
    # Run all demos
    stress_test_demo()
    error_handling_demo()
    dynamic_batching_demo()
    cache_demo()
    checkpoint_demo()
    
    print("\nAll demonstrations completed!")
