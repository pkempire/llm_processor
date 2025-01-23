from llm_processor import LLMProcessor, ProcessorConfig, DeepSeekClient

def example_high_throughput():
    """Example of processing 1M+ items efficiently"""
    config = ProcessorConfig(
        max_workers=100,
        enable_dynamic_token_batching=True,
        checkpoint_interval=1000
    )
    # ... rest of example 