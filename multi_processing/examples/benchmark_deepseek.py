# examples/deepseek_example.py

from llm_processor.processor import LLMProcessor
from llm_processor.processor_config import ProcessorConfig
from llm_processor.llm_client import DeepSeekClient

def deepseek_process_fn(prompt: str) -> dict:
    """
    Example function that calls DeepSeek API 
    using the DeepSeekClient.
    """
    client = DeepSeekClient(api_key="YOUR_DEEPSEEK_KEY")
    # The client call returns a dict like {"completion": "..."}

    
    result = client.call_api(prompt)
    return result

if __name__ == "__main__":
    config = ProcessorConfig(
        batch_size=20,
        max_workers=10,
        rate_limit=0.0  # DeepSeek has no limit
    )
    processor = LLMProcessor(config)

    prompts = [f"Prompt {i}" for i in range(100)]
    
    # Process them
    results = processor.process_batch(items=prompts, process_fn=deepseek_process_fn)

    print("Sample results:", results[:3])
    # Each item might look like {"completion": "..." }
