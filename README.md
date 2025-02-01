HyperLLM

A high-performance Python library for batch processing and concurrency of LLM API calls, achieving up to 1000x faster speeds than sequential API calls.

Built for:
1. **High-volume offline data processing** – concurrency + dynamic batching + checkpointing + caching → reduces multi-day jobs to minutes.  
2. **Real-time request handling** – dynamic concurrency, load balancing, rate limiting → keep latencies low and handle spikes.  
3. **Agent or multi-step reasoning** – easily manage repeated LLM calls in a single user flow or tool + LLM pipeline.
4. **Synthetic data generation** – quickly produce large amounts of synthetic text for training or testing.


By default, examples show usage with DeepSeek (which has no explicit rate limit), but you can easily adapt to OpenAI or any other LLM provider by swapping in a different client class.


---

### Features

- **Concurrency**: Uses Python’s `ThreadPoolExecutor` to make thousands of LLM calls in parallel, significantly reducing total runtime.  
- **Batching**: Merge multiple items into a single LLM prompt to reduce API overhead costs
- **Caching**: Optionally cache and skip repeated prompts with on-disk JSON caching.  
- **Retry**: Automatic exponential backoff for failures or rate-limit responses.  
- **Dynamic Token Batching**: Automatically chunk items so total tokens stay below a configured limit.  
- **Configurable**: A simple `ProcessorConfig` dataclass to set concurrency, batch size, dynamic token usage, etc.
- **Checkpointing**: Save progress mid-run so you can resume large jobs if your script stops unexpectedly.

---

## Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/pkempire/hyperllm.git
   cd hyperllm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Add any dependencies like `tqdm`, `requests`, or `tiktoken` if needed.)*

3. **Use it in your projects**:
   ```python
   from llm_processor import LLMProcessor, ProcessorConfig, DeepSeekClient

   # Example configuration
   config = ProcessorConfig(
       max_workers=16,
       batch_size=8,
       cache_enabled=True,
       max_retries=3
   )

   # Initialize client and processor
   client = DeepSeekClient(api_key="your_api_key_here")
   processor = LLMProcessor(client, config)
   ```

---

## Quick Start

```python
import os
from llm_processor.processor import LLMProcessor
from llm_processor.processor_config import ProcessorConfig
from llm_processor.llm_client import DeepSeekClient  # or OpenAIClient, etc.

# 1) Prepare your LLM client
api_key = os.getenv("DEEPSEEK_API_KEY") ##Or just string 
client = DeepSeekClient(api_key=api_key, model="deepseek-chat", temperature=0.1)

# 2) Create a configuration (batch size, concurrency, caching, etc.)
config = ProcessorConfig(
    max_workers=20,
    batch_size=5,
    enable_batch_prompts=True,
    cache_enabled=True,
    max_retries=2,
    # ...
)

# 3) Initialize the LLMProcessor with your client
processor = LLMProcessor(llm_client=client, config=config)

# 4) Define a processing function that calls the LLM
def process_fn(prompt):
    # Prepare system messages or additional parameters as needed
    response = client.call_api(prompt=prompt, system_prompt="You are a helpful AI assistant.")
    return response

# 5) Run the processor on a list of items (prompts)
items = [
    "What is the capital of France?",
    "Explain quantum entanglement in simple terms.",
    "Translate this sentence to Spanish: 'Hello World'.",
    # ... more ...
]

results = processor.process_batch(items, process_fn, cache_prefix="demo_job")
print("Done! Results:")
for r in results:
    print(r["content"])
    ```

---

## Dynamic Token Batching

If you want to ensure each sub-batch stays under a certain token limit:

```python
config = ProcessorConfig(
    enable_dynamic_token_batching=True,
    max_tokens_per_batch=2048,
    token_counter_fn=my_token_counter,  # or omit to use default
    # ...
)
```
Then call `processor.process_batch(...)` normally. The library will chunk your items automatically so each sub-batch is below 2048 tokens total.

---

## Contributing

1. Fork the repo and create your feature branch.
2. Add tests for your changes.
3. Submit a pull request.

---

## License

MIT License
