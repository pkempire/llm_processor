#LLM Processor Library

A high-performance Python library for batch processing and concurrency of LLM API calls.

Built for processing large datasets with LLM's efficiently, building high throuhgput live servers, real-time agents 

#1. High-volume offline data processing: concurrency + checkpointing + caching → reduces multi-hour jobs to minutes.
#2. Real-time request handling: dynamic concurrency, load balancing, rate limiting → keep latencies low, handle spikes.
#3. Agent or multi-step reasoning: managing repeated LLM calls in a single user flow, or tool + LLM interplay.

##Results 
Using DeepSeek API as they have **No Rate Limit**




 Supports:

1. **Item-Level Concurrency**  
2. **Batch-Level Concurrency** (auto combine multiple items into one prompt)  
3. **Caching** (optional disk-based)  
4. **Retries & Backoff**  
5. **Dynamic Token Batching** (avoid exceeding token limits automatically)  
6. **Progress Tracking & Metrics**

---

## Features

- **Concurrency**: Leverages `ThreadPoolExecutor` to manage thousands of LLM calls in parallel, drastically reducing total run time.
- **Batching**: Item level and batch level: Merge multiple items (e.g., short texts) into a single LLM prompt, saving overhead with network, etc.
- **Caching**: Optionally skip repeated prompts via on-disk JSON caching.
- **Retry**: Automatic exponential backoff on failures or rate-limit errors.
- **Dynamic Token Batching**: Automatic chunking of items so the total tokens in each prompt stay below a configured limit (e.g. 2,000 tokens).
- **Configurable**: A simple `ProcessorConfig` dataclass sets concurrency, batch size, etc.

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/yourname/my_llm_lib.git
   cd my_llm_lib
Install dependencies:

bash
Copy
pip install -r requirements.txt
(Add any dependencies like tqdm, requests, or tiktoken if needed.)

Use it in your projects:

python
Copy
from llm_processor.processor import LLMProcessor
from llm_processor.processor_config import ProcessorConfig
from llm_processor.llm_client import DeepSeekClient  # or OpenAIClient, etc.
Quick Start
python
Copy
from llm_processor.processor import LLMProcessor
from llm_processor.processor_config import ProcessorConfig
from llm_processor.llm_client import BaseLLMClient

# 1) Define or import your LLM client
class FakeLLMClient(BaseLLMClient):
    def call_api(self, prompt: str, system_prompt=None, **kwargs):
        # Simulate an LLM call
        return {"content": "Simulated response", "success": True}
    def validate_response(self, response):
        return True

# 2) Create a config
config = ProcessorConfig(
    max_workers=10,
    batch_size=10,
    enable_batch_prompts=True,
    enable_dynamic_token_batching=False,
    # ...
)

# 3) Create the processor
client = FakeLLMClient()
processor = LLMProcessor(llm_client=client, config=config)

# 4) Provide a "process function"
def process_subbatch(batch_of_items):
    # Build a single prompt from multiple items
    prompt = "Combine these items:\n"
    for it in batch_of_items:
        prompt += f"- {it}\n"
    response = client.call_api(prompt)
    return {"batch_result": response}

# 5) Run
items = [f"Item {i}" for i in range(50)]
results = processor.process_batch(items, process_subbatch)
print(results)
Dynamic Token Batching
If you want to ensure each sub-batch stays under a certain token limit:

python
Copy
config = ProcessorConfig(
    enable_dynamic_token_batching=True,
    max_tokens_per_batch=2048,
    token_counter_fn=my_token_counter,  # or omit to use default
    ...
)
Then call processor.process_batch(...) normally. The library will chunk your items automatically so each sub-batch is below 2048 tokens total.

Contributing
Fork the repo and create your feature branch.
Add tests for your changes.
Submit a pull request.


License
MIT License