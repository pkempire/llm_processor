# multi_processing/processor.py

import os
import json
import time
import hashlib
import concurrent.futures
from typing import List, Callable, Dict, Any, Optional, Union
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import random

from .processor_config import ProcessorConfig
from .llm_client import BaseLLMClient

from .batching_utils import dynamic_token_batch, default_token_counter

#TODO: dynamic token chunking: utomatically chunk big transcripts or texts so you don't exceed token limits.
#TODO: exponential backoff with API's. scale concurrency with 
#TODO: Checkpointing so you can pause/resume processing large datasets without redoing everything.
#3) Long-Document Summarization / Indexing
#Problem: Large docs exceed token limits or are too slow if done line by line.
#Building a pipeline that (a) splits docs into sections, (b) runs concurrency for summaries, (c) merges or organizes them.
#1. High-volume offline data processing: concurrency + checkpointing + caching → reduces multi-hour jobs to minutes.
#2. Real-time request handling: dynamic concurrency, load balancing, rate limiting → keep latencies low, handle spikes.
#3. Agent or multi-step reasoning: managing repeated LLM calls in a single user flow, or tool + LLM interplay.

class LLMProcessor:
    """High-performance batch processor for LLM operations"""

    def __init__(
        self, 
        llm_client: BaseLLMClient,
        config: Optional[ProcessorConfig] = None
    ):
        """
        Initialize processor with LLM client and config.
        
        Args:
            llm_client: Instance of BaseLLMClient implementation
            config: Optional ProcessorConfig, uses defaults if not provided
        """
        self.llm_client = llm_client
        self.config = config or ProcessorConfig()
        
        # Validate config
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Setup cache directory (if caching is on)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            'total_processed': 0,
            'cache_hits': 0,
            'errors': 0,
            'start_time': time.time(),
            'batch_times': [],
            'error_logs': []
        }

    def _generate_cache_key(self, input_data: Any, prefix: str = "") -> str:
        """Generate unique cache key including relevant metadata"""
        data_str = ""
        if isinstance(input_data, (str, int, float, bool)):
            data_str = str(input_data)
        else:
            try:
                data_str = json.dumps(input_data, sort_keys=True)
            except:
                data_str = str(input_data)
        
        context = {
            'input': data_str,
            'prefix': prefix,
            'model': getattr(self.llm_client, 'model', None)
        }
        
        hash_input = json.dumps(context, sort_keys=True)
        hash_obj = hashlib.md5(hash_input.encode())
        return f"{prefix}_{hash_obj.hexdigest()}"

    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path based on config format"""
        return os.path.join(
            self.config.cache_dir,
            f"{cache_key}.{self.config.cache_format}"
        )

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load from cache with metrics tracking"""
        if not self.config.cache_enabled:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.metrics['cache_hits'] += 1
                    return result
            except Exception as e:
                self._log_error(f"Cache read error: {e}", cache_key)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save to cache with error handling"""
        if not self.config.cache_enabled:
            return
            
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log_error(f"Cache write error: {e}", cache_key)

    def _process_with_retry(
        self,
        item: Any,
        process_fn: Callable[[Any], Dict],
        cache_prefix: str = "",
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """Process single item with retry/backoff and optional caching."""
        cache_key = self._generate_cache_key(item, cache_prefix)
        
        # 1. Check the cache first
        if use_cache and self.config.cache_enabled:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # 2. Retry loop
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                if self.config.rate_limit > 0:
                    time.sleep(self.config.rate_limit)
                
                # Actual processing
                start_time = time.time()
                result = process_fn(item)
                process_time = time.time() - start_time
                
                # Update metrics
                self.metrics['total_processed'] += 1
                self.metrics['batch_times'].append(process_time)
                
                # Cache if successful
                if result is not None and use_cache and self.config.cache_enabled:
                    self._save_to_cache(cache_key, result)
                    
                return result
                
            except Exception as e:
                error_msg = f"Error processing item: {str(e)}"
                self._log_error(error_msg, cache_key)
                
                # Exponential backoff only if more attempts remain
                if attempt < self.config.max_retries - 1:
                    delay = self._implement_backoff(attempt)
                    time.sleep(delay)
                else:
                    self.metrics['errors'] += 1

        return None

    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Dict],
        cache_prefix: str = "",
        use_cache: bool = False,
        desc: str = "Processing items"
    ) -> List[Dict]:
        """
        Master method for concurrency. Supports:
          1. Dynamic token-based batching (if enabled).
          2. Regular sub-batch concurrency (if enable_batch_prompts=True).
          3. Item-level concurrency otherwise.
        """
        # 1) Possibly do dynamic token-based chunking
        if self.config.enable_dynamic_token_batching:
            token_counter = self.config.token_counter_fn or default_token_counter
            subbatches = dynamic_token_batch(items, self.config.max_tokens_per_batch, token_counter)
            # We'll replace 'items' with these sub-batches
            # each sub-batch might contain 1..N items, depending on token usage
            items = subbatches
            # Force 'enable_batch_prompts=True' if dynamic token batching is used,
            # because now each "item" is itself a sub-batch
            self.config.enable_batch_prompts = True

        # 2) If enable_batch_prompts = True, chunk further by config.batch_size
        if self.config.enable_batch_prompts:
            # If dynamic_token_batch was used, 'items' might already be sub-batches,
            # but user might also want to chunk them again by batch_size. It's optional.
            # We'll assume we just use 'items' as is if dynamic token batching is used,
            # else we do the old chunking approach:
            if not self.config.enable_dynamic_token_batching:
                subbatches = []
                for i in range(0, len(items), self.config.batch_size):
                    batch = items[i : i + self.config.batch_size]
                    subbatches.append(batch)
                items = subbatches
            
            # Now 'items' is a list of sub-batches
            with tqdm(total=len(items), desc=desc) as pbar:
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    for subbatch in items:
                        futures.append(
                            executor.submit(
                                self._process_with_retry,
                                subbatch,
                                process_fn,
                                cache_prefix,
                                use_cache
                            )
                        )
                    
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            sub_result = f.result()
                            if sub_result is not None:
                                results.append(sub_result)
                        except Exception as e:
                            print(f"Sub-batch failed: {e}")
                        finally:
                            pbar.update(1)

                return results

        else:
            # 3) Item-level concurrency
            results = []
            with tqdm(total=len(items), desc=desc) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    for item in items:
                        futures.append(
                            executor.submit(
                                self._process_with_retry,
                                item,
                                process_fn,
                                cache_prefix,
                                use_cache
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            r = future.result()
                            if r is not None:
                                results.append(r)
                        except Exception as e:
                            print(f"Item processing failed: {e}")
                        finally:
                            pbar.update(1)

            return results
        
    def _save_progress(self, results: List[Dict], output_path: str):
        """Save progress with format handling"""
        try:
            if self.config.output_format == 'csv':
                pd.DataFrame(results).to_csv(output_path, index=False)
            elif self.config.output_format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            self._log_error(f"Error saving progress: {e}", None)

    def _save_failed_items(self, items: List[Any]):
        """Save failed items for later retry"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            failed_path = f"{self.config.error_output_path}_failed_{timestamp}.json"
            with open(failed_path, 'w') as f:
                json.dump(items, f, indent=2)
        except Exception as e:
            print(f"Error saving failed items: {e}")

    def _save_metrics(self):
        """Save processing metrics"""
        try:
            if len(self.metrics['batch_times']) > 0:
                avg_time = sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
            else:
                avg_time = 0.0

            metrics = {
                **self.metrics,
                'end_time': time.time(),
                'total_time': time.time() - self.metrics['start_time'],
                'avg_process_time': avg_time
            }
            
            if self.config.metrics_output_path:
                with open(self.config.metrics_output_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def _log_error(self, error_msg: str, cache_key: Optional[str]):
        """Log error with context"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'cache_key': cache_key
        }
        self.metrics['error_logs'].append(error_entry)
        if self.config.fail_fast:
            raise Exception(error_msg)

    def checkpoint_state(self, items: List[Any], processed_items: List[Any], checkpoint_path: str):
        """Save processing state for resume capability"""
        checkpoint = {
            'remaining_items': items,
            'processed_items': processed_items,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

    def resume_from_checkpoint(self, checkpoint_path: str) -> tuple[List[Any], List[Any]]:
        """Resume processing from a saved checkpoint"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.metrics = checkpoint['metrics']
        return checkpoint['remaining_items'], checkpoint['processed_items']

    def _implement_backoff(self, attempt: int) -> float:
        """Implement proper exponential backoff with jitter"""
        if attempt == 0:
            return 0
        base_delay = self.config.retry_delay
        max_delay = min(300, base_delay * (2 ** attempt))  # Cap at 5 minutes
        jitter = random.uniform(0, 0.1 * max_delay)
        return max_delay + jitter
