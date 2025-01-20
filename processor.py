# llm_processor/processor.py

import os
import json
import time
import hashlib
import concurrent.futures
from typing import List, Callable, Dict, Any, Optional, Union, Type
from functools import partial
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from .processor_config import ProcessorConfig
from .llm_client import BaseLLMClient

class LLMProcessor:
    """High-performance batch processor for LLM operations"""

    def __init__(self, 
                 llm_client: BaseLLMClient,
                 config: Optional[ProcessorConfig] = None):
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
            
        # Setup cache directory
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
        if isinstance(input_data, (str, int, float, bool)):
            data_str = str(input_data)
        else:
            data_str = json.dumps(input_data, sort_keys=True)
        
        # Include relevant context in cache key
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
        """Process single item with enhanced retry logic and monitoring"""
        cache_key = self._generate_cache_key(item, cache_prefix)
        
        # Check cache first if enabled
        if use_cache and self.config.cache_enabled:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                # Apply rate limiting if configured
                if self.config.rate_limit > 0:
                    time.sleep(self.config.rate_limit)
                
                # Process item
                start_time = time.time()
                result = process_fn(item)
                process_time = time.time() - start_time
                
                # Track metrics
                self.metrics['total_processed'] += 1
                self.metrics['batch_times'].append(process_time)
                
                # Cache successful result
                if result is not None and use_cache:
                    self._save_to_cache(cache_key, result)
                    
                return result
                
            except Exception as e:
                error_msg = f"Error processing item: {str(e)}"
                self._log_error(error_msg, cache_key)
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    self.metrics['errors'] += 1

        return None

    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Dict],
        transform_fn: Optional[Callable[[Dict], List[Dict]]] = None,
        cache_prefix: str = "",
        use_cache: bool = True,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process items in batches with enhanced monitoring and error handling
        """
        all_results = []
        failed_items = []
        
        # Setup progress bar
        indices = range(0, len(items), self.config.batch_size)
        pbar = tqdm(indices, disable=not self.config.show_progress)
        
        batch_start_time = time.time()
        
        for batch_i in pbar:
            batch_data = items[batch_i : batch_i + self.config.batch_size]
            
            # Process batch with thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                fn = partial(
                    self._process_with_retry,
                    process_fn=process_fn,
                    cache_prefix=cache_prefix,
                    use_cache=use_cache
                )
                batch_results = list(executor.map(fn, batch_data))
            
            # Transform results if needed
            if transform_fn:
                transformed_batch = []
                for br in batch_results:
                    if br is not None:
                        try:
                            records = transform_fn(br)
                            transformed_batch.extend(records)
                        except Exception as e:
                            self._log_error(f"Transform error: {e}", None)
                batch_results = transformed_batch
            else:
                batch_results = [br for br in batch_results if br is not None]
            
            # Track failed items
            for item, result in zip(batch_data, batch_results):
                if result is None:
                    failed_items.append(item)
            
            all_results.extend(batch_results)
            
            # Save progress periodically
            batch_index = (batch_i // self.config.batch_size) + 1
            if output_path and (batch_index % self.config.save_interval == 0):
                self._save_progress(all_results, output_path)
            
            # Update progress bar with metrics
            elapsed = time.time() - batch_start_time
            processed_count = len(all_results)
            
            postfix_dict = {
                'processed': processed_count,
                'success_rate': f"{processed_count/len(items):.1%}" if items else "0%",
                'errors': self.metrics['errors'],
            }
            
            # Only add average time if we have processed items
            if processed_count > 0:
                postfix_dict['avg_time'] = f"{elapsed/processed_count:.2f}s"
            else:
                postfix_dict['avg_time'] = "N/A"
                
            pbar.set_postfix(postfix_dict)
        
        # Save final results
        if output_path:
            self._save_progress(all_results, output_path)
        
        # Save failed items if configured
        if self.config.error_output_path and failed_items:
            self._save_failed_items(failed_items)
        
        # Save metrics if configured
        if self.config.metrics_output_path:
            self._save_metrics()
        
        return all_results

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
            metrics = {
                **self.metrics,
                'end_time': time.time(),
                'total_time': time.time() - self.metrics['start_time'],
                'avg_process_time': sum(self.metrics['batch_times']) / len(self.metrics['batch_times'])
            }
            
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