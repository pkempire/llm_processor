import unittest
import tempfile
import os
import json
import sys
from pathlib import Path

# Add the project root to Python path
# current_dir = Path(__file__).parent
# project_root = current_dir.parent
# sys.path.insert(0, str(project_root))

from multi_processing.processor import LLMProcessor
from multi_processing.processor_config import ProcessorConfig
from multi_processing.llm_client import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    def __init__(self, delay=0.1):
        self.calls = []
        self.delay = delay
        
    def call_api(self, prompt: str, system_prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system_prompt})
        return {
            "content": f"Response to: {prompt[:20]}...",
            "success": True
        }
        
    def validate_response(self, response):
        return True

class TestLLMProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ProcessorConfig(
            batch_size=2,
            max_workers=2,
            cache_dir=os.path.join(self.temp_dir, "cache"),
            metrics_output_path=os.path.join(self.temp_dir, "metrics.json"),
            enable_batch_prompts=True,
            cache_enabled=True,
            max_retries=2
        )
        self.client = MockLLMClient()
        self.processor = LLMProcessor(self.client, self.config)

    def tearDown(self):
        # Cleanup temp files
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_processing(self):
        """Test basic item processing"""
        items = ["test1", "test2"]
        
        def process_fn(item):
            return self.client.call_api(f"Process: {item}")
            
        results = self.processor.process_batch(items, process_fn)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["success"] for r in results))

    def test_caching(self):
        """Test that caching works"""
        items = ["cache_test"]
        
        def process_fn(item):
            return self.client.call_api(f"Process: {item}")
        
        # First run
        results1 = self.processor.process_batch(items, process_fn, use_cache=True)
        initial_calls = len(self.client.calls)
        
        # Second run - should use cache
        results2 = self.processor.process_batch(items, process_fn, use_cache=True)
        self.assertEqual(len(self.client.calls), initial_calls)  # No new calls
        self.assertEqual(results1, results2)

    def test_batch_processing(self):
        """Test that items are properly batched"""
        items = [f"item{i}" for i in range(5)]
        
        def process_fn(batch):
            # Should receive batches of size 2 (from config)
            self.assertLessEqual(len(batch), self.config.batch_size)
            return self.client.call_api(f"Batch: {batch}")
            
        results = self.processor.process_batch(items, process_fn)
        self.assertEqual(len(results), 3)  # Should have 3 batches (2+2+1)

    def test_retry_logic(self):
        """Test retry behavior on failures"""
        class FailingClient(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.fail_count = 2
                
            def call_api(self, prompt, **kwargs):
                if self.fail_count > 0:
                    self.fail_count -= 1
                    raise Exception("Simulated failure")
                return super().call_api(prompt, **kwargs)
        
        self.processor.llm_client = FailingClient()
        items = ["retry_test"]
        
        def process_fn(item):
            return self.processor.llm_client.call_api(f"Process: {item}")
            
        results = self.processor.process_batch(items, process_fn)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])

    def test_checkpointing(self):
        """Test checkpoint save/resume functionality"""
        items = [f"item{i}" for i in range(5)]
        processed = [f"processed{i}" for i in range(3)]
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.json")
        
        # Save checkpoint
        self.processor.checkpoint_state(items, processed, checkpoint_path)
        
        # Resume from checkpoint
        remaining, completed = self.processor.resume_from_checkpoint(checkpoint_path)
        self.assertEqual(remaining, items)
        self.assertEqual(completed, processed)

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked"""
        items = ["metric_test1", "metric_test2"]
        
        def process_fn(item):
            return self.client.call_api(f"Process: {item}")
            
        self.processor.process_batch(items, process_fn)
        
        # Check metrics were saved
        self.assertTrue(os.path.exists(self.config.metrics_output_path))
        with open(self.config.metrics_output_path) as f:
            metrics = json.load(f)
            self.assertEqual(metrics["total_processed"], 2)
            self.assertEqual(metrics["errors"], 0)

    def test_error_handling(self):
        """Test error handling and logging"""
        def failing_process(item):
            raise Exception("Test error")
            
        items = ["error_test"]
        self.config.fail_fast = False  # Don't raise exceptions
        
        results = self.processor.process_batch(items, failing_process)
        self.assertEqual(len(results), 0)
        self.assertEqual(self.processor.metrics["errors"], 1)
        self.assertEqual(len(self.processor.metrics["error_logs"]), 1)

    def test_concurrent_processing(self):
        """Test that items are processed concurrently"""
        import time
        
        class SlowMockClient(MockLLMClient):
            def call_api(self, prompt, **kwargs):
                time.sleep(0.1)  # Add delay
                return super().call_api(prompt, **kwargs)
        
        self.processor.llm_client = SlowMockClient()
        items = [f"item{i}" for i in range(4)]
        
        start = time.time()
        results = self.processor.process_batch(items, lambda x: self.processor.llm_client.call_api(f"Process: {x}"))
        duration = time.time() - start
        
        # With 2 workers, should take ~0.2s (2 batches of 2 items)
        # Without concurrency would take ~0.4s
        self.assertLess(duration, 0.3)
        self.assertEqual(len(results), 4)

if __name__ == '__main__':
    unittest.main()
