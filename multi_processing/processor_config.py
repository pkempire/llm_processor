# llm_processor/processor_config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List

@dataclass
class ProcessorConfig:
    """Configuration for LLM batch processor"""
    
    # Batch Processing
    batch_size: int = 1
    max_workers: int = 10
    enable_batch_prompts: bool = False
    
    # Dynamic token-based batching
    enable_dynamic_token_batching: bool = False
    max_tokens_per_batch: int = 2000
    token_counter_fn: Optional[Callable[[Any], int]] = None  # function to measure tokens
    
    # Retry Logic
    max_retries: int = 1
    retry_delay: float = 0.0
    
    # Caching
    cache_dir: str = "llm_cache"
    cache_format: str = "json"
    cache_enabled: bool = False
    
    # Progress & Saving
    save_interval: int = 1
    show_progress: bool = True
    output_format: str = "csv"
    
    # Rate Limiting
    rate_limit: float = 0.0  
    dynamic_rate_limit: bool = False  
    
    # Error Handling
    fail_fast: bool = False
    error_output_path: Optional[str] = None
    
    # Monitoring
    track_metrics: bool = True
    metrics_output_path: Optional[str] = None
    
    # Add these fields:
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 100  # Save checkpoint every N items
    
    # Improved backoff settings
    min_retry_delay: float = 1.0
    max_retry_delay: float = 300.0  # 5 minutes
    backoff_factor: float = 2.0
    jitter: bool = True
    
    # Rate limiting improvements
    requests_per_minute: Optional[int] = None
    concurrent_request_limit: Optional[int] = None
    
    # Cost tracking
    pricing_models: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'deepseek-chat': {
            'input': 0.50,  # $0.50 per 1M input tokens
            'output': 1.50  # $1.50 per 1M output tokens
        },
        'gpt-4': {
            'input': 30.00,
            'output': 60.00
        },
        'gpt-3.5-turbo': {
            'input': 0.50,
            'output': 1.50
        }
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessorConfig':
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })
    
    def copy(self) -> 'ProcessorConfig':
        """Create a copy of the configuration"""
        return ProcessorConfig(**self.__dict__)

    def validate(self) -> bool:
        try:
            assert self.batch_size > 0, "batch_size must be positive"
            assert self.max_workers > 0, "max_workers must be positive"
            assert self.max_retries >= 0, "max_retries must be non-negative"
            assert self.retry_delay >= 0, "retry_delay must be non-negative"
            assert self.rate_limit >= 0, "rate_limit must be non-negative"
            assert self.max_tokens_per_batch > 0, "max_tokens_per_batch must be > 0"
            return True
        except AssertionError as e:
            print(f"Invalid configuration: {e}")
            return False
