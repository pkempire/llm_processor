# multi_processing/processor_config.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProcessorConfig:
    """Configuration for LLM batch processor"""
    
    # Batch Processing
    batch_size: int = 1
    max_workers: int = 10
    enable_batch_prompts: bool = False  # <--- ADDED

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
    
    def validate(self) -> bool:
        try:
            assert self.batch_size > 0, "batch_size must be positive"
            assert self.max_workers > 0, "max_workers must be positive"
            assert self.max_retries >= 0, "max_retries must be non-negative"
            assert self.retry_delay >= 0, "retry_delay must be non-negative"
            assert self.rate_limit >= 0, "rate_limit must be non-negative"
            return True
        except AssertionError as e:
            print(f"Invalid configuration: {e}")
            return False
