# llm_processor/processor_config.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProcessorConfig:
    """Configuration for LLM batch processor"""
    
    # Batch Processing
    batch_size: int = 20
    max_workers: int = 5
    
    # Retry Logic
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Caching
    cache_dir: str = "llm_cache"
    cache_format: str = "json"
    cache_enabled: bool = True
    
    # Progress & Saving
    save_interval: int = 1
    show_progress: bool = True
    output_format: str = "csv"
    
    # Rate Limiting
    rate_limit: float = 0.0  # Seconds between API calls
    dynamic_rate_limit: bool = False  # Adjust based on response times
    
    # Error Handling
    fail_fast: bool = False  # Stop on first error
    error_output_path: Optional[str] = None  # Save failed items
    
    # Monitoring
    track_metrics: bool = True
    metrics_output_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessorConfig':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })
    
    def validate(self) -> bool:
        """Validate configuration values"""
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