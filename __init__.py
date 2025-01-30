"""High-performance LLM processing library"""

from .multi_processing.processor import LLMProcessor
from .multi_processing.processor_config import ProcessorConfig
from .multi_processing.llm_client import BaseLLMClient, DeepSeekClient

__version__ = "0.1.0"
__all__ = [
    'LLMProcessor',
    'ProcessorConfig',
    'BaseLLMClient',
    'DeepSeekClient'
]
