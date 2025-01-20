# llm_processor/llm_client.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from openai import OpenAI
import json
import time

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Subclasses implement specific provider APIs (DeepSeek, OpenAI, etc.)
    """
    
    @abstractmethod
    def call_api(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """Make an LLM API call and return structured result"""
        pass
    
    @abstractmethod
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the API response structure"""
        pass

class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API client implementation
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 temperature: float = 0.1):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
    
    def call_api(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Call DeepSeek API with retry logic and response validation
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            result = {
                'content': response.choices[0].message.content,
                'raw_response': response,
                'success': True
            }
            
            if not self.validate_response(result):
                result['success'] = False
                result['error'] = 'Invalid response structure'
            
            return result
            
        except Exception as e:
            return {
                'content': None,
                'success': False,
                'error': str(e)
            }
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate DeepSeek response structure
        """
        return (
            isinstance(response, dict) and
            'content' in response and
            'success' in response
        )

class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client implementation
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4",
                 temperature: float = 0.1):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
    
    def call_api(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Call OpenAI API with standard settings
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            result = {
                'content': response.choices[0].message.content,
                'raw_response': response,
                'success': True
            }
            
            if not self.validate_response(result):
                result['success'] = False
                result['error'] = 'Invalid response structure'
            
            return result
            
        except Exception as e:
            return {
                'content': None,
                'success': False,
                'error': str(e)
            }
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate OpenAI response structure
        """
        return (
            isinstance(response, dict) and
            'content' in response and
            'success' in response
        )