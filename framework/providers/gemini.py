"""
Google Gemini API provider implementation.

This module provides a concrete implementation of the LLMProvider interface
for Google's Gemini API, enabling the framework to use Gemini models for
text generation tasks.
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required for GeminiProvider. "
        "Install it with: pip install google-genai"
    )

from ..base import LLMProvider
from ..exceptions import LLMProviderError


# Configure logging
logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider implementation.
    
    This provider integrates with Google's Gemini API to generate text responses.
    It supports configurable model selection, parameter tuning, error handling
    with exponential backoff, and comprehensive logging.
    
    Attributes:
        api_key: Google API key for authentication
        model_name: Name of the Gemini model to use (e.g., 'gemini-2.5-flash-lite')
        default_params: Default generation parameters (temperature, max_tokens, etc.)
        client: Initialized Client instance
        max_retries: Maximum number of retry attempts for failed API calls
        initial_retry_delay: Initial delay in seconds for exponential backoff
    
    Example:
        >>> provider = GeminiProvider(
        ...     api_key="your-api-key",
        ...     model_name="gemini-2.5-flash-lite",
        ...     default_params={"temperature": 0.7, "max_output_tokens": 1024}
        ... )
        >>> response = provider.generate("Tell me a joke")
        >>> print(response)
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        default_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0
    ) -> None:
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google API key for authentication
            model_name: Name of the Gemini model (default: 'gemini-2.5-flash-lite')
            default_params: Default generation parameters
            max_retries: Maximum retry attempts for API errors (default: 3)
            initial_retry_delay: Initial delay for exponential backoff (default: 1.0s)
        
        Raises:
            ValueError: If api_key is empty or invalid
            LLMProviderError: If model initialization fails
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key
        self.model_name = model_name
        self.default_params = default_params or {}
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Initialize the client
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Initialized GeminiProvider with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise LLMProviderError(
                f"Failed to initialize Gemini client: {e}",
                provider_name="gemini"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a single prompt.
        
        Makes an API call to Gemini with the provided prompt and parameters.
        Includes error handling with exponential backoff retry logic.
        
        Args:
            prompt: The input prompt to send to the LLM
            **kwargs: Additional parameters to override defaults
                - temperature: Controls randomness (0.0 to 1.0)
                - max_output_tokens: Maximum tokens in response
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
        
        Returns:
            The generated text response
        
        Raises:
            LLMProviderError: If API call fails after all retries
            ValueError: If prompt is empty
        
        Example:
            >>> provider = GeminiProvider(api_key="...")
            >>> response = provider.generate(
            ...     "Explain quantum computing",
            ...     temperature=0.5,
            ...     max_output_tokens=512
            ... )
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Merge default params with provided kwargs
        generation_params = {**self.default_params, **kwargs}
        
        # Log the API call
        logger.debug(f"Generating response for prompt (length: {len(prompt)})")
        logger.debug(f"Generation parameters: {generation_params}")
        
        # Prepare generation config
        generation_config = self._build_generation_config(generation_params)
        
        # Execute with retry logic
        start_time = time.time()
        response_text = self._execute_with_retry(
            lambda: self._call_generate_api(prompt, generation_config)
        )
        duration = time.time() - start_time
        
        # Log success
        logger.info(
            f"Generated response (length: {len(response_text)}, "
            f"duration: {duration:.2f}s)"
        )
        
        return response_text
    
    def _build_generation_config(self, params: Dict[str, Any]) -> Optional[types.GenerateContentConfig]:
        """
        Build a GenerateContentConfig object from parameters.
        
        Args:
            params: Dictionary of generation parameters
        
        Returns:
            GenerateContentConfig object for the API call
        """
        config_dict = {}
        
        # Map common parameter names to Gemini API names
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_output_tokens",
            "max_output_tokens": "max_output_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop_sequences": "stop_sequences",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }
        
        for key, value in params.items():
            if key in param_mapping:
                config_dict[param_mapping[key]] = value
        
        return types.GenerateContentConfig(**config_dict) if config_dict else None
    
    def _call_generate_api(
        self,
        prompt: str,
        generation_config: Optional[types.GenerateContentConfig]
    ) -> str:
        """
        Make the actual API call to Gemini.
        
        Args:
            prompt: The input prompt
            generation_config: Generation configuration
        
        Returns:
            The generated text response
        
        Raises:
            Exception: Any API error (to be caught by retry logic)
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )
            
            # Parse and normalize the response
            if not response or not response.text:
                raise LLMProviderError(
                    "Empty response from Gemini API",
                    provider_name="gemini"
                )
            
            # Log token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                logger.debug(
                    f"Token usage - Prompt: {getattr(usage, 'prompt_token_count', 'N/A')}, "
                    f"Candidates: {getattr(usage, 'candidates_token_count', 'N/A')}, "
                    f"Total: {getattr(usage, 'total_token_count', 'N/A')}"
                )
            
            return response.text
        
        except Exception as e:
            # Log the error for debugging
            logger.warning(f"API call failed: {type(e).__name__}: {e}")
            raise

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Processes multiple prompts and returns their responses. Handles partial
        failures gracefully by continuing with remaining prompts and collecting
        errors for failed ones.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to override defaults
        
        Returns:
            List of generated text responses, one per prompt. Failed prompts
            will have an error message string in their position.
        
        Raises:
            ValueError: If prompts list is empty
        
        Example:
            >>> provider = GeminiProvider(api_key="...")
            >>> prompts = [
            ...     "What is AI?",
            ...     "Explain machine learning",
            ...     "Define neural networks"
            ... ]
            >>> responses = provider.generate_batch(prompts, temperature=0.7)
            >>> print(len(responses))
            3
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        logger.info(f"Processing batch of {len(prompts)} prompts")
        
        results = []
        successful = 0
        failed = 0
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                response = self.generate(prompt, **kwargs)
                results.append(response)
                successful += 1
            except Exception as e:
                # Handle partial failure gracefully
                error_msg = f"[Error generating response: {type(e).__name__}: {str(e)}]"
                logger.error(f"Failed to generate response for prompt {i+1}: {e}")
                results.append(error_msg)
                failed += 1
        
        logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed"
        )
        
        return results

    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Implements retry logic for handling transient API errors like rate
        limiting, timeouts, and temporary service unavailability.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            The result of the function call
        
        Raises:
            LLMProviderError: If all retry attempts fail
        """
        last_exception = None
        delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error: {error_type}: {e}")
                    raise LLMProviderError(
                        f"API error: {error_type}: {e}",
                        provider_name="gemini",
                        retry_count=attempt
                    )
                
                # Log retry attempt
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: "
                        f"{error_type}: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"All {self.max_retries} retry attempts failed. "
                        f"Last error: {error_type}: {e}"
                    )
        
        # All retries exhausted
        raise LLMProviderError(
            f"Failed after {self.max_retries} attempts. "
            f"Last error: {type(last_exception).__name__}: {last_exception}",
            provider_name="gemini",
            retry_count=self.max_retries
        )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception to check
        
        Returns:
            True if the error should trigger a retry, False otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Retryable error patterns
        retryable_patterns = [
            "rate limit",
            "quota",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "internal error",
            "503",
            "429",
            "500",
        ]
        
        # Check if error message contains retryable patterns
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True
        
        # Check specific exception types that are retryable
        retryable_types = [
            "TimeoutError",
            "ConnectionError",
            "ServiceUnavailable",
        ]
        
        if error_type in retryable_types:
            return True
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Gemini model.
        
        Returns:
            Dictionary containing model metadata including name, provider,
            and configuration details.
        
        Example:
            >>> provider = GeminiProvider(api_key="...")
            >>> info = provider.get_model_info()
            >>> print(info["model_name"])
            gemini-pro
        """
        return {
            "provider": "google-gemini",
            "model_name": self.model_name,
            "default_params": self.default_params.copy(),
            "max_retries": self.max_retries,
            "initial_retry_delay": self.initial_retry_delay,
        }
    
    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"GeminiProvider(model='{self.model_name}')"
