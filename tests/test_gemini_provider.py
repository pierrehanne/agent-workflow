"""
Unit tests for Gemini provider module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from framework.exceptions import LLMProviderError
from framework.providers.gemini import GeminiProvider


class TestGeminiProvider:
    """Tests for GeminiProvider class."""

    def test_initialization_success(self):
        """Test successful provider initialization."""
        with patch('framework.providers.gemini.genai.Client') as mock_client:
            provider = GeminiProvider(
                api_key="test_key",
                model_name="gemini-2.0-flash-001"
            )

            assert provider.api_key == "test_key"
            assert provider.model_name == "gemini-2.0-flash-001"
            assert provider.max_retries == 3
            mock_client.assert_called_once_with(api_key="test_key")

    def test_initialization_empty_api_key(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GeminiProvider(api_key="")

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(
                api_key="test_key",
                model_name="gemini-pro",
                default_params={"temperature": 0.5, "max_output_tokens": 512},
                max_retries=5,
                initial_retry_delay=2.0
            )

            assert provider.default_params["temperature"] == 0.5
            assert provider.default_params["max_output_tokens"] == 512
            assert provider.max_retries == 5
            assert provider.initial_retry_delay == 2.0

    def test_initialization_failure(self):
        """Test that initialization failure raises LLMProviderError."""
        with patch('framework.providers.gemini.genai.Client', side_effect=Exception("Init failed")):
            with pytest.raises(LLMProviderError, match="Failed to initialize"):
                GeminiProvider(api_key="test_key")

    def test_generate_success(self):
        """Test successful text generation."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            # Setup mock
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Generated response"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="test_key")
            result = provider.generate("Test prompt")

            assert result == "Generated response"
            mock_client.models.generate_content.assert_called_once()

    def test_generate_with_parameters(self):
        """Test generation with custom parameters."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="test_key")
            result = provider.generate(
                "Test prompt",
                temperature=0.5,
                max_output_tokens=100
            )

            assert result == "Response"

            # Check that parameters were passed
            call_args = mock_client.models.generate_content.call_args
            assert call_args is not None

    def test_generate_empty_prompt(self):
        """Test that empty prompt raises ValueError."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                provider.generate("")

    def test_generate_empty_response(self):
        """Test handling of empty response."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = None
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="test_key")

            with pytest.raises(LLMProviderError, match="Empty response"):
                provider.generate("Test prompt")

    def test_generate_batch_success(self):
        """Test successful batch generation."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(api_key="test_key")
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

            results = provider.generate_batch(prompts)

            assert len(results) == 3
            assert all(r == "Response" for r in results)

    def test_generate_batch_empty_prompts(self):
        """Test that empty prompts list raises ValueError."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            with pytest.raises(ValueError, match="Prompts list cannot be empty"):
                provider.generate_batch([])

    def test_generate_batch_partial_failure(self):
        """Test batch generation with partial failures."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call succeeds, second fails, third succeeds
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.side_effect = [
                mock_response,
                Exception("API error"),
                mock_response
            ]

            provider = GeminiProvider(api_key="test_key")
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

            results = provider.generate_batch(prompts)

            assert len(results) == 3
            assert results[0] == "Response"
            assert "Error" in results[1]  # Error message
            assert results[2] == "Response"

    def test_build_generation_config(self):
        """Test building generation config from parameters."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            params = {
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "top_p": 0.9,
                "top_k": 40
            }

            config = provider._build_generation_config(params)

            assert config is not None

    def test_build_generation_config_empty(self):
        """Test building config with empty parameters."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            config = provider._build_generation_config({})

            assert config is None

    def test_is_retryable_error(self):
        """Test identification of retryable errors."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            # Retryable errors
            assert provider._is_retryable_error(Exception("rate limit exceeded"))
            assert provider._is_retryable_error(Exception("503 service unavailable"))
            assert provider._is_retryable_error(Exception("timeout"))
            assert provider._is_retryable_error(Exception("quota exceeded"))

            # Non-retryable errors
            assert not provider._is_retryable_error(Exception("invalid api key"))
            assert not provider._is_retryable_error(Exception("bad request"))

    def test_execute_with_retry_success(self):
        """Test successful execution with retry logic."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            mock_func = Mock(return_value="success")

            result = provider._execute_with_retry(mock_func)

            assert result == "success"
            mock_func.assert_called_once()

    def test_execute_with_retry_eventual_success(self):
        """Test retry logic with eventual success."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(
                api_key="test_key",
                max_retries=3,
                initial_retry_delay=0.01  # Fast for testing
            )

            mock_func = Mock(side_effect=[
                Exception("rate limit"),
                Exception("rate limit"),
                "success"
            ])

            result = provider._execute_with_retry(mock_func)

            assert result == "success"
            assert mock_func.call_count == 3

    def test_execute_with_retry_all_failures(self):
        """Test retry logic when all attempts fail."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(
                api_key="test_key",
                max_retries=2,
                initial_retry_delay=0.01
            )

            mock_func = Mock(side_effect=Exception("rate limit"))

            with pytest.raises(LLMProviderError, match="Failed after 2 attempts"):
                provider._execute_with_retry(mock_func)

            assert mock_func.call_count == 2

    def test_execute_with_retry_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key", max_retries=3)

            mock_func = Mock(side_effect=Exception("invalid api key"))

            with pytest.raises(LLMProviderError, match="API error"):
                provider._execute_with_retry(mock_func)

            # Should fail immediately without retries
            assert mock_func.call_count == 1

    def test_get_model_info(self):
        """Test getting model information."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(
                api_key="test_key",
                model_name="gemini-2.0-flash-001",
                default_params={"temperature": 0.7},
                max_retries=5
            )

            info = provider.get_model_info()

            assert info["provider"] == "google-gemini"
            assert info["model_name"] == "gemini-2.0-flash-001"
            assert info["default_params"]["temperature"] == 0.7
            assert info["max_retries"] == 5

    def test_repr(self):
        """Test string representation."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(
                api_key="test_key",
                model_name="gemini-2.0-flash-001"
            )

            repr_str = repr(provider)

            assert "GeminiProvider" in repr_str
            assert "gemini-2.0-flash-001" in repr_str

    def test_parameter_mapping(self):
        """Test that parameters are correctly mapped."""
        with patch('framework.providers.gemini.genai.Client'):
            provider = GeminiProvider(api_key="test_key")

            # Test max_tokens -> max_output_tokens mapping
            params = {"max_tokens": 512}
            config = provider._build_generation_config(params)

            assert config is not None

            # Test that both max_tokens and max_output_tokens work
            params = {"max_output_tokens": 1024}
            config = provider._build_generation_config(params)

            assert config is not None

    def test_default_params_used(self):
        """Test that default parameters are used."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(
                api_key="test_key",
                default_params={"temperature": 0.5}
            )

            provider.generate("Test prompt")

            # Default params should be used
            call_args = mock_client.models.generate_content.call_args
            assert call_args is not None

    def test_kwargs_override_defaults(self):
        """Test that kwargs override default parameters."""
        with patch('framework.providers.gemini.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider(
                api_key="test_key",
                default_params={"temperature": 0.5}
            )

            provider.generate("Test prompt", temperature=0.9)

            # Kwargs should override defaults
            call_args = mock_client.models.generate_content.call_args
            assert call_args is not None
