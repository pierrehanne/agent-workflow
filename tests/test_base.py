"""
Unit tests for base module (WorkflowContext, Node, LLMProvider, PromptNode).
"""

from unittest.mock import Mock

import pytest

from framework.base import LLMProvider, Node, PromptNode, WorkflowContext


class TestWorkflowContext:
    """Tests for WorkflowContext class."""

    def test_initialization(self):
        """Test context initializes with empty data and metadata."""
        context = WorkflowContext()

        assert context.data == {}
        assert "start_time" in context.metadata
        assert context.metadata["total_tokens"] == 0
        assert context.metadata["model_calls"] == 0
        assert context.history == []

    def test_set_and_get(self):
        """Test setting and getting values."""
        context = WorkflowContext()

        context.set("key1", "value1")
        assert context.get("key1") == "value1"

        context.set("key2", 42)
        assert context.get("key2") == 42

    def test_get_with_default(self):
        """Test getting non-existent key returns default."""
        context = WorkflowContext()

        assert context.get("missing") is None
        assert context.get("missing", "default") == "default"

    def test_update(self):
        """Test updating multiple values at once."""
        context = WorkflowContext()

        context.update({"key1": "value1", "key2": "value2"})

        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"

    def test_history_tracking(self):
        """Test that operations are tracked in history."""
        context = WorkflowContext()

        context.set("key", "value")
        context.update({"key2": "value2"})

        history = context.get_history()
        assert len(history) == 2
        assert history[0]["operation"] == "set"
        assert history[1]["operation"] == "update"

    def test_repr(self):
        """Test string representation."""
        context = WorkflowContext()
        context.set("key1", "value1")

        repr_str = repr(context)
        assert "WorkflowContext" in repr_str
        assert "key1" in repr_str


class TestNode:
    """Tests for Node abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Node cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Node(name="test")

    def test_concrete_node_implementation(self):
        """Test that concrete implementations work."""
        class ConcreteNode(Node):
            def execute(self, context):
                return "executed"

        node = ConcreteNode(name="test", description="test node")
        assert node.name == "test"
        assert node.description == "test node"

        context = WorkflowContext()
        result = node.execute(context)
        assert result == "executed"

    def test_validate_default(self):
        """Test default validation checks name."""
        class ConcreteNode(Node):
            def execute(self, context):
                return "executed"

        node = ConcreteNode(name="test")
        assert node.validate() is True

        node_empty = ConcreteNode(name="")
        assert node_empty.validate() is False

    def test_repr(self):
        """Test string representation."""
        class ConcreteNode(Node):
            def execute(self, context):
                return "executed"

        node = ConcreteNode(name="test")
        repr_str = repr(node)
        assert "ConcreteNode" in repr_str
        assert "test" in repr_str


class TestLLMProvider:
    """Tests for LLMProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_provider_implementation(self):
        """Test that concrete implementations work."""
        class MockProvider(LLMProvider):
            def generate(self, prompt, **kwargs):
                return f"Response to: {prompt}"

            def generate_batch(self, prompts, **kwargs):
                return [f"Response to: {p}" for p in prompts]

            def get_model_info(self):
                return {"model": "mock-model"}

        provider = MockProvider()

        # Test generate
        response = provider.generate("test prompt")
        assert "test prompt" in response

        # Test generate_batch
        responses = provider.generate_batch(["prompt1", "prompt2"])
        assert len(responses) == 2

        # Test get_model_info
        info = provider.get_model_info()
        assert info["model"] == "mock-model"


class TestPromptNode:
    """Tests for PromptNode class."""

    def test_initialization(self):
        """Test PromptNode initialization."""
        mock_provider = Mock(spec=LLMProvider)

        node = PromptNode(
            name="test_node",
            prompt_template="Test {input}",
            llm_provider=mock_provider,
            output_key="result"
        )

        assert node.name == "test_node"
        assert node.prompt_template == "Test {input}"
        assert node.llm_provider == mock_provider
        assert node.output_key == "result"

    def test_format_prompt(self):
        """Test prompt formatting with context variables."""
        mock_provider = Mock(spec=LLMProvider)
        node = PromptNode(
            name="test",
            prompt_template="Hello {name}, you are {age} years old",
            llm_provider=mock_provider,
            output_key="result"
        )

        context = WorkflowContext()
        context.set("name", "Alice")
        context.set("age", 30)

        formatted = node.format_prompt(context)
        assert formatted == "Hello Alice, you are 30 years old"

    def test_format_prompt_with_input_key(self):
        """Test prompt formatting with input_key."""
        mock_provider = Mock(spec=LLMProvider)
        node = PromptNode(
            name="test",
            prompt_template="Process: {input}",
            llm_provider=mock_provider,
            input_key="data",
            output_key="result"
        )

        context = WorkflowContext()
        context.set("data", "test data")

        formatted = node.format_prompt(context)
        assert formatted == "Process: test data"

    def test_format_prompt_missing_variable(self):
        """Test that missing variables raise ValueError."""
        mock_provider = Mock(spec=LLMProvider)
        node = PromptNode(
            name="test",
            prompt_template="Hello {name}",
            llm_provider=mock_provider,
            output_key="result"
        )

        context = WorkflowContext()

        with pytest.raises(ValueError, match="Missing variable"):
            node.format_prompt(context)

    def test_execute(self):
        """Test node execution."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Generated response"

        node = PromptNode(
            name="test",
            prompt_template="Test {input}",
            llm_provider=mock_provider,
            output_key="result"
        )

        context = WorkflowContext()
        context.set("input", "data")

        result = node.execute(context)

        assert result == "Generated response"
        assert context.get("result") == "Generated response"
        mock_provider.generate.assert_called_once()

    def test_execute_with_model_params(self):
        """Test execution with model parameters."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Response"

        node = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key="result",
            model_params={"temperature": 0.5, "max_tokens": 100}
        )

        context = WorkflowContext()
        node.execute(context)

        # Check that model params were passed
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_validate(self):
        """Test node validation."""
        mock_provider = Mock(spec=LLMProvider)

        # Valid node
        valid_node = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key="result"
        )
        assert valid_node.validate() is True

        # Invalid: empty prompt
        invalid_node1 = PromptNode(
            name="test",
            prompt_template="",
            llm_provider=mock_provider,
            output_key="result"
        )
        assert invalid_node1.validate() is False

        # Invalid: no provider
        invalid_node2 = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=None,
            output_key="result"
        )
        assert invalid_node2.validate() is False

        # Invalid: empty output_key
        invalid_node3 = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key=""
        )
        assert invalid_node3.validate() is False

    def test_metadata_update(self):
        """Test that metadata is updated after execution."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Response"

        node = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key="result"
        )

        context = WorkflowContext()
        initial_calls = context.metadata.get("model_calls", 0)

        node.execute(context)

        assert context.metadata["model_calls"] == initial_calls + 1
