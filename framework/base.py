"""
Base abstractions for the Agent Workflow.

This module provides the foundational classes and interfaces that all workflow
components build upon, including context management, node abstractions, and
LLM provider interfaces.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)


class WorkflowContext:
    """
    Container for workflow execution state and data.

    The WorkflowContext maintains all data flowing through a workflow, including
    intermediate results, execution metadata, and a complete history of operations
    for debugging purposes.

    Attributes:
        data: Dictionary storing intermediate results and workflow data
        metadata: Dictionary containing execution metrics (timestamps, tokens, etc.)
        history: List of execution events for debugging and auditing

    Example:
        >>> context = WorkflowContext()
        >>> context.set("input", "Hello, world!")
        >>> context.set("processed", context.get("input").upper())
        >>> print(context.get("processed"))
        HELLO, WORLD!
    """

    def __init__(self) -> None:
        """Initialize an empty workflow context."""
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "total_tokens": 0,
            "model_calls": 0,
            "execution_time_ms": 0
        }
        self.history: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the context.

        Args:
            key: The key to store the value under
            value: The value to store

        Example:
            >>> context = WorkflowContext()
            >>> context.set("result", "Success")
        """
        self.data[key] = value
        self._add_history_entry("set", key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the context.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The value associated with the key, or default if not found

        Example:
            >>> context = WorkflowContext()
            >>> context.set("name", "Alice")
            >>> print(context.get("name"))
            Alice
            >>> print(context.get("missing", "default"))
            default
        """
        return self.data.get(key, default)

    def update(self, data: Dict[str, Any]) -> None:
        """
        Update multiple values in the context at once.

        Args:
            data: Dictionary of key-value pairs to update

        Example:
            >>> context = WorkflowContext()
            >>> context.update({"name": "Bob", "age": 30})
        """
        self.data.update(data)
        self._add_history_entry("update", None, data)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the complete execution history.

        Returns:
            List of history entries, each containing operation details

        Example:
            >>> context = WorkflowContext()
            >>> context.set("key", "value")
            >>> history = context.get_history()
            >>> print(len(history))
            1
        """
        return self.history.copy()

    def _add_history_entry(self, operation: str, key: Optional[str], value: Any) -> None:
        """
        Add an entry to the execution history.

        Args:
            operation: Type of operation performed
            key: Key involved in the operation (if applicable)
            value: Value involved in the operation
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "key": key,
            "value_type": type(value).__name__
        }
        self.history.append(entry)

    def __repr__(self) -> str:
        """Return string representation of the context."""
        return f"WorkflowContext(data_keys={list(self.data.keys())}, history_entries={len(self.history)})"


class Node(ABC):
    """
    Abstract base class for all workflow nodes.

    A Node represents a single unit of work in a workflow. Nodes can be composed
    together to create complex workflows using patterns like chaining, routing,
    and parallelization.

    Attributes:
        name: Human-readable name for the node
        description: Detailed description of what the node does

    Example:
        >>> class CustomNode(Node):
        ...     def execute(self, context: WorkflowContext) -> str:
        ...         return "Hello from custom node"
        >>> node = CustomNode(name="custom", description="A custom node")
        >>> context = WorkflowContext()
        >>> result = node.execute(context)
    """

    def __init__(self, name: str, description: str = "") -> None:
        """
        Initialize a node with a name and description.

        Args:
            name: Unique identifier for the node
            description: Human-readable description of the node's purpose
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, context: WorkflowContext) -> Any:
        """
        Execute the node's logic.

        This method must be implemented by all concrete node classes.

        Args:
            context: The workflow context containing input data and state

        Returns:
            The result of executing the node

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def validate(self) -> bool:
        """
        Validate the node's configuration.

        Override this method to implement custom validation logic.

        Returns:
            True if the node is valid, False otherwise
        """
        return bool(self.name)

    def __repr__(self) -> str:
        """Return string representation of the node."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    This interface defines the contract that all LLM provider implementations
    must follow, enabling the framework to work with different LLM backends
    (Gemini, OpenAI, Anthropic, etc.) through a consistent API.

    Example:
        >>> class CustomProvider(LLMProvider):
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         return "Generated response"
        ...     def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        ...         return ["Response 1", "Response 2"]
        ...     def get_model_info(self) -> Dict[str, Any]:
        ...         return {"model": "custom-model", "version": "1.0"}
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt: The input prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The generated text response

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            List of generated text responses, one per prompt

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.

        Returns:
            Dictionary containing model metadata (name, version, capabilities, etc.)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass


class PromptNode(Node):
    """
    Concrete node implementation for executing LLM prompts.

    PromptNode wraps an LLM provider and executes prompts with variable
    substitution from the workflow context. It supports flexible input/output
    key mapping for seamless integration into workflows.

    Attributes:
        prompt_template: Template string with {variable} placeholders
        llm_provider: The LLM provider to use for generation
        input_key: Optional key to read input from context
        output_key: Key to store the result in context
        model_params: Parameters to pass to the LLM (temperature, max_tokens, etc.)

    Example:
        >>> provider = GeminiProvider(api_key="...")
        >>> node = PromptNode(
        ...     name="summarizer",
        ...     prompt_template="Summarize: {text}",
        ...     llm_provider=provider,
        ...     output_key="summary"
        ... )
        >>> context = WorkflowContext()
        >>> context.set("text", "Long document...")
        >>> result = node.execute(context)
    """

    def __init__(
        self,
        name: str,
        prompt_template: str,
        llm_provider: LLMProvider,
        input_key: Optional[str] = None,
        output_key: str = "output",
        model_params: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> None:
        """
        Initialize a PromptNode.

        Args:
            name: Unique identifier for the node
            prompt_template: Template string with {variable} placeholders
            llm_provider: LLM provider instance to use
            input_key: Optional key to read primary input from context
            output_key: Key to store the result in context
            model_params: Optional parameters for the LLM
            description: Human-readable description
        """
        super().__init__(name, description)
        self.prompt_template = prompt_template
        self.llm_provider = llm_provider
        self.input_key = input_key
        self.output_key = output_key
        self.model_params = model_params or {}

    def execute(self, context: WorkflowContext) -> str:
        """
        Execute the prompt node.

        Formats the prompt template with context data, calls the LLM provider,
        and stores the result in the context.

        Args:
            context: The workflow context

        Returns:
            The generated text response
        """
        logger.debug(f"Executing PromptNode: {self.name}")
        start_time = datetime.now()

        # Format the prompt with context data
        prompt = self.format_prompt(context)
        logger.debug(f"Formatted prompt for {self.name} (length: {len(prompt)} chars)")

        # Generate response using LLM provider
        response = self.llm_provider.generate(prompt, **self.model_params)

        # Calculate execution time
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"PromptNode {self.name} completed in {duration_ms:.2f}ms, "
            f"response length: {len(response)} chars"
        )

        # Store result in context
        context.set(self.output_key, response)

        # Update metadata
        context.metadata["model_calls"] = context.metadata.get("model_calls", 0) + 1

        return response

    def format_prompt(self, context: WorkflowContext) -> str:
        """
        Format the prompt template with values from context.

        Supports {variable} style placeholders that are replaced with values
        from the context. If input_key is specified, it's available as {input}.

        Args:
            context: The workflow context containing variable values

        Returns:
            The formatted prompt string

        Example:
            >>> context = WorkflowContext()
            >>> context.set("name", "Alice")
            >>> node = PromptNode(
            ...     name="greeter",
            ...     prompt_template="Hello {name}!",
            ...     llm_provider=provider
            ... )
            >>> prompt = node.format_prompt(context)
            >>> print(prompt)
            Hello Alice!
        """
        # Build format dictionary from context data
        format_dict = context.data.copy()

        # Add input_key value as 'input' if specified
        if self.input_key:
            format_dict["input"] = context.get(self.input_key, "")

        # Format the template
        try:
            return self.prompt_template.format(**format_dict)
        except KeyError as e:
            raise ValueError(f"Missing variable in context for prompt template: {e}")

    def validate(self) -> bool:
        """
        Validate the prompt node configuration.

        Returns:
            True if valid, False otherwise
        """
        return (
            super().validate() and
            bool(self.prompt_template) and
            self.llm_provider is not None and
            bool(self.output_key)
        )
