"""
Exception hierarchy for the Agent Workflow.

This module defines all custom exceptions used throughout the framework,
providing clear error types for different failure scenarios.
"""


class WorkflowException(Exception):
    """
    Base exception for all workflow-related errors.

    All custom exceptions in the framework inherit from this base class,
    allowing users to catch all framework-specific errors with a single
    exception handler if desired.
    """
    pass


class NodeExecutionError(WorkflowException):
    """
    Exception raised when a node fails during execution.

    This error is raised when a node encounters an error during its
    execute() method, such as invalid input, processing failures, or
    unexpected conditions.

    Attributes:
        node_name: Name of the node that failed
        original_error: The underlying exception that caused the failure
    """

    def __init__(self, message: str, node_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.node_name = node_name
        self.original_error = original_error

    def __str__(self):
        base_msg = super().__str__()
        if self.node_name:
            base_msg = f"Node '{self.node_name}': {base_msg}"
        if self.original_error:
            base_msg = f"{base_msg} (caused by {type(self.original_error).__name__}: {self.original_error})"
        return base_msg


class RoutingError(WorkflowException):
    """
    Exception raised when routing logic fails or no valid route is found.

    This error occurs when:
    - No routing condition matches the current context
    - No default route is configured
    - A routing condition raises an exception during evaluation

    Attributes:
        context_data: Relevant context data at the time of routing failure
    """

    def __init__(self, message: str, context_data: dict = None):
        super().__init__(message)
        self.context_data = context_data or {}


class EvaluationError(WorkflowException):
    """
    Exception raised when result evaluation or optimization fails.

    This error is raised when:
    - An evaluator fails to score a result
    - An optimization strategy encounters an error
    - Maximum iterations are reached without achieving threshold

    Attributes:
        score: The evaluation score if available
        iteration: The iteration number when the error occurred
    """

    def __init__(self, message: str, score: float = None, iteration: int = None):
        super().__init__(message)
        self.score = score
        self.iteration = iteration


class LLMProviderError(WorkflowException):
    """
    Exception raised when an LLM provider encounters an error.

    This error is raised for:
    - API authentication failures
    - Rate limiting errors
    - Network connectivity issues
    - Invalid API responses
    - Model-specific errors

    Attributes:
        provider_name: Name of the LLM provider
        status_code: HTTP status code if applicable
        retry_count: Number of retries attempted
    """

    def __init__(self, message: str, provider_name: str = None,
                 status_code: int = None, retry_count: int = 0):
        super().__init__(message)
        self.provider_name = provider_name
        self.status_code = status_code
        self.retry_count = retry_count

    def __str__(self):
        base_msg = super().__str__()
        if self.provider_name:
            base_msg = f"Provider '{self.provider_name}': {base_msg}"
        if self.status_code:
            base_msg = f"{base_msg} (HTTP {self.status_code})"
        if self.retry_count > 0:
            base_msg = f"{base_msg} [after {self.retry_count} retries]"
        return base_msg


class TaskExecutionError(WorkflowException):
    """
    Exception raised when a task fails in the orchestrator-workers pattern.

    This error is raised when:
    - A worker fails to execute a task
    - Task dependencies cannot be resolved
    - Task timeout is exceeded
    - No suitable worker is available for a task

    Attributes:
        task_id: ID of the failed task
        worker_id: ID of the worker that attempted the task
        task_type: Type of the task that failed
    """

    def __init__(self, message: str, task_id: str = None,
                 worker_id: str = None, task_type: str = None):
        super().__init__(message)
        self.task_id = task_id
        self.worker_id = worker_id
        self.task_type = task_type

    def __str__(self):
        base_msg = super().__str__()
        if self.task_id:
            base_msg = f"Task '{self.task_id}': {base_msg}"
        if self.worker_id:
            base_msg = f"{base_msg} (worker: {self.worker_id})"
        if self.task_type:
            base_msg = f"{base_msg} [type: {self.task_type}]"
        return base_msg
