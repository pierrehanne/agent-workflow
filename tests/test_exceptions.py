"""
Unit tests for exceptions module.
"""

import pytest

from framework.exceptions import (
    EvaluationError,
    LLMProviderError,
    NodeExecutionError,
    RoutingError,
    TaskExecutionError,
    WorkflowException,
)


class TestWorkflowException:
    """Tests for WorkflowException base class."""

    def test_basic_exception(self):
        """Test basic exception creation and message."""
        exc = WorkflowException("Test error message")

        assert str(exc) == "Test error message"
        assert isinstance(exc, Exception)

    def test_exception_can_be_raised(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(WorkflowException) as exc_info:
            raise WorkflowException("Test error")

        assert "Test error" in str(exc_info.value)


class TestNodeExecutionError:
    """Tests for NodeExecutionError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        exc = NodeExecutionError("Execution failed")

        assert "Execution failed" in str(exc)

    def test_with_node_name(self):
        """Test error with node name."""
        exc = NodeExecutionError("Execution failed", node_name="test_node")

        error_str = str(exc)
        assert "test_node" in error_str
        assert "Execution failed" in error_str

    def test_with_original_error(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        exc = NodeExecutionError(
            "Execution failed",
            node_name="test_node",
            original_error=original
        )

        error_str = str(exc)
        assert "test_node" in error_str
        assert "ValueError" in error_str
        assert "Original error" in error_str

    def test_attributes(self):
        """Test that attributes are stored correctly."""
        original = ValueError("Original")
        exc = NodeExecutionError(
            "Failed",
            node_name="node1",
            original_error=original
        )

        assert exc.node_name == "node1"
        assert exc.original_error == original


class TestRoutingError:
    """Tests for RoutingError class."""

    def test_basic_error(self):
        """Test basic routing error."""
        exc = RoutingError("No route found")

        assert "No route found" in str(exc)

    def test_with_context_data(self):
        """Test error with context data."""
        context_data = {"key": "value", "score": 0.5}
        exc = RoutingError("No route found", context_data=context_data)

        assert exc.context_data == context_data
        assert "No route found" in str(exc)


class TestEvaluationError:
    """Tests for EvaluationError class."""

    def test_basic_error(self):
        """Test basic evaluation error."""
        exc = EvaluationError("Evaluation failed")

        assert "Evaluation failed" in str(exc)

    def test_with_score(self):
        """Test error with score."""
        exc = EvaluationError("Low score", score=0.3)

        assert exc.score == 0.3
        assert "Low score" in str(exc)

    def test_with_iteration(self):
        """Test error with iteration number."""
        exc = EvaluationError("Max iterations reached", iteration=5)

        assert exc.iteration == 5
        assert "Max iterations reached" in str(exc)

    def test_with_score_and_iteration(self):
        """Test error with both score and iteration."""
        exc = EvaluationError(
            "Failed to meet threshold",
            score=0.6,
            iteration=3
        )

        assert exc.score == 0.6
        assert exc.iteration == 3


class TestLLMProviderError:
    """Tests for LLMProviderError class."""

    def test_basic_error(self):
        """Test basic provider error."""
        exc = LLMProviderError("API call failed")

        assert "API call failed" in str(exc)

    def test_with_provider_name(self):
        """Test error with provider name."""
        exc = LLMProviderError("API call failed", provider_name="gemini")

        error_str = str(exc)
        assert "gemini" in error_str
        assert "API call failed" in error_str

    def test_with_status_code(self):
        """Test error with HTTP status code."""
        exc = LLMProviderError(
            "Rate limit exceeded",
            provider_name="gemini",
            status_code=429
        )

        error_str = str(exc)
        assert "429" in error_str
        assert "Rate limit exceeded" in error_str

    def test_with_retry_count(self):
        """Test error with retry count."""
        exc = LLMProviderError(
            "Failed after retries",
            provider_name="gemini",
            retry_count=3
        )

        error_str = str(exc)
        assert "3 retries" in error_str
        assert "Failed after retries" in error_str

    def test_all_attributes(self):
        """Test error with all attributes."""
        exc = LLMProviderError(
            "Complete failure",
            provider_name="gemini",
            status_code=500,
            retry_count=5
        )

        error_str = str(exc)
        assert "gemini" in error_str
        assert "500" in error_str
        assert "5 retries" in error_str
        assert "Complete failure" in error_str

        # Check attributes
        assert exc.provider_name == "gemini"
        assert exc.status_code == 500
        assert exc.retry_count == 5


class TestTaskExecutionError:
    """Tests for TaskExecutionError class."""

    def test_basic_error(self):
        """Test basic task execution error."""
        exc = TaskExecutionError("Task failed")

        assert "Task failed" in str(exc)

    def test_with_task_id(self):
        """Test error with task ID."""
        exc = TaskExecutionError("Task failed", task_id="task_001")

        error_str = str(exc)
        assert "task_001" in error_str
        assert "Task failed" in error_str

    def test_with_worker_id(self):
        """Test error with worker ID."""
        exc = TaskExecutionError(
            "Task failed",
            task_id="task_001",
            worker_id="worker_1"
        )

        error_str = str(exc)
        assert "task_001" in error_str
        assert "worker_1" in error_str

    def test_with_task_type(self):
        """Test error with task type."""
        exc = TaskExecutionError(
            "Task failed",
            task_id="task_001",
            task_type="summarize"
        )

        error_str = str(exc)
        assert "task_001" in error_str
        assert "summarize" in error_str

    def test_all_attributes(self):
        """Test error with all attributes."""
        exc = TaskExecutionError(
            "Complete task failure",
            task_id="task_001",
            worker_id="worker_1",
            task_type="summarize"
        )

        error_str = str(exc)
        assert "task_001" in error_str
        assert "worker_1" in error_str
        assert "summarize" in error_str
        assert "Complete task failure" in error_str

        # Check attributes
        assert exc.task_id == "task_001"
        assert exc.worker_id == "worker_1"
        assert exc.task_type == "summarize"


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_workflow_exception(self):
        """Test that all custom exceptions inherit from WorkflowException."""
        assert issubclass(NodeExecutionError, WorkflowException)
        assert issubclass(RoutingError, WorkflowException)
        assert issubclass(EvaluationError, WorkflowException)
        assert issubclass(LLMProviderError, WorkflowException)
        assert issubclass(TaskExecutionError, WorkflowException)

    def test_can_catch_all_with_base_exception(self):
        """Test that all exceptions can be caught with WorkflowException."""
        exceptions = [
            NodeExecutionError("test"),
            RoutingError("test"),
            EvaluationError("test"),
            LLMProviderError("test"),
            TaskExecutionError("test")
        ]

        for exc in exceptions:
            with pytest.raises(WorkflowException):
                raise exc
