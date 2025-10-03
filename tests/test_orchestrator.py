"""
Tests for orchestrator-workers pattern.
"""

import pytest
from unittest.mock import Mock

from framework.base import WorkflowContext
from framework.orchestrator import (
    Task,
    TaskStatus,
    Worker,
    Orchestrator,
)
from framework.exceptions import TaskExecutionError


class MockWorker(Worker):
    """Mock worker for testing."""

    def __init__(self, worker_id: str, capabilities: list[str]):
        super().__init__(worker_id, capabilities)
        self.executed_tasks = []

    def can_handle(self, task: Task) -> bool:
        """Check if worker can handle task type."""
        return task.task_type in self.capabilities

    def execute_task(self, task: Task, context: WorkflowContext) -> str:
        """Execute task and return result."""
        self.executed_tasks.append(task.task_id)
        
        # Simulate different behaviors based on task type
        if task.task_type == "error":
            raise ValueError(f"Simulated error for task {task.task_id}")
        
        return f"Result from {self.worker_id} for task {task.task_id}"


class TestTask:
    """Tests for Task class."""

    def test_task_creation(self):
        """Test that task is created with correct attributes."""
        task = Task(
            task_id="task1",
            task_type="analysis",
            data={"key": "value"}
        )
        
        assert task.task_id == "task1"
        assert task.task_type == "analysis"
        assert task.data == {"key": "value"}
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.dependencies == []

    def test_task_with_dependencies(self):
        """Test that task can have dependencies."""
        task = Task(
            task_id="task2",
            task_type="summary",
            data={},
            dependencies=["task1"]
        )
        
        assert task.dependencies == ["task1"]

    def test_task_status_update(self):
        """Test that task status can be updated."""
        task = Task(task_id="task1", task_type="test", data={})
        
        assert task.status == TaskStatus.PENDING
        
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING
        
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_task_is_ready_no_dependencies(self):
        """Test that task with no dependencies is always ready."""
        task = Task(task_id="task1", task_type="test", data={})
        
        assert task.is_ready({}) is True

    def test_task_is_ready_with_completed_dependencies(self):
        """Test that task is ready when all dependencies are completed."""
        task1 = Task(task_id="task1", task_type="test", data={})
        task1.status = TaskStatus.COMPLETED
        
        task2 = Task(task_id="task2", task_type="test", data={}, dependencies=["task1"])
        
        assert task2.is_ready({"task1": task1}) is True

    def test_task_is_not_ready_with_pending_dependencies(self):
        """Test that task is not ready when dependencies are pending."""
        task = Task(task_id="task2", task_type="test", data={}, dependencies=["task1"])
        
        assert task.is_ready({}) is False

    def test_task_repr(self):
        """Test string representation of task."""
        task = Task(task_id="task1", task_type="analysis", data={})
        
        repr_str = repr(task)
        assert "Task" in repr_str
        assert "task1" in repr_str
        assert "analysis" in repr_str


class TestWorker:
    """Tests for Worker class."""

    def test_worker_initialization(self):
        """Test that worker is initialized correctly."""
        worker = MockWorker("worker1", ["analysis", "summary"])
        
        assert worker.worker_id == "worker1"
        assert worker.capabilities == ["analysis", "summary"]
        assert worker.executed_tasks == []

    def test_can_handle_matching_type(self):
        """Test that worker can handle matching task type."""
        worker = MockWorker("worker1", ["analysis"])
        task = Task(task_id="task1", task_type="analysis", data={})
        
        assert worker.can_handle(task) is True

    def test_can_handle_non_matching_type(self):
        """Test that worker cannot handle non-matching task type."""
        worker = MockWorker("worker1", ["analysis"])
        task = Task(task_id="task1", task_type="summary", data={})
        
        assert worker.can_handle(task) is False

    def test_execute_task_success(self):
        """Test that worker executes task successfully."""
        worker = MockWorker("worker1", ["analysis"])
        task = Task(task_id="task1", task_type="analysis", data={})
        context = WorkflowContext()
        
        result = worker.execute_task(task, context)
        
        assert result == "Result from worker1 for task task1"
        assert "task1" in worker.executed_tasks

    def test_execute_task_error(self):
        """Test that worker handles task execution errors."""
        worker = MockWorker("worker1", ["error"])
        task = Task(task_id="task1", task_type="error", data={})
        context = WorkflowContext()
        
        with pytest.raises(ValueError, match="Simulated error"):
            worker.execute_task(task, context)


class TestOrchestrator:
    """Tests for Orchestrator class."""

    def test_orchestrator_initialization(self):
        """Test that orchestrator is initialized correctly."""
        orchestrator = Orchestrator(name="test_orchestrator")
        
        assert orchestrator.name == "test_orchestrator"
        assert len(orchestrator.workers) == 0
        assert orchestrator.task_queue == []
        assert orchestrator.max_concurrent_tasks == 5

    def test_orchestrator_with_custom_concurrency(self):
        """Test that orchestrator can be initialized with custom concurrency."""
        orchestrator = Orchestrator(name="test", max_concurrent_tasks=10)
        
        assert orchestrator.max_concurrent_tasks == 10

    def test_add_worker(self):
        """Test that workers can be added to orchestrator."""
        orchestrator = Orchestrator(name="test")
        worker = MockWorker("worker1", ["analysis"])
        
        result = orchestrator.add_worker(worker)
        
        assert result is orchestrator  # Fluent interface
        assert len(orchestrator.workers) == 1
        assert orchestrator.workers[0] == worker

    def test_add_multiple_workers(self):
        """Test that multiple workers can be added."""
        orchestrator = Orchestrator(name="test")
        worker1 = MockWorker("worker1", ["analysis"])
        worker2 = MockWorker("worker2", ["summary"])
        
        orchestrator.add_worker(worker1).add_worker(worker2)
        
        assert len(orchestrator.workers) == 2

    def test_add_task(self):
        """Test that tasks can be added to orchestrator."""
        orchestrator = Orchestrator(name="test")
        task = Task(task_id="task1", task_type="analysis", data={})
        
        result = orchestrator.add_task(task)
        
        assert result is orchestrator  # Fluent interface
        assert len(orchestrator.task_queue) == 1
        assert orchestrator.task_queue[0] == task

    def test_add_multiple_tasks(self):
        """Test that multiple tasks can be added."""
        orchestrator = Orchestrator(name="test")
        task1 = Task(task_id="task1", task_type="analysis", data={})
        task2 = Task(task_id="task2", task_type="summary", data={})
        
        orchestrator.add_task(task1).add_task(task2)
        
        assert len(orchestrator.task_queue) == 2

    def test_execute_single_task(self):
        """Test that orchestrator executes a single task."""
        worker = MockWorker("worker1", ["analysis"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        
        task = Task(task_id="task1", task_type="analysis", data={})
        orchestrator.add_task(task)
        
        context = WorkflowContext()
        results = orchestrator.execute(context)
        
        # Results contain completed, failed, and summary keys
        assert "completed" in results
        assert "task1" in results["completed"]
        assert "Result from worker1" in results["completed"]["task1"]
        assert task.status == TaskStatus.COMPLETED

    def test_execute_multiple_tasks(self):
        """Test that orchestrator executes multiple tasks."""
        worker1 = MockWorker("worker1", ["analysis"])
        worker2 = MockWorker("worker2", ["summary"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker1).add_worker(worker2)
        
        task1 = Task(task_id="task1", task_type="analysis", data={})
        task2 = Task(task_id="task2", task_type="summary", data={})
        
        orchestrator.add_task(task1).add_task(task2)
        context = WorkflowContext()
        results = orchestrator.execute(context)
        
        assert "completed" in results
        assert "task1" in results["completed"]
        assert "task2" in results["completed"]

    def test_execute_with_dependencies(self):
        """Test that orchestrator respects task dependencies."""
        worker = MockWorker("worker1", ["analysis", "summary"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        
        task1 = Task(task_id="task1", task_type="analysis", data={})
        task2 = Task(
            task_id="task2",
            task_type="summary",
            data={},
            dependencies=["task1"]
        )
        
        orchestrator.add_task(task2).add_task(task1)  # Add in reverse order
        context = WorkflowContext()
        results = orchestrator.execute(context)
        
        assert "completed" in results
        assert len(results["completed"]) == 2
        # task1 should be executed before task2
        assert worker.executed_tasks.index("task1") < worker.executed_tasks.index("task2")

    def test_validate_with_workers_and_tasks(self):
        """Test that orchestrator validates successfully with workers and tasks."""
        worker = MockWorker("worker1", ["analysis"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        task = Task(task_id="task1", task_type="analysis", data={})
        orchestrator.add_task(task)
        
        assert orchestrator.validate() is True

    def test_validate_without_workers(self):
        """Test that orchestrator validation fails without workers."""
        orchestrator = Orchestrator(name="test")
        
        assert orchestrator.validate() is False

    def test_context_updates(self):
        """Test that orchestrator updates context with results."""
        worker = MockWorker("worker1", ["analysis"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        
        task = Task(task_id="task1", task_type="analysis", data={})
        orchestrator.add_task(task)
        
        context = WorkflowContext()
        orchestrator.execute(context)
        
        # Check that results are stored in context
        results = context.get("test_results")
        assert results is not None
        assert "completed" in results
        assert "task1" in results["completed"]

    def test_metadata_tracking(self):
        """Test that orchestrator tracks execution metadata."""
        worker = MockWorker("worker1", ["analysis"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        
        task = Task(task_id="task1", task_type="analysis", data={})
        orchestrator.add_task(task)
        
        context = WorkflowContext()
        orchestrator.execute(context)
        
        # Check metadata
        assert "execution_time_ms" in context.metadata
        assert context.metadata["execution_time_ms"] > 0

    def test_repr(self):
        """Test string representation of orchestrator."""
        worker = MockWorker("worker1", ["analysis"])
        orchestrator = Orchestrator(name="test")
        orchestrator.add_worker(worker)
        
        repr_str = repr(orchestrator)
        assert "Orchestrator" in repr_str
        assert "test" in repr_str
