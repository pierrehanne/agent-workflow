"""
Orchestrator-workers pattern for coordinating multiple agents.

This module implements the orchestrator-workers pattern where a central
orchestrator coordinates multiple worker agents to execute subtasks in
parallel or with dependencies, enabling complex multi-agent workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .base import Node, WorkflowContext, LLMProvider
from .exceptions import TaskExecutionError


# Configure logging
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """
    Enumeration of possible task states.
    
    Attributes:
        PENDING: Task is waiting to be executed
        RUNNING: Task is currently being executed
        COMPLETED: Task has completed successfully
        FAILED: Task execution failed
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """
    Represents a unit of work to be executed by a worker.
    
    Tasks can have dependencies on other tasks, ensuring proper execution
    ordering. Each task has a unique ID, type, associated data, and tracks
    its execution status and result.
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., "summarize", "extract", "analyze")
        data: Dictionary containing task-specific data and parameters
        dependencies: List of task IDs that must complete before this task
        status: Current execution status of the task
        result: Result of task execution (None until completed)
    
    Example:
        >>> task = Task(
        ...     task_id="task_001",
        ...     task_type="summarize",
        ...     data={"text": "Long document...", "max_length": 100},
        ...     dependencies=[]
        ... )
        >>> task.status = TaskStatus.RUNNING
        >>> task.result = "Summary of the document"
        >>> task.status = TaskStatus.COMPLETED
    """
    task_id: str
    task_type: str
    data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    
    def is_ready(self, completed_tasks: Dict[str, 'Task']) -> bool:
        """
        Check if all dependencies are completed.
        
        Args:
            completed_tasks: Dictionary of completed tasks by task_id
        
        Returns:
            True if all dependencies are completed, False otherwise
        
        Example:
            >>> task1 = Task("t1", "type1", {})
            >>> task1.status = TaskStatus.COMPLETED
            >>> task2 = Task("t2", "type2", {}, dependencies=["t1"])
            >>> task2.is_ready({"t1": task1})
            True
        """
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def __repr__(self) -> str:
        """Return string representation of the task."""
        return f"Task(id='{self.task_id}', type='{self.task_type}', status={self.status.value})"



class Worker(ABC):
    """
    Abstract base class for worker agents.
    
    Workers are specialized agents that can handle specific types of tasks.
    Each worker has a unique ID and a set of capabilities that determine
    which tasks it can execute.
    
    Attributes:
        worker_id: Unique identifier for the worker
        capabilities: List of task types this worker can handle
    
    Example:
        >>> class CustomWorker(Worker):
        ...     def can_handle(self, task: Task) -> bool:
        ...         return task.task_type in self.capabilities
        ...     def execute_task(self, task: Task, context: WorkflowContext) -> Any:
        ...         return f"Processed {task.task_type}"
        >>> worker = CustomWorker(
        ...     worker_id="worker_1",
        ...     capabilities=["summarize", "extract"]
        ... )
    """
    
    def __init__(self, worker_id: str, capabilities: List[str]) -> None:
        """
        Initialize a worker with an ID and capabilities.
        
        Args:
            worker_id: Unique identifier for this worker
            capabilities: List of task types this worker can handle
        """
        self.worker_id = worker_id
        self.capabilities = capabilities
    
    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """
        Determine if this worker can handle the given task.
        
        This method should check if the task type matches the worker's
        capabilities and any other requirements.
        
        Args:
            task: The task to evaluate
        
        Returns:
            True if this worker can handle the task, False otherwise
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def execute_task(self, task: Task, context: WorkflowContext) -> Any:
        """
        Execute the given task.
        
        This method performs the actual work of the task and returns
        the result. It has access to the workflow context for reading
        inputs and storing intermediate results.
        
        Args:
            task: The task to execute
            context: The workflow context for data access
        
        Returns:
            The result of executing the task
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation of the worker."""
        return f"{self.__class__.__name__}(id='{self.worker_id}', capabilities={self.capabilities})"



class LLMWorker(Worker):
    """
    Concrete worker implementation for LLM-based tasks.
    
    LLMWorker uses an LLM provider to execute tasks by formatting prompts
    from templates and generating responses. It supports different prompt
    templates for different task types.
    
    Attributes:
        worker_id: Unique identifier for the worker
        capabilities: List of task types this worker can handle
        llm_provider: The LLM provider to use for generation
        prompt_templates: Dictionary mapping task types to prompt templates
        model_params: Optional parameters for the LLM
    
    Example:
        >>> provider = GeminiProvider(api_key="...")
        >>> worker = LLMWorker(
        ...     worker_id="llm_worker_1",
        ...     capabilities=["summarize", "extract"],
        ...     llm_provider=provider,
        ...     prompt_templates={
        ...         "summarize": "Summarize the following text: {text}",
        ...         "extract": "Extract key points from: {text}"
        ...     }
        ... )
        >>> task = Task("t1", "summarize", {"text": "Long document..."})
        >>> context = WorkflowContext()
        >>> result = worker.execute_task(task, context)
    """
    
    def __init__(
        self,
        worker_id: str,
        capabilities: List[str],
        llm_provider: LLMProvider,
        prompt_templates: Dict[str, str],
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize an LLM worker.
        
        Args:
            worker_id: Unique identifier for this worker
            capabilities: List of task types this worker can handle
            llm_provider: LLM provider instance to use
            prompt_templates: Dictionary mapping task types to prompt templates
            model_params: Optional parameters for the LLM (temperature, etc.)
        """
        super().__init__(worker_id, capabilities)
        self.llm_provider = llm_provider
        self.prompt_templates = prompt_templates
        self.model_params = model_params or {}
    
    def can_handle(self, task: Task) -> bool:
        """
        Check if this worker can handle the task.
        
        A worker can handle a task if:
        1. The task type is in the worker's capabilities
        2. A prompt template exists for the task type
        
        Args:
            task: The task to evaluate
        
        Returns:
            True if this worker can handle the task, False otherwise
        """
        return (
            task.task_type in self.capabilities and
            task.task_type in self.prompt_templates
        )
    
    def execute_task(self, task: Task, context: WorkflowContext) -> str:
        """
        Execute the task using the LLM provider.
        
        Formats the appropriate prompt template with task data,
        calls the LLM provider, and returns the generated response.
        
        Args:
            task: The task to execute
            context: The workflow context for data access
        
        Returns:
            The generated text response from the LLM
        
        Raises:
            ValueError: If no prompt template exists for the task type
        """
        # Get the prompt template for this task type
        template = self.prompt_templates.get(task.task_type)
        if not template:
            raise ValueError(f"No prompt template for task type: {task.task_type}")
        
        # Format the prompt with task data
        try:
            prompt = template.format(**task.data)
        except KeyError as e:
            raise ValueError(f"Missing required data for prompt template: {e}")
        
        # Log task execution
        logger.info(f"Worker {self.worker_id} executing task {task.task_id} (type: {task.task_type})")
        
        # Generate response using LLM provider
        response = self.llm_provider.generate(prompt, **self.model_params)
        
        # Update context metadata
        context.metadata["model_calls"] = context.metadata.get("model_calls", 0) + 1
        
        logger.info(f"Worker {self.worker_id} completed task {task.task_id}")
        
        return response



from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from threading import Lock


class Orchestrator(Node):
    """
    Coordinates multiple workers to execute tasks with dependencies.
    
    The Orchestrator manages a pool of workers and a queue of tasks,
    assigning tasks to appropriate workers based on their capabilities.
    It handles dependency resolution, concurrent execution, and result
    aggregation.
    
    Attributes:
        name: Name of the orchestrator node
        description: Description of the orchestrator's purpose
        workers: List of available workers
        task_queue: List of tasks to be executed
        completed_tasks: Dictionary of completed tasks by task_id
        failed_tasks: Dictionary of failed tasks by task_id
        max_concurrent_tasks: Maximum number of tasks to run concurrently
    
    Example:
        >>> orchestrator = Orchestrator(
        ...     name="doc_processor",
        ...     max_concurrent_tasks=3
        ... )
        >>> orchestrator.add_worker(worker1)
        >>> orchestrator.add_worker(worker2)
        >>> orchestrator.add_task(Task("t1", "summarize", {"text": "..."}))
        >>> orchestrator.add_task(Task("t2", "extract", {"text": "..."}, dependencies=["t1"]))
        >>> context = WorkflowContext()
        >>> results = orchestrator.execute(context)
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent_tasks: int = 5,
        description: str = ""
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            name: Unique identifier for the orchestrator
            max_concurrent_tasks: Maximum number of concurrent task executions
            description: Human-readable description
        """
        super().__init__(name, description)
        self.workers: List[Worker] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self._lock = Lock()
    
    def add_worker(self, worker: Worker) -> 'Orchestrator':
        """
        Add a worker to the orchestrator's worker pool.
        
        Args:
            worker: The worker to add
        
        Returns:
            Self for method chaining
        
        Example:
            >>> orchestrator = Orchestrator("orch")
            >>> orchestrator.add_worker(worker1).add_worker(worker2)
        """
        self.workers.append(worker)
        logger.info(f"Added worker {worker.worker_id} to orchestrator {self.name}")
        return self
    
    def add_task(self, task: Task) -> 'Orchestrator':
        """
        Add a task to the orchestrator's task queue.
        
        Args:
            task: The task to add
        
        Returns:
            Self for method chaining
        
        Example:
            >>> orchestrator = Orchestrator("orch")
            >>> orchestrator.add_task(task1).add_task(task2)
        """
        self.task_queue.append(task)
        logger.info(f"Added task {task.task_id} (type: {task.task_type}) to orchestrator {self.name}")
        return self
    
    def assign_task(self, task: Task) -> Optional[Worker]:
        """
        Find an appropriate worker for the given task.
        
        Selects the first worker that can handle the task based on
        the worker's can_handle method.
        
        Args:
            task: The task to assign
        
        Returns:
            A worker that can handle the task, or None if no worker is available
        
        Example:
            >>> worker = orchestrator.assign_task(task)
            >>> if worker:
            ...     result = worker.execute_task(task, context)
        """
        for worker in self.workers:
            if worker.can_handle(task):
                logger.debug(f"Assigned task {task.task_id} to worker {worker.worker_id}")
                return worker
        
        logger.warning(f"No worker available for task {task.task_id} (type: {task.task_type})")
        return None
    
    def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute all tasks using the worker pool.
        
        Manages task execution with dependency resolution and concurrent
        processing. Returns aggregated results from all completed tasks.
        
        Args:
            context: The workflow context
        
        Returns:
            Dictionary containing results from all tasks
        
        Raises:
            TaskExecutionError: If critical tasks fail
        """
        logger.info(f"Orchestrator {self.name} starting execution with {len(self.task_queue)} tasks")
        
        # Execute tasks and wait for completion
        results = self.wait_for_completion(context)
        
        # Store results in context
        context.set(f"{self.name}_results", results)
        
        logger.info(f"Orchestrator {self.name} completed: {len(self.completed_tasks)} succeeded, {len(self.failed_tasks)} failed")
        
        return results
    
    def validate(self) -> bool:
        """
        Validate the orchestrator configuration.
        
        Returns:
            True if valid (has workers and tasks), False otherwise
        """
        return (
            super().validate() and
            len(self.workers) > 0 and
            len(self.task_queue) > 0
        )

    
    def wait_for_completion(self, context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute tasks and wait for all to complete.
        
        Manages concurrent task execution with dependency resolution.
        Tasks are executed as soon as their dependencies are satisfied.
        Failed tasks are logged but don't block other independent tasks.
        
        Args:
            context: The workflow context for worker communication
        
        Returns:
            Dictionary mapping task_id to results for all completed tasks
        
        Example:
            >>> orchestrator.add_task(task1)
            >>> orchestrator.add_task(task2)
            >>> results = orchestrator.wait_for_completion(context)
            >>> print(results["task1"])
        """
        start_time = datetime.now()
        pending_tasks = self.task_queue.copy()
        running_tasks: Dict[str, Future] = {}
        task_workers: Dict[str, Worker] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            while pending_tasks or running_tasks:
                # Find tasks that are ready to execute
                ready_tasks = [
                    task for task in pending_tasks
                    if task.is_ready(self.completed_tasks)
                ]
                
                # Submit ready tasks for execution
                for task in ready_tasks:
                    if len(running_tasks) >= self.max_concurrent_tasks:
                        break
                    
                    # Assign task to a worker
                    worker = self.assign_task(task)
                    if worker:
                        task.status = TaskStatus.RUNNING
                        pending_tasks.remove(task)
                        
                        # Submit task for execution
                        future = executor.submit(self._execute_task_safely, task, worker, context)
                        running_tasks[task.task_id] = future
                        task_workers[task.task_id] = worker
                        
                        logger.info(f"Started task {task.task_id} on worker {worker.worker_id}")
                    else:
                        # No worker available for this task
                        logger.error(f"No worker can handle task {task.task_id} (type: {task.task_type})")
                        task.status = TaskStatus.FAILED
                        task.result = f"No worker available for task type: {task.task_type}"
                        self.failed_tasks[task.task_id] = task
                        pending_tasks.remove(task)
                
                # Check for completed tasks
                if running_tasks:
                    # Wait for at least one task to complete
                    done_futures = []
                    for task_id, future in list(running_tasks.items()):
                        if future.done():
                            done_futures.append((task_id, future))
                    
                    # If no tasks are done yet, wait a bit
                    if not done_futures:
                        import time
                        time.sleep(0.1)
                        continue
                    
                    # Process completed tasks
                    for task_id, future in done_futures:
                        task = next(t for t in self.task_queue if t.task_id == task_id)
                        worker = task_workers[task_id]
                        
                        try:
                            result = future.result()
                            task.status = TaskStatus.COMPLETED
                            task.result = result
                            self.completed_tasks[task_id] = task
                            
                            # Store result in context for worker communication
                            context.set(f"task_{task_id}_result", result)
                            
                            logger.info(f"Task {task_id} completed successfully")
                        except Exception as e:
                            task.status = TaskStatus.FAILED
                            task.result = str(e)
                            self.failed_tasks[task_id] = task
                            
                            logger.error(f"Task {task_id} failed: {e}")
                        
                        # Remove from running tasks
                        del running_tasks[task_id]
                        del task_workers[task_id]
                
                # If no tasks are ready and none are running, we have unresolvable dependencies
                if not ready_tasks and not running_tasks and pending_tasks:
                    logger.error(f"Deadlock detected: {len(pending_tasks)} tasks with unresolved dependencies")
                    for task in pending_tasks:
                        task.status = TaskStatus.FAILED
                        task.result = "Unresolved dependencies"
                        self.failed_tasks[task.task_id] = task
                    break
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        context.metadata["execution_time_ms"] = context.metadata.get("execution_time_ms", 0) + execution_time
        
        # Aggregate results
        results = self._aggregate_results()
        
        return results
    
    def _execute_task_safely(self, task: Task, worker: Worker, context: WorkflowContext) -> Any:
        """
        Execute a task with error handling.
        
        Wraps worker.execute_task with try-catch to ensure exceptions
        are properly captured and logged.
        
        Args:
            task: The task to execute
            worker: The worker to execute the task
            context: The workflow context
        
        Returns:
            The result of task execution
        
        Raises:
            Exception: Re-raises any exception from task execution
        """
        try:
            result = worker.execute_task(task, context)
            return result
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}", exc_info=True)
            raise
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results from all workers.
        
        Collects results from completed tasks and organizes them
        into a structured dictionary.
        
        Returns:
            Dictionary containing:
                - completed: Dict of task_id -> result for successful tasks
                - failed: Dict of task_id -> error for failed tasks
                - summary: Statistics about execution
        
        Example:
            >>> results = orchestrator._aggregate_results()
            >>> print(results["summary"]["total_tasks"])
            5
        """
        return {
            "completed": {
                task_id: task.result
                for task_id, task in self.completed_tasks.items()
            },
            "failed": {
                task_id: task.result
                for task_id, task in self.failed_tasks.items()
            },
            "summary": {
                "total_tasks": len(self.task_queue),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "success_rate": len(self.completed_tasks) / len(self.task_queue) if self.task_queue else 0
            }
        }
