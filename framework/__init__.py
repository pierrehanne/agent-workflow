"""
LLM Workflow Framework - A modular framework for building LLM-based pipelines.

This framework provides abstractions and implementations for various workflow
patterns including prompt chaining, routing, parallelization, orchestrator-workers,
and evaluator-optimizer loops.
"""

from framework.base import (
    LLMProvider,
    Node,
    PromptNode,
    WorkflowContext,
)
from framework.chaining import (
    ChainWorkflow,
)
from framework.evaluator import (
    CompositeEvaluator,
    Evaluator,
    EvaluatorNode,
    KeywordEvaluator,
    LengthEvaluator,
    LLMEvaluator,
    OptimizationStrategy,
    PromptRefinementStrategy,
    RetryStrategy,
    TemperatureAdjustmentStrategy,
)
from framework.exceptions import (
    EvaluationError,
    LLMProviderError,
    NodeExecutionError,
    RoutingError,
    TaskExecutionError,
    WorkflowException,
)
from framework.logging_config import (
    configure_logging,
    get_log_level_from_env,
    get_logger,
)
from framework.orchestrator import (
    LLMWorker,
    Orchestrator,
    Task,
    TaskStatus,
    Worker,
)
from framework.parallel import (
    ConcatenateMerge,
    DictMerge,
    ListMerge,
    MergeStrategy,
    ParallelNode,
    VotingMerge,
)
from framework.routing import (
    KeywordCondition,
    LambdaCondition,
    Route,
    RouterNode,
    RoutingCondition,
    ThresholdCondition,
)

__all__ = [
    "WorkflowContext",
    "Node",
    "LLMProvider",
    "PromptNode",
    "ChainWorkflow",
    "RoutingCondition",
    "KeywordCondition",
    "ThresholdCondition",
    "LambdaCondition",
    "Route",
    "RouterNode",
    "MergeStrategy",
    "ConcatenateMerge",
    "ListMerge",
    "DictMerge",
    "VotingMerge",
    "ParallelNode",
    "TaskStatus",
    "Task",
    "Worker",
    "LLMWorker",
    "Orchestrator",
    "Evaluator",
    "LengthEvaluator",
    "KeywordEvaluator",
    "LLMEvaluator",
    "CompositeEvaluator",
    "OptimizationStrategy",
    "RetryStrategy",
    "PromptRefinementStrategy",
    "TemperatureAdjustmentStrategy",
    "EvaluatorNode",
    "WorkflowException",
    "NodeExecutionError",
    "RoutingError",
    "EvaluationError",
    "LLMProviderError",
    "TaskExecutionError",
    "configure_logging",
    "get_logger",
    "get_log_level_from_env",
]

__version__ = "0.1.0"
