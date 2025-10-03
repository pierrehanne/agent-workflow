"""
LLM Workflow Framework - A modular framework for building LLM-based pipelines.

This framework provides abstractions and implementations for various workflow
patterns including prompt chaining, routing, parallelization, orchestrator-workers,
and evaluator-optimizer loops.
"""

from framework.base import (
    WorkflowContext,
    Node,
    LLMProvider,
    PromptNode,
)
from framework.chaining import (
    ChainWorkflow,
)
from framework.routing import (
    RoutingCondition,
    KeywordCondition,
    ThresholdCondition,
    LambdaCondition,
    Route,
    RouterNode,
)
from framework.parallel import (
    MergeStrategy,
    ConcatenateMerge,
    ListMerge,
    DictMerge,
    VotingMerge,
    ParallelNode,
)
from framework.orchestrator import (
    TaskStatus,
    Task,
    Worker,
    LLMWorker,
    Orchestrator,
)
from framework.evaluator import (
    Evaluator,
    LengthEvaluator,
    KeywordEvaluator,
    LLMEvaluator,
    CompositeEvaluator,
    OptimizationStrategy,
    RetryStrategy,
    PromptRefinementStrategy,
    TemperatureAdjustmentStrategy,
    EvaluatorNode,
)
from framework.exceptions import (
    WorkflowException,
    NodeExecutionError,
    RoutingError,
    EvaluationError,
    LLMProviderError,
    TaskExecutionError,
)
from framework.logging_config import (
    configure_logging,
    get_logger,
    get_log_level_from_env,
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
