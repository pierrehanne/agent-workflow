"""
Parallelization module for concurrent execution of workflow nodes.

This module provides components for executing multiple nodes concurrently
and merging their results using various strategies.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, List

from .base import Node, WorkflowContext
from .chaining import NodeExecutionError

logger = logging.getLogger(__name__)


class MergeStrategy(ABC):
    """
    Abstract base class for result merging strategies.

    MergeStrategy defines how results from parallel node executions
    should be combined into a single output. Subclasses implement
    specific merging logic such as concatenation, aggregation, or voting.

    Example:
        class CustomMerge(MergeStrategy):
            def merge(self, results: List[Any]) -> Any:
                return {"combined": results}
    """

    @abstractmethod
    def merge(self, results: List[Any]) -> Any:
        """
        Merge a list of results into a single output.

        Args:
            results: List of results from parallel node executions.
                    Each result can be of any type depending on the nodes.

        Returns:
            The merged result. The return type depends on the specific
            merge strategy implementation.

        Raises:
            ValueError: If results list is empty or invalid.
        """
        pass



class ConcatenateMerge(MergeStrategy):
    """
    Merge strategy that concatenates all results into a single string.

    This strategy converts each result to a string and joins them
    with an optional separator.

    Args:
        separator: String to use between concatenated results. Defaults to empty string.

    Example:
        strategy = ConcatenateMerge(separator="\\n")
        result = strategy.merge(["Hello", "World"])  # "Hello\\nWorld"
    """

    def __init__(self, separator: str = ""):
        """
        Initialize the concatenate merge strategy.

        Args:
            separator: String to use between concatenated results.
        """
        self.separator = separator

    def merge(self, results: List[Any]) -> str:
        """
        Concatenate all results into a single string.

        Args:
            results: List of results to concatenate.

        Returns:
            A single string with all results concatenated.

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError("Cannot merge empty results list")

        return self.separator.join(str(result) for result in results)


class ListMerge(MergeStrategy):
    """
    Merge strategy that returns all results as a list.

    This strategy simply aggregates all results into a list,
    preserving the order of execution completion.

    Example:
        strategy = ListMerge()
        result = strategy.merge([1, 2, 3])  # [1, 2, 3]
    """

    def merge(self, results: List[Any]) -> List[Any]:
        """
        Return all results as a list.

        Args:
            results: List of results to aggregate.

        Returns:
            The same list of results.

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError("Cannot merge empty results list")

        return results


class DictMerge(MergeStrategy):
    """
    Merge strategy that combines results into a dictionary.

    This strategy merges dictionary results or creates a dictionary
    with indexed keys for non-dictionary results.

    Args:
        key_prefix: Prefix for generated keys when results are not dicts.
                   Defaults to "result_".

    Example:
        strategy = DictMerge()
        result = strategy.merge([{"a": 1}, {"b": 2}])  # {"a": 1, "b": 2}

        result = strategy.merge(["x", "y"])  # {"result_0": "x", "result_1": "y"}
    """

    def __init__(self, key_prefix: str = "result_"):
        """
        Initialize the dictionary merge strategy.

        Args:
            key_prefix: Prefix for generated keys when results are not dicts.
        """
        self.key_prefix = key_prefix

    def merge(self, results: List[Any]) -> dict:
        """
        Merge results into a single dictionary.

        If all results are dictionaries, they are merged together.
        If results are not dictionaries, they are stored with indexed keys.

        Args:
            results: List of results to merge.

        Returns:
            A dictionary containing all merged results.

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError("Cannot merge empty results list")

        merged = {}

        # Check if all results are dictionaries
        if all(isinstance(result, dict) for result in results):
            for result in results:
                merged.update(result)
        else:
            # Create indexed dictionary for non-dict results
            for idx, result in enumerate(results):
                merged[f"{self.key_prefix}{idx}"] = result

        return merged


class VotingMerge(MergeStrategy):
    """
    Merge strategy that selects the most common result.

    This strategy counts occurrences of each result and returns
    the one that appears most frequently. Useful for consensus-based
    decision making across multiple parallel executions.

    Example:
        strategy = VotingMerge()
        result = strategy.merge(["A", "B", "A", "A"])  # "A"
    """

    def merge(self, results: List[Any]) -> Any:
        """
        Select the most common result through voting.

        Args:
            results: List of results to vote on.

        Returns:
            The result that appears most frequently. If there's a tie,
            returns the first result among the tied options.

        Raises:
            ValueError: If results list is empty.
        """
        if not results:
            raise ValueError("Cannot merge empty results list")

        # Count occurrences of each result
        from collections import Counter
        counter = Counter(str(result) for result in results)

        # Get the most common result
        most_common_str = counter.most_common(1)[0][0]

        # Return the original result object (not the string representation)
        for result in results:
            if str(result) == most_common_str:
                return result

        return results[0]  # Fallback



class ParallelNode(Node):
    """
    Node that executes multiple nodes concurrently and merges their results.

    ParallelNode uses ThreadPoolExecutor to run multiple nodes in parallel,
    which is ideal for I/O-bound operations like LLM API calls. Results are
    combined using a specified merge strategy.

    Args:
        name: Name of the parallel node.
        nodes: List of nodes to execute in parallel.
        merge_strategy: Strategy for combining results. Defaults to ListMerge.
        max_workers: Maximum number of concurrent threads. Defaults to None (uses default).
        timeout: Maximum time in seconds to wait for all tasks. Defaults to None (no timeout).
        description: Optional description of the node.

    Example:
        parallel = ParallelNode(
            name="extract_features",
            merge_strategy=DictMerge()
        )
        parallel.add_node(topic_extractor)
        parallel.add_node(entity_extractor)
        parallel.add_node(sentiment_analyzer)

        result = parallel.execute(context)  # Runs all three in parallel
    """

    def __init__(
        self,
        name: str,
        nodes: List[Node] = None,
        merge_strategy: MergeStrategy = None,
        max_workers: int = None,
        timeout: float = None,
        description: str = ""
    ):
        """
        Initialize the parallel node.

        Args:
            name: Name of the parallel node.
            nodes: List of nodes to execute in parallel.
            merge_strategy: Strategy for combining results.
            max_workers: Maximum number of concurrent threads.
            timeout: Maximum time in seconds to wait for all tasks.
            description: Optional description of the node.
        """
        super().__init__(name, description)
        self.nodes = nodes or []
        self.merge_strategy = merge_strategy or ListMerge()
        self.max_workers = max_workers
        self.timeout = timeout

    def add_node(self, node: Node) -> 'ParallelNode':
        """
        Add a node to be executed in parallel.

        This method supports fluent interface for easy workflow construction.

        Args:
            node: Node to add to parallel execution.

        Returns:
            Self for method chaining.

        Example:
            parallel.add_node(node1).add_node(node2).add_node(node3)
        """
        self.nodes.append(node)
        return self

    def validate(self) -> bool:
        """
        Validate the parallel node configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If no nodes are configured for parallel execution.
        """
        if not self.nodes:
            raise ValueError(f"ParallelNode '{self.name}' has no nodes to execute")

        # Validate each node
        for node in self.nodes:
            node.validate()

        return True

    def execute(self, context: WorkflowContext) -> Any:
        """
        Execute all nodes concurrently and merge their results.

        This method creates a ThreadPoolExecutor and submits all nodes
        for concurrent execution. Results are collected as they complete
        and merged using the configured merge strategy.

        Args:
            context: Workflow context containing execution state and data.

        Returns:
            Merged result from all parallel executions.

        Raises:
            NodeExecutionError: If parallel execution fails.
            TimeoutError: If execution exceeds the configured timeout.
        """
        import time

        self.validate()

        start_time = time.time()
        logger.info(f"Starting parallel execution of {len(self.nodes)} nodes in '{self.name}'")

        results = []
        errors = []

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all nodes for execution
                future_to_node = {
                    executor.submit(self._execute_node, node, context): node
                    for node in self.nodes
                }

                # Collect results as they complete
                for future in as_completed(future_to_node, timeout=self.timeout):
                    node = future_to_node[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.debug(f"Node '{node.name}' completed successfully")
                    except Exception as e:
                        error_msg = f"Node '{node.name}' failed: {str(e)}"
                        logger.error(error_msg)
                        errors.append({
                            "node": node.name,
                            "error": str(e),
                            "type": type(e).__name__
                        })

        except TimeoutError:
            error_msg = f"Parallel execution in '{self.name}' exceeded timeout of {self.timeout}s"
            logger.error(error_msg)

            # If we have some results, try to merge them
            if results:
                logger.warning(f"Returning partial results from {len(results)}/{len(self.nodes)} nodes")
                merged_result = self.merge_strategy.merge(results)

                # Store error information in context
                context.set(f"{self.name}_errors", errors)
                context.set(f"{self.name}_timeout", True)

                return merged_result
            else:
                raise NodeExecutionError(f"Parallel execution timeout with no completed results: {error_msg}")

        except Exception as e:
            error_msg = f"Parallel execution in '{self.name}' failed: {str(e)}"
            logger.error(error_msg)
            raise NodeExecutionError(error_msg) from e

        # Check if we have any results
        if not results:
            error_msg = f"All nodes in '{self.name}' failed. Errors: {errors}"
            logger.error(error_msg)
            raise NodeExecutionError(error_msg)

        # Log if some nodes failed but we have partial results
        if errors:
            logger.warning(
                f"Parallel execution in '{self.name}' completed with {len(errors)} failures. "
                f"Successfully completed: {len(results)}/{len(self.nodes)} nodes"
            )
            context.set(f"{self.name}_errors", errors)

        # Merge results
        try:
            merged_result = self.merge_strategy.merge(results)
        except Exception as e:
            error_msg = f"Failed to merge results in '{self.name}': {str(e)}"
            logger.error(error_msg)
            raise NodeExecutionError(error_msg) from e

        # Log execution metrics
        duration = time.time() - start_time
        logger.info(
            f"Parallel execution in '{self.name}' completed in {duration:.2f}s. "
            f"Successful: {len(results)}, Failed: {len(errors)}"
        )

        # Store metrics in context
        context.metadata[f"{self.name}_duration"] = duration
        context.metadata[f"{self.name}_successful_nodes"] = len(results)
        context.metadata[f"{self.name}_failed_nodes"] = len(errors)

        return merged_result

    def _execute_node(self, node: Node, context: WorkflowContext) -> Any:
        """
        Execute a single node with error handling.

        This is a helper method that wraps node execution to ensure
        proper error handling and logging.

        Args:
            node: Node to execute.
            context: Workflow context.

        Returns:
            Result from node execution.

        Raises:
            Exception: Any exception raised by the node.
        """
        try:
            logger.debug(f"Executing node '{node.name}' in parallel")
            return node.execute(context)
        except Exception as e:
            logger.error(f"Node '{node.name}' raised exception: {str(e)}")
            raise
