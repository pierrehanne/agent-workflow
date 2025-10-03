"""
Prompt chaining module for the LLM Workflow Framework.

This module provides the ChainWorkflow class for sequential execution of nodes,
where the output of each node is passed as input to the next node. This enables
building complex multi-step LLM pipelines with clear data flow.
"""

import logging
from typing import Any, List, Optional
from datetime import datetime

from framework.base import Node, WorkflowContext
from framework.exceptions import NodeExecutionError


# Configure module logger
logger = logging.getLogger(__name__)


class ChainWorkflow(Node):
    """
    Sequential execution of nodes with output-to-input passing.
    
    ChainWorkflow executes a series of nodes in order, automatically passing
    the output of each node as input to the next. This pattern is ideal for
    multi-step processing where each step builds on the previous result.
    
    The workflow supports two modes:
    - Standard mode: Only the final output is returned
    - Pass-through mode: All intermediate results are accumulated in context
    
    Attributes:
        nodes: List of nodes to execute sequentially
        pass_through_context: If True, accumulate all results in context
    
    Example:
        >>> chain = ChainWorkflow(name="document_processor")
        >>> chain.add_node(extract_node).add_node(summarize_node).add_node(format_node)
        >>> context = WorkflowContext()
        >>> context.set("input", "Long document text...")
        >>> result = chain.execute(context)
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        pass_through_context: bool = False
    ) -> None:
        """
        Initialize a ChainWorkflow.
        
        Args:
            name: Unique identifier for the chain
            description: Human-readable description of the chain's purpose
            pass_through_context: If True, store all intermediate results in context
        """
        super().__init__(name, description)
        self.nodes: List[Node] = []
        self.pass_through_context = pass_through_context
    
    def execute(self, context: WorkflowContext) -> Any:
        """
        Execute all nodes in the chain sequentially.
        
        Each node's output becomes available to subsequent nodes. If pass_through_context
        is enabled, intermediate results are stored with keys like "{node_name}_output".
        
        Args:
            context: The workflow context containing input data
        
        Returns:
            The output of the final node in the chain
        
        Raises:
            NodeExecutionError: If any node fails during execution
            ValueError: If the chain has no nodes
        
        Example:
            >>> chain = ChainWorkflow(name="pipeline", pass_through_context=True)
            >>> chain.add_node(node1).add_node(node2)
            >>> context = WorkflowContext()
            >>> context.set("input", "data")
            >>> result = chain.execute(context)
        """
        if not self.nodes:
            raise ValueError(f"ChainWorkflow '{self.name}' has no nodes to execute")
        
        logger.info(f"Starting chain execution: {self.name} with {len(self.nodes)} nodes")
        start_time = datetime.now()
        
        result = None
        intermediate_results = []
        
        try:
            for i, node in enumerate(self.nodes):
                logger.debug(f"Executing node {i+1}/{len(self.nodes)}: {node.name}")
                node_start_time = datetime.now()
                
                # Execute the node
                result = node.execute(context)
                
                # Calculate execution time
                node_duration = (datetime.now() - node_start_time).total_seconds() * 1000
                
                # Store intermediate result if pass_through_context is enabled
                if self.pass_through_context:
                    output_key = f"{node.name}_output"
                    context.set(output_key, result)
                    intermediate_results.append({
                        "node": node.name,
                        "output_key": output_key,
                        "result": result
                    })
                
                # Log execution details
                logger.debug(
                    f"Node {node.name} completed in {node_duration:.2f}ms, "
                    f"result type: {type(result).__name__}"
                )
                
                # Add to context history
                context.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "operation": "chain_step",
                    "node": node.name,
                    "step": i + 1,
                    "duration_ms": node_duration
                })
            
            # Calculate total execution time
            total_duration = (datetime.now() - start_time).total_seconds() * 1000
            context.metadata["execution_time_ms"] = (
                context.metadata.get("execution_time_ms", 0) + total_duration
            )
            
            logger.info(
                f"Chain {self.name} completed successfully in {total_duration:.2f}ms"
            )
            
            # Store intermediate results metadata if pass_through enabled
            if self.pass_through_context and intermediate_results:
                context.set(f"{self.name}_intermediate_results", intermediate_results)
            
            return result
            
        except Exception as e:
            # Log the error with context
            logger.error(
                f"Error in chain {self.name} at node {node.name}: {str(e)}",
                exc_info=True
            )
            
            # Add error to context history
            context.history.append({
                "timestamp": datetime.now().isoformat(),
                "operation": "chain_error",
                "node": node.name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Re-raise with additional context
            raise NodeExecutionError(
                f"Chain '{self.name}' failed at node '{node.name}': {str(e)}",
                node_name=node.name,
                original_error=e
            ) from e

    
    def add_node(self, node: Node) -> 'ChainWorkflow':
        """
        Add a node to the chain (fluent interface).
        
        This method returns self to enable method chaining for easy workflow
        construction.
        
        Args:
            node: The node to add to the chain
        
        Returns:
            Self, for method chaining
        
        Example:
            >>> chain = ChainWorkflow(name="pipeline")
            >>> chain.add_node(node1).add_node(node2).add_node(node3)
            >>> # All three nodes are now in the chain
        """
        if not isinstance(node, Node):
            raise TypeError(f"Expected Node instance, got {type(node).__name__}")
        
        self.nodes.append(node)
        logger.debug(f"Added node '{node.name}' to chain '{self.name}'")
        return self
    
    def validate(self) -> bool:
        """
        Validate the chain configuration.
        
        Checks that:
        - The chain has a valid name
        - At least one node is present
        - All nodes are valid Node instances
        - All nodes pass their own validation
        
        Returns:
            True if the chain is valid, False otherwise
        
        Example:
            >>> chain = ChainWorkflow(name="test")
            >>> chain.add_node(valid_node)
            >>> assert chain.validate() == True
        """
        # Check base validation
        if not super().validate():
            logger.warning(f"Chain '{self.name}' failed base validation")
            return False
        
        # Check that we have at least one node
        if not self.nodes:
            logger.warning(f"Chain '{self.name}' has no nodes")
            return False
        
        # Validate each node
        for i, node in enumerate(self.nodes):
            if not isinstance(node, Node):
                logger.warning(
                    f"Chain '{self.name}' node at index {i} is not a Node instance"
                )
                return False
            
            if not node.validate():
                logger.warning(
                    f"Chain '{self.name}' node '{node.name}' failed validation"
                )
                return False
        
        logger.debug(f"Chain '{self.name}' validation passed with {len(self.nodes)} nodes")
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the chain."""
        return (
            f"ChainWorkflow(name='{self.name}', "
            f"nodes={len(self.nodes)}, "
            f"pass_through={self.pass_through_context})"
        )
