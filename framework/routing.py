"""
Routing module for the LLM Workflow Framework.

This module provides conditional routing capabilities, allowing workflows to
dynamically direct execution to different nodes based on runtime conditions.
Supports keyword-based routing, threshold comparisons, and custom logic.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Callable
import logging

from framework.base import Node, WorkflowContext
from framework.exceptions import RoutingError


# Configure logging
logger = logging.getLogger(__name__)


class RoutingCondition(ABC):
    """
    Abstract base class for routing conditions.
    
    A RoutingCondition evaluates the workflow context and returns a boolean
    decision that determines whether a particular route should be taken.
    
    Example:
        >>> class CustomCondition(RoutingCondition):
        ...     def evaluate(self, context: WorkflowContext) -> bool:
        ...         return context.get("score", 0) > 0.5
        >>> condition = CustomCondition()
        >>> context = WorkflowContext()
        >>> context.set("score", 0.8)
        >>> print(condition.evaluate(context))
        True
    """
    
    @abstractmethod
    def evaluate(self, context: WorkflowContext) -> bool:
        """
        Evaluate the condition against the workflow context.
        
        Args:
            context: The workflow context containing data to evaluate
        
        Returns:
            True if the condition is met, False otherwise
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass


class KeywordCondition(RoutingCondition):
    """
    Routes based on keyword presence in context data.
    
    Evaluates to True if any of the specified keywords are found in the
    value associated with the given context key.
    
    Attributes:
        key: The context key to check for keywords
        keywords: List of keywords to search for
        case_sensitive: Whether the keyword search is case-sensitive
    
    Example:
        >>> condition = KeywordCondition(
        ...     key="text",
        ...     keywords=["urgent", "important"],
        ...     case_sensitive=False
        ... )
        >>> context = WorkflowContext()
        >>> context.set("text", "This is an URGENT message")
        >>> print(condition.evaluate(context))
        True
    """
    
    def __init__(
        self,
        key: str,
        keywords: List[str],
        case_sensitive: bool = False
    ) -> None:
        """
        Initialize a KeywordCondition.
        
        Args:
            key: The context key to check
            keywords: List of keywords to search for
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.key = key
        self.keywords = keywords
        self.case_sensitive = case_sensitive
    
    def evaluate(self, context: WorkflowContext) -> bool:
        """
        Evaluate if any keyword is present in the context value.
        
        Args:
            context: The workflow context
        
        Returns:
            True if any keyword is found, False otherwise
        """
        value = context.get(self.key, "")
        
        # Convert to string if not already
        value_str = str(value)
        
        # Perform case-insensitive search if needed
        if not self.case_sensitive:
            value_str = value_str.lower()
            keywords = [kw.lower() for kw in self.keywords]
        else:
            keywords = self.keywords
        
        # Check if any keyword is present
        return any(keyword in value_str for keyword in keywords)


class ThresholdCondition(RoutingCondition):
    """
    Routes based on numeric threshold comparison.
    
    Evaluates to True if the numeric value in the context meets the
    threshold condition (greater than, less than, equal to, etc.).
    
    Attributes:
        key: The context key containing the numeric value
        threshold: The threshold value to compare against
        operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
    
    Example:
        >>> condition = ThresholdCondition(
        ...     key="confidence",
        ...     threshold=0.8,
        ...     operator=">="
        ... )
        >>> context = WorkflowContext()
        >>> context.set("confidence", 0.9)
        >>> print(condition.evaluate(context))
        True
    """
    
    def __init__(
        self,
        key: str,
        threshold: float,
        operator: str = ">"
    ) -> None:
        """
        Initialize a ThresholdCondition.
        
        Args:
            key: The context key containing the numeric value
            threshold: The threshold value to compare against
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        
        Raises:
            ValueError: If operator is not valid
        """
        valid_operators = {">", "<", ">=", "<=", "==", "!="}
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator: {operator}. Must be one of {valid_operators}")
        
        self.key = key
        self.threshold = threshold
        self.operator = operator
    
    def evaluate(self, context: WorkflowContext) -> bool:
        """
        Evaluate if the numeric value meets the threshold condition.
        
        Args:
            context: The workflow context
        
        Returns:
            True if the condition is met, False otherwise
        
        Raises:
            ValueError: If the context value cannot be converted to float
        """
        value = context.get(self.key)
        
        if value is None:
            return False
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Value for key '{self.key}' cannot be converted to numeric: {value}")
        
        # Perform comparison based on operator
        if self.operator == ">":
            return numeric_value > self.threshold
        elif self.operator == "<":
            return numeric_value < self.threshold
        elif self.operator == ">=":
            return numeric_value >= self.threshold
        elif self.operator == "<=":
            return numeric_value <= self.threshold
        elif self.operator == "==":
            return numeric_value == self.threshold
        elif self.operator == "!=":
            return numeric_value != self.threshold
        
        return False


class LambdaCondition(RoutingCondition):
    """
    Routes based on custom lambda function logic.
    
    Provides maximum flexibility by allowing arbitrary Python functions
    to evaluate routing conditions.
    
    Attributes:
        func: A callable that takes WorkflowContext and returns bool
        description: Optional description of what the condition checks
    
    Example:
        >>> condition = LambdaCondition(
        ...     func=lambda ctx: len(ctx.get("text", "")) > 100,
        ...     description="Route if text is longer than 100 characters"
        ... )
        >>> context = WorkflowContext()
        >>> context.set("text", "Short text")
        >>> print(condition.evaluate(context))
        False
    """
    
    def __init__(
        self,
        func: Callable[[WorkflowContext], bool],
        description: str = ""
    ) -> None:
        """
        Initialize a LambdaCondition.
        
        Args:
            func: A callable that takes WorkflowContext and returns bool
            description: Optional description of the condition
        """
        self.func = func
        self.description = description
    
    def evaluate(self, context: WorkflowContext) -> bool:
        """
        Evaluate the custom function against the context.
        
        Args:
            context: The workflow context
        
        Returns:
            The boolean result of the function
        
        Raises:
            Exception: If the function raises an exception during evaluation
        """
        try:
            return bool(self.func(context))
        except Exception as e:
            logger.error(f"Error evaluating lambda condition: {e}")
            raise



class Route:
    """
    Represents a conditional route to a node.
    
    A Route pairs a routing condition with a target node, creating a
    conditional branch in the workflow.
    
    Attributes:
        condition: The condition that must be met for this route
        node: The node to execute if the condition is met
        name: Human-readable name for the route
    
    Example:
        >>> condition = KeywordCondition(key="type", keywords=["urgent"])
        >>> node = PromptNode(name="urgent_handler", ...)
        >>> route = Route(
        ...     condition=condition,
        ...     node=node,
        ...     name="urgent_route"
        ... )
    """
    
    def __init__(
        self,
        condition: RoutingCondition,
        node: Node,
        name: str
    ) -> None:
        """
        Initialize a Route.
        
        Args:
            condition: The routing condition
            node: The target node to execute
            name: Name for the route
        """
        self.condition = condition
        self.node = node
        self.name = name
    
    def __repr__(self) -> str:
        """Return string representation of the route."""
        return f"Route(name='{self.name}', node={self.node.name})"


class RouterNode(Node):
    """
    Routes execution to different nodes based on conditions.
    
    RouterNode evaluates routing conditions in order and executes the first
    node whose condition is met. If no conditions match, it executes the
    default node or raises a RoutingError.
    
    Attributes:
        routes: List of Route objects defining conditional branches
        default_node: Optional fallback node if no conditions match
    
    Example:
        >>> router = RouterNode(name="classifier")
        >>> router.add_route(
        ...     condition=KeywordCondition(key="text", keywords=["urgent"]),
        ...     node=urgent_handler,
        ...     name="urgent"
        ... )
        >>> router.add_route(
        ...     condition=KeywordCondition(key="text", keywords=["normal"]),
        ...     node=normal_handler,
        ...     name="normal"
        ... )
        >>> router.set_default(default_handler)
        >>> context = WorkflowContext()
        >>> context.set("text", "This is urgent!")
        >>> result = router.execute(context)
    """
    
    def __init__(
        self,
        name: str,
        description: str = ""
    ) -> None:
        """
        Initialize a RouterNode.
        
        Args:
            name: Unique identifier for the router
            description: Human-readable description
        """
        super().__init__(name, description)
        self.routes: List[Route] = []
        self.default_node: Optional[Node] = None
    
    def add_route(
        self,
        condition: RoutingCondition,
        node: Node,
        name: str
    ) -> "RouterNode":
        """
        Add a conditional route to the router.
        
        Routes are evaluated in the order they are added. The first route
        whose condition evaluates to True will be taken.
        
        Args:
            condition: The routing condition
            node: The target node to execute
            name: Name for the route
        
        Returns:
            Self for method chaining
        
        Example:
            >>> router = RouterNode(name="router")
            >>> router.add_route(
            ...     condition=ThresholdCondition(key="score", threshold=0.8, operator=">="),
            ...     node=high_score_node,
            ...     name="high_score"
            ... ).add_route(
            ...     condition=ThresholdCondition(key="score", threshold=0.5, operator=">="),
            ...     node=medium_score_node,
            ...     name="medium_score"
            ... )
        """
        route = Route(condition=condition, node=node, name=name)
        self.routes.append(route)
        logger.debug(f"Added route '{name}' to router '{self.name}'")
        return self
    
    def set_default(self, node: Node) -> "RouterNode":
        """
        Set the default fallback node.
        
        The default node is executed if no routing conditions match.
        
        Args:
            node: The default node to execute
        
        Returns:
            Self for method chaining
        
        Example:
            >>> router = RouterNode(name="router")
            >>> router.set_default(fallback_node)
        """
        self.default_node = node
        logger.debug(f"Set default node '{node.name}' for router '{self.name}'")
        return self
    
    def execute(self, context: WorkflowContext) -> Any:
        """
        Execute the router by evaluating conditions and routing to a node.
        
        Evaluates each route's condition in order. The first route whose
        condition is True will have its node executed. If no conditions
        match and a default node is set, the default is executed. Otherwise,
        a RoutingError is raised.
        
        Args:
            context: The workflow context
        
        Returns:
            The result from the executed node
        
        Raises:
            RoutingError: If no route matches and no default is set
        """
        logger.info(f"Router '{self.name}' evaluating {len(self.routes)} routes")
        
        # Evaluate each route in order
        for route in self.routes:
            try:
                logger.debug(f"Evaluating route '{route.name}'")
                
                if route.condition.evaluate(context):
                    logger.info(f"Router '{self.name}' matched route '{route.name}', executing node '{route.node.name}'")
                    result = route.node.execute(context)
                    
                    # Store routing decision in context metadata
                    if "routing_decisions" not in context.metadata:
                        context.metadata["routing_decisions"] = []
                    context.metadata["routing_decisions"].append({
                        "router": self.name,
                        "route": route.name,
                        "node": route.node.name
                    })
                    
                    return result
                    
            except Exception as e:
                logger.error(f"Error evaluating condition for route '{route.name}': {e}")
                # Re-raise the exception to be handled by caller
                raise RoutingError(f"Error evaluating route '{route.name}': {e}") from e
        
        # No route matched, try default
        if self.default_node:
            logger.info(f"Router '{self.name}' no routes matched, executing default node '{self.default_node.name}'")
            result = self.default_node.execute(context)
            
            # Store routing decision
            if "routing_decisions" not in context.metadata:
                context.metadata["routing_decisions"] = []
            context.metadata["routing_decisions"].append({
                "router": self.name,
                "route": "default",
                "node": self.default_node.name
            })
            
            return result
        
        # No route matched and no default
        logger.error(f"Router '{self.name}' no routes matched and no default node set")
        raise RoutingError(
            f"No routing condition matched for router '{self.name}' and no default node is set"
        )
    
    def validate(self) -> bool:
        """
        Validate the router configuration.
        
        Returns:
            True if valid (has at least one route or a default), False otherwise
        """
        return super().validate() and (len(self.routes) > 0 or self.default_node is not None)
