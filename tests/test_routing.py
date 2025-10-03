"""
Unit tests for routing module (RouterNode, RoutingCondition, etc.).
"""


import pytest

from framework.base import Node, WorkflowContext
from framework.exceptions import RoutingError
from framework.routing import (
    KeywordCondition,
    LambdaCondition,
    Route,
    RouterNode,
    ThresholdCondition,
)


class MockNode(Node):
    """Mock node for testing."""

    def __init__(self, name, return_value="result"):
        super().__init__(name)
        self.return_value = return_value
        self.executed = False

    def execute(self, context):
        self.executed = True
        return self.return_value


class TestKeywordCondition:
    """Tests for KeywordCondition class."""

    def test_keyword_match_case_insensitive(self):
        """Test keyword matching (case insensitive)."""
        condition = KeywordCondition(
            key="text",
            keywords=["urgent", "important"],
            case_sensitive=False
        )

        context = WorkflowContext()
        context.set("text", "This is an URGENT message")

        assert condition.evaluate(context) is True

    def test_keyword_match_case_sensitive(self):
        """Test keyword matching (case sensitive)."""
        condition = KeywordCondition(
            key="text",
            keywords=["urgent"],
            case_sensitive=True
        )

        context = WorkflowContext()

        # Should not match different case
        context.set("text", "This is an URGENT message")
        assert condition.evaluate(context) is False

        # Should match exact case
        context.set("text", "This is an urgent message")
        assert condition.evaluate(context) is True

    def test_keyword_no_match(self):
        """Test when no keywords match."""
        condition = KeywordCondition(
            key="text",
            keywords=["urgent", "important"]
        )

        context = WorkflowContext()
        context.set("text", "This is a normal message")

        assert condition.evaluate(context) is False

    def test_keyword_missing_key(self):
        """Test with missing context key."""
        condition = KeywordCondition(
            key="text",
            keywords=["urgent"]
        )

        context = WorkflowContext()

        assert condition.evaluate(context) is False


class TestThresholdCondition:
    """Tests for ThresholdCondition class."""

    def test_greater_than(self):
        """Test greater than operator."""
        condition = ThresholdCondition(key="score", threshold=0.5, operator=">")

        context = WorkflowContext()

        context.set("score", 0.7)
        assert condition.evaluate(context) is True

        context.set("score", 0.3)
        assert condition.evaluate(context) is False

    def test_less_than(self):
        """Test less than operator."""
        condition = ThresholdCondition(key="score", threshold=0.5, operator="<")

        context = WorkflowContext()

        context.set("score", 0.3)
        assert condition.evaluate(context) is True

        context.set("score", 0.7)
        assert condition.evaluate(context) is False

    def test_greater_equal(self):
        """Test greater than or equal operator."""
        condition = ThresholdCondition(key="score", threshold=0.5, operator=">=")

        context = WorkflowContext()

        context.set("score", 0.5)
        assert condition.evaluate(context) is True

        context.set("score", 0.6)
        assert condition.evaluate(context) is True

        context.set("score", 0.4)
        assert condition.evaluate(context) is False

    def test_invalid_operator(self):
        """Test that invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            ThresholdCondition(key="score", threshold=0.5, operator="invalid")

    def test_missing_value(self):
        """Test with missing context value."""
        condition = ThresholdCondition(key="score", threshold=0.5, operator=">")

        context = WorkflowContext()

        assert condition.evaluate(context) is False

    def test_non_numeric_value(self):
        """Test with non-numeric value raises ValueError."""
        condition = ThresholdCondition(key="score", threshold=0.5, operator=">")

        context = WorkflowContext()
        context.set("score", "not a number")

        with pytest.raises(ValueError, match="cannot be converted to numeric"):
            condition.evaluate(context)


class TestLambdaCondition:
    """Tests for LambdaCondition class."""

    def test_lambda_condition(self):
        """Test lambda condition evaluation."""
        condition = LambdaCondition(
            func=lambda ctx: ctx.get("value", 0) > 10,
            description="Check if value > 10"
        )

        context = WorkflowContext()

        context.set("value", 15)
        assert condition.evaluate(context) is True

        context.set("value", 5)
        assert condition.evaluate(context) is False

    def test_lambda_with_complex_logic(self):
        """Test lambda with complex logic."""
        condition = LambdaCondition(
            func=lambda ctx: len(ctx.get("text", "")) > 100 and "important" in ctx.get("text", "")
        )

        context = WorkflowContext()

        # Long text with keyword
        context.set("text", "a" * 101 + " important")
        assert condition.evaluate(context) is True

        # Short text with keyword
        context.set("text", "important")
        assert condition.evaluate(context) is False

        # Long text without keyword
        context.set("text", "a" * 101)
        assert condition.evaluate(context) is False


class TestRoute:
    """Tests for Route class."""

    def test_route_creation(self):
        """Test route creation."""
        condition = KeywordCondition(key="text", keywords=["test"])
        node = MockNode("test_node")

        route = Route(condition=condition, node=node, name="test_route")

        assert route.condition == condition
        assert route.node == node
        assert route.name == "test_route"

    def test_route_repr(self):
        """Test route string representation."""
        condition = KeywordCondition(key="text", keywords=["test"])
        node = MockNode("test_node")
        route = Route(condition=condition, node=node, name="test_route")

        repr_str = repr(route)
        assert "Route" in repr_str
        assert "test_route" in repr_str
        assert "test_node" in repr_str


class TestRouterNode:
    """Tests for RouterNode class."""

    def test_initialization(self):
        """Test router initialization."""
        router = RouterNode(name="test_router")

        assert router.name == "test_router"
        assert router.routes == []
        assert router.default_node is None

    def test_add_route(self):
        """Test adding routes."""
        router = RouterNode(name="test_router")
        condition = KeywordCondition(key="text", keywords=["test"])
        node = MockNode("test_node")

        result = router.add_route(condition=condition, node=node, name="test_route")

        # Should return self for chaining
        assert result is router
        assert len(router.routes) == 1
        assert router.routes[0].name == "test_route"

    def test_set_default(self):
        """Test setting default node."""
        router = RouterNode(name="test_router")
        default_node = MockNode("default")

        result = router.set_default(default_node)

        # Should return self for chaining
        assert result is router
        assert router.default_node == default_node

    def test_execute_matching_route(self):
        """Test execution with matching route."""
        router = RouterNode(name="test_router")

        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")

        condition1 = KeywordCondition(key="text", keywords=["urgent"])
        condition2 = KeywordCondition(key="text", keywords=["normal"])

        router.add_route(condition1, node1, "urgent_route")
        router.add_route(condition2, node2, "normal_route")

        context = WorkflowContext()
        context.set("text", "This is urgent")

        result = router.execute(context)

        assert result == "result1"
        assert node1.executed
        assert not node2.executed

    def test_execute_first_matching_route(self):
        """Test that first matching route is taken."""
        router = RouterNode(name="test_router")

        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")

        # Both conditions will match
        condition1 = KeywordCondition(key="text", keywords=["message"])
        condition2 = KeywordCondition(key="text", keywords=["message"])

        router.add_route(condition1, node1, "route1")
        router.add_route(condition2, node2, "route2")

        context = WorkflowContext()
        context.set("text", "This is a message")

        result = router.execute(context)

        # Should execute first route
        assert result == "result1"
        assert node1.executed
        assert not node2.executed

    def test_execute_default_route(self):
        """Test execution falls back to default."""
        router = RouterNode(name="test_router")

        node1 = MockNode("node1", return_value="result1")
        default_node = MockNode("default", return_value="default_result")

        condition1 = KeywordCondition(key="text", keywords=["urgent"])

        router.add_route(condition1, node1, "urgent_route")
        router.set_default(default_node)

        context = WorkflowContext()
        context.set("text", "This is normal")

        result = router.execute(context)

        assert result == "default_result"
        assert not node1.executed
        assert default_node.executed

    def test_execute_no_match_no_default(self):
        """Test that no match and no default raises RoutingError."""
        router = RouterNode(name="test_router")

        node1 = MockNode("node1")
        condition1 = KeywordCondition(key="text", keywords=["urgent"])

        router.add_route(condition1, node1, "urgent_route")

        context = WorkflowContext()
        context.set("text", "This is normal")

        with pytest.raises(RoutingError, match="No routing condition matched"):
            router.execute(context)

    def test_routing_decision_metadata(self):
        """Test that routing decisions are stored in metadata."""
        router = RouterNode(name="test_router")

        node1 = MockNode("node1", return_value="result1")
        condition1 = KeywordCondition(key="text", keywords=["urgent"])

        router.add_route(condition1, node1, "urgent_route")

        context = WorkflowContext()
        context.set("text", "This is urgent")

        router.execute(context)

        decisions = context.metadata.get("routing_decisions", [])
        assert len(decisions) == 1
        assert decisions[0]["router"] == "test_router"
        assert decisions[0]["route"] == "urgent_route"
        assert decisions[0]["node"] == "node1"

    def test_validate_with_routes(self):
        """Test validation with routes."""
        router = RouterNode(name="test_router")
        condition = KeywordCondition(key="text", keywords=["test"])
        node = MockNode("test_node")

        router.add_route(condition, node, "test_route")

        assert router.validate() is True

    def test_validate_with_default_only(self):
        """Test validation with only default node."""
        router = RouterNode(name="test_router")
        default_node = MockNode("default")

        router.set_default(default_node)

        assert router.validate() is True

    def test_validate_empty_router(self):
        """Test validation of empty router."""
        router = RouterNode(name="test_router")

        assert router.validate() is False
