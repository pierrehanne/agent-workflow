"""
Unit tests for parallel module (ParallelNode, MergeStrategy, etc.).
"""

import pytest
from unittest.mock import Mock
from framework.base import WorkflowContext, Node
from framework.parallel import (
    MergeStrategy,
    ConcatenateMerge,
    ListMerge,
    DictMerge,
    VotingMerge,
    ParallelNode
)
from framework.exceptions import NodeExecutionError


class MockNode(Node):
    """Mock node for testing."""
    
    def __init__(self, name, return_value="result", delay=0, should_fail=False):
        super().__init__(name)
        self.return_value = return_value
        self.delay = delay
        self.should_fail = should_fail
        self.executed = False
    
    def execute(self, context):
        import time
        if self.delay > 0:
            time.sleep(self.delay)
        
        self.executed = True
        
        if self.should_fail:
            raise Exception(f"Node {self.name} failed")
        
        return self.return_value


class TestConcatenateMerge:
    """Tests for ConcatenateMerge strategy."""
    
    def test_concatenate_with_default_separator(self):
        """Test concatenation with default separator."""
        strategy = ConcatenateMerge()
        results = ["Hello", "World", "!"]
        
        merged = strategy.merge(results)
        
        assert merged == "HelloWorld!"
    
    def test_concatenate_with_custom_separator(self):
        """Test concatenation with custom separator."""
        strategy = ConcatenateMerge(separator=" ")
        results = ["Hello", "World"]
        
        merged = strategy.merge(results)
        
        assert merged == "Hello World"
    
    def test_concatenate_with_newline(self):
        """Test concatenation with newline separator."""
        strategy = ConcatenateMerge(separator="\n")
        results = ["Line 1", "Line 2", "Line 3"]
        
        merged = strategy.merge(results)
        
        assert merged == "Line 1\nLine 2\nLine 3"
    
    def test_concatenate_empty_results(self):
        """Test that empty results raise ValueError."""
        strategy = ConcatenateMerge()
        
        with pytest.raises(ValueError, match="Cannot merge empty"):
            strategy.merge([])


class TestListMerge:
    """Tests for ListMerge strategy."""
    
    def test_list_merge(self):
        """Test list merge returns all results."""
        strategy = ListMerge()
        results = ["result1", "result2", "result3"]
        
        merged = strategy.merge(results)
        
        assert merged == results
        assert isinstance(merged, list)
    
    def test_list_merge_preserves_order(self):
        """Test that list merge preserves order."""
        strategy = ListMerge()
        results = [3, 1, 2]
        
        merged = strategy.merge(results)
        
        assert merged == [3, 1, 2]
    
    def test_list_merge_empty_results(self):
        """Test that empty results raise ValueError."""
        strategy = ListMerge()
        
        with pytest.raises(ValueError, match="Cannot merge empty"):
            strategy.merge([])


class TestDictMerge:
    """Tests for DictMerge strategy."""
    
    def test_merge_dictionaries(self):
        """Test merging dictionary results."""
        strategy = DictMerge()
        results = [
            {"key1": "value1"},
            {"key2": "value2"},
            {"key3": "value3"}
        ]
        
        merged = strategy.merge(results)
        
        assert merged == {"key1": "value1", "key2": "value2", "key3": "value3"}
    
    def test_merge_non_dict_results(self):
        """Test merging non-dictionary results."""
        strategy = DictMerge(key_prefix="result_")
        results = ["value1", "value2", "value3"]
        
        merged = strategy.merge(results)
        
        assert merged == {
            "result_0": "value1",
            "result_1": "value2",
            "result_2": "value3"
        }
    
    def test_merge_with_custom_prefix(self):
        """Test merging with custom key prefix."""
        strategy = DictMerge(key_prefix="item_")
        results = ["a", "b"]
        
        merged = strategy.merge(results)
        
        assert merged == {"item_0": "a", "item_1": "b"}
    
    def test_merge_empty_results(self):
        """Test that empty results raise ValueError."""
        strategy = DictMerge()
        
        with pytest.raises(ValueError, match="Cannot merge empty"):
            strategy.merge([])


class TestVotingMerge:
    """Tests for VotingMerge strategy."""
    
    def test_voting_selects_most_common(self):
        """Test that voting selects most common result."""
        strategy = VotingMerge()
        results = ["A", "B", "A", "A", "C"]
        
        merged = strategy.merge(results)
        
        assert merged == "A"
    
    def test_voting_with_tie(self):
        """Test voting with tie returns first."""
        strategy = VotingMerge()
        results = ["A", "B", "A", "B"]
        
        merged = strategy.merge(results)
        
        # Should return one of the tied results
        assert merged in ["A", "B"]
    
    def test_voting_single_result(self):
        """Test voting with single result."""
        strategy = VotingMerge()
        results = ["only"]
        
        merged = strategy.merge(results)
        
        assert merged == "only"
    
    def test_voting_empty_results(self):
        """Test that empty results raise ValueError."""
        strategy = VotingMerge()
        
        with pytest.raises(ValueError, match="Cannot merge empty"):
            strategy.merge([])


class TestParallelNode:
    """Tests for ParallelNode class."""
    
    def test_initialization(self):
        """Test parallel node initialization."""
        strategy = ListMerge()
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=strategy,
            max_workers=3
        )
        
        assert parallel.name == "test_parallel"
        assert parallel.merge_strategy == strategy
        assert parallel.max_workers == 3
        assert parallel.nodes == []
    
    def test_add_node(self):
        """Test adding nodes to parallel execution."""
        parallel = ParallelNode(name="test_parallel")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        result = parallel.add_node(node1)
        
        # Should return self for chaining
        assert result is parallel
        assert len(parallel.nodes) == 1
        
        parallel.add_node(node2)
        assert len(parallel.nodes) == 2
    
    def test_add_node_fluent_interface(self):
        """Test fluent interface for adding nodes."""
        parallel = ParallelNode(name="test_parallel")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        node3 = MockNode("node3")
        
        parallel.add_node(node1).add_node(node2).add_node(node3)
        
        assert len(parallel.nodes) == 3
    
    def test_execute_parallel(self):
        """Test parallel execution of nodes."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=ListMerge()
        )
        
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")
        node3 = MockNode("node3", return_value="result3")
        
        parallel.add_node(node1).add_node(node2).add_node(node3)
        
        context = WorkflowContext()
        result = parallel.execute(context)
        
        # All nodes should be executed
        assert node1.executed
        assert node2.executed
        assert node3.executed
        
        # Results should be merged
        assert set(result) == {"result1", "result2", "result3"}
    
    def test_execute_with_dict_merge(self):
        """Test parallel execution with dictionary merge."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=DictMerge(key_prefix="task_")
        )
        
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")
        
        parallel.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        result = parallel.execute(context)
        
        assert isinstance(result, dict)
        assert "task_0" in result
        assert "task_1" in result
    
    def test_execute_empty_nodes(self):
        """Test that executing with no nodes raises ValueError."""
        parallel = ParallelNode(name="test_parallel")
        context = WorkflowContext()
        
        with pytest.raises(ValueError, match="has no nodes"):
            parallel.execute(context)
    
    def test_execute_with_partial_failure(self):
        """Test execution with some nodes failing."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=ListMerge()
        )
        
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", should_fail=True)
        node3 = MockNode("node3", return_value="result3")
        
        parallel.add_node(node1).add_node(node2).add_node(node3)
        
        context = WorkflowContext()
        result = parallel.execute(context)
        
        # Should have results from successful nodes
        assert "result1" in result
        assert "result3" in result
        
        # Should have error information in context
        errors = context.get(f"{parallel.name}_errors")
        assert errors is not None
        assert len(errors) == 1
    
    def test_execute_all_failures(self):
        """Test that all failures raises NodeExecutionError."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=ListMerge()
        )
        
        node1 = MockNode("node1", should_fail=True)
        node2 = MockNode("node2", should_fail=True)
        
        parallel.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        
        with pytest.raises(NodeExecutionError, match="All nodes.*failed"):
            parallel.execute(context)
    
    def test_execute_with_timeout(self):
        """Test execution with timeout."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=ListMerge(),
            timeout=0.1  # Very short timeout
        )
        
        # Create nodes with delays
        node1 = MockNode("node1", return_value="result1", delay=0.05)
        node2 = MockNode("node2", return_value="result2", delay=0.2)  # Will timeout
        
        parallel.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        
        # Should either complete with partial results or raise error
        try:
            result = parallel.execute(context)
            # If it completes, should have timeout flag
            assert context.get(f"{parallel.name}_timeout") is True
        except NodeExecutionError:
            # Timeout with no results is also acceptable
            pass
    
    def test_metadata_update(self):
        """Test that metadata is updated during execution."""
        parallel = ParallelNode(
            name="test_parallel",
            merge_strategy=ListMerge()
        )
        
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")
        
        parallel.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        parallel.execute(context)
        
        # Should have execution metrics
        assert f"{parallel.name}_duration" in context.metadata
        assert f"{parallel.name}_successful_nodes" in context.metadata
        assert context.metadata[f"{parallel.name}_successful_nodes"] == 2
    
    def test_validate(self):
        """Test node validation."""
        parallel = ParallelNode(name="test_parallel")
        node1 = MockNode("node1")
        
        # Should raise ValueError when validating empty parallel node
        with pytest.raises(ValueError, match="has no nodes"):
            parallel.validate()
        
        # Should pass validation with nodes
        parallel.add_node(node1)
        assert parallel.validate() is True
