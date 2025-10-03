"""
Unit tests for chaining module (ChainWorkflow).
"""

import pytest
from unittest.mock import Mock
from framework.base import WorkflowContext, Node
from framework.chaining import ChainWorkflow
from framework.exceptions import NodeExecutionError


class MockNode(Node):
    """Mock node for testing."""
    
    def __init__(self, name, return_value="result", should_fail=False):
        super().__init__(name)
        self.return_value = return_value
        self.should_fail = should_fail
        self.executed = False
    
    def execute(self, context):
        self.executed = True
        if self.should_fail:
            raise Exception(f"Node {self.name} failed")
        return self.return_value


class TestChainWorkflow:
    """Tests for ChainWorkflow class."""
    
    def test_initialization(self):
        """Test chain initialization."""
        chain = ChainWorkflow(name="test_chain")
        
        assert chain.name == "test_chain"
        assert chain.nodes == []
        assert chain.pass_through_context is False
    
    def test_add_node(self):
        """Test adding nodes to chain."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        result = chain.add_node(node1)
        
        # Should return self for chaining
        assert result is chain
        assert len(chain.nodes) == 1
        
        chain.add_node(node2)
        assert len(chain.nodes) == 2
    
    def test_add_node_fluent_interface(self):
        """Test fluent interface for adding nodes."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        node3 = MockNode("node3")
        
        chain.add_node(node1).add_node(node2).add_node(node3)
        
        assert len(chain.nodes) == 3
    
    def test_add_node_type_check(self):
        """Test that adding non-Node raises TypeError."""
        chain = ChainWorkflow(name="test_chain")
        
        with pytest.raises(TypeError):
            chain.add_node("not a node")
    
    def test_execute_sequential(self):
        """Test sequential execution of nodes."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")
        node3 = MockNode("node3", return_value="result3")
        
        chain.add_node(node1).add_node(node2).add_node(node3)
        
        context = WorkflowContext()
        result = chain.execute(context)
        
        # All nodes should be executed
        assert node1.executed
        assert node2.executed
        assert node3.executed
        
        # Should return last node's result
        assert result == "result3"
    
    def test_execute_empty_chain(self):
        """Test that executing empty chain raises ValueError."""
        chain = ChainWorkflow(name="test_chain")
        context = WorkflowContext()
        
        with pytest.raises(ValueError, match="has no nodes"):
            chain.execute(context)
    
    def test_execute_with_pass_through(self):
        """Test execution with pass_through_context enabled."""
        chain = ChainWorkflow(name="test_chain", pass_through_context=True)
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", return_value="result2")
        
        chain.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        result = chain.execute(context)
        
        # Intermediate results should be stored
        assert context.get("node1_output") == "result1"
        assert context.get("node2_output") == "result2"
        
        # Final result should still be returned
        assert result == "result2"
    
    def test_execute_error_handling(self):
        """Test error handling during execution."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1", return_value="result1")
        node2 = MockNode("node2", should_fail=True)
        node3 = MockNode("node3", return_value="result3")
        
        chain.add_node(node1).add_node(node2).add_node(node3)
        
        context = WorkflowContext()
        
        with pytest.raises(NodeExecutionError) as exc_info:
            chain.execute(context)
        
        # Should indicate which node failed
        assert "node2" in str(exc_info.value)
        
        # First node should have executed
        assert node1.executed
        
        # Third node should not have executed
        assert not node3.executed
    
    def test_validate_valid_chain(self):
        """Test validation of valid chain."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        chain.add_node(node1).add_node(node2)
        
        assert chain.validate() is True
    
    def test_validate_empty_chain(self):
        """Test validation of empty chain."""
        chain = ChainWorkflow(name="test_chain")
        
        assert chain.validate() is False
    
    def test_validate_invalid_node(self):
        """Test validation with invalid node."""
        chain = ChainWorkflow(name="test_chain")
        
        # Create a node with empty name (invalid)
        invalid_node = MockNode("")
        chain.add_node(invalid_node)
        
        assert chain.validate() is False
    
    def test_history_tracking(self):
        """Test that execution history is tracked."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        chain.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        chain.execute(context)
        
        history = context.get_history()
        
        # Should have entries for chain steps
        chain_steps = [h for h in history if h.get("operation") == "chain_step"]
        assert len(chain_steps) == 2
    
    def test_metadata_update(self):
        """Test that metadata is updated during execution."""
        chain = ChainWorkflow(name="test_chain")
        node1 = MockNode("node1")
        node2 = MockNode("node2")
        
        chain.add_node(node1).add_node(node2)
        
        context = WorkflowContext()
        chain.execute(context)
        
        # Execution time should be recorded
        assert "execution_time_ms" in context.metadata
        assert context.metadata["execution_time_ms"] > 0
    
    def test_repr(self):
        """Test string representation."""
        chain = ChainWorkflow(name="test_chain", pass_through_context=True)
        node1 = MockNode("node1")
        chain.add_node(node1)
        
        repr_str = repr(chain)
        assert "ChainWorkflow" in repr_str
        assert "test_chain" in repr_str
        assert "nodes=1" in repr_str
        assert "pass_through=True" in repr_str
