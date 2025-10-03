"""
Unit tests for evaluator module (Evaluator, OptimizationStrategy, EvaluatorNode, etc.).
"""

from unittest.mock import Mock

import pytest

from framework.base import LLMProvider, Node, WorkflowContext
from framework.evaluator import (
    CompositeEvaluator,
    Evaluator,
    EvaluatorNode,
    KeywordEvaluator,
    LengthEvaluator,
    LLMEvaluator,
    PromptRefinementStrategy,
    RetryStrategy,
    TemperatureAdjustmentStrategy,
)
from framework.exceptions import EvaluationError


class MockNode(Node):
    """Mock node for testing."""

    def __init__(self, name, return_value="result", call_count=0):
        super().__init__(name)
        self.return_value = return_value
        self.call_count = call_count
        self.execution_count = 0

    def execute(self, context):
        self.execution_count += 1
        return self.return_value


class TestLengthEvaluator:
    """Tests for LengthEvaluator class."""

    def test_evaluate_within_range(self):
        """Test evaluation of text within acceptable range."""
        evaluator = LengthEvaluator(min_length=50, max_length=200)
        context = WorkflowContext()

        result = "a" * 100  # Within range
        score = evaluator.evaluate(result, context)

        assert score >= 0.7  # Should be good score

    def test_evaluate_below_minimum(self):
        """Test evaluation of text below minimum length."""
        evaluator = LengthEvaluator(min_length=50, max_length=200)
        context = WorkflowContext()

        result = "a" * 20  # Below minimum
        score = evaluator.evaluate(result, context)

        assert score < 0.5  # Should be low score

    def test_evaluate_above_maximum(self):
        """Test evaluation of text above maximum length."""
        evaluator = LengthEvaluator(min_length=50, max_length=200)
        context = WorkflowContext()

        result = "a" * 300  # Above maximum
        score = evaluator.evaluate(result, context)

        assert score < 0.8  # Should be penalized

    def test_evaluate_with_target_length(self):
        """Test evaluation with target length."""
        evaluator = LengthEvaluator(min_length=50, max_length=200, target_length=100)
        context = WorkflowContext()

        # Exactly at target
        result = "a" * 100
        score = evaluator.evaluate(result, context)
        assert score >= 0.9

        # Close to target
        result = "a" * 110
        score = evaluator.evaluate(result, context)
        assert score >= 0.7

    def test_get_feedback_too_short(self):
        """Test feedback for text that's too short."""
        evaluator = LengthEvaluator(min_length=50, max_length=200)

        result = "short"
        score = evaluator.evaluate(result, WorkflowContext())
        feedback = evaluator.get_feedback(result, score)

        assert "too short" in feedback.lower()
        assert "50" in feedback

    def test_get_feedback_too_long(self):
        """Test feedback for text that's too long."""
        evaluator = LengthEvaluator(min_length=50, max_length=200)

        result = "a" * 300
        score = evaluator.evaluate(result, WorkflowContext())
        feedback = evaluator.get_feedback(result, score)

        assert "too long" in feedback.lower()
        assert "200" in feedback


class TestKeywordEvaluator:
    """Tests for KeywordEvaluator class."""

    def test_evaluate_all_required_present(self):
        """Test evaluation when all required keywords are present."""
        evaluator = KeywordEvaluator(required_keywords=["python", "code"])
        context = WorkflowContext()

        result = "This is Python code for testing"
        score = evaluator.evaluate(result, context)

        assert score == 1.0

    def test_evaluate_some_required_missing(self):
        """Test evaluation when some required keywords are missing."""
        evaluator = KeywordEvaluator(required_keywords=["python", "code", "test"])
        context = WorkflowContext()

        result = "This is Python code"  # Missing "test"
        score = evaluator.evaluate(result, context)

        assert score < 1.0
        assert score > 0.0

    def test_evaluate_forbidden_present(self):
        """Test evaluation when forbidden keywords are present."""
        evaluator = KeywordEvaluator(forbidden_keywords=["error", "failed"])
        context = WorkflowContext()

        result = "The operation failed with an error"
        score = evaluator.evaluate(result, context)

        assert score == 0.0

    def test_evaluate_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        evaluator = KeywordEvaluator(
            required_keywords=["Python"],
            case_sensitive=False
        )
        context = WorkflowContext()

        result = "This is python code"
        score = evaluator.evaluate(result, context)

        assert score == 1.0

    def test_get_feedback_missing_keywords(self):
        """Test feedback for missing required keywords."""
        evaluator = KeywordEvaluator(required_keywords=["python", "code", "test"])

        result = "This is Python"
        score = evaluator.evaluate(result, WorkflowContext())
        feedback = evaluator.get_feedback(result, score)

        assert "missing" in feedback.lower()
        assert "code" in feedback.lower()
        assert "test" in feedback.lower()


class TestLLMEvaluator:
    """Tests for LLMEvaluator class."""

    def test_evaluate_with_llm(self):
        """Test evaluation using LLM."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "8/10"

        evaluator = LLMEvaluator(llm_provider=mock_provider)
        context = WorkflowContext()

        result = "Test output"
        score = evaluator.evaluate(result, context)

        assert score == 0.8  # 8/10 = 0.8
        mock_provider.generate.assert_called_once()

    def test_evaluate_parse_different_formats(self):
        """Test parsing different score formats."""
        mock_provider = Mock(spec=LLMProvider)
        evaluator = LLMEvaluator(llm_provider=mock_provider)
        context = WorkflowContext()

        # Test "X/10" format
        mock_provider.generate.return_value = "7/10"
        score = evaluator.evaluate("test", context)
        assert score == 0.7

        # Test "X out of 10" format
        mock_provider.generate.return_value = "9 out of 10"
        score = evaluator.evaluate("test", context)
        assert score == 0.9

        # Test "Score: X" format
        mock_provider.generate.return_value = "Score: 6"
        score = evaluator.evaluate("test", context)
        assert score == 0.6

    def test_evaluate_fallback_on_error(self):
        """Test fallback to neutral score on error."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.side_effect = Exception("API error")

        evaluator = LLMEvaluator(llm_provider=mock_provider)
        context = WorkflowContext()

        score = evaluator.evaluate("test", context)

        assert score == 0.5  # Neutral score on error

    def test_get_feedback(self):
        """Test feedback generation."""
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "The output could be more concise"

        evaluator = LLMEvaluator(llm_provider=mock_provider)

        feedback = evaluator.get_feedback("test result", 0.6)

        assert "concise" in feedback
        mock_provider.generate.assert_called_once()


class TestCompositeEvaluator:
    """Tests for CompositeEvaluator class."""

    def test_evaluate_weighted_average(self):
        """Test weighted average of multiple evaluators."""
        eval1 = LengthEvaluator(min_length=10, max_length=100)
        eval2 = KeywordEvaluator(required_keywords=["test"])

        evaluator = CompositeEvaluator([
            (eval1, 0.6),
            (eval2, 0.4)
        ])

        context = WorkflowContext()
        result = "This is a test with good length"

        score = evaluator.evaluate(result, context)

        # Should be weighted average
        assert 0.0 <= score <= 1.0

    def test_evaluate_normalizes_weights(self):
        """Test that weights are normalized."""
        eval1 = Mock(spec=Evaluator)
        eval1.evaluate.return_value = 0.8
        eval2 = Mock(spec=Evaluator)
        eval2.evaluate.return_value = 0.6

        # Weights don't sum to 1.0
        evaluator = CompositeEvaluator([
            (eval1, 2.0),
            (eval2, 2.0)
        ])

        context = WorkflowContext()
        score = evaluator.evaluate("test", context)

        # Should normalize to 0.5 each, so (0.8 + 0.6) / 2 = 0.7
        assert score == 0.7

    def test_evaluate_handles_evaluator_failure(self):
        """Test handling of evaluator failures."""
        eval1 = Mock(spec=Evaluator)
        eval1.evaluate.side_effect = Exception("Eval failed")
        eval2 = Mock(spec=Evaluator)
        eval2.evaluate.return_value = 0.8

        evaluator = CompositeEvaluator([
            (eval1, 0.5),
            (eval2, 0.5)
        ])

        context = WorkflowContext()
        score = evaluator.evaluate("test", context)

        # Should use neutral score (0.5) for failed evaluator
        # (0.5 * 0.5 + 0.8 * 0.5) = 0.65
        assert score == 0.65


class TestRetryStrategy:
    """Tests for RetryStrategy class."""

    def test_optimize_retries_node(self):
        """Test that retry strategy re-executes node."""
        node = MockNode("test", return_value="new result")
        strategy = RetryStrategy()
        context = WorkflowContext()

        result = strategy.optimize(node, context, "Try again")

        assert result == "new result"
        assert node.execution_count == 1


class TestPromptRefinementStrategy:
    """Tests for PromptRefinementStrategy class."""

    def test_optimize_refines_prompt(self):
        """Test that strategy refines prompt with feedback."""
        from framework.base import PromptNode

        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Improved result"

        node = PromptNode(
            name="test",
            prompt_template="Original prompt",
            llm_provider=mock_provider,
            output_key="result"
        )

        strategy = PromptRefinementStrategy()
        context = WorkflowContext()

        result = strategy.optimize(node, context, "Be more specific")

        assert result == "Improved result"
        # Original prompt should be restored
        assert node.prompt_template == "Original prompt"

    def test_optimize_non_prompt_node_fallback(self):
        """Test fallback for non-PromptNode."""
        node = MockNode("test", return_value="result")
        strategy = PromptRefinementStrategy()
        context = WorkflowContext()

        result = strategy.optimize(node, context, "feedback")

        # Should just execute the node
        assert result == "result"


class TestTemperatureAdjustmentStrategy:
    """Tests for TemperatureAdjustmentStrategy class."""

    def test_optimize_adjusts_temperature(self):
        """Test that strategy adjusts temperature."""
        from framework.base import PromptNode

        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Result"

        node = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key="result",
            model_params={"temperature": 0.7}
        )

        strategy = TemperatureAdjustmentStrategy(adjustment=-0.2)
        context = WorkflowContext()

        result = strategy.optimize(node, context, "feedback")

        assert result == "Result"
        # Temperature should be restored
        assert node.model_params["temperature"] == 0.7

    def test_optimize_respects_bounds(self):
        """Test that temperature stays within bounds."""
        from framework.base import PromptNode

        mock_provider = Mock(spec=LLMProvider)
        mock_provider.generate.return_value = "Result"

        node = PromptNode(
            name="test",
            prompt_template="Test",
            llm_provider=mock_provider,
            output_key="result",
            model_params={"temperature": 0.1}
        )

        strategy = TemperatureAdjustmentStrategy(
            adjustment=-0.5,
            min_temp=0.0,
            max_temp=2.0
        )
        context = WorkflowContext()

        strategy.optimize(node, context, "feedback")

        # Should not go below min_temp
        assert node.model_params["temperature"] == 0.1


class TestEvaluatorNode:
    """Tests for EvaluatorNode class."""

    def test_initialization(self):
        """Test evaluator node initialization."""
        target_node = MockNode("target")
        evaluator = LengthEvaluator(min_length=10, max_length=100)
        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.7,
            max_iterations=3
        )

        assert eval_node.name == "evaluator"
        assert eval_node.target_node == target_node
        assert eval_node.threshold == 0.7
        assert eval_node.max_iterations == 3

    def test_execute_meets_threshold_first_try(self):
        """Test execution when threshold is met on first try."""
        target_node = MockNode("target", return_value="a" * 50)
        evaluator = LengthEvaluator(min_length=10, max_length=100)
        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.7,
            max_iterations=3
        )

        context = WorkflowContext()
        result = eval_node.execute(context)

        assert result == "a" * 50
        assert target_node.execution_count == 1  # Only executed once

    def test_execute_retries_until_threshold(self):
        """Test execution retries until threshold is met."""
        # Create node that improves on retry
        target_node = MockNode("target", return_value="short")
        evaluator = LengthEvaluator(min_length=50, max_length=100)
        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.7,
            max_iterations=3
        )

        context = WorkflowContext()

        # Will not meet threshold, but should try max_iterations times
        eval_node.execute(context)

        assert target_node.execution_count == 3  # Max iterations

    def test_execute_keeps_best_result(self):
        """Test that best result is kept when keep_best=True."""
        target_node = MockNode("target", return_value="a" * 50)

        # Mock evaluator that returns different scores
        evaluator = Mock(spec=Evaluator)
        evaluator.evaluate.side_effect = [0.6, 0.8, 0.5]  # Best is second
        evaluator.get_feedback.return_value = "Improve"

        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.9,  # Won't be met
            max_iterations=3,
            keep_best=True
        )

        context = WorkflowContext()
        result = eval_node.execute(context)

        # Should return result from iteration with score 0.8
        assert result == "a" * 50

    def test_execute_returns_last_when_keep_best_false(self):
        """Test that last result is returned when keep_best=False."""
        target_node = MockNode("target", return_value="result")
        evaluator = Mock(spec=Evaluator)
        evaluator.evaluate.return_value = 0.5  # Below threshold
        evaluator.get_feedback.return_value = "Improve"

        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.7,
            max_iterations=2,
            keep_best=False
        )

        context = WorkflowContext()
        result = eval_node.execute(context)

        assert result == "result"

    def test_execute_first_iteration_failure(self):
        """Test that first iteration failure raises error."""
        target_node = Mock(spec=Node)
        target_node.execute.side_effect = Exception("Failed")

        evaluator = LengthEvaluator(min_length=10, max_length=100)
        strategy = RetryStrategy()

        eval_node = EvaluatorNode(
            name="evaluator",
            target_node=target_node,
            evaluator=evaluator,
            optimization_strategy=strategy,
            threshold=0.7,
            max_iterations=3
        )

        context = WorkflowContext()

        with pytest.raises(EvaluationError, match="First iteration failed"):
            eval_node.execute(context)
