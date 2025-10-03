"""
Evaluator-optimizer module for the LLM Workflow Framework.

This module provides abstractions and implementations for evaluating LLM outputs
and optimizing them through iterative refinement. It supports custom evaluation
criteria, optimization strategies, and automatic quality improvement loops.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from framework.base import LLMProvider, Node, WorkflowContext
from framework.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    """
    Abstract base class for result evaluators.

    Evaluators assess the quality of LLM outputs based on specific criteria
    and provide feedback for improvement. They return a numeric score and
    optionally provide textual feedback for optimization strategies.

    Example:
        >>> class CustomEvaluator(Evaluator):
        ...     def evaluate(self, result: Any, context: WorkflowContext) -> float:
        ...         return 0.8 if len(str(result)) > 100 else 0.3
        ...     def get_feedback(self, result: Any, score: float) -> str:
        ...         return "Output too short" if score < 0.5 else "Good length"
    """

    @abstractmethod
    def evaluate(self, result: Any, context: WorkflowContext) -> float:
        """
        Evaluate a result and return a score.

        Args:
            result: The result to evaluate (typically LLM output)
            context: The workflow context for additional information

        Returns:
            A score between 0.0 and 1.0, where 1.0 is perfect

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def get_feedback(self, result: Any, score: float) -> str:
        """
        Generate feedback for improving the result.

        Args:
            result: The result that was evaluated
            score: The score assigned to the result

        Returns:
            Textual feedback describing how to improve the result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass



class LengthEvaluator(Evaluator):
    """
    Evaluator that scores based on output length.

    Scores outputs based on whether they fall within a target length range.
    Useful for ensuring outputs are neither too short nor too long.

    Attributes:
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        target_length: Optional ideal length for best score

    Example:
        >>> evaluator = LengthEvaluator(min_length=50, max_length=200, target_length=100)
        >>> score = evaluator.evaluate("Short text", context)
        >>> print(score < 0.5)
        True
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        target_length: Optional[int] = None
    ) -> None:
        """
        Initialize a LengthEvaluator.

        Args:
            min_length: Minimum acceptable length (characters)
            max_length: Maximum acceptable length (characters)
            target_length: Optional ideal length for perfect score
        """
        self.min_length = min_length
        self.max_length = max_length
        self.target_length = target_length

    def evaluate(self, result: Any, context: WorkflowContext) -> float:
        """
        Evaluate result based on length.

        Args:
            result: The result to evaluate
            context: The workflow context

        Returns:
            Score between 0.0 and 1.0 based on length criteria
        """
        text = str(result)
        length = len(text)

        # Below minimum
        if length < self.min_length:
            return max(0.0, length / self.min_length * 0.5)

        # Above maximum
        if length > self.max_length:
            excess = length - self.max_length
            penalty = min(1.0, excess / self.max_length)
            return max(0.0, 0.5 - penalty * 0.5)

        # Within range
        if self.target_length:
            # Score based on distance from target
            distance = abs(length - self.target_length)
            max_distance = max(
                self.target_length - self.min_length,
                self.max_length - self.target_length
            )
            if max_distance > 0:
                score = 1.0 - (distance / max_distance * 0.3)
                return max(0.7, min(1.0, score))

        return 0.8  # Good enough if in range

    def get_feedback(self, result: Any, score: float) -> str:
        """
        Generate feedback about length.

        Args:
            result: The evaluated result
            score: The assigned score

        Returns:
            Feedback message about length
        """
        length = len(str(result))

        if length < self.min_length:
            return f"Output too short ({length} chars). Minimum: {self.min_length} chars."
        elif length > self.max_length:
            return f"Output too long ({length} chars). Maximum: {self.max_length} chars."
        elif self.target_length and abs(length - self.target_length) > 50:
            return f"Output length ({length} chars) could be closer to target ({self.target_length} chars)."
        else:
            return f"Output length ({length} chars) is acceptable."


class KeywordEvaluator(Evaluator):
    """
    Evaluator that scores based on keyword presence.

    Scores outputs based on the presence or absence of specified keywords.
    Supports both required keywords (must be present) and forbidden keywords
    (must not be present).

    Attributes:
        required_keywords: Keywords that should be present
        forbidden_keywords: Keywords that should not be present
        case_sensitive: Whether keyword matching is case-sensitive

    Example:
        >>> evaluator = KeywordEvaluator(
        ...     required_keywords=["summary", "conclusion"],
        ...     forbidden_keywords=["error", "failed"]
        ... )
        >>> score = evaluator.evaluate("Here is a summary...", context)
    """

    def __init__(
        self,
        required_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        case_sensitive: bool = False
    ) -> None:
        """
        Initialize a KeywordEvaluator.

        Args:
            required_keywords: List of keywords that must be present
            forbidden_keywords: List of keywords that must not be present
            case_sensitive: Whether to match keywords case-sensitively
        """
        self.required_keywords = required_keywords or []
        self.forbidden_keywords = forbidden_keywords or []
        self.case_sensitive = case_sensitive

    def evaluate(self, result: Any, context: WorkflowContext) -> float:
        """
        Evaluate result based on keyword presence.

        Args:
            result: The result to evaluate
            context: The workflow context

        Returns:
            Score between 0.0 and 1.0 based on keyword criteria
        """
        text = str(result)
        if not self.case_sensitive:
            text = text.lower()

        score = 1.0

        # Check required keywords
        if self.required_keywords:
            present_count = 0
            for keyword in self.required_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword in text:
                    present_count += 1

            required_score = present_count / len(self.required_keywords)
            score *= required_score

        # Check forbidden keywords
        if self.forbidden_keywords:
            forbidden_count = 0
            for keyword in self.forbidden_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword in text:
                    forbidden_count += 1

            if forbidden_count > 0:
                penalty = forbidden_count / len(self.forbidden_keywords)
                score *= (1.0 - penalty)

        return max(0.0, min(1.0, score))

    def get_feedback(self, result: Any, score: float) -> str:
        """
        Generate feedback about keyword presence.

        Args:
            result: The evaluated result
            score: The assigned score

        Returns:
            Feedback message about keywords
        """
        text = str(result)
        if not self.case_sensitive:
            text = text.lower()

        feedback_parts = []

        # Check missing required keywords
        if self.required_keywords:
            missing = []
            for keyword in self.required_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword not in text:
                    missing.append(keyword)

            if missing:
                feedback_parts.append(f"Missing required keywords: {', '.join(missing)}")

        # Check forbidden keywords present
        if self.forbidden_keywords:
            present = []
            for keyword in self.forbidden_keywords:
                check_keyword = keyword if self.case_sensitive else keyword.lower()
                if check_keyword in text:
                    present.append(keyword)

            if present:
                feedback_parts.append(f"Contains forbidden keywords: {', '.join(present)}")

        if not feedback_parts:
            return "All keyword criteria met."

        return " ".join(feedback_parts)


class LLMEvaluator(Evaluator):
    """
    Evaluator that uses an LLM to assess quality.

    Uses an LLM to evaluate the quality of outputs based on a custom evaluation
    prompt. This allows for sophisticated, context-aware evaluation that can
    assess aspects like coherence, relevance, and accuracy.

    Attributes:
        llm_provider: The LLM provider to use for evaluation
        evaluation_prompt_template: Template for the evaluation prompt
        model_params: Parameters for the LLM

    Example:
        >>> evaluator = LLMEvaluator(
        ...     llm_provider=provider,
        ...     evaluation_prompt_template="Rate this summary from 0-10: {result}"
        ... )
        >>> score = evaluator.evaluate("Summary text...", context)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        evaluation_prompt_template: str = "Evaluate the quality of this output on a scale of 0-10:\n\n{result}\n\nScore:",
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize an LLMEvaluator.

        Args:
            llm_provider: LLM provider instance to use for evaluation
            evaluation_prompt_template: Template for evaluation prompt
            model_params: Optional parameters for the LLM
        """
        self.llm_provider = llm_provider
        self.evaluation_prompt_template = evaluation_prompt_template
        self.model_params = model_params or {}

    def evaluate(self, result: Any, context: WorkflowContext) -> float:
        """
        Evaluate result using an LLM.

        Args:
            result: The result to evaluate
            context: The workflow context

        Returns:
            Score between 0.0 and 1.0 based on LLM evaluation
        """
        # Format evaluation prompt
        prompt = self.evaluation_prompt_template.format(result=str(result))

        try:
            # Get LLM evaluation
            response = self.llm_provider.generate(prompt, **self.model_params)

            # Parse score from response
            score = self._parse_score(response)
            return score
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}. Returning neutral score.")
            return 0.5

    def get_feedback(self, result: Any, score: float) -> str:
        """
        Generate feedback using the LLM.

        Args:
            result: The evaluated result
            score: The assigned score

        Returns:
            Feedback message from the LLM
        """
        feedback_prompt = f"Provide specific feedback on how to improve this output (current score: {score:.2f}):\n\n{result}\n\nFeedback:"

        try:
            feedback = self.llm_provider.generate(feedback_prompt, **self.model_params)
            return feedback.strip()
        except Exception as e:
            logger.warning(f"LLM feedback generation failed: {e}")
            return f"Score: {score:.2f}. Unable to generate detailed feedback."

    def _parse_score(self, response: str) -> float:
        """
        Parse a numeric score from LLM response.

        Args:
            response: The LLM response text

        Returns:
            Normalized score between 0.0 and 1.0
        """
        # Try to extract a number from the response
        import re

        # Look for patterns like "8/10", "8 out of 10", or just "8"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s+out of\s+10',
            r'(?:score|rating):\s*(\d+(?:\.\d+)?)',
            r'^(\d+(?:\.\d+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-1 range (assuming 0-10 scale)
                    if score <= 10:
                        return score / 10.0
                    elif score <= 100:
                        return score / 100.0
                    else:
                        return min(1.0, score / 10.0)
                except ValueError:
                    continue

        # Default to neutral score if parsing fails
        logger.warning(f"Could not parse score from response: {response[:100]}")
        return 0.5


class CompositeEvaluator(Evaluator):
    """
    Evaluator that combines multiple evaluators with weights.

    Allows combining multiple evaluation criteria into a single score by
    computing a weighted average of individual evaluator scores.

    Attributes:
        evaluators: List of (evaluator, weight) tuples

    Example:
        >>> composite = CompositeEvaluator([
        ...     (LengthEvaluator(min_length=50, max_length=200), 0.3),
        ...     (KeywordEvaluator(required_keywords=["summary"]), 0.4),
        ...     (LLMEvaluator(provider), 0.3)
        ... ])
        >>> score = composite.evaluate("Summary text...", context)
    """

    def __init__(self, evaluators: List[Tuple[Evaluator, float]]) -> None:
        """
        Initialize a CompositeEvaluator.

        Args:
            evaluators: List of (evaluator, weight) tuples. Weights should sum to 1.0
        """
        self.evaluators = evaluators

        # Normalize weights if they don't sum to 1.0
        total_weight = sum(weight for _, weight in evaluators)
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            self.evaluators = [
                (evaluator, weight / total_weight)
                for evaluator, weight in evaluators
            ]

    def evaluate(self, result: Any, context: WorkflowContext) -> float:
        """
        Evaluate result using all evaluators and compute weighted average.

        Args:
            result: The result to evaluate
            context: The workflow context

        Returns:
            Weighted average score between 0.0 and 1.0
        """
        if not self.evaluators:
            return 0.5

        weighted_sum = 0.0
        for evaluator, weight in self.evaluators:
            try:
                score = evaluator.evaluate(result, context)
                weighted_sum += score * weight
            except Exception as e:
                logger.warning(f"Evaluator {evaluator.__class__.__name__} failed: {e}")
                # Use neutral score for failed evaluators
                weighted_sum += 0.5 * weight

        return max(0.0, min(1.0, weighted_sum))

    def get_feedback(self, result: Any, score: float) -> str:
        """
        Generate combined feedback from all evaluators.

        Args:
            result: The evaluated result
            score: The assigned score

        Returns:
            Combined feedback from all evaluators
        """
        feedback_parts = [f"Overall score: {score:.2f}"]

        for evaluator, weight in self.evaluators:
            try:
                eval_score = evaluator.evaluate(result, context=WorkflowContext())
                eval_feedback = evaluator.get_feedback(result, eval_score)
                feedback_parts.append(
                    f"[{evaluator.__class__.__name__} (weight={weight:.2f}, score={eval_score:.2f})]: {eval_feedback}"
                )
            except Exception as e:
                logger.warning(f"Feedback generation failed for {evaluator.__class__.__name__}: {e}")

        return "\n".join(feedback_parts)



class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.

    Optimization strategies define how to improve results when evaluation
    scores are below threshold. They can modify prompts, adjust parameters,
    or apply other techniques to enhance output quality.

    Example:
        >>> class CustomStrategy(OptimizationStrategy):
        ...     def optimize(self, node: Node, context: WorkflowContext, feedback: str) -> Any:
        ...         # Modify context or node parameters
        ...         return node.execute(context)
    """

    @abstractmethod
    def optimize(
        self,
        node: Node,
        context: WorkflowContext,
        feedback: str
    ) -> Any:
        """
        Apply optimization strategy to improve results.

        Args:
            node: The node to optimize
            context: The workflow context
            feedback: Feedback from the evaluator about how to improve

        Returns:
            The optimized result

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass



class RetryStrategy(OptimizationStrategy):
    """
    Simple retry strategy that re-executes the node.

    This strategy simply retries the node execution without any modifications.
    Useful when LLM outputs have inherent randomness and a retry might produce
    a better result.

    Example:
        >>> strategy = RetryStrategy()
        >>> result = strategy.optimize(node, context, "Try again")
    """

    def optimize(
        self,
        node: Node,
        context: WorkflowContext,
        feedback: str
    ) -> Any:
        """
        Retry node execution without modifications.

        Args:
            node: The node to retry
            context: The workflow context
            feedback: Feedback from evaluator (not used in simple retry)

        Returns:
            Result from re-executing the node
        """
        logger.debug(f"RetryStrategy: Re-executing {node.name}")
        return node.execute(context)


class PromptRefinementStrategy(OptimizationStrategy):
    """
    Strategy that modifies the prompt based on feedback.

    This strategy appends feedback to the prompt to guide the LLM toward
    better outputs. Only works with PromptNode instances.

    Attributes:
        refinement_template: Template for adding feedback to prompt

    Example:
        >>> strategy = PromptRefinementStrategy()
        >>> result = strategy.optimize(prompt_node, context, "Be more concise")
    """

    def __init__(
        self,
        refinement_template: str = "\n\nPrevious attempt had issues: {feedback}\nPlease improve the output accordingly."
    ) -> None:
        """
        Initialize a PromptRefinementStrategy.

        Args:
            refinement_template: Template for incorporating feedback into prompt
        """
        self.refinement_template = refinement_template

    def optimize(
        self,
        node: Node,
        context: WorkflowContext,
        feedback: str
    ) -> Any:
        """
        Refine prompt with feedback and re-execute.

        Args:
            node: The node to optimize (must be PromptNode)
            context: The workflow context
            feedback: Feedback to incorporate into the prompt

        Returns:
            Result from executing with refined prompt
        """
        from framework.base import PromptNode

        if not isinstance(node, PromptNode):
            logger.warning(
                f"PromptRefinementStrategy only works with PromptNode, "
                f"got {type(node).__name__}. Falling back to simple retry."
            )
            return node.execute(context)

        # Store original prompt template
        original_template = node.prompt_template

        try:
            # Append feedback to prompt
            refinement = self.refinement_template.format(feedback=feedback)
            node.prompt_template = original_template + refinement

            logger.debug(f"PromptRefinementStrategy: Refining prompt for {node.name}")
            result = node.execute(context)

            return result
        finally:
            # Restore original prompt template
            node.prompt_template = original_template


class TemperatureAdjustmentStrategy(OptimizationStrategy):
    """
    Strategy that adjusts the temperature parameter.

    This strategy modifies the LLM temperature to encourage different output
    characteristics. Lower temperature for more focused outputs, higher for
    more creative ones. Only works with PromptNode instances.

    Attributes:
        adjustment: Amount to adjust temperature by (can be negative)
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature

    Example:
        >>> strategy = TemperatureAdjustmentStrategy(adjustment=-0.2)
        >>> result = strategy.optimize(prompt_node, context, "Be more focused")
    """

    def __init__(
        self,
        adjustment: float = -0.2,
        min_temp: float = 0.0,
        max_temp: float = 2.0
    ) -> None:
        """
        Initialize a TemperatureAdjustmentStrategy.

        Args:
            adjustment: Amount to adjust temperature (positive or negative)
            min_temp: Minimum allowed temperature value
            max_temp: Maximum allowed temperature value
        """
        self.adjustment = adjustment
        self.min_temp = min_temp
        self.max_temp = max_temp

    def optimize(
        self,
        node: Node,
        context: WorkflowContext,
        feedback: str
    ) -> Any:
        """
        Adjust temperature and re-execute.

        Args:
            node: The node to optimize (must be PromptNode)
            context: The workflow context
            feedback: Feedback from evaluator (used to determine adjustment direction)

        Returns:
            Result from executing with adjusted temperature
        """
        from framework.base import PromptNode

        if not isinstance(node, PromptNode):
            logger.warning(
                f"TemperatureAdjustmentStrategy only works with PromptNode, "
                f"got {type(node).__name__}. Falling back to simple retry."
            )
            return node.execute(context)

        # Get current temperature
        current_temp = node.model_params.get("temperature", 0.7)

        # Calculate new temperature
        new_temp = current_temp + self.adjustment
        new_temp = max(self.min_temp, min(self.max_temp, new_temp))

        # Store original temperature
        original_temp = current_temp

        try:
            # Apply new temperature
            node.model_params["temperature"] = new_temp

            logger.debug(
                f"TemperatureAdjustmentStrategy: Adjusting temperature for {node.name} "
                f"from {original_temp:.2f} to {new_temp:.2f}"
            )
            result = node.execute(context)

            return result
        finally:
            # Restore original temperature
            node.model_params["temperature"] = original_temp



class EvaluatorNode(Node):
    """
    Node that evaluates and optimizes outputs through iteration.

    EvaluatorNode wraps another node and repeatedly executes it, evaluating
    each result and applying optimization strategies until the score meets
    a threshold or maximum iterations are reached.

    Attributes:
        target_node: The node to evaluate and optimize
        evaluator: Evaluator to assess result quality
        optimization_strategy: Strategy to apply when score is below threshold
        threshold: Minimum acceptable score (0.0 to 1.0)
        max_iterations: Maximum number of optimization attempts
        keep_best: Whether to return the best result or the last one

    Example:
        >>> evaluator_node = EvaluatorNode(
        ...     name="optimized_summarizer",
        ...     target_node=summarizer_node,
        ...     evaluator=LengthEvaluator(min_length=50, max_length=200),
        ...     optimization_strategy=RetryStrategy(),
        ...     threshold=0.7,
        ...     max_iterations=3
        ... )
        >>> result = evaluator_node.execute(context)
    """

    def __init__(
        self,
        name: str,
        target_node: Node,
        evaluator: Evaluator,
        optimization_strategy: OptimizationStrategy,
        threshold: float = 0.7,
        max_iterations: int = 3,
        keep_best: bool = True,
        description: str = ""
    ) -> None:
        """
        Initialize an EvaluatorNode.

        Args:
            name: Unique identifier for the node
            target_node: The node to evaluate and optimize
            evaluator: Evaluator instance to assess quality
            optimization_strategy: Strategy to apply for optimization
            threshold: Minimum acceptable score (0.0 to 1.0)
            max_iterations: Maximum optimization attempts
            keep_best: If True, return best result; if False, return last result
            description: Human-readable description
        """
        super().__init__(name, description)
        self.target_node = target_node
        self.evaluator = evaluator
        self.optimization_strategy = optimization_strategy
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.keep_best = keep_best

    def execute(self, context: WorkflowContext) -> Any:
        """
        Execute the evaluation-optimization loop.

        Repeatedly executes the target node, evaluates results, and applies
        optimization strategies until threshold is met or max iterations reached.

        Args:
            context: The workflow context

        Returns:
            The best (or last) result from the optimization loop

        Raises:
            EvaluationError: If all iterations fail or no valid results obtained
        """
        attempts: List[Tuple[Any, float, str]] = []

        logger.info(
            f"EvaluatorNode '{self.name}': Starting evaluation loop "
            f"(threshold={self.threshold}, max_iterations={self.max_iterations})"
        )

        for iteration in range(self.max_iterations):
            try:
                # Run iteration
                result, score, feedback = self.run_iteration(context, iteration)
                attempts.append((result, score, feedback))

                logger.info(
                    f"EvaluatorNode '{self.name}': Iteration {iteration + 1}/{self.max_iterations} "
                    f"- Score: {score:.3f}, Feedback: {feedback[:100]}{'...' if len(feedback) > 100 else ''}"
                )

                # Check if threshold met
                if score >= self.threshold:
                    logger.info(
                        f"EvaluatorNode '{self.name}': Threshold met ({score:.3f} >= {self.threshold})"
                    )
                    break

                # Log optimization attempt for next iteration (if not last)
                if iteration < self.max_iterations - 1:
                    logger.debug(
                        f"EvaluatorNode '{self.name}': Score below threshold, "
                        f"will apply {self.optimization_strategy.__class__.__name__} for next iteration"
                    )

            except Exception as e:
                logger.error(
                    f"EvaluatorNode '{self.name}': Iteration {iteration + 1} failed: {e}",
                    exc_info=True
                )
                # Continue with next iteration unless this was the first
                if iteration == 0:
                    raise EvaluationError(
                        f"First iteration failed for EvaluatorNode '{self.name}': {e}"
                    ) from e
                # For subsequent iterations, log and continue
                logger.warning(
                    f"EvaluatorNode '{self.name}': Continuing to next iteration after failure"
                )

        # Select best result
        if not attempts:
            raise EvaluationError(
                f"EvaluatorNode '{self.name}': No successful attempts after {self.max_iterations} iterations"
            )

        final_result = self.select_best_result(attempts)

        # Store evaluation metadata in context
        best_score = max(score for _, score, _ in attempts)
        avg_score = sum(score for _, score, _ in attempts) / len(attempts)
        context.metadata[f"{self.name}_iterations"] = len(attempts)
        context.metadata[f"{self.name}_best_score"] = best_score
        context.metadata[f"{self.name}_avg_score"] = avg_score
        context.metadata[f"{self.name}_final_score"] = attempts[-1][1]
        context.metadata[f"{self.name}_threshold_met"] = best_score >= self.threshold

        # Log final summary
        logger.info(
            f"EvaluatorNode '{self.name}': Completed after {len(attempts)} iterations. "
            f"Best score: {best_score:.3f}, Avg score: {avg_score:.3f}, "
            f"Threshold met: {best_score >= self.threshold}"
        )

        return final_result

    def run_iteration(
        self,
        context: WorkflowContext,
        iteration: int
    ) -> Tuple[Any, float, str]:
        """
        Run a single iteration of evaluation and optimization.

        Args:
            context: The workflow context
            iteration: Current iteration number (0-indexed)

        Returns:
            Tuple of (result, score, feedback)

        Raises:
            EvaluationError: If node execution or evaluation fails
        """
        try:
            # Execute target node or apply optimization
            if iteration == 0:
                # First iteration: execute normally
                logger.debug(f"EvaluatorNode '{self.name}': Executing target node '{self.target_node.name}'")
                result = self.target_node.execute(context)
            else:
                # Subsequent iterations: apply optimization
                previous_feedback = getattr(self, '_last_feedback', '')
                logger.debug(
                    f"EvaluatorNode '{self.name}': Applying {self.optimization_strategy.__class__.__name__}"
                )
                result = self.optimization_strategy.optimize(
                    self.target_node,
                    context,
                    previous_feedback
                )
        except Exception as e:
            logger.error(f"EvaluatorNode '{self.name}': Node execution failed in iteration {iteration + 1}: {e}")
            raise EvaluationError(f"Node execution failed: {e}") from e

        try:
            # Evaluate result
            score = self.evaluator.evaluate(result, context)

            # Validate score is in valid range
            if not (0.0 <= score <= 1.0):
                logger.warning(
                    f"EvaluatorNode '{self.name}': Evaluator returned invalid score {score}, "
                    f"clamping to [0.0, 1.0]"
                )
                score = max(0.0, min(1.0, score))

            feedback = self.evaluator.get_feedback(result, score)
        except Exception as e:
            logger.error(f"EvaluatorNode '{self.name}': Evaluation failed in iteration {iteration + 1}: {e}")
            raise EvaluationError(f"Evaluation failed: {e}") from e

        # Store feedback for next iteration
        self._last_feedback = feedback

        return result, score, feedback

    def select_best_result(
        self,
        attempts: List[Tuple[Any, float, str]]
    ) -> Any:
        """
        Select the best result from all attempts.

        Args:
            attempts: List of (result, score, feedback) tuples

        Returns:
            The selected result (best or last based on keep_best setting)
        """
        if self.keep_best:
            # Return result with highest score
            best_attempt = max(attempts, key=lambda x: x[1])
            result, score, _ = best_attempt
            logger.info(
                f"EvaluatorNode '{self.name}': Returning best result (score={score:.3f})"
            )
            return result
        else:
            # Return last result
            result, score, _ = attempts[-1]
            logger.info(
                f"EvaluatorNode '{self.name}': Returning last result (score={score:.3f})"
            )
            return result

    def validate(self) -> bool:
        """
        Validate the evaluator node configuration.

        Returns:
            True if valid, False otherwise
        """
        return (
            super().validate() and
            self.target_node is not None and
            self.evaluator is not None and
            self.optimization_strategy is not None and
            0.0 <= self.threshold <= 1.0 and
            self.max_iterations > 0
        )
