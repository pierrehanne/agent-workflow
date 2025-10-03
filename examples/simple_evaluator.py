"""
Simple Evaluator-Optimizer Example

This example demonstrates how to evaluate LLM outputs and iteratively
improve them until they meet quality criteria.

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python examples/simple_evaluator.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.base import WorkflowContext, PromptNode
from framework.providers.gemini import GeminiProvider
from framework.evaluator import (
    EvaluatorNode,
    LengthEvaluator,
    KeywordEvaluator,
    CompositeEvaluator,
    RetryStrategy
)


def main():
    """Run a simple evaluator-optimizer example."""
    
    # Setup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return 1
    
    provider = GeminiProvider(api_key=api_key)
    context = WorkflowContext()
    context.set("topic", "climate change")
    
    # Create a node that generates a summary
    summary_node = PromptNode(
        name="summary_generator",
        prompt_template="""Write a brief summary about {topic}. 
Include key facts and mention 'global warming' and 'carbon emissions'.
Keep it between 100-200 characters.

Summary:""",
        llm_provider=provider,
        output_key="summary"
    )
    
    # Create evaluator with multiple criteria
    print("Creating evaluator with quality criteria...")
    
    evaluator = CompositeEvaluator([
        # Check length is appropriate (100-200 characters)
        (LengthEvaluator(min_length=100, max_length=200, target_length=150), 0.5),
        # Check for required keywords
        (KeywordEvaluator(
            required_keywords=["global warming", "carbon emissions"],
            case_sensitive=False
        ), 0.5)
    ])
    
    # Wrap the node with evaluator
    optimized_node = EvaluatorNode(
        name="optimized_summary",
        target_node=summary_node,
        evaluator=evaluator,
        optimization_strategy=RetryStrategy(),
        threshold=0.7,  # Require 70% score
        max_iterations=3,
        keep_best=True
    )
    
    # Execute
    print(f"\nGenerating optimized summary about: {context.get('topic')}")
    print("-" * 60)
    print("\nEvaluating and optimizing...")
    
    result = optimized_node.execute(context)
    
    # Display result
    print("\nFinal Optimized Summary:")
    print("=" * 60)
    print(result)
    print(f"\nLength: {len(result)} characters")
    
    # Check if keywords are present
    keywords_present = []
    for keyword in ["global warming", "carbon emissions"]:
        if keyword.lower() in result.lower():
            keywords_present.append(keyword)
    
    print(f"Keywords found: {', '.join(keywords_present) if keywords_present else 'None'}")
    
    print("\n✅ Evaluation and optimization completed!")
    return 0


if __name__ == "__main__":
    exit(main())
