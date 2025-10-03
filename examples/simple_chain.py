"""
Simple Chain Example

This example demonstrates the basic prompt chaining pattern where
outputs from one node feed into the next node sequentially.

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python examples/simple_chain.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.base import PromptNode, WorkflowContext
from framework.chaining import ChainWorkflow
from framework.providers.gemini import GeminiProvider


def main():
    """Run a simple chain workflow example."""

    # Setup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return 1

    provider = GeminiProvider(api_key=api_key)
    context = WorkflowContext()
    context.set("topic", "quantum computing")

    # Create chain: simplify -> expand -> summarize
    print("Creating a 3-step chain workflow...")

    # Step 1: Simplify the topic
    simplify_node = PromptNode(
        name="simplify",
        prompt_template="Explain {topic} in simple terms (2-3 sentences):",
        llm_provider=provider,
        output_key="simple_explanation"
    )

    # Step 2: Expand with examples
    expand_node = PromptNode(
        name="expand",
        prompt_template="Add a practical example to this explanation:\n\n{simple_explanation}\n\nExample:",
        llm_provider=provider,
        output_key="with_example"
    )

    # Step 3: Create a summary
    summarize_node = PromptNode(
        name="summarize",
        prompt_template="Summarize this in one sentence:\n\n{with_example}\n\nSummary:",
        llm_provider=provider,
        output_key="final_summary"
    )

    # Build the chain
    chain = ChainWorkflow(name="explanation_chain", pass_through_context=True)
    chain.add_node(simplify_node).add_node(expand_node).add_node(summarize_node)

    # Execute
    print(f"\nProcessing topic: {context.get('topic')}")
    print("-" * 60)

    result = chain.execute(context)

    # Display results
    print("\nStep 1 - Simple Explanation:")
    print(context.get("simple_explanation"))

    print("\nStep 2 - With Example:")
    print(context.get("with_example"))

    print("\nStep 3 - Final Summary:")
    print(result)

    print("\n✅ Chain completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
