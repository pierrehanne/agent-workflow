"""
Simple Routing Example

This example demonstrates conditional routing based on input characteristics.
Different handlers are invoked based on the sentiment of the input text.

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python examples/simple_routing.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.base import WorkflowContext, PromptNode
from framework.providers.gemini import GeminiProvider
from framework.routing import RouterNode, KeywordCondition


def main():
    """Run a simple routing workflow example."""
    
    # Setup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return 1
    
    provider = GeminiProvider(api_key=api_key)
    
    # Create different handlers for different types of feedback
    positive_handler = PromptNode(
        name="positive_response",
        prompt_template="Respond enthusiastically to this positive feedback: {feedback}",
        llm_provider=provider,
        output_key="response"
    )
    
    negative_handler = PromptNode(
        name="negative_response",
        prompt_template="Respond empathetically and offer solutions to this negative feedback: {feedback}",
        llm_provider=provider,
        output_key="response"
    )
    
    neutral_handler = PromptNode(
        name="neutral_response",
        prompt_template="Respond professionally to this neutral feedback: {feedback}",
        llm_provider=provider,
        output_key="response"
    )
    
    # Create router
    router = RouterNode(name="feedback_router")
    
    # Add routes based on keywords
    router.add_route(
        condition=KeywordCondition(
            key="feedback",
            keywords=["great", "excellent", "love", "amazing", "wonderful"],
            case_sensitive=False
        ),
        node=positive_handler,
        name="positive_route"
    )
    
    router.add_route(
        condition=KeywordCondition(
            key="feedback",
            keywords=["bad", "terrible", "hate", "awful", "disappointed"],
            case_sensitive=False
        ),
        node=negative_handler,
        name="negative_route"
    )
    
    # Set default for neutral feedback
    router.set_default(neutral_handler)
    
    # Test with different feedback types
    test_cases = [
        "This product is amazing! I love it!",
        "Terrible experience. Very disappointed.",
        "The product works as expected."
    ]
    
    print("Testing routing with different feedback types...")
    print("=" * 60)
    
    for feedback in test_cases:
        context = WorkflowContext()
        context.set("feedback", feedback)
        
        print(f"\nFeedback: {feedback}")
        print("-" * 60)
        
        response = router.execute(context)
        
        # Check which route was taken
        routing_decisions = context.metadata.get("routing_decisions", [])
        if routing_decisions:
            route_taken = routing_decisions[-1]["route"]
            print(f"Route taken: {route_taken}")
        
        print(f"Response: {response}")
    
    print("\n✅ Routing completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
