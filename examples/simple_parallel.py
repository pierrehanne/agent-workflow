"""
Simple Parallel Execution Example

This example demonstrates parallel execution of multiple independent tasks
to reduce overall execution time.

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python examples/simple_parallel.py
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.base import WorkflowContext, PromptNode
from framework.providers.gemini import GeminiProvider
from framework.parallel import ParallelNode, DictMerge


def main():
    """Run a simple parallel execution example."""
    
    # Setup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return 1
    
    provider = GeminiProvider(api_key=api_key)
    context = WorkflowContext()
    
    text = "Python is a high-level programming language known for its simplicity and readability."
    context.set("text", text)
    
    # Create multiple analysis nodes
    print("Creating parallel analysis tasks...")
    
    # Task 1: Translate to Spanish
    translate_node = PromptNode(
        name="translator",
        prompt_template="Translate to Spanish: {text}",
        llm_provider=provider,
        output_key="spanish"
    )
    
    # Task 2: Identify key concepts
    concepts_node = PromptNode(
        name="concept_extractor",
        prompt_template="List the key concepts in this text: {text}",
        llm_provider=provider,
        output_key="concepts"
    )
    
    # Task 3: Generate a question
    question_node = PromptNode(
        name="question_generator",
        prompt_template="Generate a thought-provoking question based on: {text}",
        llm_provider=provider,
        output_key="question"
    )
    
    # Create parallel node
    parallel = ParallelNode(
        name="parallel_analysis",
        merge_strategy=DictMerge(),
        max_workers=3
    )
    
    parallel.add_node(translate_node)
    parallel.add_node(concepts_node)
    parallel.add_node(question_node)
    
    # Execute and time it
    print(f"\nProcessing text: {text}")
    print("-" * 60)
    print("\nExecuting tasks in parallel...")
    
    start_time = time.time()
    results = parallel.execute(context)
    duration = time.time() - start_time
    
    # Display results
    print("\nResults:")
    print("=" * 60)
    
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)
    
    print(f"\n⏱️  Execution time: {duration:.2f}s")
    print("✅ Parallel execution completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
