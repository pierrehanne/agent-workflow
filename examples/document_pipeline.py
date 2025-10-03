"""
Comprehensive Document Processing Pipeline Example

This example demonstrates all workflow patterns in the Agent Workflow:
1. PARALLEL: Extract different aspects concurrently (topics, entities, sentiment)
2. CHAIN: Process results sequentially
3. ROUTER: Route based on document characteristics
4. EVALUATOR-OPTIMIZER: Improve summary quality through iteration
5. ORCHESTRATOR: Coordinate final processing tasks

Requirements:
- Python 3.13+
- google-generativeai library
- GEMINI_API_KEY environment variable set

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python examples/document_pipeline.py
"""

import logging
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import framework components
from framework.base import PromptNode, WorkflowContext
from framework.chaining import ChainWorkflow
from framework.evaluator import (
    CompositeEvaluator,
    EvaluatorNode,
    KeywordEvaluator,
    LengthEvaluator,
    PromptRefinementStrategy,
)
from framework.orchestrator import LLMWorker, Orchestrator, Task
from framework.parallel import DictMerge, ParallelNode
from framework.providers.gemini import GeminiProvider
from framework.routing import LambdaCondition, RouterNode

# Configure logging to show execution flow
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Sample document for processing
SAMPLE_DOCUMENT = """
Artificial Intelligence and Machine Learning: The Future of Technology

The rapid advancement of artificial intelligence (AI) and machine learning (ML) technologies
is transforming industries across the globe. From healthcare to finance, transportation to
entertainment, AI systems are becoming increasingly sophisticated and capable of performing
complex tasks that once required human intelligence.

Machine learning, a subset of AI, enables computers to learn from data without being explicitly
programmed. Deep learning, a further specialization, uses neural networks with multiple layers
to process information in ways that mimic the human brain. These technologies have led to
breakthroughs in image recognition, natural language processing, and autonomous systems.

Major tech companies like Google, Microsoft, and OpenAI are investing billions of dollars in
AI research and development. The release of large language models such as GPT-4 and Google's
Gemini has demonstrated the potential for AI to understand and generate human-like text,
revolutionizing how we interact with computers.

However, the rise of AI also brings significant challenges. Concerns about job displacement,
algorithmic bias, privacy violations, and the potential misuse of AI technologies have sparked
important ethical debates. Researchers and policymakers are working to develop frameworks for
responsible AI development that prioritize safety, fairness, and transparency.

Looking ahead, AI is expected to play an even more central role in society. Experts predict
that AI will enhance human capabilities rather than replace them, leading to new forms of
human-AI collaboration. The key will be ensuring that these powerful technologies are developed
and deployed in ways that benefit all of humanity.
"""


def setup_gemini_provider():
    """
    Set up the Gemini API provider with configuration from environment.

    Returns:
        GeminiProvider: Configured provider instance

    Raises:
        ValueError: If GEMINI_API_KEY is not set
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with: export GEMINI_API_KEY='your-api-key'"
        )

    logger.info("Initializing Gemini provider...")
    provider = GeminiProvider(
        api_key=api_key,
        model_name="gemini-2.5-flash-lite",
        default_params={
            "temperature": 0.7,
            "max_output_tokens": 1024
        }
    )

    return provider


def create_parallel_extraction_pipeline(provider):
    """
    PATTERN 1: PARALLEL EXECUTION

    Create a parallel node that extracts different aspects of the document
    concurrently. This demonstrates how to run multiple independent LLM
    calls in parallel to reduce overall execution time.

    Args:
        provider: LLM provider instance

    Returns:
        ParallelNode: Configured parallel extraction node
    """
    logger.info("Creating parallel extraction pipeline...")

    # Node 1: Extract key topics
    topic_extractor = PromptNode(
        name="topic_extractor",
        prompt_template="""Extract the 3-5 main topics from this document.
List them as bullet points.

Document:
{document}

Topics:""",
        llm_provider=provider,
        input_key="document",
        output_key="topics",
        description="Extracts main topics from document"
    )

    # Node 2: Extract named entities
    entity_extractor = PromptNode(
        name="entity_extractor",
        prompt_template="""Extract all named entities (companies, people, technologies, organizations)
from this document. List them as bullet points.

Document:
{document}

Entities:""",
        llm_provider=provider,
        input_key="document",
        output_key="entities",
        description="Extracts named entities"
    )

    # Node 3: Analyze sentiment
    sentiment_analyzer = PromptNode(
        name="sentiment_analyzer",
        prompt_template="""Analyze the overall sentiment and tone of this document.
Is it positive, negative, neutral, or mixed? Provide a brief explanation.

Document:
{document}

Sentiment Analysis:""",
        llm_provider=provider,
        input_key="document",
        output_key="sentiment",
        description="Analyzes document sentiment"
    )

    # Create parallel node with dictionary merge strategy
    # This will combine results into a single dictionary
    parallel_node = ParallelNode(
        name="parallel_extraction",
        merge_strategy=DictMerge(key_prefix="analysis_"),
        max_workers=3,
        description="Parallel extraction of topics, entities, and sentiment"
    )

    parallel_node.add_node(topic_extractor)
    parallel_node.add_node(entity_extractor)
    parallel_node.add_node(sentiment_analyzer)

    return parallel_node


def create_chained_processing_pipeline(provider):
    """
    PATTERN 2: PROMPT CHAINING

    Create a chain that processes the parallel extraction results sequentially.
    Each step builds on the previous one, demonstrating data flow through
    multiple processing stages.

    Args:
        provider: LLM provider instance

    Returns:
        ChainWorkflow: Configured chain workflow
    """
    logger.info("Creating chained processing pipeline...")

    # Node 1: Merge parallel results into a structured analysis
    merge_node = PromptNode(
        name="merge_analysis",
        prompt_template="""Based on the following analysis components, create a structured overview:

Topics: {topics}

Entities: {entities}

Sentiment: {sentiment}

Provide a brief integrated analysis (2-3 sentences):""",
        llm_provider=provider,
        output_key="integrated_analysis",
        description="Merges parallel analysis results"
    )

    # Node 2: Generate initial summary
    summary_node = PromptNode(
        name="initial_summary",
        prompt_template="""Create a concise summary of this document based on the analysis:

Document:
{document}

Analysis:
{integrated_analysis}

Summary (3-4 sentences):""",
        llm_provider=provider,
        output_key="initial_summary",
        description="Generates initial document summary"
    )

    # Create chain workflow
    chain = ChainWorkflow(
        name="processing_chain",
        pass_through_context=True,
        description="Sequential processing of analysis results"
    )

    chain.add_node(merge_node).add_node(summary_node)

    return chain


def create_routing_pipeline(provider):
    """
    PATTERN 3: CONDITIONAL ROUTING

    Create a router that directs to different summarization strategies
    based on document characteristics (length). This demonstrates dynamic
    workflow branching based on runtime conditions.

    Args:
        provider: LLM provider instance

    Returns:
        RouterNode: Configured router node
    """
    logger.info("Creating routing pipeline...")

    # Short document handler - simple summary
    short_summary_node = PromptNode(
        name="short_summary",
        prompt_template="""This is a short document. Provide a brief 2-sentence summary:

{initial_summary}

Brief Summary:""",
        llm_provider=provider,
        output_key="final_summary",
        model_params={"temperature": 0.5},
        description="Handles short documents"
    )

    # Long document handler - detailed summary with sections
    long_summary_node = PromptNode(
        name="long_summary",
        prompt_template="""This is a longer document. Expand the summary with more detail and structure:

Initial Summary: {initial_summary}

Topics: {topics}

Create a detailed summary with:
1. Overview (2 sentences)
2. Key Points (3-4 bullet points)
3. Conclusion (1 sentence)

Detailed Summary:""",
        llm_provider=provider,
        output_key="final_summary",
        model_params={"temperature": 0.6},
        description="Handles long documents"
    )

    # Create router
    router = RouterNode(
        name="summary_router",
        description="Routes based on document length"
    )

    # Route 1: Short documents (< 500 characters)
    router.add_route(
        condition=LambdaCondition(
            func=lambda ctx: len(ctx.get("document", "")) < 500,
            description="Route for short documents"
        ),
        node=short_summary_node,
        name="short_document_route"
    )

    # Route 2: Long documents (>= 500 characters)
    router.add_route(
        condition=LambdaCondition(
            func=lambda ctx: len(ctx.get("document", "")) >= 500,
            description="Route for long documents"
        ),
        node=long_summary_node,
        name="long_document_route"
    )

    # Set default to long summary
    router.set_default(long_summary_node)

    return router


def create_evaluator_optimizer_pipeline(provider, target_node):
    """
    PATTERN 4: EVALUATOR-OPTIMIZER LOOP

    Wrap a node with an evaluator that assesses output quality and
    applies optimization strategies to improve results. This demonstrates
    iterative refinement for quality assurance.

    Args:
        provider: LLM provider instance
        target_node: The node to evaluate and optimize

    Returns:
        EvaluatorNode: Configured evaluator node
    """
    logger.info("Creating evaluator-optimizer pipeline...")

    # Create composite evaluator with multiple criteria
    evaluator = CompositeEvaluator([
        # Check length is reasonable (50-500 characters)
        (LengthEvaluator(min_length=50, max_length=500, target_length=200), 0.4),
        # Check for required keywords
        (KeywordEvaluator(
            required_keywords=["AI", "technology", "machine learning"],
            case_sensitive=False
        ), 0.3),
        # Check summary doesn't contain error indicators
        (KeywordEvaluator(
            forbidden_keywords=["error", "failed", "unable"],
            case_sensitive=False
        ), 0.3)
    ])

    # Create evaluator node with prompt refinement strategy
    evaluator_node = EvaluatorNode(
        name="optimized_summary",
        target_node=target_node,
        evaluator=evaluator,
        optimization_strategy=PromptRefinementStrategy(),
        threshold=0.7,
        max_iterations=3,
        keep_best=True,
        description="Evaluates and optimizes summary quality"
    )

    return evaluator_node


def create_orchestrator_pipeline(provider):
    """
    PATTERN 5: ORCHESTRATOR-WORKERS

    Create an orchestrator that coordinates multiple workers to perform
    final processing tasks. This demonstrates multi-agent coordination
    with task dependencies.

    Args:
        provider: LLM provider instance

    Returns:
        Orchestrator: Configured orchestrator node
    """
    logger.info("Creating orchestrator pipeline...")

    # Create workers with different capabilities

    # Worker 1: Formatting specialist
    formatter_worker = LLMWorker(
        worker_id="formatter",
        capabilities=["format"],
        llm_provider=provider,
        prompt_templates={
            "format": """Format this summary for presentation with proper markdown:

{summary}

Formatted Output:"""
        },
        model_params={"temperature": 0.3}
    )

    # Worker 2: Keyword extraction specialist
    keyword_worker = LLMWorker(
        worker_id="keyword_extractor",
        capabilities=["extract_keywords"],
        llm_provider=provider,
        prompt_templates={
            "extract_keywords": """Extract 5-7 key terms from this summary that would be useful as tags:

{summary}

Keywords (comma-separated):"""
        },
        model_params={"temperature": 0.4}
    )

    # Worker 3: Title generation specialist
    title_worker = LLMWorker(
        worker_id="title_generator",
        capabilities=["generate_title"],
        llm_provider=provider,
        prompt_templates={
            "generate_title": """Create a compelling, concise title (5-10 words) for this summary:

{summary}

Title:"""
        },
        model_params={"temperature": 0.7}
    )

    # Create orchestrator
    orchestrator = Orchestrator(
        name="final_processing",
        max_concurrent_tasks=3,
        description="Coordinates final processing tasks"
    )

    # Add workers
    orchestrator.add_worker(formatter_worker)
    orchestrator.add_worker(keyword_worker)
    orchestrator.add_worker(title_worker)

    # Define tasks with dependencies
    # Task 1: Format the summary (no dependencies)
    format_task = Task(
        task_id="format_summary",
        task_type="format",
        data={"summary": "{final_summary}"},  # Will be filled at runtime
        dependencies=[]
    )

    # Task 2: Extract keywords (depends on formatting)
    keyword_task = Task(
        task_id="extract_keywords",
        task_type="extract_keywords",
        data={"summary": "{final_summary}"},
        dependencies=[]  # Can run in parallel with formatting
    )

    # Task 3: Generate title (depends on formatting)
    title_task = Task(
        task_id="generate_title",
        task_type="generate_title",
        data={"summary": "{final_summary}"},
        dependencies=[]  # Can run in parallel with others
    )

    orchestrator.add_task(format_task)
    orchestrator.add_task(keyword_task)
    orchestrator.add_task(title_task)

    return orchestrator


def run_complete_pipeline():
    """
    Execute the complete document processing pipeline demonstrating all patterns.

    This function orchestrates the entire workflow:
    1. Initialize provider and context
    2. Run parallel extraction
    3. Chain processing steps
    4. Route based on document characteristics
    5. Evaluate and optimize results
    6. Orchestrate final processing
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Step 1: Setup
        provider = setup_gemini_provider()
        context = WorkflowContext()
        context.set("document", SAMPLE_DOCUMENT)

        logger.info(f"\nProcessing document ({len(SAMPLE_DOCUMENT)} characters)...")

        # Step 2: PARALLEL - Extract features concurrently
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: PARALLEL EXTRACTION")
        logger.info("=" * 80)
        parallel_pipeline = create_parallel_extraction_pipeline(provider)
        parallel_results = parallel_pipeline.execute(context)
        logger.info(f"Parallel extraction completed. Results: {list(parallel_results.keys())}")

        # Update context with parallel results
        for key, value in parallel_results.items():
            context.set(key, value)

        # Step 3: CHAIN - Process results sequentially
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: CHAINED PROCESSING")
        logger.info("=" * 80)
        chain_pipeline = create_chained_processing_pipeline(provider)
        chain_pipeline.execute(context)
        logger.info("Chain processing completed")

        # Step 4: ROUTER - Route based on document characteristics
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: CONDITIONAL ROUTING")
        logger.info("=" * 80)
        router_pipeline = create_routing_pipeline(provider)

        # Wrap router with evaluator-optimizer
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: EVALUATOR-OPTIMIZER LOOP")
        logger.info("=" * 80)
        optimized_router = create_evaluator_optimizer_pipeline(provider, router_pipeline)
        final_summary = optimized_router.execute(context)
        logger.info("Optimized summary generated")

        # Step 5: ORCHESTRATOR - Final processing tasks
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: ORCHESTRATOR-WORKERS COORDINATION")
        logger.info("=" * 80)

        # Update orchestrator tasks with actual summary
        orchestrator = create_orchestrator_pipeline(provider)
        # Update task data with actual summary
        for task in orchestrator.task_queue:
            task.data = {"summary": final_summary}

        orchestrator_results = orchestrator.execute(context)

        # Display final results
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE - FINAL RESULTS")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("DOCUMENT PROCESSING RESULTS")
        print("=" * 80)

        print("\n📊 PARALLEL EXTRACTION RESULTS:")
        print("-" * 80)
        print(f"\nTopics:\n{context.get('topics', 'N/A')}")
        print(f"\nEntities:\n{context.get('entities', 'N/A')}")
        print(f"\nSentiment:\n{context.get('sentiment', 'N/A')}")

        print("\n🔗 CHAINED PROCESSING RESULTS:")
        print("-" * 80)
        print(f"\nIntegrated Analysis:\n{context.get('integrated_analysis', 'N/A')}")
        print(f"\nInitial Summary:\n{context.get('initial_summary', 'N/A')}")

        print("\n🎯 ROUTED & OPTIMIZED SUMMARY:")
        print("-" * 80)
        print(f"\n{final_summary}")

        print("\n🤝 ORCHESTRATED FINAL PROCESSING:")
        print("-" * 80)
        completed = orchestrator_results.get("completed", {})
        print(f"\nTitle:\n{completed.get('generate_title', 'N/A')}")
        print(f"\nKeywords:\n{completed.get('extract_keywords', 'N/A')}")
        print(f"\nFormatted Output:\n{completed.get('format_summary', 'N/A')}")

        # Display execution metrics
        duration = (datetime.now() - start_time).total_seconds()
        print("\n📈 EXECUTION METRICS:")
        print("-" * 80)
        print(f"Total Duration: {duration:.2f}s")
        print(f"Model Calls: {context.metadata.get('model_calls', 0)}")
        print(f"Routing Decisions: {len(context.metadata.get('routing_decisions', []))}")

        # Display orchestrator summary
        summary = orchestrator_results.get("summary", {})
        print("\nOrchestrator Tasks:")
        print(f"  - Total: {summary.get('total_tasks', 0)}")
        print(f"  - Completed: {summary.get('completed', 0)}")
        print(f"  - Failed: {summary.get('failed', 0)}")
        print(f"  - Success Rate: {summary.get('success_rate', 0):.1%}")

        print("\n" + "=" * 80)
        logger.info(f"Pipeline completed successfully in {duration:.2f}s")

        return context

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the example."""
    try:
        run_complete_pipeline()
        logger.info("\n✅ Example completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n❌ Example failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
