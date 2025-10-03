# Agent Workflow Examples

This directory contains example implementations demonstrating the various workflow patterns supported by the framework.

## Available Examples

### Simple Examples (Recommended for Learning)

Start with these focused examples to learn individual patterns:

#### 1. Simple Chain (`simple_chain.py`)
Demonstrates basic prompt chaining with a 3-step workflow.

```bash
python examples/simple_chain.py
```

#### 2. Simple Routing (`simple_routing.py`)
Shows conditional routing based on keyword detection in feedback.

```bash
python examples/simple_routing.py
```

#### 3. Simple Parallel (`simple_parallel.py`)
Illustrates concurrent execution of independent tasks.

```bash
python examples/simple_parallel.py
```

#### 4. Simple Evaluator (`simple_evaluator.py`)
Demonstrates quality evaluation and iterative optimization.

```bash
python examples/simple_evaluator.py
```

### Comprehensive Example

#### Document Processing Pipeline (`document_pipeline.py`)

A comprehensive example that demonstrates **all five workflow patterns** in a single, realistic document processing pipeline:

1. **Parallel Execution**: Extracts topics, entities, and sentiment concurrently
2. **Prompt Chaining**: Processes results sequentially through multiple stages
3. **Conditional Routing**: Routes to different summarization strategies based on document length
4. **Evaluator-Optimizer Loop**: Iteratively improves summary quality
5. **Orchestrator-Workers**: Coordinates final processing tasks (formatting, keywords, title generation)

#### Running the Example

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Run the example
python examples/document_pipeline.py
```

#### Expected Output

The pipeline will:
1. Extract document features in parallel (topics, entities, sentiment)
2. Merge and process these features through a chain
3. Route to appropriate summarization based on document length
4. Optimize the summary through evaluation iterations
5. Coordinate final processing tasks to generate title, keywords, and formatted output

You'll see detailed logging showing each step's execution, along with final results including:
- Extracted topics, entities, and sentiment analysis
- Integrated analysis and initial summary
- Optimized final summary
- Generated title and keywords
- Formatted output
- Execution metrics (duration, model calls, success rates)

## Pattern-Specific Examples

### 1. Parallel Execution

```python
from framework.parallel import ParallelNode, DictMerge
from framework.base import PromptNode

# Create nodes for parallel execution
node1 = PromptNode(name="task1", prompt_template="...", llm_provider=provider)
node2 = PromptNode(name="task2", prompt_template="...", llm_provider=provider)
node3 = PromptNode(name="task3", prompt_template="...", llm_provider=provider)

# Create parallel node
parallel = ParallelNode(
    name="parallel_tasks",
    merge_strategy=DictMerge(),
    max_workers=3
)
parallel.add_node(node1).add_node(node2).add_node(node3)

# Execute
context = WorkflowContext()
context.set("input", "data")
results = parallel.execute(context)
```

### 2. Prompt Chaining

```python
from framework.chaining import ChainWorkflow

# Create chain
chain = ChainWorkflow(name="pipeline", pass_through_context=True)
chain.add_node(step1).add_node(step2).add_node(step3)

# Execute - output of each step becomes input to next
result = chain.execute(context)
```

### 3. Conditional Routing

```python
from framework.routing import RouterNode, ThresholdCondition, KeywordCondition

# Create router
router = RouterNode(name="classifier")

# Add routes with conditions
router.add_route(
    condition=KeywordCondition(key="text", keywords=["urgent"]),
    node=urgent_handler,
    name="urgent_route"
)
router.add_route(
    condition=ThresholdCondition(key="score", threshold=0.8, operator=">="),
    node=high_priority_handler,
    name="high_priority"
)
router.set_default(default_handler)

# Execute - routes to appropriate node based on conditions
result = router.execute(context)
```

### 4. Evaluator-Optimizer Loop

```python
from framework.evaluator import (
    EvaluatorNode,
    LengthEvaluator,
    CompositeEvaluator,
    PromptRefinementStrategy
)

# Create evaluator
evaluator = CompositeEvaluator([
    (LengthEvaluator(min_length=50, max_length=200), 0.5),
    (KeywordEvaluator(required_keywords=["summary"]), 0.5)
])

# Wrap node with evaluator
optimized_node = EvaluatorNode(
    name="optimized",
    target_node=base_node,
    evaluator=evaluator,
    optimization_strategy=PromptRefinementStrategy(),
    threshold=0.7,
    max_iterations=3
)

# Execute - automatically retries and optimizes until threshold met
result = optimized_node.execute(context)
```

### 5. Orchestrator-Workers

```python
from framework.orchestrator import Orchestrator, Task, LLMWorker

# Create workers
worker1 = LLMWorker(
    worker_id="summarizer",
    capabilities=["summarize"],
    llm_provider=provider,
    prompt_templates={"summarize": "Summarize: {text}"}
)

worker2 = LLMWorker(
    worker_id="analyzer",
    capabilities=["analyze"],
    llm_provider=provider,
    prompt_templates={"analyze": "Analyze: {text}"}
)

# Create orchestrator
orchestrator = Orchestrator(name="coordinator", max_concurrent_tasks=2)
orchestrator.add_worker(worker1).add_worker(worker2)

# Add tasks with dependencies
task1 = Task(task_id="t1", task_type="summarize", data={"text": "..."})
task2 = Task(task_id="t2", task_type="analyze", data={"text": "..."}, dependencies=["t1"])

orchestrator.add_task(task1).add_task(task2)

# Execute - orchestrator coordinates workers and manages dependencies
results = orchestrator.execute(context)
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Parameters

You can customize LLM behavior by passing parameters to PromptNode:

```python
node = PromptNode(
    name="custom",
    prompt_template="...",
    llm_provider=provider,
    model_params={
        "temperature": 0.7,      # Creativity (0.0-1.0)
        "max_output_tokens": 512, # Response length
        "top_p": 0.9,            # Nucleus sampling
        "top_k": 40              # Top-k sampling
    }
)
```

## Tips for Building Your Own Workflows

1. **Start Simple**: Begin with a single pattern and gradually combine them
2. **Use Logging**: Enable DEBUG logging to understand execution flow
3. **Handle Errors**: Wrap workflows in try-except blocks for robust error handling
4. **Monitor Metrics**: Check context.metadata for execution statistics
5. **Iterate**: Use the evaluator-optimizer pattern for quality-critical outputs
6. **Parallelize**: Use parallel execution for independent tasks to reduce latency
7. **Test Incrementally**: Test each component before combining into complex workflows

## Common Patterns

### Sequential Processing with Quality Check

```python
# Chain with evaluator wrapper
chain = ChainWorkflow(name="pipeline")
chain.add_node(step1).add_node(step2)

optimized_chain = EvaluatorNode(
    name="quality_checked_pipeline",
    target_node=chain,
    evaluator=your_evaluator,
    optimization_strategy=RetryStrategy(),
    threshold=0.8
)
```

### Parallel Processing with Routing

```python
# Extract features in parallel, then route based on results
parallel = ParallelNode(name="extraction", merge_strategy=DictMerge())
parallel.add_node(extractor1).add_node(extractor2)

router = RouterNode(name="classifier")
router.add_route(condition=..., node=handler1, name="route1")
router.add_route(condition=..., node=handler2, name="route2")

# Execute in sequence
context = WorkflowContext()
parallel.execute(context)
router.execute(context)
```

### Multi-Stage Pipeline with All Patterns

See `main.py` for a complete example combining all patterns in a realistic workflow.

## Troubleshooting

### API Key Issues

```
ValueError: GEMINI_API_KEY environment variable not set
```

**Solution**: Set your API key before running:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Rate Limiting

If you encounter rate limiting errors, the framework automatically retries with exponential backoff. You can adjust retry settings:

```python
provider = GeminiProvider(
    api_key=api_key,
    max_retries=5,
    initial_retry_delay=2.0
)
```

### Timeout Errors

For long-running parallel operations, increase the timeout:

```python
parallel = ParallelNode(
    name="tasks",
    timeout=60.0  # 60 seconds
)
```

### Memory Issues

For large documents or many parallel tasks, consider:
- Processing in smaller chunks
- Reducing max_workers in ParallelNode
- Limiting max_concurrent_tasks in Orchestrator

## Additional Resources

- [Framework Documentation](../README.md)
- [API Reference](../framework/)
- [Design Document](../.kiro/specs/llm-workflow-framework/design.md)
- [Requirements](../.kiro/specs/llm-workflow-framework/requirements.md)
