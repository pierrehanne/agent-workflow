# LLM Workflow Framework

A clean, modular Python framework for building sophisticated LLM-based pipelines using various workflow patterns. Built with Python 3.13 and integrated with Google Gemini API.

## Features

- **Prompt Chaining**: Define sequential workflow steps where outputs feed into subsequent inputs
- **Conditional Routing**: Route requests to different prompts or models based on input characteristics
- **Parallelization**: Execute multiple LLM calls concurrently and merge results
- **Orchestrator-Workers**: Coordinate multiple worker agents for complex multi-agent systems
- **Evaluator-Optimizer**: Evaluate outputs and optimize through retries or selection
- **Extensible Architecture**: Clean abstractions following SOLID principles
- **Type-Safe**: Comprehensive type hints throughout the codebase
- **Well-Documented**: Detailed docstrings and usage examples

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. First, install uv:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

Then install the framework:

```bash
# Clone the repository
git clone <repository-url>
cd llm-workflow-framework

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development (includes testing tools)
uv pip install -e ".[dev]"
```

## Configuration

Set up your Google Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

You can obtain an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Quick Start

```python
from framework.base import WorkflowContext, PromptNode
from framework.providers.gemini import GeminiProvider
from framework.chaining import ChainWorkflow

# Initialize the LLM provider
provider = GeminiProvider(api_key="your_api_key")

# Create a simple chain workflow
context = WorkflowContext()
context.set("input", "Explain quantum computing")

# Define nodes
node1 = PromptNode(
    name="simplify",
    prompt_template="Simplify this topic: {input}",
    llm_provider=provider,
    output_key="simplified"
)

node2 = PromptNode(
    name="expand",
    prompt_template="Expand on this: {simplified}",
    llm_provider=provider,
    output_key="final"
)

# Create and execute chain
chain = ChainWorkflow(name="explain_chain")
chain.add_node(node1).add_node(node2)

result = chain.execute(context)
print(result)
```

## Workflow Patterns

### 1. Prompt Chaining

Execute prompts sequentially, passing outputs to inputs:

```python
from framework.chaining import ChainWorkflow

chain = ChainWorkflow(name="my_chain")
chain.add_node(step1).add_node(step2).add_node(step3)
result = chain.execute(context)
```

### 2. Conditional Routing

Route execution based on conditions:

```python
from framework.routing import RouterNode, KeywordCondition

router = RouterNode(name="classifier")
router.add_route(
    KeywordCondition(keywords=["technical"]),
    technical_handler,
    name="technical_route"
)
router.set_default(general_handler)
result = router.execute(context)
```

### 3. Parallel Execution

Run multiple nodes concurrently:

```python
from framework.parallel import ParallelNode, ListMerge

parallel = ParallelNode(
    name="parallel_analysis",
    merge_strategy=ListMerge()
)
parallel.add_node(analyzer1).add_node(analyzer2).add_node(analyzer3)
results = parallel.execute(context)
```

### 4. Orchestrator-Workers

Coordinate multiple worker agents:

```python
from framework.orchestrator import Orchestrator, LLMWorker, Task

orchestrator = Orchestrator(name="coordinator")
orchestrator.add_worker(LLMWorker(worker_id="w1", llm_provider=provider))
orchestrator.add_task(Task(task_id="t1", task_type="summarize", data={...}))
results = orchestrator.execute(context)
```

### 5. Evaluator-Optimizer

Evaluate and optimize outputs:

```python
from framework.evaluator import EvaluatorNode, LLMEvaluator, RetryStrategy

evaluator = EvaluatorNode(
    name="optimizer",
    target_node=generator,
    evaluator=LLMEvaluator(provider),
    optimization_strategy=RetryStrategy(),
    threshold=0.7,
    max_iterations=3
)
best_result = evaluator.execute(context)
```

## Examples

### Comprehensive Document Processing Pipeline

The `main.py` file contains a complete example demonstrating **all five workflow patterns** in a realistic document processing pipeline:

1. **Parallel Extraction**: Extracts topics, entities, and sentiment concurrently
2. **Chained Processing**: Merges and processes results sequentially
3. **Conditional Routing**: Routes to different summarization strategies based on document length
4. **Evaluator-Optimizer**: Iteratively improves summary quality
5. **Orchestrator-Workers**: Coordinates final processing tasks (formatting, keywords, title)

#### Running the Example

```bash
# Set your API key
export GEMINI_API_KEY="your_api_key_here"

# Run the example
python main.py
```

#### Expected Output

The pipeline processes a sample document about AI and machine learning, producing:

- **Parallel Extraction Results**: Topics, named entities, and sentiment analysis
- **Integrated Analysis**: Merged insights from parallel extraction
- **Initial Summary**: First-pass document summary
- **Optimized Summary**: Quality-improved summary after evaluation iterations
- **Final Processing**: Generated title, keywords, and formatted output
- **Execution Metrics**: Duration, model calls, success rates

Example output:
```
📊 PARALLEL EXTRACTION RESULTS:
Topics:
- Artificial Intelligence and Machine Learning
- Deep Learning and Neural Networks
- AI Applications and Industry Impact
...

🎯 ROUTED & OPTIMIZED SUMMARY:
The rapid advancement of AI and ML is transforming industries globally...

🤝 ORCHESTRATED FINAL PROCESSING:
Title: AI and Machine Learning: Transforming the Future of Technology
Keywords: artificial intelligence, machine learning, deep learning, automation, ethics
...

📈 EXECUTION METRICS:
Total Duration: 45.23s
Model Calls: 12
Success Rate: 100%
```

For more examples and pattern-specific code snippets, see the [examples/README.md](examples/README.md) file.

## Architecture

The framework follows a layered architecture with clear separation of concerns:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                        │
│                      (main.py)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Workflow Components                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Chain   │  │  Router  │  │ Parallel │  │Evaluator │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Orchestrator + Workers                    │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Base Abstractions                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Node       │  │ LLMProvider  │  │   Context    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Provider Layer                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         GeminiProvider (google-genai)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

```
framework/
├── base.py              # Core abstractions (Node, Context, LLMProvider)
├── chaining.py          # Sequential workflow execution
├── routing.py           # Conditional routing logic
├── parallel.py          # Concurrent execution
├── orchestrator.py      # Multi-agent coordination
├── evaluator.py         # Result evaluation and optimization
├── exceptions.py        # Custom exception hierarchy
├── logging_config.py    # Logging configuration
└── providers/
    └── gemini.py        # Google Gemini integration
```

### Design Principles

- **Separation of Concerns**: Each workflow pattern is isolated in its own module
- **Dependency Inversion**: Components depend on abstractions, not concrete implementations
- **Open/Closed Principle**: Framework is open for extension but closed for modification
- **Single Responsibility**: Each class has one clear purpose
- **Composition Over Inheritance**: Workflows are composed of reusable components

### Core Abstractions

**Node**: Base class for all workflow components. Every workflow element (chains, routers, parallel executors) extends Node.

**WorkflowContext**: Maintains state and data throughout workflow execution, including intermediate results, metadata, and execution history.

**LLMProvider**: Abstract interface for LLM backends, allowing easy integration of different providers (currently supports Gemini, extensible to OpenAI, Anthropic, etc.).

## Development

Run tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=framework --cov-report=html
```

Format code:

```bash
ruff check --fix .
```

## Requirements

- Python 3.13+
- google-genai == 1.41.0

## Areas for Contribution

- **New LLM Providers**: Add support for OpenAI, Anthropic, Cohere, etc.
- **Additional Merge Strategies**: Implement new ways to combine parallel results
- **Routing Conditions**: Create new condition types for routing logic
- **Optimization Strategies**: Add new approaches for result optimization
- **Examples**: Contribute real-world use case examples
- **Documentation**: Improve guides, tutorials, and API documentation
- **Performance**: Optimize execution speed and resource usage
- **Testing**: Increase test coverage and add edge case tests
