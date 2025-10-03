# Test Suite

This directory contains comprehensive unit tests for the LLM Workflow Framework.

## Test Coverage

### ✅ Completed Tests (161 tests)

#### `test_base.py` - Base Module Tests (20 tests)
Tests for core abstractions:
- **WorkflowContext** (6 tests): Data storage, metadata tracking, history
- **Node** (4 tests): Abstract base class, validation, concrete implementations
- **LLMProvider** (2 tests): Abstract interface, concrete implementations
- **PromptNode** (8 tests): Initialization, prompt formatting, execution, validation

#### `test_chaining.py` - Chaining Module Tests (14 tests)
Tests for sequential workflow execution:
- Chain initialization and node addition
- Sequential execution and data flow
- Pass-through context mode
- Error handling and propagation
- Validation and metadata tracking

#### `test_routing.py` - Routing Module Tests (31 tests)
Tests for conditional routing:
- **KeywordCondition** (4 tests): Case-sensitive/insensitive matching
- **ThresholdCondition** (6 tests): Numeric comparisons with various operators
- **LambdaCondition** (2 tests): Custom logic evaluation
- **Route** (2 tests): Route creation and representation
- **RouterNode** (17 tests): Route matching, default fallback, metadata tracking

#### `test_exceptions.py` - Exception Module Tests (18 tests)
Tests for custom exception hierarchy:
- **WorkflowException** (2 tests): Base exception
- **NodeExecutionError** (4 tests): Node failure handling
- **RoutingError** (2 tests): Routing failures
- **EvaluationError** (4 tests): Evaluation failures
- **LLMProviderError** (5 tests): API errors with retry tracking
- **TaskExecutionError** (4 tests): Task execution failures
- Exception hierarchy validation (2 tests)

#### `test_parallel.py` - Parallel Module Tests (26 tests) ✅
Tests for parallel execution and merge strategies:
- **ConcatenateMerge** (4 tests): String concatenation with separators
- **ListMerge** (3 tests): List aggregation
- **DictMerge** (4 tests): Dictionary merging with prefixes
- **VotingMerge** (4 tests): Consensus-based selection
- **ParallelNode** (11 tests): Concurrent execution, timeouts, error handling

#### `test_evaluator.py` - Evaluator Module Tests (29 tests) ✅
Tests for evaluator-optimizer pattern:
- **LengthEvaluator** (6 tests): Length-based scoring and feedback
- **KeywordEvaluator** (5 tests): Keyword presence/absence evaluation
- **LLMEvaluator** (4 tests): LLM-based quality assessment
- **CompositeEvaluator** (3 tests): Weighted combination of evaluators
- **RetryStrategy** (1 test): Simple retry optimization
- **PromptRefinementStrategy** (2 tests): Prompt modification with feedback
- **TemperatureAdjustmentStrategy** (2 tests): Temperature parameter tuning
- **EvaluatorNode** (6 tests): Evaluation loop, threshold checking, best result selection

#### `test_gemini_provider.py` - Gemini Provider Tests (28 tests) ✅
Tests for Gemini API provider (requires `google-genai` package):
- **Initialization** (4 tests): Provider setup, API key validation, error handling
- **Generation** (4 tests): Single prompt generation, parameters, empty responses
- **Batch Generation** (3 tests): Multiple prompts, partial failures
- **Configuration** (3 tests): Parameter mapping, config building
- **Retry Logic** (4 tests): Retryable errors, exponential backoff, max retries
- **Utility Methods** (3 tests): Model info, error detection, string representation
- **Parameter Handling** (7 tests): Default params, overrides, mappings

**Note**: `test_gemini_provider.py` requires the `google-genai` package to be installed. Install with:
```bash
uv pip install google-genai
```

### 🚧 Pending Tests

The following test files still need to be created:

- `test_orchestrator.py` - Orchestrator-workers pattern
- `test_integration.py` - End-to-end workflow tests

## Running Tests

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
python -m pytest tests/test_base.py -v
```

### Run Specific Test Class

```bash
python -m pytest tests/test_base.py::TestWorkflowContext -v
```

### Run Specific Test

```bash
python -m pytest tests/test_base.py::TestWorkflowContext::test_initialization -v
```

### Run with Coverage (requires pytest-cov)

```bash
# Install coverage tool
uv pip install pytest-cov

# Run with coverage
python -m pytest tests/ -v --cov=framework --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Structure

Each test file follows this structure:

```python
"""
Unit tests for [module_name] module.
"""

import pytest
from unittest.mock import Mock
from framework.[module] import [Classes]


class Test[ClassName]:
    """Tests for [ClassName] class."""
    
    def test_[feature](self):
        """Test [specific behavior]."""
        # Arrange
        # Act
        # Assert
```

## Testing Patterns

### Mocking LLM Providers

```python
from unittest.mock import Mock
from framework.base import LLMProvider

mock_provider = Mock(spec=LLMProvider)
mock_provider.generate.return_value = "Mocked response"
```

### Creating Mock Nodes

```python
from framework.base import Node, WorkflowContext

class MockNode(Node):
    def __init__(self, name, return_value="result"):
        super().__init__(name)
        self.return_value = return_value
        self.executed = False
    
    def execute(self, context):
        self.executed = True
        return self.return_value
```

### Testing Exceptions

```python
import pytest
from framework.exceptions import NodeExecutionError

def test_error_handling():
    with pytest.raises(NodeExecutionError) as exc_info:
        # Code that should raise exception
        raise NodeExecutionError("Test error")
    
    assert "Test error" in str(exc_info.value)
```

### Testing Context Updates

```python
from framework.base import WorkflowContext

def test_context_update():
    context = WorkflowContext()
    context.set("key", "value")
    
    assert context.get("key") == "value"
    assert len(context.get_history()) > 0
```

## Test Guidelines

### Writing Good Tests

1. **One assertion per test** (when possible)
2. **Clear test names** that describe what is being tested
3. **Arrange-Act-Assert** pattern
4. **Test both success and failure cases**
5. **Use descriptive docstrings**

### Test Naming Convention

```python
def test_[method_name]_[scenario]_[expected_result]():
    """Test that [method] [does something] when [condition]."""
```

Examples:
- `test_execute_sequential_returns_last_result()`
- `test_validate_empty_chain_returns_false()`
- `test_format_prompt_missing_variable_raises_error()`

### What to Test

✅ **Do test:**
- Public API methods
- Edge cases and boundary conditions
- Error handling
- State changes
- Return values
- Side effects (context updates, metadata)

❌ **Don't test:**
- Private methods (unless complex)
- Third-party library internals
- Simple getters/setters without logic

## Continuous Integration

Tests should be run automatically on:
- Every commit
- Pull requests
- Before releases

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.13'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]"
      - name: Run tests
        run: python -m pytest tests/ -v --cov=framework
```

## Test Metrics

Current test metrics:
- **Total Tests**: 138 (+ 28 pending google-genai install)
- **Pass Rate**: 100%
- **Execution Time**: ~0.28s
- **Coverage**: TBD (run with --cov flag)

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure all tests pass before committing
3. Aim for >80% code coverage
4. Add integration tests for complex workflows
5. Update this README with new test information

## Troubleshooting

### Import Errors

If you get import errors, ensure the framework is installed:

```bash
uv pip install -e .
```

### Pytest Not Found

Install pytest:

```bash
uv pip install pytest
```

### Mock Issues

Ensure you're using `unittest.mock` correctly:

```python
from unittest.mock import Mock, MagicMock, patch
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
