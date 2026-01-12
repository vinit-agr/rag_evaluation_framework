# Metrics

Evaluation metrics for RAG (Retrieval-Augmented Generation) systems with seamless Langsmith integration.

## Overview

The metrics system provides a flexible way to evaluate RAG retrieval performance. All metrics automatically integrate with Langsmith's evaluation framework, allowing you to track and compare metrics across different experiments.

## Quick Start

### Using Default Metrics

By default, the evaluation framework uses built-in metrics. You don't need to specify them:

```python
from rag_evaluation_framework import Evaluation

evaluator = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

# Uses default metrics automatically
results = evaluator.run(k=5)
```

### Using Custom Metrics

You can pass custom metrics to evaluate specific aspects of your RAG system:

```python
from rag_evaluation_framework import Evaluation
from rag_evaluation_framework.evaluation.metrics import ChunkLevelRecall
from rag_evaluation_framework.evaluation.metrics.base import Metrics

# Use built-in metric
results = evaluator.run(
    k=5,
    metrics={"recall": ChunkLevelRecall()}
)

# Or create your own custom metric
class MyCustomMetric(Metrics):
    def calculate(self, retrieved_chunk_ids, ground_truth_chunk_ids):
        # Your custom calculation logic
        return 0.95
    
    def extract_ground_truth_chunks_ids(self, example):
        # Extract ground truth from Langsmith Example
        return example.outputs.get("chunk_ids", [])
    
    def extract_retrieved_chunks_ids(self, run):
        # Extract retrieved chunks from Langsmith Run
        return run.outputs if isinstance(run.outputs, list) else []

results = evaluator.run(
    k=5,
    metrics={"custom": MyCustomMetric()}
)
```

## Base Metrics Class

All metrics inherit from the `Metrics` abstract base class, which provides:

### Required Methods

#### `calculate(retrieved_chunk_ids, ground_truth_chunk_ids) -> float`

Calculate the metric score based on retrieved and ground truth chunk IDs.

**Parameters:**
- `retrieved_chunk_ids` (List[str]): List of retrieved chunk IDs
- `ground_truth_chunk_ids` (List[str]): List of ground truth chunk IDs

**Returns:**
- `float`: Metric score (typically between 0.0 and 1.0)

#### `extract_ground_truth_chunks_ids(example) -> List[str]`

Extract ground truth chunk IDs from a Langsmith Example object.

**Parameters:**
- `example` (Optional[Example]): Langsmith Example containing ground truth data

**Returns:**
- `List[str]`: List of ground truth chunk IDs

#### `extract_retrieved_chunks_ids(run) -> List[str]`

Extract retrieved chunk IDs from a Langsmith Run object.

**Parameters:**
- `run` (Run): Langsmith Run containing retrieval results

**Returns:**
- `List[str]`: List of retrieved chunk IDs

### Automatic Langsmith Integration

The base class provides `to_langsmith_evaluator()` method that automatically converts any `Metrics` instance into a Langsmith evaluator function. This means:

- ✅ Any custom metric automatically works with Langsmith
- ✅ Metrics are automatically tracked in Langsmith experiments
- ✅ No manual conversion needed

```python
# This happens automatically when you pass metrics to evaluator.run()
metric = ChunkLevelRecall()
langsmith_evaluator = metric.to_langsmith_evaluator(metric_name="recall", k=5)
# Returns a function compatible with Langsmith's evaluate() API
```

## Built-in Metrics

### ChunkLevelRecall

Measures the proportion of ground truth chunks that were successfully retrieved.

**Formula:** `|retrieved_chunks ∩ ground_truth_chunks| / |ground_truth_chunks|`

**Usage:**
```python
from rag_evaluation_framework.evaluation.metrics import ChunkLevelRecall

metric = ChunkLevelRecall()
results = evaluator.run(
    k=5,
    metrics={"recall": metric}
)
```

**When to use:**
- Measure how well your retrieval system finds relevant chunks
- Higher recall means more relevant chunks are retrieved
- Useful when you want to ensure comprehensive coverage

## Creating Custom Metrics

To create a custom metric, inherit from `Metrics` and implement the three required methods:

```python
from rag_evaluation_framework.evaluation.metrics.base import Metrics
from typing import List, Optional
from langsmith.schemas import Example, Run

class PrecisionMetric(Metrics):
    """Calculate precision: proportion of retrieved chunks that are relevant."""
    
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        if len(retrieved_chunk_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        
        intersection = len(retrieved_set & ground_truth_set)
        return intersection / len(retrieved_set)
    
    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        """Extract ground truth from Langsmith Example."""
        if example is None:
            return []
        return example.outputs.get("chunk_ids", [])
    
    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        """Extract retrieved chunks from Langsmith Run."""
        return run.outputs if isinstance(run.outputs, list) else []
```

### Custom Data Extraction

If your Langsmith dataset uses a different structure, override the extraction methods:

```python
class CustomMetric(Metrics):
    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        # Custom extraction logic for your dataset format
        if example is None:
            return []
        
        # Example: ground truth stored in a different field
        outputs = example.outputs or {}
        return outputs.get("expected_chunks", [])
    
    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        # Custom extraction for your retrieval output format
        if isinstance(run.outputs, dict):
            return run.outputs.get("retrieved_ids", [])
        return []
```

## Using Multiple Metrics

You can evaluate multiple metrics in a single run:

```python
from rag_evaluation_framework.evaluation.metrics import ChunkLevelRecall

recall_metric = ChunkLevelRecall()
precision_metric = PrecisionMetric()  # Your custom metric

results = evaluator.run(
    k=5,
    metrics={
        "recall@5": recall_metric,
        "precision@5": precision_metric,
    }
)
```

All metrics will be calculated and tracked in the same Langsmith experiment.

## Integration with Evaluation

Metrics are automatically converted to Langsmith evaluators when passed to `evaluator.run()`:

```python
# Behind the scenes:
metrics = {"recall": ChunkLevelRecall()}
langsmith_evaluators = get_langsmith_evaluators(metrics, k=5)

# Then used in Langsmith evaluate():
results = evaluate(
    target=retrieval_function,
    data=dataset_name,
    evaluators=langsmith_evaluators,  # Your metrics converted automatically
)
```

## Metric Naming

Metrics are automatically named based on:
1. The key in the metrics dictionary (if provided)
2. The class name (if no key provided)
3. The `k` value (appended as `@k`)

Examples:
- `{"recall": ChunkLevelRecall()}` with `k=5` → `"recall@5"`
- `{ChunkLevelRecall()}` → `"ChunkLevelRecall@5"`

## Best Practices

1. **Consistent Data Format**: Ensure your Langsmith dataset has consistent structure for ground truth
2. **Meaningful Names**: Use descriptive names in the metrics dictionary
3. **Handle Edge Cases**: Always handle empty lists and None values in `calculate()`
4. **Test Extraction**: Verify your `extract_*` methods work with your dataset format
5. **Document Custom Metrics**: Add docstrings explaining what your metric measures

## Related Components

- **[Evaluation](evaluation.md)** - How metrics are used in the evaluation pipeline
- **[Chunker](chunker.md)** - Document chunking strategies
- **[Embedder](embedder.md)** - Embedding model integration
- **[Vector Store](vector_store.md)** - Vector database abstraction
