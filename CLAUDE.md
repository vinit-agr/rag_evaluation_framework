# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Evaluation Framework for systematically evaluating retrieval pipelines with Langsmith integration. Evaluates chunking strategies, embedding models, retrieval parameters, and re-rankers.

## Development Commands

```bash
# Install dependencies
uv pip install -e .

# Python version: 3.12 (see .python-version)
```

## Architecture

### Entry Point
`Evaluation` class in `rag_evaluation_framework/evaluation/base_eval.py` - orchestrates the pipeline:
```python
from rag_evaluation_framework import Evaluation

evaluation = Evaluation(langsmith_dataset_name="dataset", kb_data_path="./kb")
results = evaluation.run(chunker=..., embedder=..., vector_store=..., k=5, reranker=...)
```

### Component Abstractions (all in `evaluation/` subdirectories)

Each component has a `base.py` defining an abstract interface:

| Component | Base Class | Key Method | Location |
|-----------|------------|------------|----------|
| Chunker | `Chunker` | `chunk(text) -> List[str]` | `chunker/base.py` |
| Embedder | `Embedder` | `embed_docs(docs) -> List[List[float]]` | `embedder/base.py` |
| VectorStore | `VectorStore` | `search(query, k) -> List[str]` | `vector_store/base.py` |
| Reranker | `Reranker` | `rerank(docs, query, k) -> List[str]` | `reranker/base.py` |

### Metrics System

All metrics extend `Metrics` base class (`evaluation/metrics/base.py`) with three required methods:
- `calculate(retrieved_chunk_ids, ground_truth_chunk_ids) -> float`
- `extract_ground_truth_chunks_ids(example) -> List[str]`
- `extract_retrieved_chunks_ids(run) -> List[str]`

Metrics auto-convert to Langsmith evaluators via `to_langsmith_evaluator()`. Use `get_langsmith_evaluators()` from `evaluation/utils.py` to batch convert.

Built-in: `ChunkLevelRecall`, `TokenLevelRecall`

### Langsmith Integration

The framework wraps Langsmith's evaluation API. Metrics produce `EvaluationResult` objects compatible with Langsmith's `evaluate()` function. Dataset examples should contain ground truth chunk IDs in outputs.
