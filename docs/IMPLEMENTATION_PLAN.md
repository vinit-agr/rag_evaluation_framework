# RAG Evaluation Framework - Implementation Plan

## Executive Summary

This document outlines the implementation plan for completing the RAG Evaluation Framework, inspired by the [Chroma Chunking Evaluation Research](https://research.trychroma.com/evaluating-chunking) and the [chunking_evaluation library](https://github.com/brandonstarxel/chunking_evaluation). The framework aims to provide comprehensive RAG pipeline evaluation beyond just chunking, with full Langsmith integration.

---

## Current State Analysis

### What's Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| `Metrics` base class | **Complete** | Abstract base with Langsmith integration via `to_langsmith_evaluator()` |
| `ChunkLevelRecall` | **Partial** | `calculate()` implemented; missing `extract_*()` methods |
| `TokenLevelRecall` | **Empty** | Placeholder file only |
| `Chunker` base class | **Complete** | Abstract interface defined |
| `Embedder` base class | **Complete** | Abstract interface defined |
| `VectorStore` base class | **Complete** | Abstract interface defined |
| `Reranker` base class | **Complete** | Abstract interface defined |
| `ChromaVectorStore` | **Stubbed** | Methods return empty lists |
| `Evaluation` class | **Partial** | Init + validation only; `run()` is incomplete |
| `EvaluationConfig` | **Complete** | Pydantic model for config |
| `get_langsmith_evaluators` | **Complete** | Utility for batch metric conversion |

### What's Missing

1. **Core Evaluation Pipeline** - The `Evaluation.run()` method needs full implementation
2. **Token-Level Metrics** - TokenLevelRecall, TokenLevelPrecision, TokenLevelIoU, PrecisionOmega
3. **Chunk-Level Metrics** - ChunkLevelPrecision, ChunkLevelF1
4. **Concrete Component Implementations** - Working chunkers, embedders, vector stores
5. **Synthetic Data Generation** - `synthetic_datagen/` module is empty
6. **Hyperparameter Sweep** - `sweep()` method documented but not implemented

### Code Quality Issues to Fix

1. **Type hint error** in `ChunkLevelRecall.calculate()` - line 5 has `ground_truth_chunk_ids: str` but should be `List[str]`
2. **Missing abstract methods** in `ChunkLevelRecall` - needs `extract_ground_truth_chunks_ids()` and `extract_retrieved_chunks_ids()`
3. **Empty `__init__.py` files** - Should export public interfaces

---

## Research Foundation: Chroma Chunking Evaluation

### Token-Level Metrics (from Chroma Research)

The Chroma team's research introduces token-level evaluation metrics that provide more granular insight than chunk-level metrics:

**Definitions:**
- `t_e` = set of tokens in all relevant excerpts (ground truth)
- `t_r` = set of tokens in retrieved chunks

**Formulas:**

```
Recall_q(C) = |t_e ∩ t_r| / |t_e|
```
Measures what fraction of relevant tokens are successfully retrieved.

```
Precision_q(C) = |t_e ∩ t_r| / |t_r|
```
Measures what fraction of retrieved tokens are actually relevant.

```
IoU_q(C) = |t_e ∩ t_r| / (|t_e| + |t_r| - |t_e ∩ t_r|)
```
Jaccard similarity - accounts for both missed relevant tokens and irrelevant retrieved tokens.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall.

### Key Insights from Research

1. **Chunk size matters** - 200-token chunks with zero overlap performed well
2. **Overlap penalizes IoU** - Redundant tokens hurt efficiency metrics
3. **Semantic chunkers need consistent embeddings** - Use same embedding model for chunking and retrieval
4. **Token efficiency matters for LLMs** - Irrelevant tokens waste compute in downstream processing

---

## Implementation Phases

### Phase 1: Foundation Fixes & Core Metrics

**Priority: HIGH**

#### 1.1 Fix Existing Code Issues

```python
# Fix ChunkLevelRecall type hint and add missing methods
class ChunkLevelRecall(Metrics):
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        # ... existing implementation

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        if example is None:
            return []
        return example.outputs.get("chunk_ids", [])

    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        if run.outputs is None:
            return []
        return run.outputs.get("retrieved_chunk_ids", [])
```

#### 1.2 Implement ChunkLevelPrecision

```python
class ChunkLevelPrecision(Metrics):
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        if len(retrieved_chunk_ids) == 0:
            return 0.0
        retrieved_set = set(retrieved_chunk_ids)
        ground_truth_set = set(ground_truth_chunk_ids)
        return len(retrieved_set & ground_truth_set) / len(retrieved_set)
```

#### 1.3 Implement Token-Level Metrics

**TokenLevelRecall:**
```python
class TokenLevelRecall(Metrics):
    def __init__(self, tokenizer: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer)

    def calculate(self, retrieved_chunks: List[str], ground_truth_excerpts: List[str]) -> float:
        retrieved_tokens = set()
        for chunk in retrieved_chunks:
            retrieved_tokens.update(self.tokenizer.encode(chunk))

        ground_truth_tokens = set()
        for excerpt in ground_truth_excerpts:
            ground_truth_tokens.update(self.tokenizer.encode(excerpt))

        if len(ground_truth_tokens) == 0:
            return 0.0

        return len(retrieved_tokens & ground_truth_tokens) / len(ground_truth_tokens)
```

**TokenLevelPrecision, TokenLevelIoU, TokenLevelF1** - Similar pattern.

**Design Decision Needed:** Should token-level metrics work with:
- A) Chunk IDs (requires chunk content lookup)
- B) Actual text content directly
- C) Both via configuration

### Phase 2: Core Evaluation Pipeline

**Priority: HIGH**

#### 2.1 Complete Evaluation.run() Method

```python
def run(
    self,
    chunker: Optional[Chunker] = None,
    embedder: Optional[Embedder] = None,
    vector_store: Optional[VectorStore] = None,
    k: int = 5,
    reranker: Optional[Reranker] = None,
    metrics: Optional[Dict[str, Metrics]] = None,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResults:
    # 1. Load and chunk knowledge base documents
    documents = self._load_kb_documents()
    chunks = self._chunk_documents(documents, chunker)

    # 2. Embed chunks and populate vector store
    self._populate_vector_store(chunks, embedder, vector_store)

    # 3. Create retrieval function for Langsmith
    def retrieve(query: str) -> Dict:
        results = vector_store.search(query, k)
        if reranker:
            results = reranker.rerank(results, query, k)
        return {"retrieved_chunk_ids": results}

    # 4. Convert metrics to Langsmith evaluators
    evaluators = get_langsmith_evaluators(metrics or self._default_metrics(), k)

    # 5. Run Langsmith evaluation
    from langsmith import evaluate
    results = evaluate(
        target=retrieve,
        data=self.langsmith_dataset_name,
        evaluators=evaluators,
        experiment_prefix=config.experiment_name if config else None,
    )

    return EvaluationResults(results)
```

#### 2.2 Define EvaluationResults Class

```python
@dataclass
class EvaluationResults:
    raw_results: Any  # Langsmith results object
    metrics_summary: Dict[str, float]
    experiment_url: str
    config: EvaluationConfig

    def to_dataframe(self) -> pd.DataFrame: ...
    def compare(self, other: 'EvaluationResults') -> ComparisonReport: ...
```

### Phase 3: Concrete Component Implementations

**Priority: MEDIUM**

#### 3.1 Chunker Implementations

| Implementation | Description | Dependencies |
|----------------|-------------|--------------|
| `RecursiveCharacterChunker` | LangChain-style recursive splitting | None |
| `FixedTokenChunker` | Fixed token-size chunks | tiktoken |
| `SemanticChunker` | Embedding-based semantic boundaries | Embedder |
| `SentenceChunker` | Sentence boundary chunking | nltk/spacy |

```python
class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0,
                 separators: List[str] = ["\n\n", "\n", ".", "?", "!", " "]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk(self, text: str) -> List[str]:
        # Implementation
```

#### 3.2 Embedder Implementations

| Implementation | Description | Dependencies |
|----------------|-------------|--------------|
| `OpenAIEmbedder` | OpenAI text-embedding models | openai |
| `SentenceTransformerEmbedder` | HuggingFace sentence-transformers | sentence-transformers |
| `CohereEmbedder` | Cohere embed API | cohere |

```python
class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()

    def embed_docs(self, docs: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=docs, model=self.model)
        return [e.embedding for e in response.data]
```

#### 3.3 VectorStore Implementations

| Implementation | Description | Dependencies |
|----------------|-------------|--------------|
| `ChromaVectorStore` | ChromaDB integration | chromadb |
| `QdrantVectorStore` | Qdrant integration | qdrant-client |
| `InMemoryVectorStore` | Simple numpy-based store | numpy |

#### 3.4 Reranker Implementations

| Implementation | Description | Dependencies |
|----------------|-------------|--------------|
| `CohereReranker` | Cohere Rerank API | cohere |
| `CrossEncoderReranker` | HuggingFace cross-encoder | sentence-transformers |

### Phase 4: Hyperparameter Sweep

**Priority: MEDIUM**

#### 4.1 SweepConfig Class

```python
@dataclass
class SweepConfig:
    chunkers: List[Chunker]
    embedders: List[Embedder]
    vector_stores: Optional[List[VectorStore]] = None
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    rerankers: Optional[List[Optional[Reranker]]] = None
    metrics: Optional[Dict[str, Metrics]] = None
```

#### 4.2 Evaluation.sweep() Method

```python
def sweep(self, sweep_config: SweepConfig) -> SweepResults:
    results = []
    for chunker, embedder, k, reranker in itertools.product(
        sweep_config.chunkers,
        sweep_config.embedders,
        sweep_config.k_values,
        sweep_config.rerankers or [None],
    ):
        result = self.run(
            chunker=chunker,
            embedder=embedder,
            k=k,
            reranker=reranker,
            metrics=sweep_config.metrics,
        )
        results.append(result)

    return SweepResults(results)
```

### Phase 5: Synthetic Data Generation

**Priority: LOW-MEDIUM**

#### 5.1 Query-Excerpt Generation Pipeline

Based on Chroma research methodology:

```python
class SyntheticDataGenerator:
    def __init__(self, llm_client, embedding_model: Embedder):
        self.llm = llm_client
        self.embedder = embedding_model

    def generate_from_corpus(
        self,
        documents: List[str],
        queries_per_doc: int = 5,
        relevance_threshold: float = 0.40,
        dedup_threshold: float = 0.70,
    ) -> Dataset:
        # 1. Generate synthetic queries from documents
        # 2. Generate relevant excerpts for each query
        # 3. Filter by relevance (cosine similarity)
        # 4. Deduplicate similar queries
        # 5. Return Langsmith-compatible dataset
```

#### 5.2 Dataset Export to Langsmith

```python
def export_to_langsmith(self, dataset: Dataset, name: str) -> str:
    from langsmith import Client
    client = Client()
    ls_dataset = client.create_dataset(name)
    for example in dataset:
        client.create_example(
            inputs={"query": example.query},
            outputs={"chunk_ids": example.relevant_chunk_ids, "excerpts": example.excerpts},
            dataset_id=ls_dataset.id,
        )
    return ls_dataset.id
```

### Phase 6: Advanced Metrics

**Priority: LOW**

#### 6.1 Additional Metrics to Consider

| Metric | Description | Formula |
|--------|-------------|---------|
| `MRR` | Mean Reciprocal Rank | 1/rank of first relevant result |
| `NDCG` | Normalized Discounted Cumulative Gain | Standard IR metric |
| `MAP` | Mean Average Precision | Average precision across queries |
| `PrecisionOmega` | Upper bound precision assuming perfect recall | From Chroma research |

#### 6.2 Metric Groups

```python
# Convenience groupings
DEFAULT_CHUNK_METRICS = {
    "chunk_recall": ChunkLevelRecall(),
    "chunk_precision": ChunkLevelPrecision(),
}

DEFAULT_TOKEN_METRICS = {
    "token_recall": TokenLevelRecall(),
    "token_precision": TokenLevelPrecision(),
    "token_iou": TokenLevelIoU(),
}

ALL_METRICS = {**DEFAULT_CHUNK_METRICS, **DEFAULT_TOKEN_METRICS}
```

---

## Architecture Decisions

### Decision 1: Token-Level Metric Input Format

**Options:**
- **A) Work with chunk IDs** - Requires storing chunk content mapping
- **B) Work with text directly** - Simpler but changes the Metrics interface
- **C) Hybrid approach** - Metrics can accept either, with adapters

**Recommendation:** Option C - Create a `ChunkStore` abstraction that metrics can optionally use to resolve chunk IDs to content.

### Decision 2: Chunk ID Generation

**Options:**
- **A) Sequential IDs** - Simple but not stable across runs
- **B) Content hash** - Deterministic, allows deduplication
- **C) Composite key** - `{doc_id}:{chunk_index}` format

**Recommendation:** Option B - Use content hash (e.g., first 8 chars of SHA256) for deterministic chunk identification.

### Decision 3: Langsmith Dataset Schema

**Proposed Schema:**
```python
# Input (query)
{
    "query": str,
}

# Output (ground truth)
{
    "chunk_ids": List[str],           # For chunk-level metrics
    "excerpts": List[str],            # Actual text for token-level metrics
    "metadata": {
        "doc_id": str,
        "relevance_score": float,     # Optional
    }
}
```

### Decision 4: Error Handling Strategy

**Options:**
- **A) Fail fast** - Raise exceptions immediately
- **B) Collect errors** - Continue evaluation, report errors in results
- **C) Configurable** - Let user choose behavior

**Recommendation:** Option C - Default to fail-fast for development, option to collect errors for production sweeps.

---

## Dependency Management

### Required Dependencies (to add to pyproject.toml)

```toml
[project.optional-dependencies]
# Core evaluation
eval = [
    "tiktoken>=0.5.0",           # Token counting for token-level metrics
    "numpy>=1.24.0",             # Numerical operations
]

# Vector stores
chroma = ["chromadb>=0.4.0"]
qdrant = ["qdrant-client>=1.6.0"]

# Embedders
openai = ["openai>=1.0.0"]
sentence-transformers = ["sentence-transformers>=2.2.0"]
cohere = ["cohere>=4.0.0"]

# Rerankers
rerankers = ["cohere>=4.0.0"]

# Data generation
datagen = [
    "openai>=1.0.0",
    "anthropic>=0.18.0",
]

# Full installation
all = [
    "rag-evaluation-framework[eval,chroma,openai,rerankers,datagen]"
]
```

---

## Testing Strategy

### Unit Tests

```
tests/
├── test_metrics/
│   ├── test_chunk_level_recall.py
│   ├── test_chunk_level_precision.py
│   ├── test_token_level_recall.py
│   ├── test_token_level_precision.py
│   └── test_token_level_iou.py
├── test_chunkers/
│   ├── test_recursive_character_chunker.py
│   └── test_fixed_token_chunker.py
├── test_evaluation/
│   ├── test_evaluation_run.py
│   └── test_evaluation_sweep.py
└── conftest.py  # Fixtures for mock Langsmith objects
```

### Integration Tests

```python
# Test full pipeline with mock components
def test_full_evaluation_pipeline():
    eval = Evaluation(
        langsmith_dataset_name="test-dataset",
        kb_data_path="./test_kb"
    )
    results = eval.run(
        chunker=MockChunker(),
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        k=5,
    )
    assert results.metrics_summary["chunk_recall@5"] >= 0.0
```

---

## Implementation Priority Matrix

| Phase | Component | Priority | Effort | Dependencies |
|-------|-----------|----------|--------|--------------|
| 1.1 | Fix ChunkLevelRecall | HIGH | Low | None |
| 1.2 | ChunkLevelPrecision | HIGH | Low | None |
| 1.3 | TokenLevelRecall | HIGH | Medium | tiktoken |
| 1.3 | TokenLevelPrecision | HIGH | Low | TokenLevelRecall |
| 1.3 | TokenLevelIoU | HIGH | Low | TokenLevelRecall |
| 2.1 | Evaluation.run() | HIGH | High | Phase 1 |
| 2.2 | EvaluationResults | HIGH | Medium | Phase 2.1 |
| 3.1 | RecursiveCharacterChunker | MEDIUM | Medium | None |
| 3.2 | OpenAIEmbedder | MEDIUM | Low | openai |
| 3.3 | ChromaVectorStore | MEDIUM | Medium | chromadb |
| 4 | Sweep functionality | MEDIUM | High | Phase 2-3 |
| 5 | Synthetic data generation | LOW | High | openai/anthropic |
| 6 | Advanced metrics (MRR, NDCG) | LOW | Medium | None |

---

## Success Criteria

### MVP (Minimum Viable Product)
- [ ] All chunk-level metrics working (Recall, Precision)
- [ ] At least TokenLevelRecall implemented
- [ ] `Evaluation.run()` executes full pipeline with Langsmith
- [ ] One working chunker implementation
- [ ] One working embedder implementation
- [ ] ChromaVectorStore working

### V1.0 Release
- [ ] All token-level metrics (Recall, Precision, IoU, F1)
- [ ] Hyperparameter sweep functionality
- [ ] Multiple chunker implementations
- [ ] Multiple embedder implementations
- [ ] Documentation with examples
- [ ] Test coverage > 80%

### Future Enhancements
- [ ] Synthetic data generation module
- [ ] Advanced metrics (MRR, NDCG, MAP)
- [ ] Visualization dashboard
- [ ] CLI interface
- [ ] Async evaluation support

---

## Open Questions for Clarification

### Architecture Questions

1. **Langsmith Dataset Schema**: What is the expected structure of your Langsmith evaluation datasets? Specifically:
   - What field names do you use for ground truth chunk IDs?
   - Do you store actual text excerpts or just chunk IDs?
   - How do you currently identify chunks (sequential IDs, hashes, composite keys)?

2. **Token-Level Metric Scope**: For token-level metrics, should we:
   - Compare tokens from chunk IDs (requires chunk content storage/lookup)?
   - Compare tokens from actual text strings passed directly?
   - Support both approaches?

3. **Chunk Content Storage**: Token-level metrics need access to chunk text content. Should we:
   - Store chunk content in the vector store metadata?
   - Maintain a separate ChunkStore/ChunkRegistry?
   - Require users to pass chunk content directly?

### Feature Prioritization

4. **Which concrete implementations are highest priority?**
   - Chunkers: RecursiveCharacter, FixedToken, Semantic, Sentence?
   - Embedders: OpenAI, SentenceTransformers, Cohere?
   - Vector Stores: Chroma, Qdrant, Pinecone, In-Memory?
   - Rerankers: Cohere, CrossEncoder?

5. **Hyperparameter Sweep**: Is the sweep functionality important for V1, or can it be deferred?

6. **Synthetic Data Generation**: How important is the `synthetic_datagen` module? Do you have existing datasets, or is generation critical?

### Integration Questions

7. **Existing Evaluation Datasets**: Do you have Langsmith datasets already populated that we need to support? If so, what's their schema?

8. **Langsmith Experiment Tracking**: Any specific requirements for experiment naming, metadata, or organization?

9. **Comparison with chunking_evaluation**: Are there specific features from the [chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation) library you want to replicate or improve upon?

### Technical Questions

10. **Tokenizer Choice**: Should we standardize on OpenAI's `cl100k_base` tokenizer for token-level metrics, or support configurable tokenizers?

11. **Error Handling**: Prefer fail-fast (exceptions) or graceful degradation (collect errors, continue)?

12. **Async Support**: Is async evaluation important for your use case (large datasets, parallel processing)?

13. **Configuration Management**: Do you want YAML/JSON config file support, or is Python-only configuration sufficient?

---

## References

- [Chroma Chunking Evaluation Research](https://research.trychroma.com/evaluating-chunking)
- [chunking_evaluation GitHub](https://github.com/brandonstarxel/chunking_evaluation)
- [Langsmith Evaluation Documentation](https://docs.smith.langchain.com/evaluation)
- [OpenAI tiktoken](https://github.com/openai/tiktoken)

---

*Document Version: 1.0*
*Created: January 2026*
*Last Updated: January 2026*
