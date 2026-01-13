# RAG Evaluation Framework - Architecture Brainstorm

## Core Insight: Evaluation Type as First-Class Concept

The evaluation type (chunk-level vs token-level) should be a **foundational choice** that shapes the entire pipeline, not an afterthought. This means:

1. Different LangSmith dataset schemas
2. Different synthetic data generation strategies
3. Different chunker interfaces (or adapters)
4. Different metric implementations
5. Strong typing that makes incompatible combinations impossible

---

## Two Evaluation Paradigms

### Chunk-Level Evaluation
- **Question**: "Did we retrieve the right chunks?"
- **Ground truth**: List of chunk IDs that are relevant
- **Metric basis**: Set intersection of chunk IDs
- **Simpler**, but binary (chunk is relevant or not)

### Token-Level Evaluation (Character Spans)
- **Question**: "Did we retrieve the right *content*?"
- **Ground truth**: Character ranges in source documents
- **Metric basis**: Character overlap between spans
- **More granular**, captures partial relevance

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TEXT CORPUS                               │
│                  (folder of markdown files)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   CHOOSE EVALUATION TYPE      │
              │  (chunk-level | token-level)  │
              └───────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
┌─────────────────────┐               ┌─────────────────────┐
│  CHUNK-LEVEL PATH   │               │  TOKEN-LEVEL PATH   │
└─────────────────────┘               └─────────────────────┘
           │                                     │
           ▼                                     ▼
┌─────────────────────┐               ┌─────────────────────┐
│ SyntheticDataGen    │               │ SyntheticDataGen    │
│ (ChunkLevel)        │               │ (TokenLevel)        │
│                     │               │                     │
│ Output:             │               │ Output:             │
│ - query             │               │ - query             │
│ - relevant_chunk_ids│               │ - relevant_spans    │
│                     │               │   (doc_id, start,   │
│                     │               │    end, text)       │
└─────────────────────┘               └─────────────────────┘
           │                                     │
           ▼                                     ▼
┌─────────────────────┐               ┌─────────────────────┐
│ LangSmith Dataset   │               │ LangSmith Dataset   │
│ (ChunkLevelSchema)  │               │ (TokenLevelSchema)  │
└─────────────────────┘               └─────────────────────┘
           │                                     │
           ▼                                     ▼
┌─────────────────────┐               ┌─────────────────────┐
│ Evaluation          │               │ Evaluation          │
│ (ChunkLevel)        │               │ (TokenLevel)        │
│                     │               │                     │
│ Uses:               │               │ Uses:               │
│ - Chunker           │               │ - PositionAware     │
│ - Embedder          │               │   Chunker           │
│ - VectorStore       │               │ - Embedder          │
│ - Reranker          │               │ - VectorStore       │
│                     │               │ - Reranker          │
│ Metrics:            │               │                     │
│ - ChunkRecall       │               │ Metrics:            │
│ - ChunkPrecision    │               │ - SpanRecall        │
│ - ChunkF1           │               │ - SpanPrecision     │
└─────────────────────┘               │ - SpanIoU           │
                                      └─────────────────────┘
```

---

## Type Definitions

### Core Types (Shared)

```python
from typing import TypedDict, Literal, List, Optional
from dataclasses import dataclass
from pathlib import Path

EvaluationType = Literal["chunk-level", "token-level"]

@dataclass
class Document:
    """A source document from the corpus."""
    id: str
    path: Path
    content: str

@dataclass
class Corpus:
    """Collection of documents to evaluate against."""
    documents: List[Document]
    base_path: Path

    @classmethod
    def from_folder(cls, folder: Path, glob: str = "**/*.md") -> "Corpus":
        """Load all markdown files from a folder."""
        ...
```

### Chunk-Level Types

```python
@dataclass
class Chunk:
    """A chunk with ID but no position tracking."""
    id: str  # hash of content or composite key
    content: str
    doc_id: str
    metadata: dict = field(default_factory=dict)

class ChunkLevelGroundTruth(TypedDict):
    """Ground truth for chunk-level evaluation."""
    query: str
    relevant_chunk_ids: List[str]

class ChunkLevelDatasetExample(TypedDict):
    """LangSmith dataset example for chunk-level."""
    inputs: dict  # {"query": str}
    outputs: dict  # {"relevant_chunk_ids": List[str]}

class ChunkLevelRunOutput(TypedDict):
    """Output from retrieval pipeline for chunk-level."""
    retrieved_chunk_ids: List[str]
```

### Token-Level Types (Character Spans)

```python
@dataclass
class CharacterSpan:
    """A span of characters in a source document."""
    doc_id: str
    start: int  # inclusive
    end: int    # exclusive
    text: str   # the actual text (for convenience/validation)

    def overlaps(self, other: "CharacterSpan") -> bool:
        """Check if two spans overlap."""
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: "CharacterSpan") -> int:
        """Calculate character overlap."""
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

@dataclass
class PositionAwareChunk:
    """A chunk that knows its position in the source document."""
    id: str
    content: str
    doc_id: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    def to_span(self) -> CharacterSpan:
        return CharacterSpan(
            doc_id=self.doc_id,
            start=self.start,
            end=self.end,
            text=self.content
        )

class TokenLevelGroundTruth(TypedDict):
    """Ground truth for token-level evaluation."""
    query: str
    relevant_spans: List[dict]  # List of {doc_id, start, end, text}

class TokenLevelDatasetExample(TypedDict):
    """LangSmith dataset example for token-level."""
    inputs: dict  # {"query": str}
    outputs: dict  # {"relevant_spans": List[{doc_id, start, end, text}]}

class TokenLevelRunOutput(TypedDict):
    """Output from retrieval pipeline for token-level."""
    retrieved_spans: List[dict]  # List of {doc_id, start, end, text}
```

---

## Interface Definitions

### Chunker Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Protocol

class Chunker(ABC):
    """Base chunker - returns chunks without position info."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        ...

class PositionAwareChunker(ABC):
    """Chunker that tracks character positions."""

    @abstractmethod
    def chunk_with_positions(self, doc: Document) -> List[PositionAwareChunk]:
        """Split document into position-aware chunks."""
        ...

# Adapter to make any Chunker position-aware
class ChunkerPositionAdapter(PositionAwareChunker):
    """Wraps a regular Chunker to track positions."""

    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    def chunk_with_positions(self, doc: Document) -> List[PositionAwareChunk]:
        chunks = self.chunker.chunk(doc.content)
        result = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find chunk in original text
            start = doc.content.find(chunk_text, current_pos)
            if start == -1:
                # Fallback: chunk was modified (e.g., whitespace normalized)
                # This is a limitation - may need smarter matching
                start = current_pos
            end = start + len(chunk_text)

            result.append(PositionAwareChunk(
                id=self._generate_id(doc.id, chunk_text),
                content=chunk_text,
                doc_id=doc.id,
                start=start,
                end=end,
            ))
            current_pos = end

        return result

    def _generate_id(self, doc_id: str, content: str) -> str:
        import hashlib
        hash_input = f"{doc_id}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
```

**Open Question**: Should we require ALL chunkers to be position-aware, or use the adapter pattern?

Option A: Single interface, always position-aware
- Simpler mental model
- All implementations must track positions
- Could be annoying for simple use cases

Option B: Two interfaces + adapter (shown above)
- More flexible
- Adapter handles position tracking automatically
- Risk: adapter may fail with chunkers that modify text (normalization, etc.)

**Recommendation**: Option B with clear documentation about adapter limitations.

---

### Synthetic Data Generation

```python
from abc import ABC, abstractmethod
from typing import Union, overload

class SyntheticDataGenerator(ABC):
    """Base class for synthetic data generation."""

    def __init__(self, llm_client, corpus: Corpus):
        self.llm = llm_client
        self.corpus = corpus

class ChunkLevelDataGenerator(SyntheticDataGenerator):
    """Generate synthetic QA pairs with chunk-level ground truth."""

    def __init__(
        self,
        llm_client,
        corpus: Corpus,
        chunker: Chunker,  # Need to chunk first to get chunk IDs
    ):
        super().__init__(llm_client, corpus)
        self.chunker = chunker
        self._chunk_index: Dict[str, Chunk] = {}  # chunk_id -> Chunk

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: Optional[str] = None,
    ) -> List[ChunkLevelGroundTruth]:
        """
        Generate synthetic queries with relevant chunk IDs.

        Process:
        1. Chunk all documents, build chunk index
        2. For each document, generate queries using LLM
        3. For each query, identify relevant chunks (LLM or embedding similarity)
        4. Return/upload ground truth pairs
        """
        ...

class TokenLevelDataGenerator(SyntheticDataGenerator):
    """Generate synthetic QA pairs with character span ground truth."""

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: Optional[str] = None,
    ) -> List[TokenLevelGroundTruth]:
        """
        Generate synthetic queries with relevant character spans.

        Process:
        1. For each document, generate queries using LLM
        2. For each query, ask LLM to extract relevant excerpts
        3. Find character positions of excerpts in source document
        4. Return/upload ground truth pairs

        Note: No pre-chunking needed! Ground truth is excerpt positions,
        independent of how we chunk at evaluation time.
        """
        ...
```

**Key Insight**: Token-level synthetic data generation is **chunker-independent**. We generate relevant excerpts directly from documents. This means:
- Same ground truth dataset works with ANY chunking strategy
- Can fairly compare different chunkers against same baseline
- This is a major advantage of token-level evaluation!

For chunk-level, we must chunk first, which means:
- Ground truth is tied to a specific chunking strategy
- Changing chunkers requires regenerating ground truth
- Less fair for chunker comparison

---

### Evaluation Classes

```python
from typing import Generic, TypeVar, Union
from dataclasses import dataclass

GT = TypeVar('GT', ChunkLevelGroundTruth, TokenLevelGroundTruth)

@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    metrics: Dict[str, float]
    experiment_url: Optional[str]
    raw_results: Any  # Langsmith results

class BaseEvaluation(ABC, Generic[GT]):
    """Base evaluation class."""

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ):
        self.corpus = corpus
        self.langsmith_dataset_name = langsmith_dataset_name

class ChunkLevelEvaluation(BaseEvaluation[ChunkLevelGroundTruth]):
    """Evaluation using chunk-level metrics."""

    def run(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        k: int = 5,
        reranker: Optional[Reranker] = None,
        metrics: Optional[List[ChunkLevelMetric]] = None,
    ) -> EvaluationResult:
        """
        Run chunk-level evaluation.

        Pipeline:
        1. Chunk corpus using chunker
        2. Generate chunk IDs (content hash)
        3. Embed and index chunks
        4. For each query in dataset:
           - Retrieve top-k chunks
           - Compare retrieved chunk IDs vs ground truth chunk IDs
        5. Compute metrics (recall, precision, F1)
        """
        ...

class TokenLevelEvaluation(BaseEvaluation[TokenLevelGroundTruth]):
    """Evaluation using token-level (character span) metrics."""

    def run(
        self,
        chunker: Union[Chunker, PositionAwareChunker],
        embedder: Embedder,
        vector_store: VectorStore,
        k: int = 5,
        reranker: Optional[Reranker] = None,
        metrics: Optional[List[TokenLevelMetric]] = None,
    ) -> EvaluationResult:
        """
        Run token-level evaluation.

        Pipeline:
        1. Chunk corpus using chunker (wrapped with PositionAdapter if needed)
        2. Track chunk positions in source documents
        3. Embed and index chunks
        4. For each query in dataset:
           - Retrieve top-k chunks
           - Convert chunks to character spans
           - Compare retrieved spans vs ground truth spans (overlap)
        5. Compute metrics (span recall, precision, IoU)
        """
        # Wrap chunker if needed
        if isinstance(chunker, Chunker):
            chunker = ChunkerPositionAdapter(chunker)
        ...
```

---

### Metrics

```python
from abc import ABC, abstractmethod

class ChunkLevelMetric(ABC):
    """Metric for chunk-level evaluation."""

    @abstractmethod
    def calculate(
        self,
        retrieved_chunk_ids: List[str],
        ground_truth_chunk_ids: List[str]
    ) -> float:
        ...

class ChunkRecall(ChunkLevelMetric):
    def calculate(self, retrieved: List[str], ground_truth: List[str]) -> float:
        if not ground_truth:
            return 0.0
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        return len(retrieved_set & ground_truth_set) / len(ground_truth_set)

class ChunkPrecision(ChunkLevelMetric):
    def calculate(self, retrieved: List[str], ground_truth: List[str]) -> float:
        if not retrieved:
            return 0.0
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        return len(retrieved_set & ground_truth_set) / len(retrieved_set)

class ChunkF1(ChunkLevelMetric):
    def calculate(self, retrieved: List[str], ground_truth: List[str]) -> float:
        recall = ChunkRecall().calculate(retrieved, ground_truth)
        precision = ChunkPrecision().calculate(retrieved, ground_truth)
        if recall + precision == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class TokenLevelMetric(ABC):
    """Metric for token-level (character span) evaluation."""

    @abstractmethod
    def calculate(
        self,
        retrieved_spans: List[CharacterSpan],
        ground_truth_spans: List[CharacterSpan]
    ) -> float:
        ...

class SpanRecall(TokenLevelMetric):
    """What fraction of ground truth characters were retrieved?"""

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not ground_truth:
            return 0.0

        total_gt_chars = sum(span.end - span.start for span in ground_truth)

        # Calculate overlap
        overlap_chars = 0
        for gt_span in ground_truth:
            for ret_span in retrieved:
                overlap_chars += gt_span.overlap_chars(ret_span)

        # Note: Need to handle overlapping retrieved spans (dedup)
        # This is simplified - real implementation needs interval merging
        return min(overlap_chars / total_gt_chars, 1.0)

class SpanPrecision(TokenLevelMetric):
    """What fraction of retrieved characters were relevant?"""

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not retrieved:
            return 0.0

        total_ret_chars = sum(span.end - span.start for span in retrieved)

        overlap_chars = 0
        for ret_span in retrieved:
            for gt_span in ground_truth:
                overlap_chars += ret_span.overlap_chars(gt_span)

        return min(overlap_chars / total_ret_chars, 1.0)

class SpanIoU(TokenLevelMetric):
    """Intersection over Union of character spans."""

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not retrieved and not ground_truth:
            return 1.0
        if not retrieved or not ground_truth:
            return 0.0

        # Need proper interval arithmetic here
        # Union = total unique characters covered by either
        # Intersection = characters covered by both
        ...
```

---

## LangSmith Dataset Schemas

### Chunk-Level Dataset

```json
{
  "name": "rag-eval-chunk-level-v1",
  "description": "Ground truth for chunk-level RAG evaluation",
  "example_schema": {
    "inputs": {
      "query": "string"
    },
    "outputs": {
      "relevant_chunk_ids": ["string"],
      "metadata": {
        "source_docs": ["string"],
        "generation_model": "string"
      }
    }
  }
}
```

Example:
```json
{
  "inputs": {"query": "What are the benefits of RAG?"},
  "outputs": {
    "relevant_chunk_ids": ["a3f2b1c8", "7d9e4f2a", "1b3c5d7e"],
    "metadata": {
      "source_docs": ["rag_overview.md"],
      "generation_model": "gpt-4"
    }
  }
}
```

### Token-Level Dataset

```json
{
  "name": "rag-eval-token-level-v1",
  "description": "Ground truth for token-level RAG evaluation (character spans)",
  "example_schema": {
    "inputs": {
      "query": "string"
    },
    "outputs": {
      "relevant_spans": [
        {
          "doc_id": "string",
          "start": "integer",
          "end": "integer",
          "text": "string"
        }
      ],
      "metadata": {
        "generation_model": "string"
      }
    }
  }
}
```

Example:
```json
{
  "inputs": {"query": "What are the benefits of RAG?"},
  "outputs": {
    "relevant_spans": [
      {
        "doc_id": "rag_overview.md",
        "start": 1520,
        "end": 1847,
        "text": "RAG combines the benefits of retrieval systems with generative models..."
      },
      {
        "doc_id": "rag_overview.md",
        "start": 2103,
        "end": 2298,
        "text": "Key advantages include reduced hallucination and access to current information..."
      }
    ],
    "metadata": {
      "generation_model": "gpt-4"
    }
  }
}
```

---

## User-Facing API

### Option 1: Factory Pattern

```python
from rag_evaluation_framework import create_evaluation, EvaluationType

# Chunk-level
eval = create_evaluation(
    eval_type="chunk-level",
    corpus_path="./knowledge_base",
    langsmith_dataset="my-dataset",
)

# Token-level
eval = create_evaluation(
    eval_type="token-level",
    corpus_path="./knowledge_base",
    langsmith_dataset="my-dataset",
)

# Same run() interface
results = eval.run(
    chunker=RecursiveCharacterChunker(chunk_size=200),
    embedder=OpenAIEmbedder(),
    vector_store=ChromaVectorStore(),
    k=5,
)
```

### Option 2: Explicit Classes

```python
from rag_evaluation_framework import (
    ChunkLevelEvaluation,
    TokenLevelEvaluation,
    Corpus,
)

corpus = Corpus.from_folder("./knowledge_base")

# Chunk-level
eval = ChunkLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-chunk-dataset",
)

# Token-level
eval = TokenLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-token-dataset",
)
```

### Option 3: Single Class with Type Parameter

```python
from rag_evaluation_framework import Evaluation

# Chunk-level
eval = Evaluation[ChunkLevel](
    corpus_path="./knowledge_base",
    langsmith_dataset="my-dataset",
)

# Token-level
eval = Evaluation[TokenLevel](
    corpus_path="./knowledge_base",
    langsmith_dataset="my-dataset",
)
```

**Recommendation**: Option 2 (Explicit Classes)
- Most Pythonic
- Clear what you're getting
- IDE autocomplete works well
- Type checker catches mismatches

---

## Synthetic Data Generation API

```python
from rag_evaluation_framework import (
    Corpus,
    ChunkLevelDataGenerator,
    TokenLevelDataGenerator,
)

corpus = Corpus.from_folder("./knowledge_base")

# For chunk-level evaluation
# Note: requires chunker because ground truth is chunk IDs
chunk_gen = ChunkLevelDataGenerator(
    llm_client=OpenAI(),
    corpus=corpus,
    chunker=RecursiveCharacterChunker(chunk_size=200),
)

chunk_dataset = chunk_gen.generate(
    queries_per_doc=5,
    upload_to_langsmith=True,
    dataset_name="my-chunk-eval-dataset",
)

# For token-level evaluation
# Note: NO chunker needed - ground truth is character spans
token_gen = TokenLevelDataGenerator(
    llm_client=OpenAI(),
    corpus=corpus,
)

token_dataset = token_gen.generate(
    queries_per_doc=5,
    upload_to_langsmith=True,
    dataset_name="my-token-eval-dataset",
)
```

---

## Full Workflow Example

### Token-Level (Recommended for Chunker Comparison)

```python
from rag_evaluation_framework import (
    Corpus,
    TokenLevelDataGenerator,
    TokenLevelEvaluation,
    RecursiveCharacterChunker,
    FixedTokenChunker,
    SemanticChunker,
    OpenAIEmbedder,
    ChromaVectorStore,
)
from openai import OpenAI

# 1. Load corpus
corpus = Corpus.from_folder("./knowledge_base")

# 2. Generate synthetic data (one-time)
generator = TokenLevelDataGenerator(
    llm_client=OpenAI(),
    corpus=corpus,
)
generator.generate(
    queries_per_doc=10,
    upload_to_langsmith=True,
    dataset_name="my-rag-eval-token-level",
)

# 3. Run evaluation with different chunkers
eval = TokenLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-rag-eval-token-level",
)

chunkers_to_test = [
    RecursiveCharacterChunker(chunk_size=200, overlap=0),
    RecursiveCharacterChunker(chunk_size=200, overlap=50),
    RecursiveCharacterChunker(chunk_size=500, overlap=0),
    FixedTokenChunker(tokens=100),
    SemanticChunker(embedder=OpenAIEmbedder()),
]

results = []
for chunker in chunkers_to_test:
    result = eval.run(
        chunker=chunker,
        embedder=OpenAIEmbedder(),
        vector_store=ChromaVectorStore(),
        k=5,
    )
    results.append(result)
    print(f"{chunker}: Recall={result.metrics['span_recall']:.3f}")
```

### Chunk-Level (Simpler, but Chunker-Dependent Ground Truth)

```python
from rag_evaluation_framework import (
    Corpus,
    ChunkLevelDataGenerator,
    ChunkLevelEvaluation,
    RecursiveCharacterChunker,
    OpenAIEmbedder,
    ChromaVectorStore,
)

# 1. Load corpus
corpus = Corpus.from_folder("./knowledge_base")

# 2. Choose chunker (this is fixed for this evaluation)
chunker = RecursiveCharacterChunker(chunk_size=200)

# 3. Generate synthetic data with this chunker
generator = ChunkLevelDataGenerator(
    llm_client=OpenAI(),
    corpus=corpus,
    chunker=chunker,  # Required!
)
generator.generate(
    queries_per_doc=10,
    upload_to_langsmith=True,
    dataset_name="my-rag-eval-chunk-level",
)

# 4. Run evaluation (must use same chunker!)
eval = ChunkLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-rag-eval-chunk-level",
)

result = eval.run(
    chunker=chunker,  # Must match!
    embedder=OpenAIEmbedder(),
    vector_store=ChromaVectorStore(),
    k=5,
)
```

---

## Open Questions

### 1. Chunk ID Stability for Chunk-Level Evaluation

When using chunk-level evaluation, how do we ensure chunk IDs are stable?

Options:
- **Content hash**: `sha256(text)[:12]` - deterministic but changes if text changes
- **Position hash**: `sha256(doc_id + start + end)[:12]` - stable to content edits elsewhere
- **Composite key**: `{doc_id}:{chunk_index}` - simple but order-dependent

**Recommendation**: Content hash for most cases. Document that regenerating dataset is needed if corpus changes.

### 2. Handling Overlapping Spans in Token-Level Metrics

When chunks overlap (common with sliding window), how do we count characters?

```
Chunk 1: [----chars 0-100----]
Chunk 2:        [----chars 50-150----]
Ground truth:   [--chars 60-90--]
```

Do we count chars 60-90 once or twice?

**Recommendation**: Merge overlapping retrieved spans before comparison. Count each character at most once.

### 3. Cross-Document Ground Truth

Can a single query have relevant spans from multiple documents?

```json
{
  "query": "Compare RAG and fine-tuning",
  "relevant_spans": [
    {"doc_id": "rag.md", "start": 100, "end": 200},
    {"doc_id": "fine_tuning.md", "start": 50, "end": 150}
  ]
}
```

**Recommendation**: Yes, support this. It's realistic and the span-based approach handles it naturally.

### 4. VectorStore Position Tracking

For token-level evaluation, the VectorStore needs to return position info. Options:

A) Store positions in metadata, return with results
B) Maintain separate chunk registry, look up after retrieval
C) Return chunk IDs, look up positions from registry

**Recommendation**: Option A - Store in metadata. Most vector stores support this.

```python
class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[PositionAwareChunk], embeddings: List[List[float]]):
        """Add chunks with their positions stored in metadata."""
        ...

    @abstractmethod
    def search(self, query_embedding: List[float], k: int) -> List[PositionAwareChunk]:
        """Return chunks with position info."""
        ...
```

### 5. Adapter Failure Cases

The `ChunkerPositionAdapter` may fail when:
- Chunker normalizes whitespace
- Chunker adds/removes characters
- Chunker reorders content

How to handle?

Options:
- Warn user, skip problematic chunks
- Use fuzzy matching (slower)
- Require explicit position-aware chunkers

**Recommendation**: Warn and skip, with clear documentation. Most chunkers preserve text.

---

## Summary: Chunk-Level vs Token-Level

| Aspect | Chunk-Level | Token-Level |
|--------|-------------|-------------|
| Ground truth format | Chunk IDs | Character spans |
| Chunker for data gen | Required | Not needed |
| Compare chunkers fairly | No (tied to GT chunker) | Yes (chunker-independent GT) |
| Implementation complexity | Lower | Higher |
| Metric granularity | Binary (chunk relevant or not) | Continuous (% overlap) |
| Interface changes needed | None | Chunker position tracking |
| Best for | Quick iteration, simple cases | Research, chunker comparison |

**Recommendation**:
- Start with **Token-Level** as the primary approach - more principled, better for comparing chunking strategies
- Offer **Chunk-Level** as a simpler alternative when users don't need fine-grained metrics

---

## Next Steps

1. **Decide** on the API style (Option 1/2/3 above)
2. **Define** final type definitions in `types.py`
3. **Implement** `PositionAwareChunker` interface and adapter
4. **Implement** `TokenLevelDataGenerator`
5. **Implement** span-based metrics
6. **Implement** `TokenLevelEvaluation.run()`
7. **Update** VectorStore interface for position metadata
8. **Write** comprehensive tests
9. **Document** with examples
