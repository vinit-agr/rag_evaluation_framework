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
- **Ground truth**: List of position-aware chunk IDs (which map to character ranges)
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
│ - relevant_chunk_ids│               │ - relevant_pa_chunk │
│   (chunk_xxxxx)     │               │   _ids (pa_chunk_xx)│
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

These are the foundational types used throughout the framework. They provide strong typing
and clear semantics for all data structures.

```python
from typing import TypedDict, Literal, List, Optional, Dict, Any, NewType
from dataclasses import dataclass, field

# =============================================================================
# PRIMITIVE TYPE ALIASES
# =============================================================================
# These type aliases provide semantic meaning and type safety beyond bare strings.
# Using these instead of `str` makes the code self-documenting and helps catch
# type mismatches at development time.

# Unique identifier for a document in the corpus.
# Format: typically the filename or a hash of the file path.
# Example: "rag_overview.md", "doc_a1b2c3d4"
DocumentId = NewType("DocumentId", str)

# Unique identifier for a query/question.
# Format: typically a UUID or hash of the query text.
# Example: "query_f47ac10b"
QueryId = NewType("QueryId", str)

# The actual query/question text that will be used for retrieval.
# Example: "What are the benefits of RAG?"
QueryText = NewType("QueryText", str)

# Unique identifier for a standard chunk (without position tracking).
# Format: "chunk_" prefix + first 12 chars of SHA256 hash of content.
# Example: "chunk_a3f2b1c8d9e0"
# The prefix makes it easy to identify this as a chunk ID at a glance.
ChunkId = NewType("ChunkId", str)

# Unique identifier for a position-aware chunk (with character span tracking).
# Format: "pa_chunk_" prefix + first 12 chars of SHA256 hash of content.
# Example: "pa_chunk_7d9e4f2a1b3c"
# The "pa_" prefix distinguishes these from regular chunk IDs, making it
# immediately clear when you're working with position-aware data.
PositionAwareChunkId = NewType("PositionAwareChunkId", str)

# =============================================================================
# EVALUATION TYPE
# =============================================================================

# The type of evaluation to perform. This is a foundational choice that
# determines the shape of ground truth data, metrics used, and chunker requirements.
EvaluationType = Literal["chunk-level", "token-level"]


# =============================================================================
# DOCUMENT AND CORPUS
# =============================================================================

@dataclass
class Document:
    """
    A source document from the corpus.

    Represents a single text file (typically markdown) that will be chunked
    and indexed for retrieval evaluation.

    Attributes:
        id: Unique identifier for this document. Used to reference the document
            in chunk IDs and ground truth data. Typically derived from filename.
        content: The full text content of the document.
        metadata: Arbitrary key-value pairs for additional document information.
            Examples: {"author": "John", "date": "2024-01-15", "source": "wiki"}
    """
    id: DocumentId
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    """
    Collection of documents to evaluate against.

    The corpus represents the entire knowledge base that will be chunked,
    embedded, and indexed. Synthetic queries are generated from this corpus,
    and retrieval performance is measured against it.

    Attributes:
        documents: List of all documents in the corpus.
        metadata: Arbitrary key-value pairs for corpus-level information.
            Examples: {"name": "product_docs", "version": "2.0"}
    """
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_folder(cls, folder_path: str, glob_pattern: str = "**/*.md") -> "Corpus":
        """
        Load all markdown files from a folder into a Corpus.

        Args:
            folder_path: Path to the folder containing documents.
            glob_pattern: Glob pattern for matching files. Default matches all
                markdown files recursively.

        Returns:
            A Corpus containing all matched documents.
        """
        ...


# =============================================================================
# CHUNK TYPES
# =============================================================================

@dataclass
class Chunk:
    """
    A chunk of text extracted from a document (without position tracking).

    Used in chunk-level evaluation where we only care about chunk identity,
    not the exact character positions in the source document.

    Attributes:
        id: Unique identifier for this chunk. Format: "chunk_" + content hash.
            Example: "chunk_a3f2b1c8d9e0"
        content: The actual text content of this chunk.
        doc_id: Reference to the parent document this chunk was extracted from.
        metadata: Arbitrary key-value pairs for additional chunk information.
            Examples: {"chunk_index": 5, "section": "introduction"}
    """
    id: ChunkId
    content: str
    doc_id: DocumentId
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CharacterSpan:
    """
    A span of characters in a source document.

    Represents a contiguous range of text within a document, defined by
    start and end character positions. Used for computing overlap in
    token-level evaluation metrics.

    Attributes:
        doc_id: The document this span belongs to.
        start: Starting character position (inclusive, 0-indexed).
        end: Ending character position (exclusive).

    Example:
        For document content "Hello, World!", CharacterSpan("doc1", 0, 5)
        represents the text "Hello".
    """
    doc_id: DocumentId
    start: int  # inclusive, 0-indexed
    end: int    # exclusive

    def overlaps(self, other: "CharacterSpan") -> bool:
        """
        Check if this span overlaps with another span.

        Two spans overlap if they share at least one character position
        AND belong to the same document.

        Returns:
            True if spans overlap, False otherwise.
        """
        if self.doc_id != other.doc_id:
            return False
        return self.start < other.end and other.start < self.end

    def overlap_chars(self, other: "CharacterSpan") -> int:
        """
        Calculate the number of overlapping characters with another span.

        Returns:
            Number of characters in the intersection. Returns 0 if no overlap.
        """
        if not self.overlaps(other):
            return 0
        return min(self.end, other.end) - max(self.start, other.start)

    def length(self) -> int:
        """Return the length of this span in characters."""
        return self.end - self.start


@dataclass
class PositionAwareChunk:
    """
    A chunk that knows its exact position in the source document.

    Used in token-level evaluation where we need to compute character-level
    overlap between retrieved chunks and ground truth spans.

    Attributes:
        id: Unique identifier for this chunk. Format: "pa_chunk_" + content hash.
            Example: "pa_chunk_7d9e4f2a1b3c"
        content: The actual text content of this chunk.
        doc_id: Reference to the parent document this chunk was extracted from.
        start: Starting character position in the source document (inclusive).
        end: Ending character position in the source document (exclusive).
        metadata: Arbitrary key-value pairs for additional chunk information.

    Note:
        The content should exactly match document[start:end]. This invariant
        is important for correct metric calculation.
    """
    id: PositionAwareChunkId
    content: str
    doc_id: DocumentId
    start: int  # inclusive, 0-indexed
    end: int    # exclusive
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_span(self) -> CharacterSpan:
        """
        Convert this chunk to a CharacterSpan for metric calculation.

        Returns:
            A CharacterSpan with the same document and position info.
        """
        return CharacterSpan(
            doc_id=self.doc_id,
            start=self.start,
            end=self.end,
        )


# =============================================================================
# QUERY TYPES
# =============================================================================

@dataclass
class Query:
    """
    A query/question for retrieval evaluation.

    Represents a single question that will be used to test the retrieval
    pipeline. Contains both the query text and optional metadata.

    Attributes:
        id: Unique identifier for this query.
        text: The actual question text.
        metadata: Arbitrary key-value pairs for additional query information.
            Examples: {"source_doc": "overview.md", "difficulty": "hard"}
    """
    id: QueryId
    text: QueryText
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Chunk-Level Types

These types are used specifically for chunk-level evaluation, where ground truth
and retrieval results are expressed as lists of chunk IDs.

```python
# =============================================================================
# CHUNK-LEVEL GROUND TRUTH AND RESULTS
# =============================================================================

@dataclass
class ChunkLevelGroundTruth:
    """
    Ground truth data for a single query in chunk-level evaluation.

    Maps a query to the list of chunk IDs that are considered relevant.
    Used to measure retrieval performance at the chunk level.

    Attributes:
        query: The query this ground truth is for.
        relevant_chunk_ids: List of chunk IDs that are relevant to this query.
            Format: ["chunk_a3f2b1c8d9e0", "chunk_7d9e4f2a1b3c", ...]
    """
    query: Query
    relevant_chunk_ids: List[ChunkId]


class ChunkLevelDatasetExample(TypedDict):
    """
    LangSmith dataset example schema for chunk-level evaluation.

    This is the format used when storing/retrieving data from LangSmith.
    Follows LangSmith's inputs/outputs convention.
    """
    inputs: Dict[str, QueryText]        # {"query": "What is RAG?"}
    outputs: Dict[str, List[ChunkId]]   # {"relevant_chunk_ids": ["chunk_xxx", ...]}


class ChunkLevelRunOutput(TypedDict):
    """
    Output from the retrieval pipeline for chunk-level evaluation.

    This is what the retrieval function returns for each query.
    """
    retrieved_chunk_ids: List[ChunkId]  # ["chunk_xxx", "chunk_yyy", ...]
```

### Token-Level Types

These types are used specifically for token-level evaluation, where ground truth
and retrieval results reference position-aware chunks (character spans).

```python
# =============================================================================
# TOKEN-LEVEL GROUND TRUTH AND RESULTS
# =============================================================================

@dataclass
class TokenLevelGroundTruth:
    """
    Ground truth data for a single query in token-level evaluation.

    Maps a query to the list of position-aware chunk IDs that contain
    relevant content. The actual character spans can be looked up from
    the chunk registry using these IDs.

    Attributes:
        query: The query this ground truth is for.
        relevant_chunk_ids: List of position-aware chunk IDs that are relevant.
            Format: ["pa_chunk_a3f2b1c8d9e0", "pa_chunk_7d9e4f2a1b3c", ...]

    Note:
        We store only chunk IDs (not the full span data) to avoid duplicating
        text content in the dataset. The actual spans can be resolved by
        looking up chunks from the ChunkRegistry.
    """
    query: Query
    relevant_chunk_ids: List[PositionAwareChunkId]


class TokenLevelDatasetExample(TypedDict):
    """
    LangSmith dataset example schema for token-level evaluation.

    This is the format used when storing/retrieving data from LangSmith.
    Only stores chunk IDs to minimize data duplication.
    """
    inputs: Dict[str, QueryText]                    # {"query": "What is RAG?"}
    outputs: Dict[str, List[PositionAwareChunkId]]  # {"relevant_chunk_ids": ["pa_chunk_xxx", ...]}


class TokenLevelRunOutput(TypedDict):
    """
    Output from the retrieval pipeline for token-level evaluation.

    This is what the retrieval function returns for each query.
    """
    retrieved_chunk_ids: List[PositionAwareChunkId]  # ["pa_chunk_xxx", "pa_chunk_yyy", ...]


# =============================================================================
# CHUNK REGISTRY
# =============================================================================

class ChunkRegistry:
    """
    Registry for looking up chunk content and positions by ID.

    Since we only store chunk IDs in ground truth and run outputs (to avoid
    data duplication), we need a way to resolve IDs back to full chunk objects.
    The ChunkRegistry serves this purpose.

    This is especially important for token-level evaluation, where we need
    the character span information to compute overlap metrics.

    Usage:
        registry = ChunkRegistry()
        registry.register(chunk)

        # Later, when computing metrics:
        chunk = registry.get(chunk_id)
        span = chunk.to_span()
    """

    def __init__(self):
        self._chunks: Dict[ChunkId, Chunk] = {}
        self._pa_chunks: Dict[PositionAwareChunkId, PositionAwareChunk] = {}

    def register_chunk(self, chunk: Chunk) -> None:
        """Register a standard chunk."""
        self._chunks[chunk.id] = chunk

    def register_pa_chunk(self, chunk: PositionAwareChunk) -> None:
        """Register a position-aware chunk."""
        self._pa_chunks[chunk.id] = chunk

    def get_chunk(self, chunk_id: ChunkId) -> Optional[Chunk]:
        """Look up a standard chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_pa_chunk(self, chunk_id: PositionAwareChunkId) -> Optional[PositionAwareChunk]:
        """Look up a position-aware chunk by ID."""
        return self._pa_chunks.get(chunk_id)

    def get_span(self, chunk_id: PositionAwareChunkId) -> Optional[CharacterSpan]:
        """Get the character span for a position-aware chunk."""
        chunk = self.get_pa_chunk(chunk_id)
        if chunk is None:
            return None
        return chunk.to_span()
```

---

## Interface Definitions

### Chunker Interfaces

We maintain two separate interfaces: a simple `Chunker` for basic use cases, and a
`PositionAwareChunker` for token-level evaluation. An adapter bridges the two.

**Decision**: Keep two separate interfaces with adapter pattern for maximum flexibility.

```python
from abc import ABC, abstractmethod
from typing import List
import hashlib

class Chunker(ABC):
    """
    Base chunker interface - returns text chunks without position tracking.

    Use this for chunk-level evaluation or when you don't need character
    position information. Simpler to implement than PositionAwareChunker.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: The full text to chunk.

        Returns:
            List of chunk text strings.
        """
        ...


class PositionAwareChunker(ABC):
    """
    Chunker that tracks character positions in the source document.

    Required for token-level evaluation where we need to compute
    character-level overlap between retrieved and relevant content.
    """

    @abstractmethod
    def chunk_with_positions(self, doc: Document) -> List[PositionAwareChunk]:
        """
        Split document into position-aware chunks.

        Args:
            doc: The document to chunk.

        Returns:
            List of PositionAwareChunk objects with character positions.
        """
        ...


class ChunkerPositionAdapter(PositionAwareChunker):
    """
    Adapter that wraps a regular Chunker to make it position-aware.

    This allows using any existing Chunker implementation for token-level
    evaluation without modifying the chunker itself.

    Limitations:
        - May fail if the chunker normalizes whitespace or modifies text
        - May fail if the chunker reorders or combines content
        - Logs a warning and skips chunks that can't be located

    For best results, use chunkers that preserve the original text exactly.
    """

    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    def chunk_with_positions(self, doc: Document) -> List[PositionAwareChunk]:
        chunks = self.chunker.chunk(doc.content)
        result = []
        current_pos = 0

        for chunk_text in chunks:
            # Find chunk in original text starting from current position
            start = doc.content.find(chunk_text, current_pos)

            if start == -1:
                # Chunk text not found - chunker may have modified it
                # Log warning and skip this chunk
                import warnings
                warnings.warn(
                    f"Could not locate chunk in source document. "
                    f"Chunk may have been modified by chunker. Skipping. "
                    f"Chunk preview: {chunk_text[:50]}..."
                )
                continue

            end = start + len(chunk_text)

            result.append(PositionAwareChunk(
                id=self._generate_id(chunk_text),
                content=chunk_text,
                doc_id=doc.id,
                start=start,
                end=end,
            ))
            current_pos = end

        return result

    def _generate_id(self, content: str) -> PositionAwareChunkId:
        """
        Generate a position-aware chunk ID from content.

        Format: "pa_chunk_" + first 12 chars of SHA256 hash.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return PositionAwareChunkId(f"pa_chunk_{content_hash}")


def generate_chunk_id(content: str) -> ChunkId:
    """
    Generate a standard chunk ID from content.

    Format: "chunk_" + first 12 chars of SHA256 hash.

    Using content hash ensures:
    - Deterministic: same content always produces same ID
    - Deduplication: identical chunks have identical IDs
    - Stable: ID doesn't change based on processing order
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return ChunkId(f"chunk_{content_hash}")


def generate_pa_chunk_id(content: str) -> PositionAwareChunkId:
    """
    Generate a position-aware chunk ID from content.

    Format: "pa_chunk_" + first 12 chars of SHA256 hash.

    The "pa_" prefix distinguishes these from regular chunk IDs,
    making it immediately clear when working with position-aware data.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return PositionAwareChunkId(f"pa_chunk_{content_hash}")
```

---

### Synthetic Data Generation

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict

class SyntheticDataGenerator(ABC):
    """Base class for synthetic data generation."""

    def __init__(self, llm_client, corpus: Corpus):
        self.llm = llm_client
        self.corpus = corpus


class ChunkLevelDataGenerator(SyntheticDataGenerator):
    """
    Generate synthetic QA pairs with chunk-level ground truth.

    This generator requires a chunker because chunk IDs must exist before
    we can reference them in ground truth. The LLM generates queries AND
    identifies relevant chunks simultaneously (chunk-level citation).
    """

    def __init__(
        self,
        llm_client,
        corpus: Corpus,
        chunker: Chunker,  # Required: must chunk first to get chunk IDs
    ):
        super().__init__(llm_client, corpus)
        self.chunker = chunker
        self._chunk_registry = ChunkRegistry()

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: Optional[str] = None,
    ) -> List[ChunkLevelGroundTruth]:
        """
        Generate synthetic queries with relevant chunk IDs.

        Process:
        1. Chunk all documents, build chunk registry with IDs
        2. For each document's chunks:
           a. Present chunks with their IDs to the LLM
           b. Ask LLM to generate queries that can be answered by specific chunks
           c. LLM returns both the query AND the relevant chunk IDs (citations)
        3. Validate that returned chunk IDs exist in registry
        4. Upload to LangSmith and/or return ground truth pairs

        The key insight is that query generation and chunk citation happen
        together in a single LLM call, ensuring accurate ground truth.

        Example LLM prompt:
            "Here are chunks from a document:
             [chunk_a1b2c3d4]: 'RAG combines retrieval with generation...'
             [chunk_e5f6g7h8]: 'The benefits include reduced hallucination...'

             Generate 3 questions that can be answered using these chunks.
             For each question, list the chunk IDs that contain the answer.

             Format:
             Q: <question>
             Chunks: chunk_xxx, chunk_yyy"
        """
        ...


class TokenLevelDataGenerator(SyntheticDataGenerator):
    """
    Generate synthetic QA pairs with character span ground truth.

    This generator does NOT require a chunker upfront. Instead, it:
    1. Generates queries from document content
    2. Asks LLM to extract relevant excerpts (raw text)
    3. Finds character positions of excerpts in source document
    4. Creates position-aware chunks from these excerpts

    This approach is chunker-independent, allowing fair comparison of
    different chunking strategies against the same ground truth.
    """

    def __init__(
        self,
        llm_client,
        corpus: Corpus,
        # Note: NO chunker required - ground truth is excerpt positions
    ):
        super().__init__(llm_client, corpus)
        self._chunk_registry = ChunkRegistry()

    def generate(
        self,
        queries_per_doc: int = 5,
        upload_to_langsmith: bool = True,
        dataset_name: Optional[str] = None,
    ) -> List[TokenLevelGroundTruth]:
        """
        Generate synthetic queries with relevant character spans.

        Process:
        1. For each document:
           a. Ask LLM to generate queries about the document
           b. For each query, ask LLM to extract verbatim relevant excerpts
        2. For each excerpt:
           a. Find exact character positions in source document
           b. Create PositionAwareChunk with these positions
           c. Register chunk in registry
        3. Upload to LangSmith (only chunk IDs, not full text)
        4. Return ground truth with chunk IDs (resolve via registry)

        Advantages:
        - Same ground truth works with ANY chunking strategy
        - Can fairly compare different chunkers
        - Ground truth is based on actual relevant content, not chunk boundaries

        Example LLM prompt for excerpt extraction:
            "Document: <full document text>

             Question: What are the benefits of RAG?

             Extract the exact passages from the document that answer this
             question. Copy the text verbatim - do not paraphrase."
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

**Decision**: Use explicit separate classes for `ChunkLevelEvaluation` and `TokenLevelEvaluation`.
This is the most Pythonic approach and provides clear type safety.

```python
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC

@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    metrics: Dict[str, float]
    experiment_url: Optional[str]
    raw_results: Any  # Langsmith results object


class ChunkLevelEvaluation:
    """
    Evaluation using chunk-level metrics.

    Compares retrieved chunk IDs against ground truth chunk IDs.
    Metrics are binary: a chunk is either relevant or not.
    """

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ):
        self.corpus = corpus
        self.langsmith_dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Chunker,
        embedder: Embedder,
        k: int = 5,
        vector_store: Optional[VectorStore] = None,  # Optional, defaults to ChromaVectorStore
        reranker: Optional[Reranker] = None,         # Optional, defaults to None
        metrics: Optional[List[ChunkLevelMetric]] = None,
    ) -> EvaluationResult:
        """
        Run chunk-level evaluation.

        Args:
            chunker: Chunker to use for splitting documents.
            embedder: Embedder for generating vector representations.
            k: Number of chunks to retrieve per query.
            vector_store: Vector store for indexing/search. Defaults to ChromaVectorStore.
            reranker: Optional reranker to apply after retrieval.
            metrics: List of metrics to compute. Defaults to [ChunkRecall, ChunkPrecision, ChunkF1].

        Pipeline:
        1. Chunk corpus using chunker
        2. Generate chunk IDs (content hash with "chunk_" prefix)
        3. Embed and index chunks in vector store
        4. For each query in dataset:
           - Retrieve top-k chunks
           - Optionally rerank results
           - Compare retrieved chunk IDs vs ground truth chunk IDs
        5. Compute metrics (recall, precision, F1)

        Returns:
            EvaluationResult with computed metrics and experiment URL.
        """
        # Default vector store to ChromaDB if not provided
        if vector_store is None:
            vector_store = ChromaVectorStore()
        ...


class TokenLevelEvaluation:
    """
    Evaluation using token-level (character span) metrics.

    Compares character overlap between retrieved chunks and ground truth spans.
    Metrics are continuous: measures what fraction of relevant content was retrieved.
    """

    def __init__(
        self,
        corpus: Corpus,
        langsmith_dataset_name: str,
    ):
        self.corpus = corpus
        self.langsmith_dataset_name = langsmith_dataset_name

    def run(
        self,
        chunker: Union[Chunker, PositionAwareChunker],
        embedder: Embedder,
        k: int = 5,
        vector_store: Optional[VectorStore] = None,  # Optional, defaults to ChromaVectorStore
        reranker: Optional[Reranker] = None,         # Optional, defaults to None
        metrics: Optional[List[TokenLevelMetric]] = None,
    ) -> EvaluationResult:
        """
        Run token-level evaluation.

        Args:
            chunker: Chunker to use. Will be wrapped with PositionAdapter if needed.
            embedder: Embedder for generating vector representations.
            k: Number of chunks to retrieve per query.
            vector_store: Vector store for indexing/search. Defaults to ChromaVectorStore.
            reranker: Optional reranker to apply after retrieval.
            metrics: List of metrics to compute. Defaults to [SpanRecall, SpanPrecision, SpanIoU].

        Pipeline:
        1. Chunk corpus using chunker (wrapped with PositionAdapter if needed)
        2. Track chunk positions in source documents
        3. Embed and index chunks (store positions in vector store metadata)
        4. For each query in dataset:
           - Retrieve top-k chunks (with position metadata)
           - Optionally rerank results
           - Convert chunks to character spans
           - Compare retrieved spans vs ground truth spans (character overlap)
        5. Compute metrics (span recall, precision, IoU)

        Note on overlapping spans:
            Retrieved spans are merged before comparison. Each character
            is counted at most once to avoid inflating metrics.

        Returns:
            EvaluationResult with computed metrics and experiment URL.
        """
        # Default vector store to ChromaDB if not provided
        if vector_store is None:
            vector_store = ChromaVectorStore()

        # Wrap chunker if needed
        if isinstance(chunker, Chunker):
            chunker = ChunkerPositionAdapter(chunker)
        ...
```

---

### Metrics

```python
from abc import ABC, abstractmethod
from typing import List

class ChunkLevelMetric(ABC):
    """Metric for chunk-level evaluation."""

    @abstractmethod
    def calculate(
        self,
        retrieved_chunk_ids: List[ChunkId],
        ground_truth_chunk_ids: List[ChunkId]
    ) -> float:
        ...


class ChunkRecall(ChunkLevelMetric):
    """What fraction of relevant chunks were retrieved?"""

    def calculate(self, retrieved: List[ChunkId], ground_truth: List[ChunkId]) -> float:
        if not ground_truth:
            return 0.0
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        return len(retrieved_set & ground_truth_set) / len(ground_truth_set)


class ChunkPrecision(ChunkLevelMetric):
    """What fraction of retrieved chunks were relevant?"""

    def calculate(self, retrieved: List[ChunkId], ground_truth: List[ChunkId]) -> float:
        if not retrieved:
            return 0.0
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        return len(retrieved_set & ground_truth_set) / len(retrieved_set)


class ChunkF1(ChunkLevelMetric):
    """Harmonic mean of chunk precision and recall."""

    def calculate(self, retrieved: List[ChunkId], ground_truth: List[ChunkId]) -> float:
        recall = ChunkRecall().calculate(retrieved, ground_truth)
        precision = ChunkPrecision().calculate(retrieved, ground_truth)
        if recall + precision == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class TokenLevelMetric(ABC):
    """
    Metric for token-level (character span) evaluation.

    These metrics work with CharacterSpan objects and compute overlap
    at the character level for more granular evaluation.
    """

    @abstractmethod
    def calculate(
        self,
        retrieved_spans: List[CharacterSpan],
        ground_truth_spans: List[CharacterSpan]
    ) -> float:
        ...


class SpanRecall(TokenLevelMetric):
    """
    What fraction of ground truth characters were retrieved?

    Measures completeness: did we retrieve all the relevant content?

    Note: Overlapping retrieved spans are merged before calculation.
    Each character is counted at most once.
    """

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not ground_truth:
            return 0.0

        # Merge overlapping retrieved spans to avoid double-counting
        merged_retrieved = self._merge_spans(retrieved)

        total_gt_chars = sum(span.length() for span in ground_truth)

        # Calculate overlap (each GT char counted at most once)
        overlap_chars = self._calculate_total_overlap(ground_truth, merged_retrieved)

        return min(overlap_chars / total_gt_chars, 1.0)

    def _merge_spans(self, spans: List[CharacterSpan]) -> List[CharacterSpan]:
        """Merge overlapping spans within the same document."""
        # Implementation: sort by (doc_id, start), merge overlapping intervals
        ...

    def _calculate_total_overlap(
        self,
        spans_a: List[CharacterSpan],
        spans_b: List[CharacterSpan]
    ) -> int:
        """Calculate total character overlap, counting each char at most once."""
        ...


class SpanPrecision(TokenLevelMetric):
    """
    What fraction of retrieved characters were relevant?

    Measures efficiency: how much of what we retrieved was actually useful?

    Note: Overlapping retrieved spans are merged before calculation.
    Each character is counted at most once.
    """

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not retrieved:
            return 0.0

        # Merge overlapping retrieved spans
        merged_retrieved = self._merge_spans(retrieved)

        total_ret_chars = sum(span.length() for span in merged_retrieved)

        overlap_chars = self._calculate_total_overlap(merged_retrieved, ground_truth)

        return min(overlap_chars / total_ret_chars, 1.0)

    # ... same helper methods as SpanRecall


class SpanIoU(TokenLevelMetric):
    """
    Intersection over Union of character spans.

    Balances both precision and recall in a single metric.
    IoU = |intersection| / |union|

    Note: All spans are merged before calculation.
    """

    def calculate(
        self,
        retrieved: List[CharacterSpan],
        ground_truth: List[CharacterSpan]
    ) -> float:
        if not retrieved and not ground_truth:
            return 1.0
        if not retrieved or not ground_truth:
            return 0.0

        merged_retrieved = self._merge_spans(retrieved)
        merged_gt = self._merge_spans(ground_truth)

        intersection = self._calculate_total_overlap(merged_retrieved, merged_gt)

        total_retrieved = sum(span.length() for span in merged_retrieved)
        total_gt = sum(span.length() for span in merged_gt)
        union = total_retrieved + total_gt - intersection

        return intersection / union if union > 0 else 0.0
```

---

## LangSmith Dataset Schemas

### Chunk-Level Dataset

Stores only chunk IDs to minimize data size. Chunk content can be resolved via ChunkRegistry.

```json
{
  "name": "rag-eval-chunk-level-v1",
  "description": "Ground truth for chunk-level RAG evaluation",
  "example_schema": {
    "inputs": {
      "query": "string"
    },
    "outputs": {
      "relevant_chunk_ids": ["string (format: chunk_xxxxxxxxxx)"],
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
    "relevant_chunk_ids": ["chunk_a3f2b1c8d9e0", "chunk_7d9e4f2a1b3c", "chunk_1b3c5d7e9f0a"],
    "metadata": {
      "source_docs": ["rag_overview.md"],
      "generation_model": "gpt-4"
    }
  }
}
```

### Token-Level Dataset

Stores only position-aware chunk IDs. Character spans are resolved via ChunkRegistry lookup.

```json
{
  "name": "rag-eval-token-level-v1",
  "description": "Ground truth for token-level RAG evaluation (character spans)",
  "example_schema": {
    "inputs": {
      "query": "string"
    },
    "outputs": {
      "relevant_chunk_ids": ["string (format: pa_chunk_xxxxxxxxxx)"],
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
    "relevant_chunk_ids": ["pa_chunk_a3f2b1c8d9e0", "pa_chunk_7d9e4f2a1b3c"],
    "metadata": {
      "generation_model": "gpt-4"
    }
  }
}
```

**Note**: The actual text and character positions are NOT stored in the LangSmith dataset.
They can be looked up from the ChunkRegistry using the chunk IDs. This avoids:
- Duplicating text content across datasets and run outputs
- Bloating LangSmith storage with redundant data
- Making the dataset schema simpler and more consistent

---

## User-Facing API

**Decision**: Use explicit separate classes (`ChunkLevelEvaluation` and `TokenLevelEvaluation`).

```python
from rag_evaluation_framework import (
    Corpus,
    ChunkLevelEvaluation,
    TokenLevelEvaluation,
    RecursiveCharacterChunker,
    OpenAIEmbedder,
    ChromaVectorStore,
    CohereReranker,
)

corpus = Corpus.from_folder("./knowledge_base")

# =============================================================================
# CHUNK-LEVEL EVALUATION
# =============================================================================

eval = ChunkLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-chunk-dataset",
)

result = eval.run(
    chunker=RecursiveCharacterChunker(chunk_size=200),
    embedder=OpenAIEmbedder(),
    k=5,
    # vector_store=ChromaVectorStore(),  # Optional, defaults to ChromaVectorStore
    # reranker=CohereReranker(),          # Optional, defaults to None
)

# =============================================================================
# TOKEN-LEVEL EVALUATION
# =============================================================================

eval = TokenLevelEvaluation(
    corpus=corpus,
    langsmith_dataset_name="my-token-dataset",
)

result = eval.run(
    chunker=RecursiveCharacterChunker(chunk_size=200),
    embedder=OpenAIEmbedder(),
    k=5,
    # vector_store=ChromaVectorStore(),  # Optional, defaults to ChromaVectorStore
    # reranker=CohereReranker(),          # Optional, defaults to None
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
# Note: NO chunker required - ground truth is chunker-independent!
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
        k=5,
        # vector_store defaults to ChromaVectorStore
        # reranker defaults to None
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
)

# 1. Load corpus
corpus = Corpus.from_folder("./knowledge_base")

# 2. Choose chunker (this is fixed for this evaluation)
chunker = RecursiveCharacterChunker(chunk_size=200)

# 3. Generate synthetic data with this chunker
# LLM generates queries AND identifies relevant chunk IDs together
generator = ChunkLevelDataGenerator(
    llm_client=OpenAI(),
    corpus=corpus,
    chunker=chunker,  # Required! Ground truth is tied to this chunker.
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
    chunker=chunker,  # Must match the chunker used for data generation!
    embedder=OpenAIEmbedder(),
    k=5,
)
```

---

## Resolved Design Decisions

### 1. Chunk ID Format

**Decision**: Use content hash with descriptive prefixes.

- **Standard chunks**: `chunk_` + first 12 chars of SHA256 hash
  - Example: `chunk_a3f2b1c8d9e0`
- **Position-aware chunks**: `pa_chunk_` + first 12 chars of SHA256 hash
  - Example: `pa_chunk_7d9e4f2a1b3c`

Benefits:
- Prefixes make it immediately clear what type of chunk you're dealing with
- Content hash ensures determinism and deduplication
- 12 chars provides sufficient uniqueness for most corpora

### 2. Handling Overlapping Spans in Token-Level Metrics

**Decision**: Merge overlapping retrieved spans before comparison. Count each character at most once.

```
Chunk 1: [----chars 0-100----]
Chunk 2:        [----chars 50-150----]
Ground truth:   [--chars 60-90--]

After merging: [----chars 0-150----]
Overlap with GT: chars 60-90 = 30 chars (counted once)
```

This prevents sliding window chunkers from artificially inflating metrics.

### 3. Cross-Document Ground Truth

**Decision**: Yes, support queries with relevant spans from multiple documents.

```json
{
  "query": "Compare RAG and fine-tuning",
  "relevant_chunk_ids": ["pa_chunk_a1b2c3d4", "pa_chunk_e5f6g7h8"]
}
```

Where the chunks reference different source documents. This is realistic and the
span-based approach handles it naturally.

### 4. VectorStore Position Tracking

**Decision**: Store positions in vector store metadata, return with results.

```python
class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        chunks: List[PositionAwareChunk],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add chunks with their positions stored in metadata.

        The implementation should store doc_id, start, end in metadata
        so they can be returned with search results.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int
    ) -> List[PositionAwareChunk]:
        """
        Return chunks with position info reconstructed from metadata.
        """
        ...
```

Most vector stores (Chroma, Qdrant, Pinecone) support arbitrary metadata, so this
is widely compatible.

### 5. Adapter Failure Cases

**Decision**: Warn and skip problematic chunks, with clear documentation.

When the `ChunkerPositionAdapter` cannot find a chunk's text in the source document
(e.g., because the chunker normalized whitespace), it:
1. Logs a warning with the chunk preview
2. Skips that chunk
3. Continues processing remaining chunks

Most chunkers preserve text exactly, so this is rarely an issue. Documentation
will clearly state this limitation.

### 6. Chunker Interface

**Decision**: Keep two separate interfaces with adapter pattern.

- `Chunker`: Simple interface, returns `List[str]`
- `PositionAwareChunker`: Full interface, returns `List[PositionAwareChunk]`
- `ChunkerPositionAdapter`: Wraps `Chunker` to make it position-aware

This provides maximum flexibility:
- Simple chunkers remain simple
- Token-level evaluation can use any chunker via the adapter
- Users can implement `PositionAwareChunker` directly for full control

---

## Summary: Chunk-Level vs Token-Level

| Aspect | Chunk-Level | Token-Level |
|--------|-------------|-------------|
| Ground truth format | Chunk IDs (`chunk_xxx`) | PA Chunk IDs (`pa_chunk_xxx`) |
| Chunker for data gen | Required | Not needed |
| Compare chunkers fairly | No (tied to GT chunker) | Yes (chunker-independent GT) |
| Implementation complexity | Lower | Higher |
| Metric granularity | Binary (chunk relevant or not) | Continuous (% overlap) |
| Interface changes needed | None | Chunker position tracking |
| Best for | Quick iteration, simple cases | Research, chunker comparison |

**Recommendation**:
- Use **Token-Level** as the primary approach for comparing chunking strategies
- Use **Chunk-Level** when you need simpler setup and don't need fine-grained metrics

---

## Next Steps

1. **Define** final type definitions in `types.py`
2. **Implement** `PositionAwareChunker` interface and adapter
3. **Implement** `ChunkRegistry` for chunk lookup
4. **Implement** `TokenLevelDataGenerator` with excerpt extraction
5. **Implement** `ChunkLevelDataGenerator` with citation-style query generation
6. **Implement** span-based metrics with interval merging
7. **Implement** `TokenLevelEvaluation.run()`
8. **Implement** `ChunkLevelEvaluation.run()`
9. **Update** VectorStore interface for position metadata
10. **Write** comprehensive tests
11. **Document** with examples
