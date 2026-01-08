# RAG Evaluation Framework

This is a RAG Evaluation Framework which helps you quickly run evaluations using Langsmith SDK. 

## Pipeline:

1. Pre-processing Data (kb aka knowledge base)
2. Synthetic Data Generation
3. Chunking Strategy
4. Embedding model
	4.1 Custom Embedding model (for adding vector store or db)
5. @k parameter aka retrieved documents
6. Re-ranker (optional)

## API

```py
from rag_evaluation_framework import Evaluation

evaluation = Evaluation(
    langsmith_dataset_name="my-dataset",
    kb_data_path="./knowledge_base"
)

eval_results = evaluation.run(
    chunker=my_chunker,
    embedder=my_embedder,
    vector_store=my_vector_store,  # optional, defaults to Chroma
    k=5,
    reranker=my_reranker  # optional
)
```


