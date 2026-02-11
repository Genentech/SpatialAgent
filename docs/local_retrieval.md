# Local Embedding Setup

SpatialAgent uses local embeddings by default for tool retrieval and database search tools. No API keys or rate limits needed.

## Available Models

| Short Name | Full Path | Description |
|------------|-----------|-------------|
| `qwen3-0.6b` (default) | Qwen/Qwen3-Embedding-0.6B | Best quality, on par with text-embedding-3-small |
| `pubmedbert` | pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb | Biomedical NLI, 768 dim |
| `biomedbert` | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract | Biomedical abstracts, 768 dim |

These are defined in `spatialagent/agent/make_llm.py` as `LOCAL_EMBEDDING_MODELS`.

## Configuration

Local embeddings are enabled by default. To switch to API-based embeddings:

```bash
export USE_LOCAL_EMBEDDINGS=false
```

To change the local model:

```bash
export LOCAL_EMBEDDING_MODEL=pubmedbert
```

## Usage

### Automatic (default)

The `ToolRegistry` in `tool_system.py` uses `qwen3-0.6b` by default for tool retrieval. Database search tools (`search_panglao`, `search_cellmarker2`, `extract_czi_markers`) also use local embeddings when `USE_LOCAL_EMBEDDINGS` is not set to `false`.

### Programmatic

```python
from spatialagent.agent import make_llm_emb, make_llm_emb_local

# Use default local model (qwen3-0.6b)
emb = make_llm_emb_local()

# Use a specific model
emb = make_llm_emb_local("pubmedbert")

# make_llm_emb() also defaults to local
emb = make_llm_emb()  # Returns local model unless USE_LOCAL_EMBEDDINGS=false

# Embed texts
vectors = emb.embed_documents(["CD4+ T cell", "B lymphocyte"])
query_vec = emb.embed_query("immune cell marker genes")
```

### Custom Models

Any HuggingFace sentence-transformers model works:

```python
from spatialagent.agent import LocalEmbeddings

emb = LocalEmbeddings("sentence-transformers/all-mpnet-base-v2")
```

## API Fallback

When `USE_LOCAL_EMBEDDINGS=false`, `make_llm_emb()` falls back to Azure OpenAI or a custom endpoint. See `make_llm.py` for configuration details.

## Troubleshooting

**Slow first load**: Models download from HuggingFace on first use. Set cache directory:
```bash
export HF_HOME=/path/to/cache
```

**Out of memory**: Use a smaller model like `pubmedbert` (768 dim, ~440MB) instead of larger alternatives.

**GPU**: sentence-transformers automatically uses GPU if available. Force CPU with:
```python
emb = LocalEmbeddings("qwen3-0.6b")
emb.model = emb.model.to("cpu")
```
