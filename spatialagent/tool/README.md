# SpatialAgent Tools

72 specialized tools for spatial transcriptomics and single-cell analysis.

## Tool Categories

### Database Tools (9) - `databases.py`

| Tool | Description |
|------|-------------|
| `search_panglao` | Query PanglaoDB for marker genes by cell type and tissue |
| `search_cellmarker2` | Query CellMarker 2.0 for tissue-specific markers |
| `search_czi_datasets` | Search CZI CELLxGENE Census for matching reference datasets |
| `extract_czi_markers` | Extract cell types and marker genes from CZI for panel design |
| `download_czi_reference` | Download CZI reference h5ad for Harmony annotation |
| `query_tissue_expression` | Query gene expression levels in specific tissues (ARCHS4) |
| `query_celltype_genesets` | Get gene sets associated with cell types (Enrichr) |
| `validate_genes_expression` | Validate marker genes against expression data |
| `query_disease_genes` | Query disease-associated genes from GWAS/OpenTargets |

### Literature Tools (7) - `literature.py`

| Tool | Description |
|------|-------------|
| `query_pubmed` | Search PubMed using Biopython Entrez (handles special chars) |
| `query_arxiv` | Search arXiv preprints |
| `search_semantic_scholar` | Search Semantic Scholar (free API, includes citation counts) |
| `web_search` | Unified web search using provider APIs (Anthropic/OpenAI/Google) |
| `extract_url_content` | Extract content from a URL |
| `extract_pdf_content` | Extract text from PDF files |
| `fetch_supplementary_from_doi` | Fetch supplementary materials from DOI |

*Disabled:* `query_scholar` (rate limits), `search_duckduckgo` (blocked on institutional networks)

### Analytics Tools (48) - `analytics.py`

#### Preprocessing & Basic Analysis (5)
| Tool | Description |
|------|-------------|
| `preprocess_spatial_data` | QC, normalization, HVG selection, PCA, UMAP |
| `harmony_transfer_labels` | Batch correction and label transfer from reference |
| `scanpy_score_genes` | Score gene signatures per cell |
| `scanpy_ingest` | Map query data to reference using Scanpy ingest |
| `scanpy_bbknn` | Batch-balanced KNN for integration |

#### Spatial Clustering (3)
| Tool | Description |
|------|-------------|
| `run_utag_clustering` | Graph-based spatial clustering (UTAG algorithm) |
| `spagcn_clustering` | SpaGCN spatial clustering with histology integration |
| `graphst_clustering` | GraphST self-supervised spatial clustering |

#### Cell-Cell Communication - LIANA (5)
| Tool | Description |
|------|-------------|
| `liana_inference` | Run LIANA ligand-receptor inference |
| `liana_spatial` | Spatial CCI analysis with LIANA |
| `liana_misty` | LIANA MISTy multi-view learning |
| `liana_tensor` | LIANA tensor decomposition (Cell2Cell + TensorLy) |
| `liana_plot` | Generate LIANA visualization plots |

#### Cell-Cell Communication - CellPhoneDB (5)
| Tool | Description |
|------|-------------|
| `cellphonedb_prepare` | Prepare data for CellPhoneDB analysis |
| `cellphonedb_analysis` | Run CellPhoneDB statistical analysis |
| `cellphonedb_degs_analysis` | DEGs-based CellPhoneDB analysis |
| `cellphonedb_filter` | Filter CellPhoneDB results |
| `cellphonedb_plot` | Generate CellPhoneDB plots (heatmaps, dotplots) |

#### Spatial Statistics - Squidpy (8)
| Tool | Description |
|------|-------------|
| `squidpy_spatial_neighbors` | Compute spatial neighborhood graph |
| `squidpy_nhood_enrichment` | Neighborhood enrichment analysis |
| `squidpy_co_occurrence` | Cell type co-occurrence analysis |
| `squidpy_spatial_autocorr` | Spatial autocorrelation (Moran's I, Geary's C) |
| `squidpy_ripley` | Ripley's statistics for point patterns |
| `squidpy_centrality` | Graph centrality scores |
| `squidpy_interaction_matrix` | Cell type interaction matrix |
| `squidpy_ligrec` | Squidpy ligand-receptor analysis |

#### Spatial Mapping - Tangram (5)
| Tool | Description |
|------|-------------|
| `tangram_preprocess` | Preprocess data for Tangram mapping |
| `tangram_map_cells` | Map single cells to spatial locations |
| `tangram_project_annotations` | Project cell type annotations to spatial data |
| `tangram_project_genes` | Project gene expression to spatial data |
| `tangram_evaluate` | Evaluate Tangram mapping quality |

#### Deconvolution (4)
| Tool | Description |
|------|-------------|
| `destvi_deconvolution` | DestVI spatial deconvolution |
| `cell2location_mapping` | Cell2location spatial mapping |
| `stereoscope_deconvolution` | Stereoscope deconvolution |
| `gimvi_imputation` | gimVI gene imputation |

#### Trajectory Inference (5)
| Tool | Description |
|------|-------------|
| `scvelo_velocity` | RNA velocity analysis with scVelo |
| `scvelo_velocity_embedding` | Velocity embedding visualization |
| `cellrank_terminal_states` | Identify terminal states with CellRank |
| `cellrank_fate_probabilities` | Compute fate probabilities |
| `paga_trajectory` | PAGA trajectory inference |

#### Multi-Modal Integration (3)
| Tool | Description |
|------|-------------|
| `totalvi_integration` | TotalVI RNA+protein integration |
| `multivi_integration` | MultiVI multi-modal integration |
| `mofa_integration` | MOFA+ factor analysis |

#### Summary & Dynamics (5)
| Tool | Description |
|------|-------------|
| `aggregate_gene_voting` | LLM-based marker gene voting |
| `infer_dynamics` | Cross-condition dynamics analysis |
| `summarize_conditions` | Condition-level summaries |
| `summarize_celltypes` | Cell type distribution analysis |
| `summarize_tissue_regions` | Spatial niche summaries |

### Interpretation Tools (3) - `interpretation.py`

| Tool | Description |
|------|-------------|
| `annotate_cell_types` | Two-level hierarchical cell type annotation using LLM |
| `annotate_tissue_niches` | Batch niche annotation with anatomical reference |
| `interpret_figure` | Vision LLM figure interpretation |

### Subagent Tools (2) - `subagent.py`

| Tool | Description |
|------|-------------|
| `report_subagent` | Multi-pass autonomous analysis for publication-quality reports |
| `verification_subagent` | Verify claims against evidence |

### Code Execution (2) - `coding.py`

| Tool | Description |
|------|-------------|
| `execute_python` | Stateful Python REPL (numpy, pandas, scanpy pre-imported) |
| `execute_bash` | Bash command execution |

### Foundry Tools (1) - `foundry.py`

| Tool | Description |
|------|-------------|
| `inspect_tool_code` | View source code of any tool |

## Usage

### Through Agent (Recommended)

```python
from spatialagent.agent import SpatialAgent, make_llm

llm = make_llm("claude-sonnet-4-5-20250929")
agent = SpatialAgent(llm=llm, save_path="./experiments/")

# Agent auto-selects tools based on the task
result = agent.run("Annotate cell types in my spatial data at './data/spatial.h5ad'")
```

### Web Search Configuration

The `web_search` tool supports multiple providers (Anthropic, OpenAI, Google). By default, it uses `gemini-3-flash-preview` for fast and cost-effective searches regardless of the agent's LLM.

```python
# Default: uses gemini-3-flash-preview for web search (fast & cheap)
agent = SpatialAgent(llm=llm)

# Use agent's model for web search (same provider as agent)
agent = SpatialAgent(llm=llm, web_search_model=None)

# Force a specific model for web search
agent = SpatialAgent(llm=llm, web_search_model="claude-sonnet-4-5-20250929")
```

| `web_search_model` | Web search uses |
|--------------------|-----------------|
| `"gemini-3-flash-preview"` (default) | Google Gemini 3 Flash |
| `None` | Agent's model/provider |
| `"claude-*"` | Anthropic |
| `"gpt-*"` | OpenAI |

### Direct Invocation

```python
from spatialagent.tool import databases, analytics, literature

# Database query
result = databases.search_czi_datasets.invoke({
    "query": "human lung normal",
    "n_datasets": 3
})

# Literature search
papers = literature.query_pubmed.invoke({
    "query": "spatial transcriptomics cell-cell communication",
    "max_results": 10
})

# Analytics
result = analytics.liana_inference.invoke({
    "adata_path": "./data/spatial.h5ad",
    "groupby": "cell_type"
})

# Web search (unified provider function)
from spatialagent.tool.literature import web_search
result = web_search("latest scanpy version", provider="google")
```

## Architecture

Tools use the `@tool` decorator pattern with lazy imports for heavy libraries.

Tools support both calling conventions:
```python
# Dict style (used by Claude models)
tool_name({"param1": value1, "param2": value2})

# Kwargs style (used by GPT/Gemini models)
tool_name(param1=value1, param2=value2)
```

## File Structure

```
tool/
├── __init__.py          # Exports all tools
├── databases.py         # Database tools (9)
├── literature.py        # Literature tools (7)
├── analytics.py         # Analytics tools (48)
├── interpretation.py    # Interpretation tools (3)
├── subagent.py          # Subagent tools (2)
├── coding.py            # Code execution (2)
├── foundry.py           # Tool inspection (1)
└── utils.py             # Embedding similarity utilities
```
