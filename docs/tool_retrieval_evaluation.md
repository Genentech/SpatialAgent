# Tool Retrieval Evaluation

Evaluation of LLM-based tool selection across 8 models on 8 representative bioinformatics tasks.

## Configuration

- **Min tools**: 5, **Max tools**: 20
- **Total available tools**: 72

## Test Tasks

| Task | Expected Tools |
|------|----------------|
| Panel Design | query_pubmed, search_czi_datasets, extract_czi_markers, search_panglao, search_cellmarker2, validate_genes_expression, aggregate_gene_voting |
| Cell Type Annotation | annotate_cell_types, search_panglao, search_cellmarker2 |
| Trajectory Analysis | scvelo_velocity, cellrank_fate_probabilities, paga_trajectory |
| Spatial Domain Detection | spagcn_clustering, graphst_clustering |
| Cell-Cell Communication | cellphonedb_analysis, liana_inference |
| Multimodal Integration | totalvi_integration, multivi_integration |
| Spatial Niche Analysis | annotate_tissue_niches, squidpy_spatial_neighbors, squidpy_nhood_enrichment |
| Literature Research | query_pubmed |

## Results

| Model | Model ID | Recall | Precision | F1 | Perfect Tasks |
|-------|----------|--------|-----------|-----|---------------|
| **gemini3pro** | gemini-3-pro-preview | 100.0% | 42.3% | 0.581 | 8/8 |
| **gemini2.5pro** | gemini-2.5-pro | 100.0% | 41.0% | 0.565 | 8/8 |
| **opus4** | claude-opus-4-1-20250805 | 100.0% | 39.5% | 0.550 | 8/8 |
| **sonnet45** | claude-sonnet-4-5-20250929 | 100.0% | 34.7% | 0.503 | 8/8 |
| **sonnet4** | claude-sonnet-4-20250514 | 100.0% | 34.4% | 0.501 | 8/8 |
| gpt4o | gpt-4o | 94.6% | 38.0% | 0.525 | 7/8 |
| gpt5 | gpt-5 | 93.8% | 32.9% | 0.477 | 7/8 |
| gpt4.1 | gpt-4.1 | 89.9% | 35.9% | 0.494 | 5/8 |

## Key Findings

- **Claude and Gemini models achieve 100% recall** (all expected tools selected across all tasks)
- **Gemini 3 Pro** has the best F1 due to higher precision (fewer unnecessary tools)
- **GPT models miss some tools**: GPT-4o misses on Panel Design, GPT-5 on CCI, GPT-4.1 on multiple tasks

## Recommendations

1. Use Claude or Gemini for tool selection to ensure 100% recall
2. Gemini models offer the best precision (fewer unnecessary tools selected)
3. Avoid GPT-4.1 for tool selection (89.9% recall)

## Implementation

The `LLMToolSelector` class in `spatialagent/agent/tool_system.py` handles selection:

```python
class LLMToolSelector:
    def __init__(
        self,
        registry: ToolRegistry,
        min_tools: int = 5,
        max_tools: int = 20,
        always_loaded_tools: List[str] = None,
    ):
        ...
```

The selection prompt asks the LLM to pick relevant tools from the catalog, considering database/search, analysis, visualization, and utility tools.
