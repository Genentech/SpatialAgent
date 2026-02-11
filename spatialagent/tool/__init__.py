"""
Modern function-based tools for SpatialAgent.

All tools use simple functions with @tool decorator instead of classes.
Organized by functional type: databases, analytics, interpretation, literature, coding, and foundry.

Note: Most tools are direct @tool decorated functions. Coding tools use creator functions
because they need initialization parameters (save_path, data_path).
"""

# Database & Reference Query Tools (direct @tool functions)
from .databases import (
    search_panglao,
    search_cellmarker2,
    search_czi_datasets,
    extract_czi_markers,
    download_czi_reference,
    query_tissue_expression,
    query_celltype_genesets,
    validate_genes_expression,
    query_disease_genes,
)

# Literature Research Tools (direct @tool functions)
from .literature import (
    query_pubmed,
    query_arxiv,
    search_semantic_scholar,
    web_search,  # Unified web search using Anthropic/OpenAI/Google server-side tools
    # query_scholar,  # Disabled - hangs due to Google Scholar rate limits
    # search_duckduckgo,  # Disabled - blocked on many networks, overlaps with academic search
    extract_url_content,
    extract_pdf_content,
    fetch_supplementary_from_doi,
)

# Computational & Statistical Analysis Tools (direct @tool functions)
from .analytics import (
    preprocess_spatial_data,
    harmony_transfer_labels,
    run_utag_clustering,
    aggregate_gene_voting,
    infer_dynamics,
    summarize_conditions,
    summarize_celltypes,
    summarize_tissue_regions,
    # Tangram tools
    tangram_preprocess,
    tangram_map_cells,
    tangram_project_annotations,
    tangram_project_genes,
    tangram_evaluate,
    # CellPhoneDB tools
    cellphonedb_prepare,
    cellphonedb_analysis,
    cellphonedb_degs_analysis,
    cellphonedb_filter,
    cellphonedb_plot,
    # LIANA tools
    liana_tensor,
    liana_inference,
    liana_spatial,
    liana_misty,
    liana_plot,
    # Squidpy tools
    squidpy_spatial_neighbors,
    squidpy_nhood_enrichment,
    squidpy_co_occurrence,
    squidpy_spatial_autocorr,
    squidpy_ripley,
    squidpy_centrality,
    squidpy_interaction_matrix,
    squidpy_ligrec,
    # scvi-tools spatial deconvolution
    destvi_deconvolution,
    cell2location_mapping,
    stereoscope_deconvolution,
    gimvi_imputation,
    # Spatial domain detection (SpaGCN, GraphST)
    spagcn_clustering,
    graphst_clustering,
    # Scanpy tools
    scanpy_score_genes,
    scanpy_ingest,
    scanpy_bbknn,
    # Trajectory inference
    scvelo_velocity,
    scvelo_velocity_embedding,
    cellrank_terminal_states,
    cellrank_fate_probabilities,
    paga_trajectory,
    # Multimodal integration
    totalvi_integration,
    multivi_integration,
    mofa_integration,
)

# LLM-Powered Interpretation Tools (direct @tool functions)
from .interpretation import (
    annotate_cell_types,
    annotate_tissue_niches,
    interpret_figure,
)

# Subagent Tools (autonomous multi-step analysis)
from .subagent import (
    report_subagent,
    verification_subagent,
)

# CodeAct: Python REPL and Bash (creator functions - need initialization)
from .coding import create_python_repl_tool, create_bash_tool

# Code inspection: Retrieve and adapt tool source code (direct @tool function)
from .foundry import inspect_tool_code

__all__ = [
    # Database Tools
    "search_panglao",
    "search_cellmarker2",
    "search_czi_datasets",
    "extract_czi_markers",
    "download_czi_reference",
    "query_tissue_expression",
    "query_celltype_genesets",
    "validate_genes_expression",
    "query_disease_genes",
    # Literature Research Tools
    "query_pubmed",
    "query_arxiv",
    "search_semantic_scholar",
    "web_search",
    # "query_scholar",  # Disabled
    # "search_duckduckgo",  # Disabled
    "extract_url_content",
    "extract_pdf_content",
    "fetch_supplementary_from_doi",
    # Analytics Tools
    "preprocess_spatial_data",
    "harmony_transfer_labels",
    "run_utag_clustering",
    "aggregate_gene_voting",
    "infer_dynamics",
    "summarize_conditions",
    "summarize_celltypes",
    "summarize_tissue_regions",
    # Tangram Tools
    "tangram_preprocess",
    "tangram_map_cells",
    "tangram_project_annotations",
    "tangram_project_genes",
    "tangram_evaluate",
    # CellPhoneDB Tools
    "cellphonedb_prepare",
    "cellphonedb_analysis",
    "cellphonedb_degs_analysis",
    "cellphonedb_filter",
    "cellphonedb_plot",
    # LIANA Tools
    "liana_tensor",
    "liana_inference",
    "liana_spatial",
    "liana_misty",
    "liana_plot",
    # Squidpy Tools
    "squidpy_spatial_neighbors",
    "squidpy_nhood_enrichment",
    "squidpy_co_occurrence",
    "squidpy_spatial_autocorr",
    "squidpy_ripley",
    "squidpy_centrality",
    "squidpy_interaction_matrix",
    "squidpy_ligrec",
    # scvi-tools Spatial Deconvolution
    "destvi_deconvolution",
    "cell2location_mapping",
    "stereoscope_deconvolution",
    "gimvi_imputation",
    # Spatial Domain Detection
    "spagcn_clustering",
    "graphst_clustering",
    # Scanpy Tools
    "scanpy_score_genes",
    "scanpy_ingest",
    "scanpy_bbknn",
    # Trajectory Inference
    "scvelo_velocity",
    "scvelo_velocity_embedding",
    "cellrank_terminal_states",
    "cellrank_fate_probabilities",
    "paga_trajectory",
    # Multimodal Integration
    "totalvi_integration",
    "multivi_integration",
    "mofa_integration",
    # Interpretation Tools
    "annotate_cell_types",
    "annotate_tissue_niches",
    "interpret_figure",
    # Subagent Tools
    "report_subagent",
    "verification_subagent",
    # CodeAct (creator functions)
    "create_python_repl_tool",
    "create_bash_tool",
    # Support
    "inspect_tool_code",
]
