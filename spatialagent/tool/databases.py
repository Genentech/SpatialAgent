"""
Database and Reference Query Tools

External knowledge retrieval tools that query biological databases
for marker genes, cell type information, and reference datasets.

All tools are standalone functions following Biomni pattern.
"""

# Lightweight imports - keep at module level
import os
import pandas as pd
import numpy as np
from typing import Annotated, Literal
from os.path import exists
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.tools import tool
from pydantic import Field

from .utils import find_most_similar, _embed_with_retry, _get_cache_key, _load_cached_embeddings, _save_cached_embeddings


# Module-level config (set via configure_database_tools)
_config = {
    "data_path": "./data",
}

def configure_database_tools(data_path: str = "./data"):
    """Configure paths for database tools. Call this before using the tools."""
    _config["data_path"] = data_path

def get_data_path() -> str:
    """Get the configured data path."""
    return _config["data_path"]

# NOTE: text-embedding-3-small outperforms local models (pubmedbert, bge-large) for
# database cell type matching. Benchmark (2025-12-30) showed:
#   - text-embedding-3-small: 100% accuracy (5/5 queries)
#   - pubmedbert (local): 60% accuracy (3/5 queries) - fails on immune cells
# Local models incorrectly match "T cells" -> "Neurons", "B cells" -> "Satellite glial cells"
# See: experiments/embedding_benchmark/DATABASE_BENCHMARK_RESULTS.md
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# =============================================================================
# Tool 1: PanglaoDB Search
# =============================================================================

@tool
def search_panglao(
    cell_types: Annotated[str, Field(description="Cell types to search - either comma-separated list (e.g., 'T cell, B cell, macrophage') or path to CSV file with 'cell_type' column")],
    organism: Annotated[Literal["Hs", "Mm"], Field(description="Species: Hs (human) or Mm (mouse)")],
    tissue: Annotated[str, Field(description="Target tissue context (e.g., brain, liver, pancreas)")],
    save_path: Annotated[str, Field(description="Directory to save results (optional)")] = None,
    iter_round: Annotated[int, Field(ge=1, le=3, description="Iteration round for panel design (1-3)")] = None,
) -> str:
    """Search PanglaoDB for marker genes of given cell types.

    Examples:
        - Direct input: cell_types="T cell, B cell, macrophage"
        - From CSV: cell_types="/path/to/czi_reference_celltype_1.csv"
        - For panel design: iter_round=1 produces pangdb_celltype_1.csv

    Returns:
        str: Formatted string with marker genes. Example output:
            PanglaoDB Results (Hs, brain):

            T cell (matched: T cells):
              Marker genes (15): ['CD3D', 'CD3E', 'CD4', ...]

            B cell (matched: B cells):
              Marker genes (12): ['CD19', 'MS4A1', 'CD79A', ...]

            Saved to: /path/to/pangdb_celltype_1.csv
    """
    from ..agent import make_llm_emb, get_effective_embedding_model

    # Get the actual embedding model that will be used (may be overridden by env vars)
    effective_model = get_effective_embedding_model(DEFAULT_EMBEDDING_MODEL)

    # Create embedding models with appropriate input_type for Cohere
    llm_embed_query = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_query")
    llm_embed_doc = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_document")

    # Parse cell_types input - either CSV path or comma-separated list
    if cell_types.endswith('.csv') and os.path.exists(cell_types):
        df_input = pd.read_csv(cell_types).astype(str)
        if 'cell_type' not in df_input.columns:
            return f"ERROR: CSV file must have 'cell_type' column. Found columns: {list(df_input.columns)}"
        cell_type_list = df_input['cell_type'].unique().tolist()
    else:
        cell_type_list = [ct.strip() for ct in cell_types.split(',')]

    if not cell_type_list:
        return "ERROR: No cell types provided"

    # Load PanglaoDB
    panglao_path = f"{get_data_path()}/PanglaoDB_markers_27_Mar_2020.tsv"
    df_panglao = pd.read_csv(panglao_path, sep="\t")
    df_panglao = df_panglao[df_panglao["species"].str.contains(organism, na=False)]

    # Create semantic descriptions for matching
    query_descriptions = [f"{organism}; {ct}; {tissue}" for ct in cell_type_list]
    df_panglao["description_all"] = (
        df_panglao[["species", "cell type", "organ"]].astype(str).agg("; ".join, axis=1)
    )
    db_descriptions = list(df_panglao["description_all"].unique())

    # Match using embeddings with correct input_type
    # Use effective_model for cache key to handle env var overrides (e.g., USE_LOCAL_EMBEDDINGS)
    matched = find_most_similar(
        llm_embed_query, query_descriptions, db_descriptions,
        llm_emb_doc=llm_embed_doc,
        database=f"panglao_{organism}",
        embedding_model=effective_model
    )

    # Extract marker genes (filter out nan values and duplicates)
    # Use column names expected by score_gene_importance tool
    res = {"cell_type": [], "cell_type_pangdb": [], "marker_genes": []}
    for cell_type, query_desc, panglao_match in zip(cell_type_list, query_descriptions, matched):
        res["cell_type"].append(cell_type)
        res["cell_type_pangdb"].append(panglao_match.split(";")[1].strip())
        markers = df_panglao.loc[
            df_panglao["description_all"] == panglao_match,
            "official gene symbol"
        ].dropna().unique().tolist()
        res["marker_genes"].append(markers)

    # Save if path provided
    df_result = pd.DataFrame(res)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # Use iteration-specific filename if iter_round is provided
        if iter_round is not None:
            save_csv = f"{save_path}/pangdb_celltype_{iter_round}.csv"
        else:
            save_csv = f"{save_path}/pangdb_markers.csv"
        df_result.to_csv(save_csv, index=False)

    # Build clear output showing marker genes for each cell type
    output_lines = [f"PanglaoDB Results ({organism}, {tissue}):"]
    for _, row in df_result.iterrows():
        genes = row['marker_genes']
        n_genes = len(genes) if isinstance(genes, list) else 0
        output_lines.append(f"\n{row['cell_type']} (matched: {row['cell_type_pangdb']}):")
        output_lines.append(f"  Marker genes ({n_genes}): {genes}")

    if save_path:
        output_lines.append(f"\nSaved to: {save_csv}")

    output = "\n".join(output_lines)
    print(output)
    return output


# =============================================================================
# Tool 2: CZI Dataset Retrieval
# =============================================================================

@tool
def search_czi_datasets(
    query: Annotated[str, Field(description="Query describing tissue, condition, and organism (e.g., 'liver, breast cancer, Homo sapiens')")],
    n_datasets: Annotated[int, Field(description="Number of top datasets to return")] = 1,
) -> str:
    """Search CZI CELLxGENE Census for reference single-cell datasets.

    Input is a string containing: tissue, condition, and organism.

    Returns:
        str: Formatted string with dataset info. Example output:
            CZI CELLxGENE Search Results for 'heart, Homo sapiens':

            Dataset 1:
              dataset_id: e6a11140-2545-46bc-929e-da243eed2cae
              dataset_title: Tabula Sapiens - Heart
              collection_name: Tabula Sapiens
              organism: Homo sapiens
              tissue: cardiac atrium;cardiac ventricle
              disease: normal
              similarity_score: 0.666

        To extract dataset_id from the output string:
            for line in result.split('\\n'):
                if 'dataset_id:' in line:
                    dataset_id = line.split('dataset_id:')[1].strip()
                    break
    """
    from ..agent import make_llm_emb, get_effective_embedding_model

    # Get the actual embedding model that will be used (may be overridden by env vars)
    effective_model = get_effective_embedding_model(DEFAULT_EMBEDDING_MODEL)

    # Create embedding models with appropriate input_type for Cohere
    llm_embed_query = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_query")
    llm_embed_doc = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_document")

    # Load CZI Census metadata
    metadata_path = f"{get_data_path()}/czi_census_datasets_v4_short.csv"

    if not os.path.exists(metadata_path):
        return f"ERROR: CZI metadata file not found at {metadata_path}"

    df = pd.read_csv(metadata_path)

    # Create descriptions for each dataset
    df["description"] = (
        df["organism"].astype(str) + "; " +
        df["tissue"].astype(str) + "; " +
        df["disease"].astype(str) + "; " +
        df["dataset_title"].astype(str)
    )

    descriptions = df["description"].tolist()

    # Embed query with retry
    query_embedding = _embed_with_retry(llm_embed_query, [query])

    # Check for cached description embeddings (use effective_model for cache key)
    cache_key = _get_cache_key("czi_census", effective_model, len(descriptions))
    desc_embeddings = _load_cached_embeddings(cache_key)

    if desc_embeddings is None:
        # Embed descriptions with retry
        desc_embeddings = _embed_with_retry(llm_embed_doc, descriptions)
        _save_cached_embeddings(cache_key, desc_embeddings)

    # Find top-k most similar
    similarities = cosine_similarity(query_embedding, desc_embeddings)[0]
    top_indices = np.argsort(similarities)[-n_datasets:][::-1]

    # Format results
    results = [f"CZI CELLxGENE Search Results for '{query}':"]
    for idx in top_indices:
        row = df.iloc[idx]
        similarity = similarities[idx]

        result_str = (
            f"\nDataset {len(results)}:\n"
            f"  dataset_id: {row['dataset_id']}\n"
            f"  dataset_title: {row['dataset_title']}\n"
            f"  collection_name: {row['collection_name']}\n"
            f"  organism: {row['organism']}\n"
            f"  tissue: {row['tissue']}\n"
            f"  disease: {row['disease']}\n"
            f"  similarity_score: {similarity:.3f}"
        )
        results.append(result_str)

    output = "\n".join(results)
    print(output)
    return output


# =============================================================================
# Tool 3: CellMarker2 Search
# =============================================================================

@tool
def search_cellmarker2(
    cell_types: Annotated[str, Field(description="Cell types to search - either comma-separated list (e.g., 'hepatocyte, Kupffer cell') or path to CSV file with 'cell_type' column")],
    organism: Annotated[str, Field(description="Organism: Human or Mouse")],
    tissue: Annotated[str, Field(description="Target tissue context (e.g., liver, brain)")],
    save_path: Annotated[str, Field(description="Directory to save results (optional)")] = None,
    iter_round: Annotated[int, Field(ge=1, le=3, description="Iteration round for panel design (1-3)")] = None,
) -> str:
    """Search CellMarker2 database for marker genes of given cell types.

    Examples:
        - Direct input: cell_types="hepatocyte, Kupffer cell, stellate cell"
        - From CSV: cell_types="/path/to/czi_reference_celltype_1.csv"
        - For panel design: iter_round=1 produces cellmarker_celltype_1.csv

    Returns:
        str: Formatted string with marker genes. Example output:
            CellMarker2 Results (Human, liver):

            hepatocyte (matched: Hepatocyte):
              Marker genes (25): ['ALB', 'APOA1', 'APOB', ...]

            Kupffer cell (matched: Kupffer cell):
              Marker genes (18): ['CD68', 'MARCO', 'CLEC4F', ...]

            Saved to: /path/to/cellmarker_celltype_1.csv
    """
    from ..agent import make_llm_emb, get_effective_embedding_model

    # Get the actual embedding model that will be used (may be overridden by env vars)
    effective_model = get_effective_embedding_model(DEFAULT_EMBEDDING_MODEL)

    # Create embedding models with appropriate input_type for Cohere
    llm_embed_query = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_query")
    llm_embed_doc = make_llm_emb(DEFAULT_EMBEDDING_MODEL, input_type="search_document")

    # Parse cell_types input - either CSV path or comma-separated list
    if cell_types.endswith('.csv') and os.path.exists(cell_types):
        df_input = pd.read_csv(cell_types).astype(str)
        if 'cell_type' not in df_input.columns:
            return f"ERROR: CSV file must have 'cell_type' column. Found columns: {list(df_input.columns)}"
        cell_type_list = df_input['cell_type'].unique().tolist()
    else:
        cell_type_list = [ct.strip() for ct in cell_types.split(',')]

    if not cell_type_list:
        return "ERROR: No cell types provided"

    # Load CellMarker2
    df_cellmarker2 = pd.read_csv(f"{get_data_path()}/Cell_marker_All.csv")
    df_cellmarker2 = df_cellmarker2[df_cellmarker2["species"].str.contains(organism, na=False)]
    df_cellmarker2 = df_cellmarker2.astype("str")

    # Create semantic descriptions for matching
    query_descriptions = [f"{organism}; {ct}; {tissue}" for ct in cell_type_list]
    df_cellmarker2["description_all"] = (
        df_cellmarker2[["species", "cell_type", "cell_name", "tissue_class", "tissue_type"]]
        .astype(str)
        .agg("; ".join, axis=1)
    )
    db_descriptions = list(df_cellmarker2["description_all"].unique())

    # Match using embeddings with correct input_type
    # Use effective_model for cache key to handle env var overrides (e.g., USE_LOCAL_EMBEDDINGS)
    matched = find_most_similar(
        llm_embed_query, query_descriptions, db_descriptions,
        llm_emb_doc=llm_embed_doc,
        database=f"cellmarker2_{organism}",
        embedding_model=effective_model
    )

    # Extract markers (filter out nan values and duplicates)
    # Use column names expected by score_gene_importance tool
    res = {"cell_type": [], "cell_type_cellmarker": [], "marker_genes": []}
    for cell_type, query_desc, cm2_match in zip(cell_type_list, query_descriptions, matched):
        res["cell_type"].append(cell_type)
        res["cell_type_cellmarker"].append(cm2_match.split(";")[2].strip())
        markers = df_cellmarker2.loc[
            df_cellmarker2["description_all"] == cm2_match, "Symbol"
        ].dropna().unique().tolist()
        # Filter out 'nan' strings as well (from .astype(str) conversion)
        markers = [m for m in markers if m.lower() != 'nan']
        res["marker_genes"].append(markers)

    # Save if path provided
    df_result = pd.DataFrame(res)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # Use iteration-specific filename if iter_round is provided
        if iter_round is not None:
            save_csv = f"{save_path}/cellmarker_celltype_{iter_round}.csv"
        else:
            save_csv = f"{save_path}/cellmarker2_markers.csv"
        df_result.to_csv(save_csv, index=False)

    # Build clear output showing marker genes for each cell type
    output_lines = [f"CellMarker2 Results ({organism}, {tissue}):"]
    for _, row in df_result.iterrows():
        genes = row['marker_genes']
        n_genes = len(genes) if isinstance(genes, list) else 0
        output_lines.append(f"\n{row['cell_type']} (matched: {row['cell_type_cellmarker']}):")
        output_lines.append(f"  Marker genes ({n_genes}): {genes}")

    if save_path:
        output_lines.append(f"\nSaved to: {save_csv}")

    output = "\n".join(output_lines)
    print(output)
    return output


# =============================================================================
# Tool 4: CZI Data Reader
# =============================================================================

@tool
def extract_czi_markers(
    save_path: Annotated[str, Field(description="Experiment directory")],
    dataset_id: Annotated[str, Field(description="CZI dataset ID or comma-separated list")],
    iter_round: Annotated[int, Field(ge=1, le=3, description="Iteration round (1-3)")],
    organism: Annotated[Literal["Homo sapiens", "Mus musculus"], Field(description="Organism species")] = "Mus musculus",
) -> str:
    """Read and process CZI database reference datasets to extract cell types and marker genes.

    This tool downloads the dataset from CZI Census, extracts cell types and their marker genes,
    and saves the results to: {save_path}/czi_reference_celltype_{iter_round}.csv

    Use different iter_round values (1, 2, 3) when performing iterative panel design to avoid
    overwriting previous results.

    Returns:
        str: Status message. Example output:
            Successfully processed 1 CZI dataset(s) with 18 cell types. Saved to /path/to/czi_reference_celltype_1.csv
            Note: 15 cell types have marker genes from CellGuide, 3 do not (may need PanglaoDB/CellMarker2 lookup).

        The saved CSV file contains columns: cell_type, cell_type_id, n_cells, marker_genes, cano_marker_genes
    """
    # Heavy imports - only load when this tool is called
    import cellxgene_census
    import requests

    CELL_GUIDE_BASE_URI = "https://cellguide.cellxgene.cziscience.com"
    LATEST_SNAPSHOT = requests.get(f"{CELL_GUIDE_BASE_URI}/latest_snapshot_identifier").text.replace('\n', '')

    def get_cellguide_file(relpth, snapshot=LATEST_SNAPSHOT):
        req = requests.get(f"{CELL_GUIDE_BASE_URI}/{snapshot}/{relpth}")
        if req.text == "":
            raise ValueError(f"No record found for {snapshot}/{relpth}")
        return req

    save_csv = f"{save_path}/czi_reference_celltype_{iter_round}.csv"

    if exists(save_csv):
        msg = f"CZI reference data already exists at {save_csv}"
        print(msg)
        return msg

    print(f"[extract_czi_markers] Processing {dataset_id}...")

    # Handle multiple dataset IDs
    dataset_ids = [d.strip() for d in dataset_id.split(",")]

    # Read from CZI Census
    census = cellxgene_census.open_soma(census_version="latest")

    all_results = []

    for did in dataset_ids:
        # Query dataset
        query = f'dataset_id == "{did}"'
        adata = cellxgene_census.get_anndata(census, organism, obs_value_filter=query)

        # Get cell types
        cell_types = adata.obs["cell_type"].value_counts()

        for cell_type, count in cell_types.items():
            # Get cell_type_id with error handling for empty results
            cell_type_mask = adata.obs["cell_type"] == cell_type
            cell_type_ids = adata.obs[cell_type_mask]["cell_type_ontology_term_id"].values

            if len(cell_type_ids) == 0:
                continue  # Skip if no cell_type_ontology_term_id found

            cell_type_id = cell_type_ids[0]

            # Get marker genes from CellGuide
            # Note: CellGuide uses underscore format (CL_0000182) not colon format (CL:0000182)
            cellguide_id = cell_type_id.replace(":", "_")

            comp_genes = []
            cano_genes = []

            # Limit marker genes to top N for readability (CellGuide can return 500+ genes)
            MAX_MARKER_GENES = 100

            # Helper to convert gene symbols based on organism
            # CellGuide returns mouse-format symbols (title case like 'Grin2b')
            # Human genes should be uppercase (GRIN2B), mouse stays title case
            def normalize_gene_symbol(gene: str) -> str:
                if organism == "Homo sapiens":
                    return gene.upper()
                return gene  # Keep mouse format as-is

            try:
                comp_markers = get_cellguide_file(f"computational_marker_genes/{cellguide_id}.json")
                if comp_markers.status_code == 200 and comp_markers.text:
                    comp_markers_df = pd.DataFrame.from_records(comp_markers.json())
                    # Gene symbol is in 'symbol' column, not 'marker_gene'
                    if "symbol" in comp_markers_df.columns:
                        comp_genes = [normalize_gene_symbol(g) for g in comp_markers_df["symbol"].tolist()[:MAX_MARKER_GENES]]
                    elif "marker_gene" in comp_markers_df.columns:
                        comp_genes = [normalize_gene_symbol(g) for g in comp_markers_df["marker_gene"].tolist()[:MAX_MARKER_GENES]]
            except Exception:
                pass  # CellGuide may not have data for all cell types

            try:
                cano_markers = get_cellguide_file(f"canonical_marker_genes/{cellguide_id}.json")
                if cano_markers.status_code == 200 and cano_markers.text:
                    cano_markers_df = pd.DataFrame.from_records(cano_markers.json())
                    if "symbol" in cano_markers_df.columns:
                        cano_genes = [normalize_gene_symbol(g) for g in cano_markers_df["symbol"].tolist()[:MAX_MARKER_GENES]]
                    elif "marker_gene" in cano_markers_df.columns:
                        cano_genes = [normalize_gene_symbol(g) for g in cano_markers_df["marker_gene"].tolist()[:MAX_MARKER_GENES]]
            except Exception:
                pass

            all_results.append({
                "cell_type": cell_type,
                "cell_type_id": cell_type_id,
                "n_cells": count,
                "marker_genes": comp_genes,
                "cano_marker_genes": cano_genes,
            })

    # Save
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(save_csv, index=False)

    # Count how many cell types have marker genes
    n_with_markers = sum(1 for r in all_results if r["marker_genes"] or r["cano_marker_genes"])
    n_without_markers = len(all_results) - n_with_markers

    msg = f"Successfully processed {len(dataset_ids)} CZI dataset(s) with {len(all_results)} cell types. Saved to {save_csv}"
    if n_without_markers > 0:
        msg += f"\nNote: {n_with_markers} cell types have marker genes from CellGuide, {n_without_markers} do not (may need PanglaoDB/CellMarker2 lookup)."

    print(msg)
    return msg


# =============================================================================
# Tool 5: Download CZI Reference (for Harmony integration)
# =============================================================================

@tool
def download_czi_reference(
    dataset_id: Annotated[str, Field(description="CZI dataset ID from search_czi_datasets")],
    organism: Annotated[str, Field(description="Species: 'Homo sapiens' or 'Mus musculus'")],
    save_path: Annotated[str, Field(description="Experiment directory")],
) -> str:
    """Download CZI reference scRNA-seq data for cell type annotation via Harmony integration.

    This tool downloads a reference scRNA-seq dataset from CZI Census and saves it as h5ad.
    The downloaded reference can then be used with harmony_transfer_labels to transfer
    cell type annotations to spatial transcriptomics data.

    Workflow:
    1. Use search_czi_datasets to find a matching reference dataset
    2. Use download_czi_reference to download the reference h5ad (this tool)
    3. Use harmony_transfer_labels to transfer cell types from reference to spatial data

    Output:
    - Saves reference h5ad to: {save_path}/czi_reference/sc_reference_{dataset_id}.h5ad
    - Returns path to the downloaded reference file
    """
    # Heavy imports - only load when this tool is called
    import cellxgene_census
    import scanpy as sc

    # Create reference directory
    ref_dir = f"{save_path}/czi_reference"
    os.makedirs(ref_dir, exist_ok=True)

    # Output path for reference h5ad
    ref_adata_path = f"{ref_dir}/sc_reference_{dataset_id}.h5ad"

    # Check if already exists
    if exists(ref_adata_path):
        # Load to get cell count info
        adata = sc.read_h5ad(ref_adata_path)
        n_cells = adata.shape[0]
        n_celltypes = len(adata.obs["cell_type"].unique())
        msg = f"Reference data already exists at {ref_adata_path} ({n_cells:,} cells, {n_celltypes} cell types)"
        print(msg)
        return msg

    print(f"[download_czi_reference] Downloading dataset {dataset_id}...")

    try:
        # Open CZI Census
        census = cellxgene_census.open_soma(census_version="latest")

        try:
            # Download the dataset
            adata = cellxgene_census.get_anndata(
                census,
                organism=organism,
                obs_value_filter=f'dataset_id == "{dataset_id}"'
            )

            if adata.shape[0] == 0:
                msg = f"ERROR: No cells found for dataset_id '{dataset_id}'. Please verify the dataset ID."
                print(msg)
                return msg

            # Filter out genes with zero expression
            gene_sums = np.array(adata.X.sum(axis=0)).flatten()
            adata = adata[:, gene_sums > 0]

            # Filter out "unknown" cell types
            adata = adata[adata.obs["cell_type"] != "unknown"]

            # Filter cell types with too few cells (< 10)
            cell_type_counts = adata.obs["cell_type"].value_counts()
            valid_types = cell_type_counts[cell_type_counts >= 10].index
            adata = adata[adata.obs["cell_type"].isin(valid_types)]

            if adata.shape[0] == 0:
                msg = f"ERROR: No valid cells remaining after filtering for dataset '{dataset_id}'."
                print(msg)
                return msg

            # Convert categorical columns to save properly
            for col in adata.obs.columns:
                adata.obs[col] = pd.Categorical(adata.obs[col].astype(str))
            for col in adata.var.columns:
                adata.var[col] = pd.Categorical(adata.var[col].astype(str))

            # Save reference h5ad
            adata.write_h5ad(ref_adata_path, compression="gzip")

            # Get summary info
            n_cells = adata.shape[0]
            n_genes = adata.shape[1]
            cell_types = adata.obs["cell_type"].unique().tolist()
            n_celltypes = len(cell_types)

            # Also save a summary file
            summary_path = f"{ref_dir}/reference_info_{dataset_id}.txt"
            summary = f"""CZI Reference Dataset Summary
=============================
Dataset ID: {dataset_id}
Organism: {organism}
Cells: {n_cells:,}
Genes: {n_genes:,}
Cell types ({n_celltypes}): {', '.join(sorted(cell_types)[:20])}{'...' if n_celltypes > 20 else ''}

Reference file: {ref_adata_path}
"""
            with open(summary_path, 'w') as f:
                f.write(summary)

            msg = f"""Successfully downloaded CZI reference dataset.
- Path: {ref_adata_path}
- Cells: {n_cells:,}
- Genes: {n_genes:,}
- Cell types: {n_celltypes}

Next step: Use harmony_transfer_labels with ref_path="{ref_adata_path}" to transfer annotations."""
            print(msg)
            return msg

        finally:
            census.close()

    except Exception as e:
        msg = f"ERROR downloading CZI reference: {str(e)}"
        print(msg)
        return msg


# =============================================================================
# Tool 6: Query Tissue Expression (ARCHS4)
# =============================================================================

@tool
def query_tissue_expression(
    gene: Annotated[str, Field(description="Gene symbol to query (e.g., 'GFAP', 'SLC17A7')")],
    top_k: Annotated[int, Field(ge=1, le=50, description="Number of top tissues to return")] = 10,
) -> str:
    """Query ARCHS4 database for tissue-specific gene expression.

    Returns median TPM (transcripts per million) across human tissues for a gene.
    Useful for validating that marker genes are expressed in the target tissue.

    Examples:
        - query_tissue_expression({"gene": "GFAP", "top_k": 5})  # Astrocyte marker
        - query_tissue_expression({"gene": "SLC17A7"})  # Neuronal marker
    """
    import gget
    from .utils import parse_list_string

    # Handle case where LLM passes a stringified list like "['CD3D', 'MS4A6A']"
    genes = parse_list_string(gene)
    if len(genes) > 1:
        # Process multiple genes and combine results
        all_results = []
        for g in genes:
            result = query_tissue_expression(g, top_k)
            all_results.append(result)
        return "\n\n".join(all_results)

    # Single gene - use the parsed value
    gene = genes[0] if genes else gene

    try:
        # Fetch tissue expression data from ARCHS4
        data = gget.archs4(gene, which="tissue")

        if data is None or data.empty:
            msg = f"No expression data found for gene '{gene}' in ARCHS4."
            print(msg)
            return msg

        # Format results
        results = [f"Tissue expression for {gene} (top {top_k} tissues by median TPM):"]
        for idx, row in data.head(top_k).iterrows():
            tissue = row.get("id", row.get("tissue", "Unknown"))
            median_tpm = row.get("median", row.get("median_tpm", 0))
            results.append(f"  {tissue}: {median_tpm:.2f} TPM")

        output = "\n".join(results)
        print(output)
        return output

    except Exception as e:
        msg = f"Error querying ARCHS4 for gene '{gene}': {e}"
        print(msg)
        return msg


# =============================================================================
# Tool 7: Query Cell Type Gene Sets (Enrichr)
# =============================================================================

@tool
def query_celltype_genesets(
    tissue: Annotated[str, Field(description="Tissue type to search for cell type markers (e.g., 'brain', 'liver')")],
    top_k: Annotated[int, Field(ge=1, le=20, description="Number of top gene sets to return")] = 10,
) -> str:
    """Query Enrichr/PanglaoDB for cell type-specific gene sets in a tissue.

    Returns curated cell type marker gene sets relevant to the specified tissue.
    Uses the PanglaoDB_Augmented_2021 database from Enrichr.

    Examples:
        - query_celltype_genesets({"tissue": "brain", "top_k": 10})
        - query_celltype_genesets({"tissue": "liver"})
    """
    import gget

    # Use common marker genes for the tissue to find relevant cell type gene sets
    tissue_seed_genes = {
        "brain": ["GFAP", "SLC17A7", "GAD1", "MBP", "AIF1", "RBFOX3"],
        "liver": ["ALB", "CYP3A4", "KRT19", "CLEC4G", "CD68", "ACTA2"],
        "lung": ["SFTPC", "SCGB1A1", "PECAM1", "CD68", "ACTA2", "KRT5"],
        "heart": ["MYH7", "TNNT2", "PECAM1", "VWF", "CD68", "ACTA2"],
        "kidney": ["SLC12A1", "AQP2", "NPHS1", "PECAM1", "CD68", "ACTA2"],
        "pancreas": ["INS", "GCG", "KRT19", "AMY2A", "CD68", "ACTA2"],
        "skin": ["KRT14", "KRT1", "TYRP1", "CD68", "PECAM1", "ACTA2"],
        "intestine": ["FABP2", "LGR5", "MUC2", "CHGA", "CD68", "ACTA2"],
    }

    # Get seed genes for the tissue, or use generic markers
    seed_genes = tissue_seed_genes.get(
        tissue.lower(),
        ["PTPRC", "CD68", "PECAM1", "ACTA2", "KRT18", "VIM"]  # Generic markers
    )

    try:
        # Query Enrichr with PanglaoDB cell type database
        df = gget.enrichr(
            seed_genes,
            database="PanglaoDB_Augmented_2021",
            plot=False
        )

        if df is None or df.empty:
            return f"No cell type gene sets found for tissue '{tissue}'."

        # Filter for tissue-relevant results and format output
        results = [f"Cell type gene sets relevant to {tissue} (from PanglaoDB):"]

        for idx, row in df.head(top_k).iterrows():
            path_name = row.get("path_name", "Unknown")
            p_val = row.get("p_val", 1.0)
            genes = row.get("overlapping_genes", [])

            # Extract cell type name from path_name (format: "Cell Type_Tissue_Species")
            cell_type = path_name.split("_")[0] if "_" in path_name else path_name

            results.append(f"\n  {cell_type}:")
            results.append(f"    P-value: {p_val:.2e}")
            results.append(f"    Marker genes: {', '.join(genes[:10])}")

        output = "\n".join(results)
        print(output)
        return output

    except Exception as e:
        error_msg = f"Error querying cell type gene sets for '{tissue}': {e}"
        print(error_msg)
        return error_msg


# =============================================================================
# Tool 8: Validate Gene Expression in Tissue
# =============================================================================

@tool
def validate_genes_expression(
    genes: Annotated[str, Field(description="Comma-separated list of gene symbols to validate")],
    target_tissue: Annotated[str, Field(description="Target tissue to check expression in (e.g., 'brain', 'frontal cortex')")],
) -> str:
    """Validate that a list of genes are expressed in the target tissue.

    Queries ARCHS4 for each gene and checks if the target tissue is in the top expressing tissues.
    Returns a summary of which genes are validated as expressed vs not expressed.

    Examples:
        - validate_genes_expression({"genes": "GFAP, SLC17A7, GAD1", "target_tissue": "brain"})
    """
    import gget
    from .utils import parse_list_string

    # Parse genes - handles "gene1, gene2" and "['gene1', 'gene2']" formats
    gene_list = parse_list_string(genes, uppercase=True)
    target_tissue_lower = target_tissue.lower()

    expressed = []
    not_expressed = []
    not_found = []

    for gene in gene_list:
        try:
            data = gget.archs4(gene, which="tissue")

            if data is None or data.empty:
                not_found.append(gene)
                continue

            # Check if target tissue is in top 20 expressing tissues
            top_tissues = data.head(20)
            tissue_names = [str(t).lower() for t in top_tissues.get("id", top_tissues.get("tissue", []))]

            # Check if any top tissue contains the target tissue name
            is_expressed = any(target_tissue_lower in t for t in tissue_names)

            if is_expressed:
                # Get the rank and TPM for the matching tissue
                for idx, row in top_tissues.iterrows():
                    tissue = str(row.get("id", row.get("tissue", ""))).lower()
                    if target_tissue_lower in tissue:
                        tpm = row.get("median", row.get("median_tpm", 0))
                        expressed.append(f"{gene} (TPM: {tpm:.1f})")
                        break
                else:
                    expressed.append(gene)
            else:
                not_expressed.append(gene)

        except Exception:
            not_found.append(gene)

    # Format results
    results = [f"Gene expression validation for '{target_tissue}':\n"]
    results.append(f"Expressed ({len(expressed)}/{len(gene_list)}): {', '.join(expressed)}")

    if not_expressed:
        results.append(f"\nNot in top tissues ({len(not_expressed)}): {', '.join(not_expressed)}")

    if not_found:
        results.append(f"\nNot found in ARCHS4 ({len(not_found)}): {', '.join(not_found)}")

    output = "\n".join(results)
    print(output)
    return output


# =============================================================================
# Tool 9: Query Disease Genes from Real Databases
# =============================================================================


def _query_gwas_catalog(disease_trait: str, max_genes: int = 50) -> dict:
    """Query GWAS Catalog REST API for disease-associated genes.

    API: https://www.ebi.ac.uk/gwas/rest/api
    """
    import requests

    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    genes = []
    associations = []

    try:
        # Search for studies related to the disease/trait
        search_url = f"{base_url}/efoTraits/search/findByEfoUri"

        # First, search for associations by trait
        assoc_url = f"{base_url}/associations/search/findByDiseaseTrait"
        params = {"diseaseTrait": disease_trait, "size": 100}

        response = requests.get(assoc_url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if "_embedded" in data and "associations" in data["_embedded"]:
                for assoc in data["_embedded"]["associations"]:
                    # Get p-value
                    p_value = assoc.get("pvalue", 1.0)

                    # Get genes from the association
                    if "loci" in assoc:
                        for locus in assoc["loci"]:
                            if "authorReportedGenes" in locus:
                                for gene in locus["authorReportedGenes"]:
                                    gene_name = gene.get("geneName", "")
                                    if gene_name and gene_name not in genes:
                                        genes.append(gene_name)
                                        associations.append({
                                            "gene": gene_name,
                                            "p_value": p_value,
                                            "trait": disease_trait
                                        })

        # Alternative: search by keyword in studies
        if len(genes) < 10:
            study_url = f"{base_url}/studies/search/findByDiseaseTrait"
            params = {"diseaseTrait": disease_trait, "size": 50}
            response = requests.get(study_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "_embedded" in data and "studies" in data["_embedded"]:
                    for study in data["_embedded"]["studies"][:20]:
                        # Get associations for this study
                        if "_links" in study and "associations" in study["_links"]:
                            assoc_link = study["_links"]["associations"]["href"]
                            assoc_resp = requests.get(assoc_link, timeout=10)
                            if assoc_resp.status_code == 200:
                                assoc_data = assoc_resp.json()
                                if "_embedded" in assoc_data:
                                    for assoc in assoc_data["_embedded"].get("associations", []):
                                        if "loci" in assoc:
                                            for locus in assoc["loci"]:
                                                for gene in locus.get("authorReportedGenes", []):
                                                    gene_name = gene.get("geneName", "")
                                                    if gene_name and gene_name not in genes:
                                                        genes.append(gene_name)

                        if len(genes) >= max_genes:
                            break

        return {
            "source": "GWAS_Catalog",
            "query": disease_trait,
            "genes": genes[:max_genes],
            "n_genes": len(genes[:max_genes]),
            "associations": associations[:max_genes]
        }

    except Exception as e:
        return {"source": "GWAS_Catalog", "error": str(e), "genes": []}


def _query_opentargets(disease_query: str, max_genes: int = 50) -> dict:
    """Query OpenTargets GraphQL API for disease-associated genes.

    API: https://api.platform.opentargets.org/api/v4/graphql
    """
    import requests

    api_url = "https://api.platform.opentargets.org/api/v4/graphql"

    try:
        # First, search for diseases matching the query
        search_query = """
        query searchDiseases($queryString: String!) {
            search(queryString: $queryString, entityNames: ["disease"], page: {size: 5, index: 0}) {
                hits {
                    id
                    name
                    entity
                }
            }
        }
        """

        response = requests.post(
            api_url,
            json={"query": search_query, "variables": {"queryString": disease_query}},
            timeout=30
        )

        if response.status_code != 200:
            return {"source": "OpenTargets", "error": f"Search failed: {response.status_code}", "genes": []}

        search_data = response.json()
        hits = search_data.get("data", {}).get("search", {}).get("hits", [])

        if not hits:
            return {"source": "OpenTargets", "error": f"No diseases found for '{disease_query}'", "genes": []}

        # Get the first disease match
        disease_id = hits[0]["id"]
        disease_name = hits[0]["name"]

        # Query associated targets (genes) for this disease
        targets_query = """
        query diseaseAssociations($diseaseId: String!, $size: Int!) {
            disease(efoId: $diseaseId) {
                id
                name
                associatedTargets(page: {size: $size, index: 0}) {
                    count
                    rows {
                        target {
                            id
                            approvedSymbol
                            approvedName
                        }
                        score
                        datatypeScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """

        response = requests.post(
            api_url,
            json={
                "query": targets_query,
                "variables": {"diseaseId": disease_id, "size": max_genes}
            },
            timeout=30
        )

        if response.status_code != 200:
            return {"source": "OpenTargets", "error": f"Targets query failed: {response.status_code}", "genes": []}

        data = response.json()
        disease_data = data.get("data", {}).get("disease", {})

        if not disease_data:
            return {"source": "OpenTargets", "error": "No disease data returned", "genes": []}

        rows = disease_data.get("associatedTargets", {}).get("rows", [])

        genes = []
        gene_scores = []
        for row in rows:
            target = row.get("target", {})
            gene_symbol = target.get("approvedSymbol", "")
            if gene_symbol:
                genes.append(gene_symbol)
                gene_scores.append({
                    "gene": gene_symbol,
                    "name": target.get("approvedName", ""),
                    "score": row.get("score", 0)
                })

        return {
            "source": "OpenTargets",
            "query": disease_query,
            "disease_id": disease_id,
            "disease_name": disease_name,
            "genes": genes,
            "n_genes": len(genes),
            "gene_details": gene_scores
        }

    except Exception as e:
        return {"source": "OpenTargets", "error": str(e), "genes": []}


@tool
def query_disease_genes(
    disease: Annotated[str, Field(description="Disease or trait to search (e.g., 'Alzheimer disease', 'schizophrenia', 'type 2 diabetes', 'cognitive function')")],
    source: Annotated[str, Field(description="Database to query: 'opentargets', 'gwas', or 'all' (queries both)")] = "all",
    max_genes: Annotated[int, Field(ge=10, le=200, description="Maximum number of genes to return")] = 50,
) -> str:
    """Query disease-associated genes from real databases (GWAS Catalog, OpenTargets).

    This tool queries live databases for genes associated with diseases or traits:
    - **OpenTargets**: Comprehensive drug target and disease association database
    - **GWAS Catalog**: Curated GWAS hits from published studies

    Examples:
        - query_disease_genes({"disease": "Alzheimer disease"})
        - query_disease_genes({"disease": "schizophrenia", "source": "opentargets"})
        - query_disease_genes({"disease": "cognitive function", "source": "gwas"})
        - query_disease_genes({"disease": "prefrontal cortex", "source": "all"})

    Common disease/trait queries for brain:
        - "Alzheimer disease", "Parkinson disease", "schizophrenia"
        - "bipolar disorder", "major depressive disorder"
        - "cognitive function", "intelligence", "memory"
        - "brain volume", "cortical thickness"

    Returns genes ranked by association strength with the disease/trait.
    """
    results = [f"Querying disease-associated genes for: '{disease}'\n"]
    all_genes = set()
    gene_sources = {}

    # Query OpenTargets
    if source.lower() in ["opentargets", "all"]:
        results.append("=" * 60)
        results.append("OpenTargets Database")
        results.append("=" * 60)

        ot_result = _query_opentargets(disease, max_genes)

        if "error" in ot_result:
            results.append(f"  Error: {ot_result['error']}")
        else:
            results.append(f"  Disease matched: {ot_result.get('disease_name', 'N/A')}")
            results.append(f"  Disease ID: {ot_result.get('disease_id', 'N/A')}")

        if ot_result.get("genes"):
            genes = ot_result["genes"]
            results.append(f"  Genes found: {len(genes)}")
            results.append(f"  Top genes: {', '.join(genes[:20])}")
            if len(genes) > 20:
                results.append(f"  ... and {len(genes) - 20} more")

            for g in genes:
                all_genes.add(g)
                gene_sources[g] = gene_sources.get(g, []) + ["OpenTargets"]

    # Query GWAS Catalog
    if source.lower() in ["gwas", "all"]:
        results.append("\n" + "=" * 60)
        results.append("GWAS Catalog (EBI)")
        results.append("=" * 60)

        gwas_result = _query_gwas_catalog(disease, max_genes)

        if "error" in gwas_result:
            results.append(f"  Error: {gwas_result['error']}")

        if gwas_result.get("genes"):
            genes = gwas_result["genes"]
            results.append(f"  Genes found: {len(genes)}")
            results.append(f"  Top genes: {', '.join(genes[:20])}")
            if len(genes) > 20:
                results.append(f"  ... and {len(genes) - 20} more")

            for g in genes:
                all_genes.add(g)
                gene_sources[g] = gene_sources.get(g, []) + ["GWAS"]

    # Summary
    results.append("\n" + "=" * 60)
    results.append("SUMMARY")
    results.append("=" * 60)
    results.append(f"Total unique genes: {len(all_genes)}")

    # Genes found in multiple sources (higher confidence)
    multi_source = [g for g, sources in gene_sources.items() if len(sources) > 1]
    if multi_source:
        results.append(f"\nHigh-confidence genes (found in multiple databases):")
        results.append(f"  {', '.join(multi_source)}")

    results.append(f"\nAll genes: {', '.join(sorted(all_genes))}")

    output = "\n".join(results)
    print(output)
    return output
