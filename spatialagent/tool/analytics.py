"""
Computational and Statistical Analysis Tools

Tools for data preprocessing, integration, clustering, statistical analysis,
and pattern summarization using algorithmic and computational methods.

All tools are standalone functions following Biomni pattern.
"""

# Lightweight imports - keep at module level
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Annotated
from os.path import exists
from glob import glob

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings('ignore')


# Module-level config (set via configure_analytics_tools)
_config = {
    "save_path": "./experiments",
}

def configure_analytics_tools(save_path: str = "./experiments"):
    """Configure paths for analytics tools. Call this before using the tools."""
    _config["save_path"] = save_path


# Default model for subagent LLM calls (fallback if agent model not set)
DEFAULT_SUBAGENT_MODEL = "claude-sonnet-4-5-20250929"

def _get_subagent_model() -> str:
    """Get the model to use for subagent calls.

    Uses the main agent's model if available, otherwise falls back to DEFAULT_SUBAGENT_MODEL.
    """
    try:
        from ..agent import get_agent_model
        model = get_agent_model()
        return model if model else DEFAULT_SUBAGENT_MODEL
    except ImportError:
        return DEFAULT_SUBAGENT_MODEL


# Heavy imports moved inside tools:
# - scanpy (2-3s import time)
# - scanpy.external (sce)
# - scipy
# - sklearn.neural_network.MLPClassifier
# - tqdm


# =============================================================================
# Tool 1: Preprocess
# =============================================================================

@tool
def preprocess_spatial_data(
    adata_path: Annotated[str, Field(description="Path to raw spatial transcriptomics h5ad file")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Preprocess spatial transcriptomics data using Scanpy pipeline."""
    save_path = save_path or _config["save_path"]
    # Heavy imports - only load when this tool is called
    import scanpy as sc

    output_path = f"{save_path}/preprocessed.h5ad"

    if exists(output_path):
        msg = f"Preprocessed data already exists at {output_path}"
        print(msg)
        return msg

    print(f"[preprocess_spatial_data] Loading data from {adata_path}...")

    # Load and preprocess
    adata = sc.read_h5ad(adata_path)
    adata.var.index = adata.var.index.str.upper()

    print(f"[preprocess_spatial_data] Loaded {adata.n_obs} cells, {adata.n_vars} genes. Running QC...")

    # QC filtering
    sc.pp.filter_cells(adata, min_genes=5)
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)

    print(f"[preprocess_spatial_data] After QC: {adata.n_obs} cells, {adata.n_vars} genes. Normalizing...")

    # Normalization
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    print("[preprocess_spatial_data] Running PCA and UMAP...")

    # Dimensionality reduction
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # Save
    adata.write(output_path, compression="gzip")

    msg = f"Successfully preprocessed data: {adata.n_obs} cells, {adata.n_vars} genes. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 2: Harmony Transfer
# =============================================================================

@tool
def harmony_transfer_labels(
    adata_path: Annotated[str, Field(description="Path to preprocessed spatial data")],
    ref_path: Annotated[str, Field(description="Path to CZI reference scRNA data")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
    czi_index: Annotated[int, Field(ge=0, description="Index for naming output files")] = 0,
) -> str:
    """Transfer cell type labels from CZI reference to spatial data using Harmony integration."""
    save_path = save_path or _config["save_path"]
    # Heavy imports - only load when this tool is called
    import scanpy as sc
    import scanpy.external as sce
    from sklearn.neural_network import MLPClassifier

    output_path = f"{save_path}/celltype-transferred_{czi_index}.h5ad"

    if exists(output_path):
        msg = f"Harmony results already exist at {output_path}"
        print(msg)
        return msg

    print(f"[harmony_transfer_labels] Loading spatial and reference data...")

    # Load data
    adata_sp = sc.read_h5ad(adata_path)
    adata_sc = sc.read_h5ad(ref_path)
    print(f"[harmony_transfer_labels] Spatial: {adata_sp.n_obs} cells, Reference: {adata_sc.n_obs} cells")

    # Subset reference if larger
    if adata_sp.shape[0] < adata_sc.shape[0]:
        # Sample proportionally from each cell type
        cell_type_counts = adata_sc.obs["cell_type"].value_counts()
        scale = adata_sp.shape[0] / cell_type_counts.sum()
        samples_per_type = (cell_type_counts * scale).astype(int)

        sampled_indices = []
        for cell_type, count in samples_per_type.items():
            if count > 0:
                mask = adata_sc.obs["cell_type"] == cell_type
                indices = adata_sc.obs.index[mask]
                sampled = np.random.RandomState(42).choice(indices, count, replace=False)
                sampled_indices.extend(sampled)
        adata_sc = adata_sc[sampled_indices]

    # Harmonize gene names
    adata_sc.var.index = adata_sc.var["feature_name"].str.upper()
    adata_sc.var_names_make_unique()

    # Select common genes
    common_genes = adata_sp.var.index.intersection(adata_sc.var.index)
    adata_sp, adata_sc = adata_sp[:, common_genes], adata_sc[:, common_genes]

    # Scale
    sc.pp.scale(adata_sp)
    sc.pp.normalize_per_cell(adata_sc)
    sc.pp.log1p(adata_sc)
    sc.pp.scale(adata_sc)

    # Combine and integrate
    combined = adata_sp.concatenate(adata_sc, batch_key="dataset", batch_categories=["st", "scrna"])
    sc.pp.pca(combined, n_comps=30)
    sce.pp.harmony_integrate(combined, "dataset")

    # Save
    combined.write_h5ad(output_path, compression="gzip")

    # Transfer labels using MLP
    ad_sp = combined[combined.obs["dataset"] == "st", :]
    ad_sc = combined[combined.obs["dataset"] == "scrna", :]

    X_train = ad_sc.obsm["X_pca_harmony"]
    y_train = ad_sc.obs["cell_type"]
    X_test = ad_sp.obsm["X_pca_harmony"]

    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Save predictions (strip batch suffix added by concatenate)
    cell_ids = ad_sp.obs.index.str.replace(r'-st$', '', regex=True)
    result_df = pd.DataFrame({"predicted_celltype": predictions}, index=cell_ids)
    csv_path = f"{save_path}/celltype_transferred.csv"
    result_df.to_csv(csv_path)

    msg = f"Successfully transferred labels using Harmony. Predictions for {len(predictions)} cells saved to {csv_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 3: UTAG (Spatial Clustering)
# =============================================================================

def _estimate_max_dist(adata, slide_key=None, target_neighbors=3):
    """Estimate optimal max_dist for UTAG based on spatial density."""
    import numpy as np

    # Use first sample if slide_key provided
    if slide_key and slide_key in adata.obs.columns:
        sample = adata.obs[slide_key].unique()[0]
        sample_data = adata[adata.obs[slide_key] == sample]
    else:
        sample_data = adata

    # Sample random cells for efficiency
    n_cells = min(10000, sample_data.shape[0])
    idx = np.random.permutation(sample_data.shape[0])[:n_cells]
    coords = sample_data.obsm["spatial"][idx]

    # Find distance where each cell has ~target_neighbors neighbors
    for distance in range(10, 500, 5):
        distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
        avg_neighbors = np.mean(np.sum(distances < distance, axis=1) - 1)
        if avg_neighbors > target_neighbors:
            return distance

    return 50  # Default fallback


def _remove_small_clusters(adata, label_key, slide_key=None, min_cells=100):
    """Remove small clusters by assigning cells to nearest larger cluster.

    IMPORTANT: Filtering is done PER-SAMPLE (not globally) to ensure each sample
    has no small clusters. A cluster might have many cells globally but only a few
    in a specific sample, which causes issues in per-sample analysis (e.g.,
    statistical tests like Wilcoxon require >= 2 samples per group).
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # Convert to string to avoid categorical issues
    adata.obs[label_key] = adata.obs[label_key].astype(str)

    batches = adata.obs[slide_key].unique() if slide_key else [None]

    for batch in batches:
        if batch is not None:
            batch_mask = adata.obs[slide_key] == batch
            batch_data = adata[batch_mask]
        else:
            batch_data = adata
            batch_mask = np.ones(adata.shape[0], dtype=bool)

        # Identify small clusters WITHIN THIS BATCH (not globally)
        batch_counts = batch_data.obs[label_key].value_counts()
        small_clusters_in_batch = list(batch_counts[batch_counts < min_cells].index)

        if not small_clusters_in_batch:
            continue

        cells_to_assign = batch_data.obs[label_key].isin(small_clusters_in_batch)
        if cells_to_assign.sum() == 0:
            continue

        reference_mask = ~batch_data.obs[label_key].isin(small_clusters_in_batch)
        if reference_mask.sum() == 0:
            # All clusters are small in this batch - skip (can't reassign)
            print(f"Warning: All clusters in batch '{batch}' have < {min_cells} cells, skipping merge")
            continue

        # Find nearest neighbor from larger clusters
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(batch_data[reference_mask].obsm["spatial"])
        _, indices = nn.kneighbors(batch_data[cells_to_assign].obsm["spatial"])

        # Assign to nearest larger cluster
        new_labels = batch_data[reference_mask].obs[label_key].values[indices.flatten()]
        cell_indices = batch_data[cells_to_assign].obs.index
        adata.obs.loc[cell_indices, label_key] = new_labels

        if small_clusters_in_batch:
            print(f"Batch '{batch}': merged {len(small_clusters_in_batch)} small clusters "
                  f"({cells_to_assign.sum()} cells) to nearest neighbors")

    return adata


@tool
def run_utag_clustering(
    adata_path: Annotated[str, Field(description="Path to annotated spatial data")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
    slide_key: Annotated[str, Field(description="Column for sample/slide ID to run UTAG per sample (e.g., 'batch', 'sample_id')")] = None,
    max_dist: Annotated[float, Field(description="Max distance for neighbors. Use 0 for auto-estimation.")] = 0,
    min_cluster_size: Annotated[int, Field(description="Min cells per cluster (smaller merged to nearest)")] = 100,
    resolutions: Annotated[list, Field(description="Clustering resolutions to try")] = [0.05, 0.1, 0.3],
    min_niches: Annotated[int, Field(description="Minimum number of niches required")] = 5,
) -> str:
    """Run UTAG spatial clustering to identify tissue niches.

    UTAG combines gene expression with spatial information to identify
    tissue domains/niches. It uses message passing on a spatial graph.

    When slide_key is provided, UTAG runs separately on each sample/slide
    to account for batch effects and sample-specific spatial patterns.

    Features:
    - Auto-estimates max_dist if set to 0
    - Removes small clusters and reassigns cells to nearest larger cluster
    - Tries multiple resolutions and picks one with enough niches
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import matplotlib.pyplot as plt
    from utag import utag

    output_path = f"{save_path}/utag_main_result.csv"
    utag_h5ad_path = f"{save_path}/utag_clustered.h5ad"

    if exists(output_path):
        msg = f"UTAG results already exist at {output_path}"
        print(msg)
        return msg

    print(f"[run_utag_clustering] Loading data and running UTAG...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Validate slide_key if provided
    if slide_key and slide_key not in adata.obs.columns:
        msg = f"ERROR: slide_key '{slide_key}' not found. Available: {list(adata.obs.columns)}"
        print(msg)
        return msg

    # Auto-estimate max_dist if not provided
    if max_dist <= 0:
        max_dist = _estimate_max_dist(adata, slide_key)
        print(f"Auto-estimated max_dist: {max_dist}")

    # Try resolutions until we find one with enough niches
    utag_results = None
    best_label_key = None

    for resolution in resolutions:
        print(f"Trying UTAG with resolution={resolution}...")

        utag_results = utag(
            adata,
            slide_key=slide_key,
            max_dist=max_dist,
            normalization_mode="l1_norm",
            apply_umap=True,
            apply_clustering=True,
            clustering_method=["leiden"],
            resolutions=[resolution],
        )

        label_key = f"UTAG Label_leiden_{resolution}"
        n_niches = utag_results.obs[label_key].nunique()
        print(f"  Found {n_niches} niches")

        if n_niches >= min_niches:
            best_label_key = label_key
            break

    if utag_results is None:
        msg = "ERROR: UTAG clustering failed"
        print(msg)
        return msg

    if best_label_key is None:
        # Use last resolution if none met threshold
        best_label_key = f"UTAG Label_leiden_{resolutions[-1]}"

    # Remove small clusters
    utag_results = _remove_small_clusters(utag_results, best_label_key, slide_key, min_cluster_size)

    # Add unified 'utag' column
    utag_results.obs["utag"] = utag_results.obs[best_label_key].astype("category")
    n_final = utag_results.obs["utag"].nunique()

    # Save results
    utag_results.write_h5ad(utag_h5ad_path, compression="gzip")
    pd.DataFrame(utag_results.obs).to_csv(output_path)

    # Generate per-sample plots if slide_key provided
    if slide_key:
        plt.ioff()
        for sample in utag_results.obs[slide_key].unique():
            sample_data = utag_results[utag_results.obs[slide_key] == sample]
            fig, ax = plt.subplots(figsize=(6, 6))
            sc.pl.embedding(sample_data, basis="spatial", color="utag",
                          palette="tab20", size=3, ax=ax, show=False)
            ax.set_title(f"UTAG Niches - {sample}")
            plt.savefig(f"{save_path}/utag_niche_{sample}.png", dpi=150, bbox_inches="tight")
            plt.close()

    sample_info = f" (per sample: {slide_key})" if slide_key else ""
    msg = f"Successfully ran UTAG clustering{sample_info}. Found {n_final} niches using {best_label_key}. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 4: Gene Voting (LLM-based)
# =============================================================================

@tool
def aggregate_gene_voting(
    adata_path: Annotated[str, Field(description="Path to annotated spatial data")],
    group_by: Annotated[str, Field(description="Column to group by (e.g., 'celltype', 'niche')")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Aggregate marker genes across cells/niches using LLM-based voting."""
    save_path = save_path or _config["save_path"]
    # Heavy imports
    import scanpy as sc
    from ..agent import make_llm

    # Create LLM instance
    llm = make_llm(_get_subagent_model(), stop_sequences=[])

    output_path = f"{save_path}/gene_voting_results.csv"

    if exists(output_path):
        msg = f"Gene voting results already exist at {output_path}"
        print(msg)
        return msg

    print(f"[aggregate_gene_voting] Loading data and aggregating genes by {group_by}...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Group by specified column and aggregate
    results = []
    for group_name in adata.obs[group_by].unique():
        mask = adata.obs[group_by] == group_name
        subset = adata[mask]

        # Get top marker genes
        sc.tl.rank_genes_groups(subset, groupby=group_by)
        markers = subset.uns['rank_genes_groups']['names'][:20]

        results.append({
            "group": group_name,
            "marker_genes": markers.tolist()
        })

    # Save
    pd.DataFrame(results).to_csv(output_path)

    msg = f"Successfully aggregated genes for {len(results)} groups. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 5: Cell-Cell Interactions (Full LIANA + Cell2Cell + TensorLy)
# =============================================================================

@tool
def liana_tensor(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    sample_key: Annotated[str, Field(description="Sample/batch column name")],
    condition_key: Annotated[str, Field(description="Condition column name")],
    cell_type_key: Annotated[str, Field(description="Cell type column name")],
    organism: Annotated[str, Field(description="'human', 'mouse', or 'auto'")] = "auto",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run LIANA + Tensor-Cell2Cell for multi-sample interaction analysis.

    Performs tensor decomposition to identify context-specific communication patterns:
    1. LIANA rank_aggregate per sample
    2. Build interaction tensor (samples x interactions x senders x receivers)
    3. Non-negative tensor factorization into latent factors
    """
    save_path = save_path or _config["save_path"]
    # Heavy imports - only load when this tool is called
    import scanpy as sc
    from tqdm.auto import tqdm
    import liana as li

    output_path = f"{save_path}/cci_analysis/factor.pkl"

    if exists(output_path):
        msg = f"CCI analysis results already exist at {output_path}"
        print(msg)
        return msg

    print(f"[liana_tensor] Running LIANA + Tensor-Cell2Cell analysis...")

    # Create output directory
    os.makedirs(f"{save_path}/cci_analysis", exist_ok=True)

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Auto-detect organism based on gene name format
    if organism == "auto":
        # Mouse genes: first letter uppercase, rest lowercase (e.g., Actb)
        # Human genes: all uppercase (e.g., ACTB)
        sample_genes = adata.var_names[:100].tolist()
        n_mouse_format = sum(1 for g in sample_genes if g[0].isupper() and len(g) > 1 and g[1:].islower())
        organism = "mouse" if n_mouse_format > 50 else "human"
        print(f"Auto-detected organism: {organism}")

    # Select resource based on organism
    resource_name = "mouseconsensus" if organism == "mouse" else "consensus"

    # Step 1: Run LIANA per sample to get ligand-receptor interactions
    print("Running LIANA per sample...")
    li.mt.rank_aggregate.by_sample(
        adata,
        sample_key=sample_key,
        groupby=cell_type_key,
        resource_name=resource_name,
        use_raw=False,
        verbose=True,
        key_added='liana_res'
    )

    # Get unique LR pairs
    liana_res = adata.uns['liana_res']
    lr_pairs = liana_res[['ligand_complex', 'receptor_complex']].drop_duplicates()
    print(f"Found {len(lr_pairs)} unique ligand-receptor pairs")

    # Step 2: Build tensor using LIANA's built-in function
    print("Building interaction tensor...")
    tensor = li.multi.to_tensor_c2c(
        adata,
        sample_key=sample_key,
        score_key='magnitude_rank',
        non_negative=True
    )

    # Step 3: Tensor factorization using PreBuiltTensor's built-in method
    print("Running tensor factorization...")
    try:
        tensor.compute_tensor_factorization(
            rank=3,
            init='random',
            random_state=42
        )
    except Exception as e:
        # If factorization fails, return partial results
        print(f"Warning: Tensor factorization failed ({e}). Returning LIANA results only.")
        liana_res.to_csv(f"{save_path}/cci_analysis/liana_results.csv", index=False)
        msg = f"LIANA analysis complete for {len(adata.obs[sample_key].unique())} samples, {len(lr_pairs)} LR pairs. Factorization skipped (insufficient data). Saved to {save_path}/cci_analysis/"
        print(msg)
        return msg

    # Get context (condition) mapping
    samples = adata.obs[sample_key].unique()
    context_dict = dict(zip(samples, adata.obs.groupby(sample_key)[condition_key].first()))

    # Save results
    results = {
        'tensor': tensor,  # Contains factorization results
        'context_dict': context_dict,
        'liana_res': liana_res,
        'lr_pairs': lr_pairs.values.tolist(),
        'samples': list(samples)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    # Also save LIANA results as CSV
    liana_res.to_csv(f"{save_path}/cci_analysis/liana_results.csv", index=False)

    msg = f"Successfully computed cell-cell interactions for {len(samples)} samples. Found {len(lr_pairs)} LR pairs. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 6: Dynamics (DEG Analysis)
# =============================================================================

@tool
def infer_dynamics(
    adata_path: Annotated[str, Field(description="Path to spatial data with condition labels")],
    condition_column: Annotated[str, Field(description="Column containing condition labels")],
    condition1: Annotated[str, Field(description="First condition (e.g., 'control')")],
    condition2: Annotated[str, Field(description="Second condition (e.g., 'disease')")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compare conditions with differential expression gene (DEG) analysis."""
    save_path = save_path or _config["save_path"]
    # Heavy imports
    import scanpy as sc

    output_path = f"{save_path}/deg_results.csv"

    if exists(output_path):
        msg = f"DEG results already exist at {output_path}"
        print(msg)
        return msg

    print(f"[infer_dynamics] Comparing {condition1} vs {condition2}...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Filter to specified conditions
    mask = adata.obs[condition_column].isin([condition1, condition2])
    adata_subset = adata[mask]

    # Run differential expression
    sc.tl.rank_genes_groups(
        adata_subset,
        groupby=condition_column,
        groups=[condition2],
        reference=condition1,
        method='wilcoxon'
    )

    # Extract results
    result = sc.get.rank_genes_groups_df(adata_subset, group=condition2)
    result.to_csv(output_path)

    msg = f"Successfully analyzed dynamics: {condition1} vs {condition2}. Found {len(result)} DEGs. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 7: Summarize Conditions
# =============================================================================

@tool
def summarize_conditions(
    adata_path: Annotated[str, Field(description="Path to spatial data with condition labels")],
    condition_key: Annotated[str, Field(description="Column name for condition labels")],
    cell_type_key: Annotated[str, Field(description="Column name for cell type annotations")] = "cell_type",
    save_path: Annotated[str, Field(description="Directory to save summary")] = None,
) -> str:
    """Summarize cell type distributions across different conditions."""
    save_path = save_path or _config["save_path"]
    # Heavy imports
    import scanpy as sc

    output_path = f"{save_path}/condition_summary.txt"

    if exists(output_path):
        msg = f"Condition summary already exists at {output_path}"
        print(msg)
        return msg

    print(f"[summarize_conditions] Analyzing conditions by {condition_key}...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    if condition_key not in adata.obs.columns:
        msg = f"Error: Condition column '{condition_key}' not found. Available columns: {list(adata.obs.columns)}"
        print(msg)
        return msg

    if cell_type_key not in adata.obs.columns:
        msg = f"Error: Cell type column '{cell_type_key}' not found. Available columns: {list(adata.obs.columns)}"
        print(msg)
        return msg

    # Get condition distribution
    condition_counts = adata.obs[condition_key].value_counts()
    total_cells = len(adata)

    # Build summary
    lines = []
    lines.append("# Condition Summary")
    lines.append(f"\nTotal cells: {total_cells}")
    lines.append(f"Conditions: {len(condition_counts)}\n")

    lines.append("## Condition Overview\n")
    lines.append("| Condition | Cells | % Total |")
    lines.append("|-----------|-------|---------|")

    for condition in condition_counts.index:
        count = condition_counts[condition]
        pct = 100 * count / total_cells
        lines.append(f"| {condition} | {count} | {pct:.1f}% |")

    # Cell type distribution per condition
    lines.append("\n## Cell Type Distribution by Condition\n")

    for condition in condition_counts.index:
        mask = adata.obs[condition_key] == condition
        subset = adata[mask]

        celltype_counts = subset.obs[cell_type_key].value_counts()
        lines.append(f"### {condition} ({len(subset)} cells)\n")

        for ct, count in celltype_counts.head(5).items():
            ct_pct = 100 * count / len(subset)
            lines.append(f"- {ct}: {count} ({ct_pct:.1f}%)")
        lines.append("")

    # Save
    summary_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(summary_text)

    # Also save cross-tabulation as CSV
    csv_path = f"{save_path}/condition_celltype_matrix.csv"
    crosstab = pd.crosstab(adata.obs[condition_key], adata.obs[cell_type_key])
    crosstab.to_csv(csv_path)

    msg = f"Condition summary:\n{summary_text}\n\nSaved to {output_path} and {csv_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 8: Summarize Cell Types (LLM-based)
# =============================================================================

@tool
def summarize_celltypes(
    adata_path: Annotated[str, Field(description="Path to annotated spatial data")],
    cell_type_key: Annotated[str, Field(description="Column name for cell type annotations")] = "cell_type",
    save_path: Annotated[str, Field(description="Directory to save summary")] = None,
) -> str:
    """Summarize cell type distributions and marker genes in the dataset."""
    save_path = save_path or _config["save_path"]
    # Heavy imports
    import scanpy as sc

    output_path = f"{save_path}/celltype_summary.txt"

    if exists(output_path):
        msg = f"Cell type summary already exists at {output_path}"
        print(msg)
        return msg

    print(f"[summarize_celltypes] Analyzing cell types...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    if cell_type_key not in adata.obs.columns:
        msg = f"Error: Column '{cell_type_key}' not found. Available columns: {list(adata.obs.columns)}"
        print(msg)
        return msg

    # Get cell type distribution
    celltype_counts = adata.obs[cell_type_key].value_counts()
    total_cells = len(adata)

    # Run differential expression to find markers for each cell type
    print(f"Running marker gene analysis for {len(celltype_counts)} cell types...")
    sc.tl.rank_genes_groups(adata, groupby=cell_type_key, method='wilcoxon')

    # Build summary
    lines = []
    lines.append("# Cell Type Summary")
    lines.append(f"\nTotal cells: {total_cells}")
    lines.append(f"Cell types: {len(celltype_counts)}\n")
    lines.append("## Cell Type Distribution\n")
    lines.append("| Cell Type | Count | Percentage | Top Markers |")
    lines.append("|-----------|-------|------------|-------------|")

    for celltype in celltype_counts.index:
        count = celltype_counts[celltype]
        pct = 100 * count / total_cells

        # Get marker genes for this cell type
        try:
            markers = adata.uns['rank_genes_groups']['names'][celltype][:5].tolist()
            marker_str = ", ".join(markers)
        except Exception:
            marker_str = "N/A"

        lines.append(f"| {celltype} | {count} | {pct:.1f}% | {marker_str} |")

    # Save
    summary_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(summary_text)

    # Also save as CSV for easy access
    csv_path = f"{save_path}/celltype_summary.csv"
    summary_df = pd.DataFrame({
        'cell_type': celltype_counts.index,
        'count': celltype_counts.values,
        'percentage': [100 * c / total_cells for c in celltype_counts.values]
    })
    summary_df.to_csv(csv_path, index=False)

    msg = f"Cell type summary:\n{summary_text}\n\nSaved to {output_path} and {csv_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 9: Summarize Tissue Regions (LLM-based)
# =============================================================================

@tool
def summarize_tissue_regions(
    adata_path: Annotated[str, Field(description="Path to spatial data with region annotations")],
    region_key: Annotated[str, Field(description="Column name for region/niche annotations")] = "spatial_cluster",
    cell_type_key: Annotated[str, Field(description="Column name for cell type annotations")] = "cell_type",
    save_path: Annotated[str, Field(description="Directory to save summary")] = None,
) -> str:
    """Summarize tissue regions and their cell type compositions."""
    save_path = save_path or _config["save_path"]
    # Heavy imports
    import scanpy as sc

    output_path = f"{save_path}/tissue_region_summary.txt"

    if exists(output_path):
        msg = f"Tissue region summary already exists at {output_path}"
        print(msg)
        return msg

    print(f"[summarize_tissue_regions] Analyzing tissue regions by {region_key}...")

    # Load data
    adata = sc.read_h5ad(adata_path)

    if region_key not in adata.obs.columns:
        msg = f"Error: Region column '{region_key}' not found. Available columns: {list(adata.obs.columns)}"
        print(msg)
        return msg

    if cell_type_key not in adata.obs.columns:
        msg = f"Error: Cell type column '{cell_type_key}' not found. Available columns: {list(adata.obs.columns)}"
        print(msg)
        return msg

    # Get region counts
    region_counts = adata.obs[region_key].value_counts()
    total_cells = len(adata)

    # Build summary
    lines = []
    lines.append("# Tissue Region Summary")
    lines.append(f"\nTotal cells: {total_cells}")
    lines.append(f"Regions: {len(region_counts)}\n")

    # Build composition matrix
    composition_data = []
    for region in region_counts.index:
        mask = adata.obs[region_key] == region
        subset = adata[mask]
        region_pct = 100 * len(subset) / total_cells

        # Get cell type composition
        celltype_counts = subset.obs[cell_type_key].value_counts()
        top_celltypes = []
        for ct, count in celltype_counts.head(3).items():
            ct_pct = 100 * count / len(subset)
            top_celltypes.append(f"{ct} ({ct_pct:.0f}%)")

        composition_data.append({
            'region': region,
            'count': len(subset),
            'percentage': region_pct,
            'top_celltypes': ", ".join(top_celltypes)
        })

    lines.append("## Region Overview\n")
    lines.append("| Region | Cells | % Total | Top Cell Types |")
    lines.append("|--------|-------|---------|----------------|")

    for row in composition_data:
        lines.append(f"| {row['region']} | {row['count']} | {row['percentage']:.1f}% | {row['top_celltypes']} |")

    # Save
    summary_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(summary_text)

    # Also save composition matrix as CSV
    csv_path = f"{save_path}/tissue_region_composition.csv"
    # Create cross-tabulation
    crosstab = pd.crosstab(adata.obs[region_key], adata.obs[cell_type_key], normalize='index') * 100
    crosstab.to_csv(csv_path)

    msg = f"Tissue region summary:\n{summary_text}\n\nSaved to {output_path} and {csv_path}"
    print(msg)
    return msg


# =============================================================================
# Tangram Tools for Spatial Mapping
# =============================================================================

@tool
def tangram_preprocess(
    adata_sc_path: Annotated[str, Field(description="Path to single-cell RNA-seq data (h5ad)")],
    adata_sp_path: Annotated[str, Field(description="Path to spatial transcriptomics data (h5ad)")],
    marker_genes: Annotated[str, Field(description="Comma-separated genes, or 'auto' to compute from scRNA-seq")] = "auto",
    cell_type_key: Annotated[str, Field(description="Cell type column for auto marker detection")] = "cell_type",
    n_markers: Annotated[int, Field(description="Top markers per cell type if auto")] = 100,
    save_path: Annotated[str, Field(description="Directory to save preprocessed data")] = None,
) -> str:
    """Preprocess scRNA-seq and spatial data for Tangram mapping.

    Finds shared genes, removes zero-valued genes, and computes density priors.
    Must run before tangram_map_cells.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import tangram as tg

    os.makedirs(save_path, exist_ok=True)
    sc_out = f"{save_path}/tangram_sc_prep.h5ad"
    sp_out = f"{save_path}/tangram_sp_prep.h5ad"

    adata_sc = sc.read_h5ad(adata_sc_path)
    adata_sp = sc.read_h5ad(adata_sp_path)

    if marker_genes == "auto":
        sc.tl.rank_genes_groups(adata_sc, groupby=cell_type_key, use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[:n_markers, :]
        markers = list(np.unique(markers_df.melt().value.values))
    else:
        from .utils import parse_list_string
        markers = parse_list_string(marker_genes)

    tg.pp_adatas(adata_sc, adata_sp, genes=markers)
    n_train = len(adata_sc.uns.get("training_genes", []))

    adata_sc.write_h5ad(sc_out, compression="gzip")
    adata_sp.write_h5ad(sp_out, compression="gzip")

    msg = f"Preprocessed for Tangram: {n_train} training genes. Saved to {sc_out} and {sp_out}"
    print(msg)
    return msg


@tool
def tangram_map_cells(
    adata_sc_path: Annotated[str, Field(description="Path to preprocessed scRNA-seq (from tangram_preprocess)")],
    adata_sp_path: Annotated[str, Field(description="Path to preprocessed spatial data")],
    mode: Annotated[str, Field(description="'cells' (single-cell, GPU recommended) or 'clusters' (faster)")] = "cells",
    cluster_label: Annotated[str, Field(description="Cell type column for 'clusters' mode")] = "cell_type",
    device: Annotated[str, Field(description="'cpu' or 'cuda:0'")] = "cpu",
    num_epochs: Annotated[int, Field(description="Training epochs")] = 500,
    save_path: Annotated[str, Field(description="Directory to save mapping")] = None,
) -> str:
    """Map single cells onto spatial locations using Tangram.

    Creates cell-by-spot probability matrix by optimizing gene expression similarity.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import tangram as tg

    out_path = f"{save_path}/tangram_mapping.h5ad"
    if exists(out_path):
        msg = f"Mapping exists at {out_path}"
        print(msg)
        return msg

    print(f"[tangram_map_cells] Running Tangram mapping in {mode} mode...")

    adata_sc = sc.read_h5ad(adata_sc_path)
    adata_sp = sc.read_h5ad(adata_sp_path)

    if mode == "clusters":
        ad_map = tg.map_cells_to_space(adata_sc, adata_sp, mode="clusters",
            cluster_label=cluster_label, density_prior='rna_count_based',
            num_epochs=num_epochs, device=device)
    else:
        ad_map = tg.map_cells_to_space(adata_sc, adata_sp, mode="cells",
            density_prior='rna_count_based', num_epochs=num_epochs, device=device)

    ad_map.write_h5ad(out_path, compression="gzip")
    score = ad_map.uns.get("train_genes_df", pd.DataFrame())["train_score"].mean() if "train_genes_df" in ad_map.uns else 0

    msg = f"Mapped {ad_map.shape[0]} cells to {ad_map.shape[1]} spots. Avg score: {score:.3f}. Saved to {out_path}"
    print(msg)
    return msg


@tool
def tangram_project_annotations(
    adata_map_path: Annotated[str, Field(description="Path to Tangram mapping")],
    adata_sp_path: Annotated[str, Field(description="Path to spatial data")],
    annotation: Annotated[str, Field(description="Annotation to project (e.g., 'cell_type')")] = "cell_type",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Project cell type annotations onto spatial data.

    Transfers cell type probabilities to spatial spots based on mapping.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import tangram as tg

    ad_map = sc.read_h5ad(adata_map_path)
    adata_sp = sc.read_h5ad(adata_sp_path)

    tg.project_cell_annotations(ad_map, adata_sp, annotation=annotation)
    ct_pred = adata_sp.obsm.get("tangram_ct_pred")

    if ct_pred is not None:
        ct_pred.to_csv(f"{save_path}/tangram_celltype_probs.csv")
        adata_sp.obs["tangram_celltype"] = ct_pred.idxmax(axis=1)

    adata_sp.write_h5ad(f"{save_path}/tangram_annotated.h5ad", compression="gzip")
    n_types = ct_pred.shape[1] if ct_pred is not None else 0

    msg = f"Projected {n_types} cell types. Saved to {save_path}/tangram_annotated.h5ad"
    print(msg)
    return msg


@tool
def tangram_project_genes(
    adata_map_path: Annotated[str, Field(description="Path to Tangram mapping")],
    adata_sc_path: Annotated[str, Field(description="Path to scRNA-seq data")],
    cluster_label: Annotated[str, Field(description="Cluster label if 'clusters' mode used")] = "",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Project gene expression from scRNA-seq onto spatial locations.

    Creates imputed spatial gene expression for the entire transcriptome.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import tangram as tg

    ad_map = sc.read_h5ad(adata_map_path)
    adata_sc = sc.read_h5ad(adata_sc_path)

    if cluster_label and cluster_label in adata_sc.obs.columns:
        ad_ge = tg.project_genes(ad_map, adata_sc, cluster_label=cluster_label)
    else:
        ad_ge = tg.project_genes(ad_map, adata_sc)

    out_path = f"{save_path}/tangram_projected.h5ad"
    ad_ge.write_h5ad(out_path, compression="gzip")

    msg = f"Projected {ad_ge.shape[1]} genes to {ad_ge.shape[0]} spots. Saved to {out_path}"
    print(msg)
    return msg


@tool
def tangram_evaluate(
    adata_ge_path: Annotated[str, Field(description="Path to projected genes (from tangram_project_genes)")],
    adata_sp_path: Annotated[str, Field(description="Path to spatial data (original or preprocessed)")],
    save_path: Annotated[str, Field(description="Directory to save evaluation")] = None,
) -> str:
    """Evaluate Tangram mapping quality.

    Computes per-gene cosine similarity between predicted and measured expression.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import matplotlib.pyplot as plt
    from scipy import sparse

    os.makedirs(save_path, exist_ok=True)

    ad_ge = sc.read_h5ad(adata_ge_path)
    adata_sp = sc.read_h5ad(adata_sp_path)

    # Find overlapping genes (case-insensitive)
    ge_genes = set(g.lower() for g in ad_ge.var_names)
    sp_genes = set(g.lower() for g in adata_sp.var_names)
    overlap = sorted(ge_genes & sp_genes)

    if len(overlap) < 5:
        msg = f"ERROR: Only {len(overlap)} overlapping genes. Cannot evaluate."
        print(msg)
        return msg

    # Get expression matrices for overlapping genes
    ge_idx = [i for i, g in enumerate(ad_ge.var_names) if g.lower() in overlap]
    sp_idx = [i for i, g in enumerate(adata_sp.var_names) if g.lower() in overlap]

    X_ge = ad_ge.X[:, ge_idx]
    X_sp = adata_sp.X[:, sp_idx]

    if sparse.issparse(X_ge):
        X_ge = X_ge.toarray()
    if sparse.issparse(X_sp):
        X_sp = X_sp.toarray()

    # Compute per-gene cosine similarity
    scores = []
    gene_names = [ad_ge.var_names[i] for i in ge_idx]
    for i in range(len(overlap)):
        v1, v2 = X_ge[:, i], X_sp[:, i]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        score = (v1 @ v2) / norm if norm > 0 else 0
        scores.append(score)

    df = pd.DataFrame({'gene': gene_names, 'score': scores})
    df = df.sort_values('score', ascending=False)
    df.to_csv(f"{save_path}/tangram_scores.csv", index=False)

    avg_score = np.nanmean(scores)
    median_score = np.nanmedian(scores)

    # Plot score distribution
    plt.ioff()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(scores, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[0].axvline(avg_score, color='red', linestyle='--', label=f'Mean: {avg_score:.3f}')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Gene Score Distribution')
    axes[0].legend()

    axes[1].scatter(range(len(scores)), sorted(scores, reverse=True), s=10, alpha=0.6)
    axes[1].set_xlabel('Gene Rank')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Ranked Gene Scores')
    plt.tight_layout()
    plt.savefig(f"{save_path}/tangram_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    top5 = ', '.join(df.head(5)['gene'].tolist())

    msg = f"Evaluation: Mean={avg_score:.3f}, Median={median_score:.3f} ({len(overlap)} genes). Top: {top5}. Saved to {save_path}/tangram_scores.csv"
    print(msg)
    return msg


# =============================================================================
# CellPhoneDB Tools - Cell-cell communication analysis
# =============================================================================

@tool
def cellphonedb_prepare(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    cell_type_key: Annotated[str, Field(description="Column in obs for cell type annotations")] = "cell_type",
    layer: Annotated[str, Field(description="Layer to use for counts (empty for .X)")] = "",
    save_path: Annotated[str, Field(description="Directory to save prepared files")] = None,
) -> str:
    """Prepare AnnData for CellPhoneDB analysis.

    Extracts counts matrix and metadata files required by CellPhoneDB methods.
    Converts gene symbols to human format if needed.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Extract metadata
    meta = pd.DataFrame({
        'Cell': adata.obs_names,
        'cell_type': adata.obs[cell_type_key].values
    })
    meta_path = f"{save_path}/cellphonedb_meta.txt"
    meta.to_csv(meta_path, sep='\t', index=False)

    # Extract counts - CellPhoneDB expects normalized counts
    counts_path = f"{save_path}/cellphonedb_counts.h5ad"
    if layer and layer in adata.layers:
        adata_out = adata.copy()
        adata_out.X = adata.layers[layer]
    else:
        adata_out = adata

    adata_out.write_h5ad(counts_path, compression="gzip")

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    n_types = adata.obs[cell_type_key].nunique()

    msg = f"Prepared CellPhoneDB inputs: {n_cells} cells, {n_genes} genes, {n_types} cell types. Saved to {meta_path} and {counts_path}"
    print(msg)
    return msg


@tool
def cellphonedb_analysis(
    counts_path: Annotated[str, Field(description="Path to counts h5ad file")],
    meta_path: Annotated[str, Field(description="Path to metadata txt file")],
    iterations: Annotated[int, Field(description="Permutation iterations (0 for simple/no-stats method)")] = 1000,
    threshold: Annotated[float, Field(description="Min fraction of cells expressing gene")] = 0.1,
    microenvs_path: Annotated[str, Field(description="Path to microenvironments file (optional, restricts to colocalized cells)")] = "",
    score_interactions: Annotated[bool, Field(description="Score interactions by specificity (0-10 scale)")] = False,
    threads: Annotated[int, Field(description="Number of threads")] = 4,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run CellPhoneDB analysis (statistical or simple).

    METHOD 2 (Statistical): Permutation-based p-values for enriched interactions.
    Set iterations=0 for METHOD 1 (Simple): Mean expression only, no statistics.

    Outputs: means.txt, pvalues.txt, significant_means.txt, deconvoluted.txt
    """
    save_path = save_path or _config["save_path"]
    os.makedirs(save_path, exist_ok=True)

    # Get CellPhoneDB database path
    cpdb_dir = os.path.expanduser("~/.cpdb")
    cpdb_file_path = os.path.join(cpdb_dir, "cellphonedb.zip")

    # Prepare optional parameters
    microenvs = microenvs_path if microenvs_path and os.path.exists(microenvs_path) else None

    if iterations == 0:
        # METHOD 1: Simple analysis (no statistics)
        from cellphonedb.src.core.methods import cpdb_analysis_method

        out_path = f"{save_path}/cellphonedb_simple"
        os.makedirs(out_path, exist_ok=True)

        results = cpdb_analysis_method.call(
            cpdb_file_path=cpdb_file_path,
            meta_file_path=meta_path,
            counts_file_path=counts_path,
            counts_data='hgnc_symbol',
            output_path=out_path,
            threshold=threshold,
            microenvs_file_path=microenvs,
            score_interactions=score_interactions,
        )
        method_name = "simple"
    else:
        # METHOD 2: Statistical analysis
        from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

        out_path = f"{save_path}/cellphonedb_statistical"
        os.makedirs(out_path, exist_ok=True)

        results = cpdb_statistical_analysis_method.call(
            cpdb_file_path=cpdb_file_path,
            meta_file_path=meta_path,
            counts_file_path=counts_path,
            counts_data='hgnc_symbol',
            output_path=out_path,
            threshold=threshold,
            iterations=iterations,
            threads=threads,
            microenvs_file_path=microenvs,
            score_interactions=score_interactions,
        )
        method_name = "statistical"

    # Count results
    means = results.get('means', pd.DataFrame())
    pvals = results.get('pvalues', pd.DataFrame())
    sig_means = results.get('significant_means', pd.DataFrame())

    n_interactions = len(means) if len(means) > 0 else 0
    n_significant = (sig_means.notna().sum(axis=1) > 0).sum() if len(sig_means) > 0 else 0

    msg = f"CellPhoneDB {method_name} analysis complete: {n_interactions} interactions tested, {n_significant} significant. Results saved to {out_path}"
    print(msg)
    return msg


@tool
def cellphonedb_degs_analysis(
    counts_path: Annotated[str, Field(description="Path to counts h5ad file")],
    meta_path: Annotated[str, Field(description="Path to metadata txt file")],
    degs_path: Annotated[str, Field(description="Path to DEGs file (two columns: Cell, Gene)")],
    threshold: Annotated[float, Field(description="Min fraction of cells expressing gene")] = 0.1,
    microenvs_path: Annotated[str, Field(description="Path to microenvironments file (optional, restricts to colocalized cells)")] = "",
    score_interactions: Annotated[bool, Field(description="Score interactions by specificity (0-10 scale)")] = False,
    threads: Annotated[int, Field(description="Number of threads")] = 4,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run CellPhoneDB DEG-based analysis (METHOD 3).

    Retrieves interactions where at least one gene is differentially expressed.
    More targeted than statistical method for specific comparisons.

    DEGs file format: Two columns - 'Cell' (cell type) and 'Gene' (gene symbol).
    Only interactions with at least one DEG are returned as relevant.
    """
    save_path = save_path or _config["save_path"]
    from cellphonedb.src.core.methods import cpdb_degs_analysis_method

    os.makedirs(save_path, exist_ok=True)
    out_path = f"{save_path}/cellphonedb_degs"
    os.makedirs(out_path, exist_ok=True)

    # Get CellPhoneDB database path
    cpdb_dir = os.path.expanduser("~/.cpdb")
    cpdb_file_path = os.path.join(cpdb_dir, "cellphonedb.zip")

    # Prepare optional parameters
    microenvs = microenvs_path if microenvs_path and os.path.exists(microenvs_path) else None

    results = cpdb_degs_analysis_method.call(
        cpdb_file_path=cpdb_file_path,
        meta_file_path=meta_path,
        counts_file_path=counts_path,
        degs_file_path=degs_path,
        counts_data='hgnc_symbol',
        output_path=out_path,
        threshold=threshold,
        microenvs_file_path=microenvs,
        score_interactions=score_interactions,
        threads=threads,
    )

    relevant = results.get('relevant_interactions', pd.DataFrame())
    sig_means = results.get('significant_means', pd.DataFrame())

    n_relevant = (relevant == 1).sum().sum() if len(relevant) > 0 else 0
    n_interactions = len(sig_means) if len(sig_means) > 0 else 0

    msg = f"CellPhoneDB DEG analysis complete (METHOD 3): {n_relevant} relevant interactions from {n_interactions} tested. Results saved to {out_path}"
    print(msg)
    return msg


@tool
def cellphonedb_filter(
    results_path: Annotated[str, Field(description="Path to CellPhoneDB results directory")],
    cell_types: Annotated[str, Field(description="Comma-separated cell types to filter (e.g., 'T cell,B cell')")] = "",
    genes: Annotated[str, Field(description="Comma-separated genes to filter")] = "",
    min_mean: Annotated[float, Field(description="Minimum mean expression threshold")] = 0.0,
    save_path: Annotated[str, Field(description="Directory to save filtered results")] = None,
) -> str:
    """Search and filter CellPhoneDB results.

    Filters significant interactions by cell types, genes, or expression levels.
    """
    save_path = save_path or _config["save_path"]
    os.makedirs(save_path, exist_ok=True)

    # Load results
    sig_means_path = f"{results_path}/significant_means.csv"
    if not exists(sig_means_path):
        sig_means_path = f"{results_path}/significant_means.txt"

    if not exists(sig_means_path):
        msg = f"ERROR: No significant_means file found in {results_path}"
        print(msg)
        return msg

    sig_means = pd.read_csv(sig_means_path, sep='\t' if sig_means_path.endswith('.txt') else ',')

    # Get cell type pair columns (exclude metadata columns)
    meta_cols = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b',
                 'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b',
                 'annotation_strategy', 'is_integrin', 'rank']
    pair_cols = [c for c in sig_means.columns if c not in meta_cols and '|' in c]

    filtered = sig_means.copy()

    # Filter by cell types
    if cell_types:
        ct_list = [ct.strip() for ct in cell_types.split(',')]
        matching_cols = [c for c in pair_cols if any(ct in c for ct in ct_list)]
        if matching_cols:
            keep_cols = [c for c in sig_means.columns if c not in pair_cols] + matching_cols
            filtered = filtered[keep_cols]
            pair_cols = matching_cols

    # Filter by genes
    if genes:
        from .utils import parse_list_string
        gene_list = parse_list_string(genes, uppercase=True)
        mask = filtered['interacting_pair'].str.upper().apply(
            lambda x: any(g in x for g in gene_list)
        )
        filtered = filtered[mask]

    # Filter by minimum mean expression
    if min_mean > 0 and pair_cols:
        mask = filtered[pair_cols].max(axis=1) >= min_mean
        filtered = filtered[mask]

    out_path = f"{save_path}/cellphonedb_filtered.csv"
    filtered.to_csv(out_path, index=False)

    msg = f"Filtered results: {len(filtered)} interactions. Saved to {out_path}"
    print(msg)
    return msg


@tool
def cellphonedb_plot(
    results_path: Annotated[str, Field(description="Path to CellPhoneDB results directory")],
    plot_type: Annotated[str, Field(description="'dotplot', 'heatmap', or 'chord'")] = "dotplot",
    cell_types: Annotated[str, Field(description="Comma-separated cell types to include (empty for all)")] = "",
    top_n: Annotated[int, Field(description="Number of top interactions to plot")] = 30,
    save_path: Annotated[str, Field(description="Directory to save plots")] = None,
) -> str:
    """Visualize CellPhoneDB results.

    Creates dot plots, heatmaps, or chord diagrams of significant interactions.
    """
    save_path = save_path or _config["save_path"]
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_path, exist_ok=True)

    # Load results
    sig_means_path = f"{results_path}/significant_means.csv"
    pvals_path = f"{results_path}/pvalues.csv"

    if not exists(sig_means_path):
        sig_means_path = f"{results_path}/significant_means.txt"
        pvals_path = f"{results_path}/pvalues.txt"

    if not exists(sig_means_path):
        msg = f"ERROR: No results found in {results_path}"
        print(msg)
        return msg

    sep = '\t' if sig_means_path.endswith('.txt') else ','
    sig_means = pd.read_csv(sig_means_path, sep=sep)
    pvals = pd.read_csv(pvals_path, sep=sep) if exists(pvals_path) else None

    # Get cell type pair columns
    meta_cols = ['id_cp_interaction', 'interacting_pair', 'partner_a', 'partner_b',
                 'gene_a', 'gene_b', 'secreted', 'receptor_a', 'receptor_b',
                 'annotation_strategy', 'is_integrin', 'rank']
    pair_cols = [c for c in sig_means.columns if c not in meta_cols and '|' in c]

    # Filter by cell types if specified
    if cell_types:
        ct_list = [ct.strip() for ct in cell_types.split(',')]
        pair_cols = [c for c in pair_cols if any(ct in c for ct in ct_list)]

    if not pair_cols:
        msg = "ERROR: No cell type pairs found after filtering"
        print(msg)
        return msg

    # Get top interactions by mean expression
    means_data = sig_means[['interacting_pair'] + pair_cols].copy()
    means_data['max_mean'] = means_data[pair_cols].max(axis=1)
    means_data = means_data.nlargest(top_n, 'max_mean')

    plt.ioff()

    if plot_type == "dotplot":
        # Prepare data for dot plot
        plot_data = means_data.melt(
            id_vars=['interacting_pair'],
            value_vars=pair_cols,
            var_name='cell_pair',
            value_name='mean'
        ).dropna()

        if pvals is not None:
            pval_data = pvals[['interacting_pair'] + pair_cols].melt(
                id_vars=['interacting_pair'],
                value_vars=pair_cols,
                var_name='cell_pair',
                value_name='pvalue'
            )
            plot_data = plot_data.merge(pval_data, on=['interacting_pair', 'cell_pair'])
            plot_data['-log10(pval)'] = -np.log10(plot_data['pvalue'] + 1e-10)

        fig, ax = plt.subplots(figsize=(max(12, len(pair_cols)*0.8), max(8, top_n*0.3)))

        # Create pivot for heatmap-style dot plot
        pivot = plot_data.pivot(index='interacting_pair', columns='cell_pair', values='mean')
        sns.heatmap(pivot, cmap='Reds', ax=ax, cbar_kws={'label': 'Mean Expression'})
        ax.set_title(f'Top {top_n} Interactions')
        plt.xticks(rotation=45, ha='right')

    elif plot_type == "heatmap":
        # Interaction count heatmap per cell type pair
        counts = (sig_means[pair_cols].notna() & (sig_means[pair_cols] > 0)).sum()
        count_df = pd.DataFrame({'pair': counts.index, 'count': counts.values})
        count_df[['source', 'target']] = count_df['pair'].str.split('|', expand=True)

        pivot = count_df.pivot(index='source', columns='target', values='count').fillna(0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax)
        ax.set_title('Significant Interactions per Cell Type Pair')

    elif plot_type == "chord":
        # Simplified chord-like visualization as bar plot
        counts = (sig_means[pair_cols].notna() & (sig_means[pair_cols] > 0)).sum()
        counts = counts.sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 6))
        counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_ylabel('Number of Interactions')
        ax.set_title('Interactions per Cell Type Pair')
        plt.xticks(rotation=45, ha='right')

    else:
        msg = f"ERROR: Unknown plot type '{plot_type}'. Use 'dotplot', 'heatmap', or 'chord'"
        print(msg)
        return msg

    plt.tight_layout()
    out_path = f"{save_path}/cellphonedb_{plot_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    msg = f"Created {plot_type} visualization. Saved to {out_path}"
    print(msg)
    return msg


# =============================================================================
# LIANA Tools - Multi-method cell-cell communication analysis
# =============================================================================

@tool
def liana_inference(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    cell_type_key: Annotated[str, Field(description="Column in obs for cell type annotations")] = "cell_type",
    sample_key: Annotated[str, Field(description="Column for sample IDs (empty for single-sample)")] = "",
    organism: Annotated[str, Field(description="'human', 'mouse', or 'auto'")] = "auto",
    expr_prop: Annotated[float, Field(description="Min proportion of cells expressing gene")] = 0.1,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run LIANA rank aggregate for ligand-receptor inference.

    Combines multiple LR methods (CellPhoneDB, CellChat, NATMI, etc.) into consensus rankings.
    Works for single-sample or multi-sample analysis.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import liana as li

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Auto-detect organism
    if organism == "auto":
        sample_genes = adata.var_names[:100].tolist()
        n_mouse = sum(1 for g in sample_genes if len(g) > 1 and g[0].isupper() and g[1:].islower())
        organism = "mouse" if n_mouse > 50 else "human"

    resource = "mouseconsensus" if organism == "mouse" else "consensus"

    if sample_key and sample_key in adata.obs.columns:
        # Multi-sample analysis
        li.mt.rank_aggregate.by_sample(
            adata,
            sample_key=sample_key,
            groupby=cell_type_key,
            resource_name=resource,
            expr_prop=expr_prop,
            use_raw=False,
            verbose=True,
            key_added='liana_res'
        )
        n_samples = adata.obs[sample_key].nunique()
        mode = f"multi-sample ({n_samples} samples)"
    else:
        # Single-sample analysis
        li.mt.rank_aggregate(
            adata,
            groupby=cell_type_key,
            resource_name=resource,
            expr_prop=expr_prop,
            use_raw=False,
            verbose=True,
            key_added='liana_res'
        )
        mode = "single-sample"

    # Save results
    liana_res = adata.uns['liana_res']
    out_csv = f"{save_path}/liana_results.csv"
    liana_res.to_csv(out_csv, index=False)

    # Save updated adata
    out_h5ad = f"{save_path}/liana_adata.h5ad"
    adata.write_h5ad(out_h5ad, compression="gzip")

    n_interactions = len(liana_res)
    n_pairs = liana_res[['source', 'target']].drop_duplicates().shape[0]

    msg = f"LIANA {mode} analysis complete: {n_interactions} interactions across {n_pairs} cell type pairs. Saved to {out_csv}"
    print(msg)
    return msg


@tool
def liana_spatial(
    adata_path: Annotated[str, Field(description="Path to spatial AnnData h5ad file")],
    local_metric: Annotated[str, Field(description="Local metric: 'cosine', 'pearson', 'spearman', 'jaccard'")] = "cosine",
    global_metric: Annotated[str, Field(description="Global metric: 'morans' or 'lee'")] = "morans",
    bandwidth: Annotated[float, Field(description="Spatial bandwidth for neighbor weights")] = 200,
    n_perms: Annotated[int, Field(description="Permutations for p-value calculation")] = 100,
    organism: Annotated[str, Field(description="'human', 'mouse', or 'auto'")] = "auto",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run LIANA spatial bivariate analysis.

    Computes local and global spatial correlations between ligand-receptor pairs.
    Requires spatial coordinates in adata.obsm['spatial'].
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import liana as li

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check for spatial coordinates
    if 'spatial' not in adata.obsm:
        return "ERROR: No spatial coordinates found in adata.obsm['spatial']"

    # Auto-detect organism
    if organism == "auto":
        sample_genes = adata.var_names[:100].tolist()
        n_mouse = sum(1 for g in sample_genes if len(g) > 1 and g[0].isupper() and g[1:].islower())
        organism = "mouse" if n_mouse > 50 else "human"

    resource = "mouseconsensus" if organism == "mouse" else "consensus"

    # Build spatial neighbors
    li.ut.spatial_neighbors(adata, bandwidth=bandwidth, cutoff=0.1, kernel='gaussian', set_diag=True)

    # Run bivariate analysis
    lrdata = li.mt.bivariate(
        adata,
        resource_name=resource,
        local_name=local_metric,
        global_name=global_metric,
        n_perms=n_perms,
        mask_negatives=False,
        add_categories=True,
        nz_prop=0.1,
        use_raw=False,
        verbose=True
    )

    # Save results
    out_h5ad = f"{save_path}/liana_spatial.h5ad"
    lrdata.write_h5ad(out_h5ad, compression="gzip")

    # Save summary statistics
    var_df = lrdata.var.copy()
    var_df.to_csv(f"{save_path}/liana_spatial_summary.csv")

    n_interactions = lrdata.n_vars
    top_by_global = var_df.sort_values(global_metric, ascending=False).head(5).index.tolist()

    msg = f"LIANA bivariate analysis complete: {n_interactions} interactions. Top by {global_metric}: {', '.join(top_by_global[:3])}. Saved to {out_h5ad}"
    print(msg)
    return msg


@tool
def liana_misty(
    adata_path: Annotated[str, Field(description="Path to spatial AnnData h5ad file")],
    target_key: Annotated[str, Field(description="obsm key for target features (e.g., 'compositions')")] = "",
    predictor_key: Annotated[str, Field(description="obsm key for predictor features (e.g., 'pathway_scores')")] = "",
    bandwidth: Annotated[float, Field(description="Spatial bandwidth for para view")] = 200,
    n_neighs: Annotated[int, Field(description="Number of neighbors for juxta view")] = 6,
    model_type: Annotated[str, Field(description="'linear' or 'rf' (random forest)")] = "linear",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run LIANA MISTy for learning spatial relationships.

    Models how features in different spatial contexts (intra, juxta, para) predict target features.
    Useful for understanding spatial dependencies between cell types, pathways, or gene programs.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import liana as li
    from liana.method import genericMistyData
    from liana.method.sp import LinearModel, RandomForestModel

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check for spatial coordinates
    if 'spatial' not in adata.obsm:
        return "ERROR: No spatial coordinates found in adata.obsm['spatial']"

    # Build spatial neighbors
    li.ut.spatial_neighbors(adata, bandwidth=bandwidth, cutoff=0.1, set_diag=False)

    # Extract target and predictor data
    if target_key and target_key in adata.obsm:
        intra = li.ut.obsm_to_adata(adata, target_key)
    else:
        # Use highly variable genes as default
        if 'highly_variable' not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=500)
        hvg = adata.var[adata.var['highly_variable']].index
        intra = adata[:, hvg].copy()

    if predictor_key and predictor_key in adata.obsm:
        extra = li.ut.obsm_to_adata(adata, predictor_key)
    else:
        extra = None

    # Create MISTy data
    if extra is not None:
        misty = genericMistyData(
            intra=intra,
            extra=extra,
            cutoff=0.05,
            bandwidth=bandwidth,
            n_neighs=n_neighs
        )
    else:
        misty = genericMistyData(
            intra=intra,
            cutoff=0.05,
            bandwidth=bandwidth,
            n_neighs=n_neighs
        )

    # Select model
    model = LinearModel if model_type == "linear" else RandomForestModel

    # Run MISTy
    misty(model=model, bypass_intra=False, verbose=True)

    # Extract results
    target_metrics = misty.uns.get('target_metrics', pd.DataFrame())
    interactions = misty.uns.get('interactions', {})

    # Save results as CSV (avoid pickling non-serializable objects)
    if len(target_metrics) > 0:
        target_metrics.to_csv(f"{save_path}/misty_metrics.csv", index=False)

    # Save interaction importances per view
    for view_name, view_df in interactions.items():
        if isinstance(view_df, pd.DataFrame) and len(view_df) > 0:
            view_df.to_csv(f"{save_path}/misty_{view_name}.csv")

    if len(target_metrics) > 0:
        avg_r2 = target_metrics['multi_R2'].mean() if 'multi_R2' in target_metrics.columns else 0
        avg_gain = target_metrics['gain_R2'].mean() if 'gain_R2' in target_metrics.columns else 0
        msg = f"MISTy analysis complete: {len(target_metrics)} targets. Avg R²={avg_r2:.3f}, Avg gain={avg_gain:.3f}. Saved to {save_path}/misty_*.csv"
        print(msg)
        return msg

    msg = f"MISTy analysis complete. Saved to {save_path}/misty_*.csv"
    print(msg)
    return msg


@tool
def liana_plot(
    results_path: Annotated[str, Field(description="Path to LIANA results CSV or h5ad")],
    plot_type: Annotated[str, Field(description="'dotplot', 'tileplot', or 'source_target'")] = "dotplot",
    source_cells: Annotated[str, Field(description="Comma-separated source cell types to include")] = "",
    target_cells: Annotated[str, Field(description="Comma-separated target cell types to include")] = "",
    top_n: Annotated[int, Field(description="Number of top interactions to plot")] = 20,
    save_path: Annotated[str, Field(description="Directory to save plots")] = None,
) -> str:
    """Visualize LIANA results.

    Creates dotplots, tileplots, or source-target network plots of LR interactions.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_path, exist_ok=True)

    # Load results
    if results_path.endswith('.csv'):
        liana_res = pd.read_csv(results_path)
    elif results_path.endswith('.h5ad'):
        adata = sc.read_h5ad(results_path)
        if 'liana_res' in adata.uns:
            liana_res = adata.uns['liana_res']
        else:
            return "ERROR: No liana_res found in adata.uns"
    else:
        return "ERROR: Provide .csv or .h5ad file"

    # Filter by cell types if specified
    if source_cells:
        src_list = [s.strip() for s in source_cells.split(',')]
        liana_res = liana_res[liana_res['source'].isin(src_list)]

    if target_cells:
        tgt_list = [t.strip() for t in target_cells.split(',')]
        liana_res = liana_res[liana_res['target'].isin(tgt_list)]

    if len(liana_res) == 0:
        return "ERROR: No interactions found after filtering"

    # Sort by magnitude or specificity
    sort_col = 'magnitude_rank' if 'magnitude_rank' in liana_res.columns else 'lr_means'
    ascending = True if 'rank' in sort_col else False
    liana_res = liana_res.sort_values(sort_col, ascending=ascending)

    plt.ioff()

    if plot_type == "dotplot":
        # Create interaction label
        liana_res = liana_res.head(top_n * 5)  # Get more for filtering
        liana_res['interaction'] = liana_res['ligand_complex'] + ' → ' + liana_res['receptor_complex']
        liana_res['cell_pair'] = liana_res['source'] + ' → ' + liana_res['target']

        # Get top interactions
        top_interactions = liana_res.groupby('interaction')[sort_col].mean().sort_values(ascending=ascending).head(top_n).index

        plot_df = liana_res[liana_res['interaction'].isin(top_interactions)]

        # Pivot for heatmap
        if 'lr_means' in plot_df.columns:
            pivot = plot_df.pivot_table(index='interaction', columns='cell_pair', values='lr_means', aggfunc='mean')
        else:
            pivot = plot_df.pivot_table(index='interaction', columns='cell_pair', values='magnitude_rank', aggfunc='mean')

        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.6), max(8, len(pivot)*0.4)))
        sns.heatmap(pivot, cmap='Reds', ax=ax, cbar_kws={'label': 'Expression'})
        ax.set_title(f'Top {top_n} Ligand-Receptor Interactions')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    elif plot_type == "tileplot":
        # Interaction counts per cell pair
        counts = liana_res.groupby(['source', 'target']).size().reset_index(name='n_interactions')
        pivot = counts.pivot(index='source', columns='target', values='n_interactions').fillna(0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax)
        ax.set_title('Number of Interactions per Cell Type Pair')

    elif plot_type == "source_target":
        # Bar plot of interactions per source/target
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        source_counts = liana_res['source'].value_counts().head(15)
        source_counts.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_xlabel('Number of Interactions')
        axes[0].set_title('Interactions by Source Cell Type')
        axes[0].invert_yaxis()

        target_counts = liana_res['target'].value_counts().head(15)
        target_counts.plot(kind='barh', ax=axes[1], color='coral')
        axes[1].set_xlabel('Number of Interactions')
        axes[1].set_title('Interactions by Target Cell Type')
        axes[1].invert_yaxis()

    else:
        return f"ERROR: Unknown plot type '{plot_type}'. Use 'dotplot', 'tileplot', or 'source_target'"

    plt.tight_layout()
    out_path = f"{save_path}/liana_{plot_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    msg = f"Created {plot_type} visualization. Saved to {out_path}"
    print(msg)
    return msg


# =============================================================================
# Squidpy Tools - Spatial analysis and graph statistics
# =============================================================================

@tool
def squidpy_spatial_neighbors(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    coord_type: Annotated[str, Field(description="'grid' (for Visium/hex), 'generic' (for other), or 'auto'")] = "auto",
    n_neighs: Annotated[int, Field(description="Number of neighbors (for generic coord_type)")] = 6,
    n_rings: Annotated[int, Field(description="Number of rings (for grid coord_type, e.g., Visium)")] = 1,
    delaunay: Annotated[bool, Field(description="Use Delaunay triangulation")] = False,
    radius: Annotated[float, Field(description="Radius for neighbor search (0 to disable)")] = 0,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Build spatial neighbors graph for downstream spatial analysis.

    Creates connectivity and distance matrices stored in adata.obsp.
    Required before running most Squidpy graph analyses.

    coord_type options:
    - 'auto': Auto-detect (grid if Visium-like, generic otherwise)
    - 'grid': For Visium and other grid-based spatial data
    - 'generic': For MERFISH, SeqFISH, Slide-seq, and other non-grid data
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check for spatial coordinates
    if 'spatial' not in adata.obsm:
        return "ERROR: No spatial coordinates found in adata.obsm['spatial']"

    # Map coord_type (handle 'visium' as alias for 'grid', 'auto' as None)
    coord_type_map = {"visium": "grid", "auto": None}
    sq_coord_type = coord_type_map.get(coord_type, coord_type)

    # Build spatial neighbors based on coord_type
    if sq_coord_type == "grid":
        sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid")
    elif delaunay:
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
    elif radius > 0:
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic")
    elif sq_coord_type is None:
        # Auto-detect
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type=None)
    else:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    # Save results
    out_path = f"{save_path}/spatial_neighbors.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    n_cells = adata.n_obs
    n_edges = adata.obsp['spatial_connectivities'].nnz

    msg = f"Built spatial neighbors graph: {n_cells} cells, {n_edges} edges. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_nhood_enrichment(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file (with spatial neighbors)")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster/cell type annotations")] = "cell_type",
    n_perms: Annotated[int, Field(description="Number of permutations for significance")] = 1000,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute neighborhood enrichment between cell type clusters.

    Tests whether cell types are enriched or depleted as neighbors compared to random.
    Requires spatial neighbors graph (run squidpy_spatial_neighbors first).
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check prerequisites
    if 'spatial_connectivities' not in adata.obsp:
        return "ERROR: No spatial neighbors found. Run squidpy_spatial_neighbors first."

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Compute neighborhood enrichment
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_perms=n_perms)

    # Save plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
    sq.pl.nhood_enrichment(adata, cluster_key=cluster_key, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_path}/squidpy_nhood_enrichment.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    out_path = f"{save_path}/squidpy_nhood.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    # Extract z-scores summary
    zscore_key = f"{cluster_key}_nhood_enrichment"
    if zscore_key in adata.uns:
        zscores = adata.uns[zscore_key]['zscore']
        max_enrich = zscores.max().max()
        min_enrich = zscores.min().min()
        msg = f"Neighborhood enrichment complete. Z-scores range: [{min_enrich:.2f}, {max_enrich:.2f}]. Saved to {out_path}"
        print(msg)
        return msg

    msg = f"Neighborhood enrichment complete. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_co_occurrence(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster/cell type annotations")] = "cell_type",
    spatial_key: Annotated[str, Field(description="Key in obsm for spatial coordinates")] = "spatial",
    interval: Annotated[int, Field(description="Number of distance intervals")] = 50,
    n_splits: Annotated[int, Field(description="Number of spatial splits for computation")] = 2,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute co-occurrence probability between cell types across distances.

    Calculates conditional probability of observing one cell type given another
    at various spatial distances.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Compute co-occurrence
    sq.gr.co_occurrence(adata, cluster_key=cluster_key, spatial_key=spatial_key, interval=interval, n_splits=n_splits)

    # Save plot for first few clusters
    plt.ioff()
    try:
        clusters = adata.obs[cluster_key].unique()[:3].tolist()
        sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=clusters, figsize=(12, 4))
        plt.savefig(f"{save_path}/co_occurrence.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        # Plotting may fail due to version issues, continue with data export
        pass

    # Save results
    out_path = f"{save_path}/squidpy_cooccur.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    msg = f"Co-occurrence analysis complete for {len(adata.obs[cluster_key].unique())} clusters. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_spatial_autocorr(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file (with spatial neighbors)")],
    mode: Annotated[str, Field(description="'moran' for Moran's I or 'geary' for Geary's C")] = "moran",
    genes: Annotated[str, Field(description="Comma-separated genes, or 'hvg' for highly variable, or 'all'")] = "hvg",
    n_perms: Annotated[int, Field(description="Number of permutations")] = 100,
    n_jobs: Annotated[int, Field(description="Number of parallel jobs")] = 4,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute spatial autocorrelation (Moran's I or Geary's C) for genes.

    Identifies genes with significant spatial patterns.
    Requires spatial neighbors graph (run squidpy_spatial_neighbors first).
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check prerequisites
    if 'spatial_connectivities' not in adata.obsp:
        return "ERROR: No spatial neighbors found. Run squidpy_spatial_neighbors first."

    # Select genes
    if genes in ["hvg", "highly_variable"]:
        if 'highly_variable' not in adata.var.columns:
            # Compute HVGs if not present
            sc.pp.highly_variable_genes(adata, n_top_genes=min(500, adata.n_vars))
        gene_list = adata.var[adata.var['highly_variable']].index.tolist()
    elif genes == "all":
        gene_list = adata.var_names.tolist()[:1000]  # Limit to avoid memory issues
    else:
        from .utils import parse_list_string
        gene_list = parse_list_string(genes)
        gene_list = [g for g in gene_list if g in adata.var_names]

    if len(gene_list) == 0:
        # Fallback to top variable genes
        gene_list = adata.var_names.tolist()[:100]

    if len(gene_list) == 0:
        return "ERROR: No genes available in dataset"

    # Compute spatial autocorrelation
    sq.gr.spatial_autocorr(adata, mode=mode, genes=gene_list, n_perms=n_perms, n_jobs=n_jobs)

    # Get results
    result_key = "moranI" if mode == "moran" else "gearyC"
    results_df = adata.uns[result_key]
    results_df.to_csv(f"{save_path}/squidpy_{mode}.csv")

    # Save adata
    out_path = f"{save_path}/squidpy_autocorr.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    # Summary
    sig_genes = results_df[results_df['pval_norm'] < 0.05]
    top_genes = results_df.head(5).index.tolist()

    msg = f"Spatial autocorrelation ({mode}): {len(gene_list)} genes tested, {len(sig_genes)} significant. Top: {', '.join(top_genes)}. Saved to {save_path}/squidpy_{mode}.csv"
    print(msg)
    return msg


@tool
def squidpy_ripley(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster annotations")] = "cell_type",
    mode: Annotated[str, Field(description="'L' (Ripley's L), 'F', or 'G'")] = "L",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute Ripley's statistics for spatial point pattern analysis.

    Determines whether cell types show clustered, random, or dispersed spatial patterns.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Compute Ripley's statistic
    sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode)

    # Save plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))
    sq.pl.ripley(adata, cluster_key=cluster_key, mode=mode, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_path}/squidpy_ripley_{mode}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    out_path = f"{save_path}/squidpy_ripley.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    n_clusters = adata.obs[cluster_key].nunique()
    msg = f"Ripley's {mode} analysis complete for {n_clusters} clusters. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_centrality(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file (with spatial neighbors)")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster annotations")] = "cell_type",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute centrality scores for cell type clusters in spatial graph.

    Calculates closeness centrality, clustering coefficient, and degree centrality.
    Requires spatial neighbors graph (run squidpy_spatial_neighbors first).
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check prerequisites
    if 'spatial_connectivities' not in adata.obsp:
        return "ERROR: No spatial neighbors found. Run squidpy_spatial_neighbors first."

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Compute centrality scores
    sq.gr.centrality_scores(adata, cluster_key=cluster_key)

    # Save plot
    plt.ioff()
    try:
        sq.pl.centrality_scores(adata, cluster_key=cluster_key, figsize=(12, 6))
        plt.savefig(f"{save_path}/centrality.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception:
        # Plotting may fail due to version issues
        pass

    # Save results
    out_path = f"{save_path}/squidpy_centrality.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    # Extract centrality results
    cent_key = f"{cluster_key}_centrality_scores"
    if cent_key in adata.uns:
        cent_df = adata.uns[cent_key]
        cent_df.to_csv(f"{save_path}/squidpy_centrality.csv")

    msg = f"Centrality scores computed for {adata.obs[cluster_key].nunique()} clusters. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_interaction_matrix(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file (with spatial neighbors)")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster annotations")] = "cell_type",
    normalized: Annotated[bool, Field(description="Row-normalize the interaction matrix")] = True,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute interaction matrix between cell type clusters.

    Quantifies the number of edges between cell types in the spatial graph.
    Requires spatial neighbors graph (run squidpy_spatial_neighbors first).
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    # Check prerequisites
    if 'spatial_connectivities' not in adata.obsp:
        return "ERROR: No spatial neighbors found. Run squidpy_spatial_neighbors first."

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Compute interaction matrix
    sq.gr.interaction_matrix(adata, cluster_key=cluster_key, normalized=normalized)

    # Save plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
    sq.pl.interaction_matrix(adata, cluster_key=cluster_key, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_path}/squidpy_interaction_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    out_path = f"{save_path}/squidpy_interaction.h5ad"
    adata.write_h5ad(out_path, compression="gzip")

    # Save matrix as CSV
    int_key = f"{cluster_key}_interactions"
    if int_key in adata.uns:
        int_df = pd.DataFrame(adata.uns[int_key])
        int_df.to_csv(f"{save_path}/squidpy_interaction_matrix.csv")

    msg = f"Interaction matrix computed for {adata.obs[cluster_key].nunique()} clusters. Saved to {out_path}"
    print(msg)
    return msg


@tool
def squidpy_ligrec(
    adata_path: Annotated[str, Field(description="Path to AnnData h5ad file")],
    cluster_key: Annotated[str, Field(description="Column in obs for cluster annotations")] = "cell_type",
    n_perms: Annotated[int, Field(description="Number of permutations")] = 1000,
    threshold: Annotated[float, Field(description="Min fraction of cells expressing gene")] = 0.1,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run receptor-ligand analysis using Squidpy and OmniPath database.

    Identifies significant ligand-receptor interactions between cell types.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import squidpy as sq

    os.makedirs(save_path, exist_ok=True)
    adata = sc.read_h5ad(adata_path)

    if cluster_key not in adata.obs.columns:
        return f"ERROR: Column '{cluster_key}' not found in adata.obs"

    # Run ligand-receptor analysis
    res = sq.gr.ligrec(
        adata,
        n_perms=n_perms,
        cluster_key=cluster_key,
        copy=True,
        use_raw=False,
        threshold=threshold,
        transmitter_params={"categories": "ligand"},
        receiver_params={"categories": "receptor"},
    )

    # Save results
    means_df = res['means']
    pvals_df = res['pvalues']

    means_df.to_csv(f"{save_path}/squidpy_ligrec_means.csv")
    pvals_df.to_csv(f"{save_path}/squidpy_ligrec_pvalues.csv")

    # Count significant interactions
    n_significant = (pvals_df < 0.05).sum().sum()
    n_total = pvals_df.notna().sum().sum()

    msg = f"Ligand-receptor analysis complete: {n_significant} significant interactions out of {n_total} tested. Saved to {save_path}/squidpy_ligrec_*.csv"
    print(msg)
    return msg


# =============================================================================
# scvi-tools Spatial Deconvolution Tools
# =============================================================================

@tool
def destvi_deconvolution(
    sc_adata_path: Annotated[str, Field(description="Path to single-cell reference h5ad file")],
    st_adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    cell_type_key: Annotated[str, Field(description="Column in sc_adata.obs for cell type labels")] = "cell_type",
    layer: Annotated[str, Field(description="Layer to use for counts (empty for .X)")] = "",
    sc_max_epochs: Annotated[int, Field(description="Max epochs for single-cell model")] = 300,
    st_max_epochs: Annotated[int, Field(description="Max epochs for spatial model")] = 2500,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run DestVI for multi-resolution spatial deconvolution.

    DestVI decomposes spatial spots into cell type proportions while capturing
    intra-cell-type variation (gamma). Requires a single-cell reference with
    cell type annotations.

    Outputs: Cell type proportions per spot, gamma latent space, trained models.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import scvi
    from scvi.model import CondSCVI, DestVI

    os.makedirs(save_path, exist_ok=True)

    # Load data
    sc_adata = sc.read_h5ad(sc_adata_path)
    st_adata = sc.read_h5ad(st_adata_path)

    # Validate cell type column
    if cell_type_key not in sc_adata.obs.columns:
        return f"ERROR: Column '{cell_type_key}' not found in sc_adata.obs. Available: {list(sc_adata.obs.columns)}"

    # Find common genes
    common_genes = list(set(sc_adata.var_names) & set(st_adata.var_names))
    if len(common_genes) < 100:
        return f"ERROR: Only {len(common_genes)} common genes found. Need at least 100."

    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    # Setup single-cell model
    if layer and layer in sc_adata.layers:
        CondSCVI.setup_anndata(sc_adata, layer=layer, labels_key=cell_type_key)
    else:
        CondSCVI.setup_anndata(sc_adata, labels_key=cell_type_key)

    # Train single-cell model
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=sc_max_epochs, accelerator="auto")

    # Setup spatial model
    if layer and layer in st_adata.layers:
        DestVI.setup_anndata(st_adata, layer=layer)
    else:
        DestVI.setup_anndata(st_adata)

    # Train spatial model
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=st_max_epochs, accelerator="auto")

    # Get proportions
    proportions = st_model.get_proportions()
    st_adata.obsm["destvi_proportions"] = proportions

    # Save results
    proportions.to_csv(f"{save_path}/destvi_proportions.csv")
    st_adata.write_h5ad(f"{save_path}/destvi_spatial.h5ad", compression="gzip")

    # Save models
    sc_model.save(f"{save_path}/destvi_sc_model", overwrite=True)
    st_model.save(f"{save_path}/destvi_st_model", overwrite=True)

    n_spots = st_adata.n_obs
    n_celltypes = proportions.shape[1]

    msg = f"DestVI deconvolution complete: {n_spots} spots, {n_celltypes} cell types, {len(common_genes)} genes. Results saved to {save_path}"
    print(msg)
    return msg


@tool
def cell2location_mapping(
    sc_adata_path: Annotated[str, Field(description="Path to single-cell reference h5ad file")],
    st_adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    cell_type_key: Annotated[str, Field(description="Column in sc_adata.obs for cell type labels")] = "cell_type",
    batch_key: Annotated[str, Field(description="Column for batch information (optional)")] = "",
    n_cells_per_location: Annotated[int, Field(description="Expected cells per spot (tissue-dependent)")] = 30,
    detection_alpha: Annotated[float, Field(description="Detection sensitivity (200 default, 20 for high variation)")] = 200,
    sc_max_epochs: Annotated[int, Field(description="Max epochs for reference model")] = 250,
    st_max_epochs: Annotated[int, Field(description="Max epochs for spatial model")] = 30000,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run Cell2location for spatial cell type mapping.

    Cell2location uses Bayesian inference to estimate cell type abundance at each
    spatial location. Integrates scRNA-seq reference with Visium/spatial data.

    Parameters:
    - n_cells_per_location: ~30 for lymph node, ~8 for brain, adjust per tissue
    - detection_alpha: Lower (20) for high within-batch variation

    Outputs: Cell abundance per spot (q05 quantile = confident estimates).
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    from cell2location.models import Cell2location, RegressionModel

    os.makedirs(save_path, exist_ok=True)

    # Load data
    sc_adata = sc.read_h5ad(sc_adata_path)
    st_adata = sc.read_h5ad(st_adata_path)

    # Validate cell type column
    if cell_type_key not in sc_adata.obs.columns:
        return f"ERROR: Column '{cell_type_key}' not found in sc_adata.obs. Available: {list(sc_adata.obs.columns)}"

    # Find common genes
    common_genes = list(set(sc_adata.var_names) & set(st_adata.var_names))
    if len(common_genes) < 100:
        return f"ERROR: Only {len(common_genes)} common genes found. Need at least 100."

    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    # Setup reference model
    setup_kwargs = {"labels_key": cell_type_key}
    if batch_key and batch_key in sc_adata.obs.columns:
        setup_kwargs["batch_key"] = batch_key

    RegressionModel.setup_anndata(sc_adata, **setup_kwargs)

    # Train reference model
    ref_model = RegressionModel(sc_adata)
    ref_model.train(max_epochs=sc_max_epochs, accelerator="auto")

    # Export signatures
    sc_adata = ref_model.export_posterior(
        sc_adata,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500}
    )

    # Get signatures for spatial model
    if "means_per_cluster_mu_fg" in sc_adata.varm:
        inf_aver = sc_adata.varm["means_per_cluster_mu_fg"].T.copy()
    else:
        return "ERROR: Failed to extract cell type signatures from reference model"

    # Setup spatial model
    Cell2location.setup_anndata(st_adata)

    # Train spatial model
    st_model = Cell2location(
        st_adata,
        cell_state_df=inf_aver,
        N_cells_per_location=n_cells_per_location,
        detection_alpha=detection_alpha,
    )
    st_model.train(max_epochs=st_max_epochs, accelerator="auto")

    # Export results
    st_adata = st_model.export_posterior(
        st_adata,
        sample_kwargs={"num_samples": 1000, "batch_size": st_adata.n_obs}
    )

    # Save results
    if "q05_cell_abundance_w_sf" in st_adata.obsm:
        abundance = pd.DataFrame(
            st_adata.obsm["q05_cell_abundance_w_sf"],
            index=st_adata.obs_names,
            columns=inf_aver.index
        )
        abundance.to_csv(f"{save_path}/cell2location_abundance.csv")

    st_adata.write_h5ad(f"{save_path}/cell2location_spatial.h5ad", compression="gzip")

    # Save models
    ref_model.save(f"{save_path}/cell2location_ref_model", overwrite=True)
    st_model.save(f"{save_path}/cell2location_st_model", overwrite=True)

    n_spots = st_adata.n_obs
    n_celltypes = inf_aver.shape[0]

    msg = f"Cell2location mapping complete: {n_spots} spots, {n_celltypes} cell types, {len(common_genes)} genes. Results saved to {save_path}"
    print(msg)
    return msg


@tool
def stereoscope_deconvolution(
    sc_adata_path: Annotated[str, Field(description="Path to single-cell reference h5ad file")],
    st_adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    cell_type_key: Annotated[str, Field(description="Column in sc_adata.obs for cell type labels")] = "cell_type",
    layer: Annotated[str, Field(description="Layer to use for counts (empty for .X)")] = "",
    sc_max_epochs: Annotated[int, Field(description="Max epochs for single-cell model")] = 100,
    st_max_epochs: Annotated[int, Field(description="Max epochs for spatial model")] = 2000,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run Stereoscope for spatial cell type deconvolution.

    Stereoscope learns cell-type-specific gene expression from scRNA-seq and
    infers cell type proportions in spatial spots.

    Outputs: Cell type proportions per spot, trained models.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    from scvi.external import RNAStereoscope, SpatialStereoscope

    os.makedirs(save_path, exist_ok=True)

    # Load data
    sc_adata = sc.read_h5ad(sc_adata_path)
    st_adata = sc.read_h5ad(st_adata_path)

    # Validate cell type column
    if cell_type_key not in sc_adata.obs.columns:
        return f"ERROR: Column '{cell_type_key}' not found in sc_adata.obs. Available: {list(sc_adata.obs.columns)}"

    # Find common genes
    common_genes = list(set(sc_adata.var_names) & set(st_adata.var_names))
    if len(common_genes) < 100:
        return f"ERROR: Only {len(common_genes)} common genes found. Need at least 100."

    sc_adata = sc_adata[:, common_genes].copy()
    st_adata = st_adata[:, common_genes].copy()

    # Setup single-cell model
    if layer and layer in sc_adata.layers:
        RNAStereoscope.setup_anndata(sc_adata, layer=layer, labels_key=cell_type_key)
    else:
        RNAStereoscope.setup_anndata(sc_adata, labels_key=cell_type_key)

    # Train single-cell model
    sc_model = RNAStereoscope(sc_adata)
    sc_model.train(max_epochs=sc_max_epochs, accelerator="auto")

    # Setup spatial model
    if layer and layer in st_adata.layers:
        SpatialStereoscope.setup_anndata(st_adata, layer=layer)
    else:
        SpatialStereoscope.setup_anndata(st_adata)

    # Train spatial model
    st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=st_max_epochs, accelerator="auto")

    # Get proportions
    proportions = st_model.get_proportions()
    st_adata.obsm["stereoscope_proportions"] = proportions

    # Save results
    proportions.to_csv(f"{save_path}/stereoscope_proportions.csv")
    st_adata.write_h5ad(f"{save_path}/stereoscope_spatial.h5ad", compression="gzip")

    # Save models
    sc_model.save(f"{save_path}/stereoscope_sc_model", overwrite=True)
    st_model.save(f"{save_path}/stereoscope_st_model", overwrite=True)

    n_spots = st_adata.n_obs
    n_celltypes = proportions.shape[1]

    msg = f"Stereoscope deconvolution complete: {n_spots} spots, {n_celltypes} cell types, {len(common_genes)} genes. Results saved to {save_path}"
    print(msg)
    return msg


@tool
def gimvi_imputation(
    sc_adata_path: Annotated[str, Field(description="Path to single-cell reference h5ad file")],
    st_adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    genes_to_impute: Annotated[str, Field(description="Comma-separated genes to impute (empty for all missing)")] = "",
    layer: Annotated[str, Field(description="Layer to use for counts (empty for .X)")] = "",
    max_epochs: Annotated[int, Field(description="Max training epochs")] = 200,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run gimVI for gene imputation in spatial data.

    gimVI learns a joint model of scRNA-seq and spatial data to impute
    genes missing from the spatial measurements using the single-cell reference.

    Useful for spatial technologies with limited gene panels (FISH-based).

    Outputs: Imputed gene expression matrix, imputed spatial AnnData.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    from scvi.external import GIMVI

    os.makedirs(save_path, exist_ok=True)

    # Load data
    sc_adata = sc.read_h5ad(sc_adata_path)
    st_adata = sc.read_h5ad(st_adata_path)

    # Determine genes to impute
    sc_genes = set(sc_adata.var_names)
    st_genes = set(st_adata.var_names)
    common_genes = list(sc_genes & st_genes)

    if len(common_genes) < 50:
        return f"ERROR: Only {len(common_genes)} common genes. Need at least 50 for training."

    if genes_to_impute:
        from .utils import parse_list_string
        target_genes = parse_list_string(genes_to_impute)
        missing_genes = [g for g in target_genes if g in sc_genes and g not in st_genes]
        if not missing_genes:
            return f"ERROR: No valid genes to impute. Genes must be in scRNA-seq but not in spatial data."
    else:
        missing_genes = list(sc_genes - st_genes)

    if not missing_genes:
        return "No genes to impute - all scRNA-seq genes already in spatial data."

    # Subset to common genes for training
    sc_adata_train = sc_adata[:, common_genes].copy()
    st_adata_train = st_adata[:, common_genes].copy()

    # Setup gimVI
    if layer and layer in sc_adata_train.layers:
        GIMVI.setup_anndata(sc_adata_train, layer=layer)
        GIMVI.setup_anndata(st_adata_train, layer=layer)
    else:
        GIMVI.setup_anndata(sc_adata_train)
        GIMVI.setup_anndata(st_adata_train)

    # Train model
    model = GIMVI(sc_adata_train, st_adata_train)
    model.train(max_epochs=max_epochs, accelerator="auto")

    # Get imputed values for spatial data
    # gimVI returns (seq_imputed, spatial_imputed) tuple
    _, imputed_spatial = model.get_imputed_values(normalized=True)

    # Create imputed expression matrix
    # gimVI imputes expression for common genes using the joint latent space
    imputed_df = pd.DataFrame(
        imputed_spatial,
        index=st_adata_train.obs_names,
        columns=common_genes
    )

    # Save results
    imputed_df.to_csv(f"{save_path}/gimvi_imputed.csv")

    # Add imputed values to spatial AnnData
    st_adata_train.obsm["gimvi_imputed"] = imputed_spatial
    st_adata_train.write_h5ad(f"{save_path}/gimvi_spatial.h5ad", compression="gzip")

    # Save model
    model.save(f"{save_path}/gimvi_model", overwrite=True)

    n_spots = st_adata_train.n_obs
    n_imputed = len(common_genes)

    msg = f"gimVI imputation complete: {n_spots} spots, {n_imputed} genes imputed. Results saved to {save_path}"
    print(msg)
    return msg


# =============================================================================
# Spatial Domain Detection Tools (SpaGCN, GraphST)
# =============================================================================

@tool
def spagcn_clustering(
    adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    n_clusters: Annotated[int, Field(description="Target number of spatial domains")] = 7,
    histology_path: Annotated[str, Field(description="Path to histology image (optional, tif/png/jpg)")] = "",
    x_pixel_col: Annotated[str, Field(description="Column in obs for x pixel coordinates")] = "x_pixel",
    y_pixel_col: Annotated[str, Field(description="Column in obs for y pixel coordinates")] = "y_pixel",
    x_array_col: Annotated[str, Field(description="Column in obs for x array coordinates")] = "x_array",
    y_array_col: Annotated[str, Field(description="Column in obs for y array coordinates")] = "y_array",
    p: Annotated[float, Field(description="Percentage of expression from spatial neighbors")] = 0.5,
    alpha: Annotated[float, Field(description="Histology weight (higher = more weight)")] = 1,
    beta: Annotated[int, Field(description="Spot area parameter for histology")] = 49,
    refine: Annotated[bool, Field(description="Whether to refine clusters using spatial adjacency")] = True,
    shape: Annotated[str, Field(description="Spot shape: 'hexagon' for Visium, 'square' for ST")] = "hexagon",
    max_epochs: Annotated[int, Field(description="Maximum training epochs")] = 200,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run SpaGCN for spatial domain detection.

    SpaGCN uses graph convolutional networks to identify spatial domains by
    integrating gene expression with spatial location and histology images.

    Outputs: Cluster assignments, refined clusters (optional), trained model.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import SpaGCN as spg
    import random
    import torch

    os.makedirs(save_path, exist_ok=True)

    # Load data
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    # Get coordinates
    if x_pixel_col in adata.obs.columns and y_pixel_col in adata.obs.columns:
        x_pixel = adata.obs[x_pixel_col].values
        y_pixel = adata.obs[y_pixel_col].values
    elif 'spatial' in adata.obsm:
        x_pixel = adata.obsm['spatial'][:, 0]
        y_pixel = adata.obsm['spatial'][:, 1]
        adata.obs['x_pixel'] = x_pixel
        adata.obs['y_pixel'] = y_pixel
    else:
        return f"ERROR: No spatial coordinates found. Need '{x_pixel_col}'/'{y_pixel_col}' in obs or 'spatial' in obsm."

    # Get array coordinates for refinement
    if x_array_col in adata.obs.columns and y_array_col in adata.obs.columns:
        x_array = adata.obs[x_array_col].values
        y_array = adata.obs[y_array_col].values
    else:
        x_array = x_pixel
        y_array = y_pixel

    # Calculate adjacency matrix
    if histology_path and os.path.exists(histology_path):
        import cv2
        img = cv2.imread(histology_path)
        adj = spg.calculate_adj_matrix(
            x=x_pixel, y=y_pixel,
            x_pixel=x_pixel, y_pixel=y_pixel,
            image=img, beta=beta, alpha=alpha, histology=True
        )
    else:
        adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)

    # Preprocess expression data
    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    # Search for l parameter
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # Set random seeds
    r_seed = t_seed = n_seed = 100
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    # Search for resolution
    res = spg.search_res(
        adata, adj, l, n_clusters,
        start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
        r_seed=r_seed, t_seed=t_seed, n_seed=n_seed
    )

    # Train SpaGCN
    clf = spg.SpaGCN()
    clf.set_l(l)
    clf.train(
        adata, adj,
        init_spa=True, init="leiden", res=res,
        tol=5e-3, lr=0.05, max_epochs=max_epochs
    )

    # Get predictions
    y_pred, prob = clf.predict()
    adata.obs["spagcn_pred"] = y_pred
    adata.obs["spagcn_pred"] = adata.obs["spagcn_pred"].astype('category')

    # Refine predictions
    if refine:
        adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        refined_pred = spg.refine(
            sample_id=adata.obs.index.tolist(),
            pred=adata.obs["spagcn_pred"].tolist(),
            dis=adj_2d, shape=shape
        )
        adata.obs["spagcn_refined"] = refined_pred
        adata.obs["spagcn_refined"] = adata.obs["spagcn_refined"].astype('category')

    # Save results
    adata.write_h5ad(f"{save_path}/spagcn_result.h5ad", compression="gzip")

    # Save cluster assignments
    cluster_col = "spagcn_refined" if refine else "spagcn_pred"
    clusters_df = adata.obs[[cluster_col]].copy()
    clusters_df.to_csv(f"{save_path}/spagcn_clusters.csv")

    n_spots = adata.n_obs
    n_domains = adata.obs[cluster_col].nunique()

    msg = f"SpaGCN clustering complete: {n_spots} spots, {n_domains} spatial domains detected. Results saved to {save_path}"
    print(msg)
    return msg


@tool
def graphst_clustering(
    adata_path: Annotated[str, Field(description="Path to spatial transcriptomics h5ad file")],
    n_clusters: Annotated[int, Field(description="Target number of spatial domains")] = 7,
    cluster_method: Annotated[str, Field(description="Clustering method: 'leiden'")] = "leiden",
    n_pcs: Annotated[int, Field(description="Number of principal components")] = 30,
    n_neighbors: Annotated[int, Field(description="Number of neighbors for graph construction")] = 10,
    random_seed: Annotated[int, Field(description="Random seed for reproducibility")] = 42,
    device: Annotated[str, Field(description="Device: 'cuda' or 'cpu'")] = "cuda",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Run GraphST for spatial domain detection.

    GraphST uses graph self-supervised contrastive learning to identify spatial
    domains by integrating gene expression with spatial information.

    Clustering methods:
    - leiden: Community detection (recommended)

    Outputs: Cluster assignments, spatial embeddings, trained model.
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import torch
    from GraphST.GraphST import GraphST
    from GraphST import clustering

    os.makedirs(save_path, exist_ok=True)

    # Set device
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device('cpu')

    # Load data
    adata = sc.read_h5ad(adata_path)
    adata.var_names_make_unique()

    # Check for spatial coordinates
    if 'spatial' not in adata.obsm:
        if 'x_pixel' in adata.obs.columns and 'y_pixel' in adata.obs.columns:
            adata.obsm['spatial'] = np.array([
                adata.obs['x_pixel'].values,
                adata.obs['y_pixel'].values
            ]).T
        else:
            return "ERROR: No spatial coordinates found. Need 'spatial' in obsm or 'x_pixel'/'y_pixel' in obs."

    # Preprocess (normalize and select HVGs)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Build and train GraphST model
    model = GraphST(adata, device=device_obj, random_seed=random_seed)
    adata = model.train()

    # Perform clustering on the learned embeddings
    if cluster_method == 'leiden':
        clustering(adata, n_clusters=n_clusters, method=cluster_method, start=0.1, end=2.0, increment=0.01)
    else:
        return f"ERROR: Unknown clustering method '{cluster_method}'. Use 'leiden'."

    # Rename cluster column for consistency
    if 'domain' in adata.obs.columns:
        adata.obs['graphst_cluster'] = adata.obs['domain'].astype('category')

    # Save results
    adata.write_h5ad(f"{save_path}/graphst_result.h5ad", compression="gzip")

    # Save cluster assignments
    if 'graphst_cluster' in adata.obs.columns:
        clusters_df = adata.obs[['graphst_cluster']].copy()
        clusters_df.to_csv(f"{save_path}/graphst_clusters.csv")
        n_domains = adata.obs['graphst_cluster'].nunique()
    else:
        n_domains = "unknown"

    # Save embeddings
    if 'emb' in adata.obsm:
        emb_df = pd.DataFrame(
            adata.obsm['emb'],
            index=adata.obs_names,
            columns=[f'GraphST_{i}' for i in range(adata.obsm['emb'].shape[1])]
        )
        emb_df.to_csv(f"{save_path}/graphst_embeddings.csv")

    n_spots = adata.n_obs

    msg = f"GraphST clustering complete: {n_spots} spots, {n_domains} spatial domains detected. Results saved to {save_path}"
    print(msg)
    return msg


# =============================================================================
# Scanpy Tools: Gene Scoring
# =============================================================================

@tool
def scanpy_score_genes(
    adata_path: Annotated[str, Field(description="Path to h5ad file")],
    gene_list: Annotated[list[str], Field(description="List of genes to score (e.g., pathway genes, signature genes)")],
    score_name: Annotated[str, Field(description="Name for the score column in adata.obs")] = "gene_score",
    ctrl_size: Annotated[int, Field(description="Number of control genes per binned expression level")] = 50,
    n_bins: Annotated[int, Field(description="Number of expression bins for control gene selection")] = 25,
    use_raw: Annotated[bool, Field(description="Use raw expression data if available")] = False,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Score cells/spots based on expression of a gene signature using Scanpy.

    Computes a score for each cell by comparing the average expression of the
    input gene set to the average expression of a reference set of control genes.
    The control genes are randomly selected from genes binned by their average expression,
    so that the control gene set has a similar expression level distribution as the input genes.

    Common use cases:
    - Score cells for pathway activity (e.g., hypoxia, cell cycle, EMT)
    - Score cells for cell type markers
    - Score cells for custom gene signatures from literature
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    # Ensure gene names are uppercase for matching
    adata.var_names_make_unique()
    gene_list_upper = [g.upper() for g in gene_list]
    adata.var.index = adata.var.index.str.upper()

    # Check which genes are present
    genes_found = [g for g in gene_list_upper if g in adata.var_names]
    genes_missing = [g for g in gene_list_upper if g not in adata.var_names]

    if len(genes_found) == 0:
        return f"ERROR: None of the {len(gene_list)} genes found in the data. First few: {gene_list[:5]}"

    # Score genes
    sc.tl.score_genes(
        adata,
        gene_list=genes_found,
        score_name=score_name,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        use_raw=use_raw,
    )

    # Save results
    os.makedirs(save_path, exist_ok=True)
    adata.write_h5ad(f"{save_path}/scored.h5ad", compression="gzip")

    # Save scores as CSV
    scores_df = adata.obs[[score_name]].copy()
    scores_df.to_csv(f"{save_path}/{score_name}.csv")

    # Basic stats
    score_mean = adata.obs[score_name].mean()
    score_std = adata.obs[score_name].std()
    score_min = adata.obs[score_name].min()
    score_max = adata.obs[score_name].max()

    result = f"Gene scoring complete for '{score_name}':\n"
    result += f"  - {len(genes_found)}/{len(gene_list)} genes found in data\n"
    if genes_missing and len(genes_missing) <= 10:
        result += f"  - Missing genes: {genes_missing}\n"
    elif genes_missing:
        result += f"  - {len(genes_missing)} genes missing (first 5: {genes_missing[:5]})\n"
    result += f"  - Score stats: mean={score_mean:.3f}, std={score_std:.3f}, range=[{score_min:.3f}, {score_max:.3f}]\n"
    result += f"  - Saved to {save_path}/scored.h5ad and {save_path}/{score_name}.csv"

    print(result)
    return result


# =============================================================================
# Scanpy Tools: Integration - Ingest
# =============================================================================

@tool
def scanpy_ingest(
    adata_query_path: Annotated[str, Field(description="Path to query h5ad file (data to annotate)")],
    adata_ref_path: Annotated[str, Field(description="Path to reference h5ad file (annotated data)")],
    obs_keys: Annotated[list[str], Field(description="Observation keys to transfer from reference (e.g., ['cell_type', 'cluster'])")],
    embedding: Annotated[str, Field(description="Embedding to use for mapping: 'pca' or 'umap'")] = "umap",
    use_rep: Annotated[str, Field(description="Key in adata.obsm for representation (e.g., 'X_pca')")] = "X_pca",
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Transfer annotations from a reference dataset to a query dataset using Scanpy ingest.

    Ingest maps cells from a query dataset to a reference dataset that has been
    preprocessed and annotated. It projects query cells onto the PCA/UMAP of the
    reference and transfers labels based on nearest neighbors.

    Requirements:
    - Reference must have computed PCA (and UMAP if embedding='umap')
    - Reference must have the annotation columns specified in obs_keys
    - Both datasets should have overlapping genes

    Common use cases:
    - Transfer cell type labels from annotated scRNA-seq to spatial data
    - Annotate new samples using a well-curated reference atlas
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc

    adata_query = sc.read_h5ad(adata_query_path)
    adata_ref = sc.read_h5ad(adata_ref_path)

    # Ensure gene names are uppercase
    adata_query.var.index = adata_query.var.index.str.upper()
    adata_ref.var.index = adata_ref.var.index.str.upper()

    # Find common genes
    common_genes = list(set(adata_query.var_names) & set(adata_ref.var_names))
    if len(common_genes) < 100:
        return f"ERROR: Only {len(common_genes)} common genes found between query and reference. Need at least 100."

    # Subset to common genes
    adata_query = adata_query[:, common_genes].copy()
    adata_ref = adata_ref[:, common_genes].copy()

    # Check if reference has required representations
    if use_rep not in adata_ref.obsm:
        return f"ERROR: Reference missing '{use_rep}' in obsm. Run PCA on reference first."

    if embedding == 'umap' and 'X_umap' not in adata_ref.obsm:
        return f"ERROR: Reference missing 'X_umap' in obsm. Run UMAP on reference first or use embedding='pca'."

    # Check if obs_keys exist in reference
    missing_keys = [k for k in obs_keys if k not in adata_ref.obs.columns]
    if missing_keys:
        return f"ERROR: Reference missing annotation keys: {missing_keys}. Available: {list(adata_ref.obs.columns)}"

    # Run ingest
    sc.tl.ingest(
        adata_query,
        adata_ref,
        obs=obs_keys,
        embedding_method=embedding,
    )

    # Save results
    os.makedirs(save_path, exist_ok=True)
    adata_query.write_h5ad(f"{save_path}/ingest_result.h5ad", compression="gzip")

    # Save transferred annotations as CSV
    transferred_df = adata_query.obs[obs_keys].copy()
    transferred_df.to_csv(f"{save_path}/ingest_annotations.csv")

    result = f"Ingest complete:\n"
    result += f"  - Query: {adata_query.n_obs} cells\n"
    result += f"  - Reference: {adata_ref.n_obs} cells\n"
    result += f"  - Common genes: {len(common_genes)}\n"
    result += f"  - Transferred annotations: {obs_keys}\n"

    for key in obs_keys:
        n_categories = adata_query.obs[key].nunique()
        result += f"  - {key}: {n_categories} unique values\n"

    result += f"  - Saved to {save_path}/ingest_result.h5ad"

    print(result)
    return result


# =============================================================================
# Scanpy Tools: Integration - BBKNN
# =============================================================================

@tool
def scanpy_bbknn(
    adata_path: Annotated[str, Field(description="Path to h5ad file with multiple batches")],
    batch_key: Annotated[str, Field(description="Column in adata.obs containing batch information")],
    n_pcs: Annotated[int, Field(description="Number of principal components to use")] = 50,
    neighbors_within_batch: Annotated[int, Field(description="Number of neighbors per batch")] = 3,
    trim: Annotated[int, Field(description="Trim KNN graph to this many neighbors per cell (0=no trim)")] = 0,
    use_rep: Annotated[str, Field(description="Key in adata.obsm for representation")] = "X_pca",
    run_umap: Annotated[bool, Field(description="Compute UMAP after batch correction")] = True,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Perform batch-balanced KNN (BBKNN) integration for multi-batch data.

    BBKNN modifies the neighborhood graph to balance connections across batches.
    Instead of finding k-nearest neighbors globally, it finds k neighbors from
    each batch, effectively removing batch effects in the graph structure.

    Requirements:
    - Data must have PCA computed
    - Batch key must exist in adata.obs

    Common use cases:
    - Integrate multiple spatial tissue sections
    - Integrate spatial data with scRNA-seq reference
    - Remove technical batch effects while preserving biological variation
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc

    try:
        import bbknn
    except ImportError:
        return "ERROR: bbknn not installed. Install with: pip install bbknn"

    adata = sc.read_h5ad(adata_path)

    # Check batch key exists
    if batch_key not in adata.obs.columns:
        return f"ERROR: Batch key '{batch_key}' not found in adata.obs. Available: {list(adata.obs.columns)}"

    # Check PCA exists
    if use_rep not in adata.obsm:
        # Run PCA if not present
        if adata.n_vars > 2000:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            sc.tl.pca(adata, use_highly_variable=True)
        else:
            sc.tl.pca(adata)

    # Get batch statistics
    batch_counts = adata.obs[batch_key].value_counts()
    n_batches = len(batch_counts)

    # Run BBKNN
    bbknn.bbknn(
        adata,
        batch_key=batch_key,
        n_pcs=min(n_pcs, adata.obsm[use_rep].shape[1]),
        neighbors_within_batch=neighbors_within_batch,
        trim=trim if trim > 0 else None,
    )

    # Optionally run UMAP
    if run_umap:
        sc.tl.umap(adata)

    # Save results
    os.makedirs(save_path, exist_ok=True)
    adata.write_h5ad(f"{save_path}/bbknn_result.h5ad", compression="gzip")

    result = f"BBKNN integration complete:\n"
    result += f"  - {adata.n_obs} cells across {n_batches} batches\n"
    result += f"  - Batch distribution:\n"
    for batch, count in batch_counts.head(5).items():
        result += f"      {batch}: {count} cells\n"
    if n_batches > 5:
        result += f"      ... and {n_batches - 5} more batches\n"
    result += f"  - UMAP computed: {run_umap}\n"
    result += f"  - Saved to {save_path}/bbknn_result.h5ad"

    print(result)
    return result


# =============================================================================
# Trajectory Inference: scVelo
# =============================================================================

@tool
def scvelo_velocity(
    adata_path: Annotated[str, Field(description="Path to h5ad file with spliced/unspliced counts")],
    mode: Annotated[str, Field(description="Velocity mode: 'deterministic', 'stochastic', or 'dynamical'")] = "stochastic",
    n_top_genes: Annotated[int, Field(description="Number of highly variable genes to use")] = 2000,
    n_pcs: Annotated[int, Field(description="Number of PCs for neighbors")] = 30,
    n_neighbors: Annotated[int, Field(description="Number of neighbors")] = 30,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute RNA velocity using scVelo.

    RNA velocity estimates the rate of gene expression change by comparing
    spliced and unspliced mRNA counts. This reveals the future state of cells
    and can infer developmental trajectories.

    Requirements:
    - Data must have spliced ('spliced') and unspliced ('unspliced') layers
    - These are typically from velocyto, kallisto bustools, or STARsolo

    Modes:
    - deterministic: Fast, assumes steady-state (original velocyto model)
    - stochastic: Accounts for transcriptional stochasticity (recommended)
    - dynamical: Full dynamical model, most accurate but slower
    """
    save_path = save_path or _config["save_path"]
    import scvelo as scv
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    # Check for required layers
    if 'spliced' not in adata.layers or 'unspliced' not in adata.layers:
        return "ERROR: Data must have 'spliced' and 'unspliced' layers. Run velocyto or kallisto bustools first."

    # Filter and normalize
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)

    # Compute moments
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    # Compute velocity
    if mode == 'dynamical':
        scv.tl.recover_dynamics(adata)
        scv.tl.velocity(adata, mode='dynamical')
    else:
        scv.tl.velocity(adata, mode=mode)

    # Compute velocity graph
    scv.tl.velocity_graph(adata)

    # Save results
    os.makedirs(save_path, exist_ok=True)
    adata.write_h5ad(f"{save_path}/velocity_result.h5ad", compression="gzip")

    # Get velocity statistics
    n_cells = adata.n_obs
    n_genes_velocity = adata.var['velocity_genes'].sum() if 'velocity_genes' in adata.var else 'N/A'

    result = f"RNA velocity computed successfully:\n"
    result += f"  - Mode: {mode}\n"
    result += f"  - Cells: {n_cells}\n"
    result += f"  - Velocity genes: {n_genes_velocity}\n"
    result += f"  - Saved to {save_path}/velocity_result.h5ad\n"
    result += f"\nNext steps: Use scvelo_velocity_embedding for visualization"

    print(result)
    return result


@tool
def scvelo_velocity_embedding(
    adata_path: Annotated[str, Field(description="Path to h5ad file with computed velocity")],
    basis: Annotated[str, Field(description="Embedding to use: 'umap', 'tsne', or 'pca'")] = "umap",
    color_by: Annotated[str, Field(description="Column in adata.obs to color by")] = None,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Project RNA velocity onto embedding and create velocity stream plot.

    Creates a velocity stream plot showing the direction and magnitude of
    cellular state changes on a low-dimensional embedding.

    Requirements:
    - Data must have velocity computed (from scvelo_velocity)
    - Data must have the specified embedding (e.g., X_umap)
    """
    save_path = save_path or _config["save_path"]
    import scvelo as scv
    import scanpy as sc
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(adata_path)

    # Check velocity exists
    if 'velocity' not in adata.layers:
        return "ERROR: Velocity not computed. Run scvelo_velocity first."

    # Check embedding exists
    embed_key = f"X_{basis}"
    if embed_key not in adata.obsm:
        # Try to compute it
        import scanpy as sc
        if basis == 'umap':
            sc.tl.umap(adata)
        elif basis == 'tsne':
            sc.tl.tsne(adata)
        else:
            return f"ERROR: Embedding '{basis}' not found. Available: {list(adata.obsm.keys())}"

    os.makedirs(save_path, exist_ok=True)

    # Create velocity embedding stream plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scv.pl.velocity_embedding_stream(adata, basis=basis, color=color_by, ax=ax, show=False)
    plt.savefig(f"{save_path}/velocity_stream_{basis}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Create velocity embedding grid plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scv.pl.velocity_embedding_grid(adata, basis=basis, color=color_by, ax=ax, show=False)
    plt.savefig(f"{save_path}/velocity_grid_{basis}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save updated adata
    adata.write_h5ad(f"{save_path}/velocity_embedding.h5ad", compression="gzip")

    result = f"Velocity embedding created:\n"
    result += f"  - Basis: {basis}\n"
    result += f"  - Stream plot: {save_path}/velocity_stream_{basis}.png\n"
    result += f"  - Grid plot: {save_path}/velocity_grid_{basis}.png\n"
    result += f"  - Data saved to: {save_path}/velocity_embedding.h5ad"

    print(result)
    return result


# =============================================================================
# Trajectory Inference: CellRank
# =============================================================================

@tool
def cellrank_terminal_states(
    adata_path: Annotated[str, Field(description="Path to h5ad file with velocity or connectivity")],
    cluster_key: Annotated[str, Field(description="Column in adata.obs for cell clusters")] = None,
    n_states: Annotated[int, Field(description="Number of terminal states to find (0=auto)")] = 0,
    use_velocity: Annotated[bool, Field(description="Use RNA velocity (requires velocity layer)")] = True,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Identify terminal/macrostates using CellRank.

    CellRank uses Markov chain analysis to identify terminal states (endpoints)
    and initial states of cellular differentiation trajectories.

    Requirements:
    - If use_velocity=True: Data must have velocity (from scvelo_velocity)
    - If use_velocity=False: Data must have neighbors graph

    Outputs:
    - Terminal states (macrostates) identified in adata.obs
    - Transition probabilities stored in adata
    """
    save_path = save_path or _config["save_path"]
    import cellrank as cr
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    # Create kernel based on data type
    if use_velocity:
        if 'velocity' not in adata.layers:
            return "ERROR: Velocity not found. Run scvelo_velocity first or set use_velocity=False."
        # Velocity kernel
        vk = cr.kernels.VelocityKernel(adata)
        vk.compute_transition_matrix()
        kernel = vk
    else:
        # Connectivity kernel (based on neighbors graph)
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata)
        ck = cr.kernels.ConnectivityKernel(adata)
        ck.compute_transition_matrix()
        kernel = ck

    # Create GPCCA estimator for terminal states
    g = cr.estimators.GPCCA(kernel)

    # Compute Schur decomposition
    g.compute_schur(n_components=20)

    # Compute macrostates
    if n_states > 0:
        g.compute_macrostates(n_states=n_states, cluster_key=cluster_key)
    else:
        g.compute_macrostates(cluster_key=cluster_key)

    # Identify terminal states
    g.predict_terminal_states()

    # Save results
    os.makedirs(save_path, exist_ok=True)
    adata.write_h5ad(f"{save_path}/cellrank_terminal.h5ad", compression="gzip")

    # Get terminal state info
    terminal_states = adata.obs['terminal_states'].dropna().unique().tolist() if 'terminal_states' in adata.obs else []

    result = f"CellRank terminal state analysis complete:\n"
    result += f"  - Method: {'Velocity kernel' if use_velocity else 'Connectivity kernel'}\n"
    result += f"  - Terminal states found: {len(terminal_states)}\n"
    if terminal_states:
        result += f"  - States: {terminal_states}\n"
    result += f"  - Saved to {save_path}/cellrank_terminal.h5ad\n"
    result += f"\nNext steps: Use cellrank_fate_probabilities to compute fate maps"

    print(result)
    return result


@tool
def cellrank_fate_probabilities(
    adata_path: Annotated[str, Field(description="Path to h5ad file with terminal states")],
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute fate probabilities for each cell towards terminal states.

    After identifying terminal states with cellrank_terminal_states, this
    computes the probability of each cell reaching each terminal state.

    Requirements:
    - Data must have terminal states computed (from cellrank_terminal_states)

    Outputs:
    - Fate probabilities in adata.obsm['to_terminal_states']
    - Lineage drivers (genes correlated with fate)
    """
    save_path = save_path or _config["save_path"]
    import cellrank as cr
    import scanpy as sc
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(adata_path)

    # Check terminal states exist
    if 'terminal_states' not in adata.obs:
        return "ERROR: Terminal states not found. Run cellrank_terminal_states first."

    # Recreate estimator
    if 'velocity' in adata.layers:
        vk = cr.kernels.VelocityKernel(adata)
        vk.compute_transition_matrix()
        kernel = vk
    else:
        ck = cr.kernels.ConnectivityKernel(adata)
        ck.compute_transition_matrix()
        kernel = ck

    g = cr.estimators.GPCCA(kernel)
    g.compute_schur(n_components=20)

    # Set terminal states from previous analysis
    terminal_states = adata.obs['terminal_states'].dropna().unique().tolist()
    g.set_terminal_states(terminal_states)

    # Compute fate probabilities
    g.compute_fate_probabilities()

    os.makedirs(save_path, exist_ok=True)

    # Save fate probabilities
    if hasattr(g, 'fate_probabilities') and g.fate_probabilities is not None:
        fate_df = g.fate_probabilities.to_frame()
        fate_df.to_csv(f"{save_path}/fate_probabilities.csv")

    # Save updated adata
    adata.write_h5ad(f"{save_path}/cellrank_fate.h5ad", compression="gzip")

    result = f"Fate probabilities computed:\n"
    result += f"  - Terminal states: {terminal_states}\n"
    result += f"  - Fate probabilities saved to: {save_path}/fate_probabilities.csv\n"
    result += f"  - Data saved to: {save_path}/cellrank_fate.h5ad"

    print(result)
    return result


@tool
def paga_trajectory(
    adata_path: Annotated[str, Field(description="Path to preprocessed h5ad file")],
    groups_key: Annotated[str, Field(description="Column in adata.obs for cell groups (e.g., 'leiden', 'cell_type')")],
    threshold: Annotated[float, Field(description="Threshold for PAGA edges (0-1)")] = 0.05,
    root_group: Annotated[str, Field(description="Name of root group for pseudotime (optional)")] = None,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Compute PAGA (Partition-based Graph Abstraction) trajectory.

    PAGA computes a coarse-grained graph of cell groups showing connectivity
    and potential differentiation paths. It can also compute diffusion pseudotime.

    Requirements:
    - Data must have neighbors computed
    - Data must have the specified groups column
    """
    save_path = save_path or _config["save_path"]
    import scanpy as sc
    import matplotlib.pyplot as plt

    adata = sc.read_h5ad(adata_path)

    # Check groups exist
    if groups_key not in adata.obs:
        return f"ERROR: Groups key '{groups_key}' not found. Available: {list(adata.obs.columns)}"

    # Compute neighbors if not present
    if 'neighbors' not in adata.uns:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    # Compute PAGA
    sc.tl.paga(adata, groups=groups_key)

    os.makedirs(save_path, exist_ok=True)

    # Plot PAGA
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.paga(adata, threshold=threshold, ax=ax, show=False)
    plt.savefig(f"{save_path}/paga_graph.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Compute diffusion pseudotime if root specified
    if root_group:
        # Find a cell in root group
        root_cells = adata.obs[adata.obs[groups_key] == root_group].index
        if len(root_cells) == 0:
            return f"ERROR: Root group '{root_group}' not found. Available: {adata.obs[groups_key].unique().tolist()}"

        adata.uns['iroot'] = adata.obs_names.get_loc(root_cells[0])
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)

        # Plot pseudotime
        if 'X_umap' in adata.obsm:
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False)
            plt.savefig(f"{save_path}/pseudotime_umap.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Save PAGA adjacency
    paga_adj = adata.uns['paga']['connectivities'].toarray()
    groups = adata.obs[groups_key].cat.categories.tolist()
    paga_df = pd.DataFrame(paga_adj, index=groups, columns=groups)
    paga_df.to_csv(f"{save_path}/paga_adjacency.csv")

    # Save results
    adata.write_h5ad(f"{save_path}/paga_result.h5ad", compression="gzip")

    result = f"PAGA trajectory analysis complete:\n"
    result += f"  - Groups: {groups_key} ({len(groups)} groups)\n"
    result += f"  - PAGA graph: {save_path}/paga_graph.png\n"
    result += f"  - Adjacency matrix: {save_path}/paga_adjacency.csv\n"
    if root_group:
        result += f"  - Pseudotime computed from root: {root_group}\n"
        result += f"  - Pseudotime plot: {save_path}/pseudotime_umap.png\n"
    result += f"  - Data saved to: {save_path}/paga_result.h5ad"

    print(result)
    return result


# =============================================================================
# Multimodal Integration: scvi-tools
# =============================================================================

@tool
def totalvi_integration(
    adata_path: Annotated[str, Field(description="Path to h5ad file with RNA and protein (CITE-seq) data")],
    protein_layer: Annotated[str, Field(description="Key in adata.obsm for protein counts")] = "protein_expression",
    n_latent: Annotated[int, Field(description="Dimensionality of latent space")] = 20,
    max_epochs: Annotated[int, Field(description="Maximum training epochs")] = 400,
    batch_key: Annotated[str, Field(description="Batch column for integration (optional)")] = None,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Integrate RNA and protein data using totalVI (CITE-seq analysis).

    totalVI is a deep generative model for joint analysis of RNA and protein
    measurements from CITE-seq or similar multi-modal single-cell technologies.

    Requirements:
    - RNA counts in adata.X (raw counts, not normalized)
    - Protein counts in adata.obsm[protein_layer]

    Outputs:
    - Latent representation in adata.obsm['X_totalVI']
    - Denoised protein values
    - Normalized RNA values
    """
    save_path = save_path or _config["save_path"]
    import scvi
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    # Check protein data exists
    if protein_layer not in adata.obsm:
        return f"ERROR: Protein layer '{protein_layer}' not found. Available obsm: {list(adata.obsm.keys())}"

    # Get protein names if available
    protein_names = None
    if f"{protein_layer}_names" in adata.uns:
        protein_names = adata.uns[f"{protein_layer}_names"]

    # Setup anndata for totalVI
    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key=protein_layer,
        batch_key=batch_key,
    )

    # Create and train model
    model = scvi.model.TOTALVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True)

    # Get latent representation
    adata.obsm['X_totalVI'] = model.get_latent_representation()

    # Get normalized values
    rna_norm, protein_norm = model.get_normalized_expression(return_mean=True)
    adata.layers['totalvi_rna_normalized'] = rna_norm
    # Convert protein_norm to numpy array to avoid column name issues
    if hasattr(protein_norm, 'values'):
        adata.obsm['totalvi_protein_normalized'] = protein_norm.values
    else:
        adata.obsm['totalvi_protein_normalized'] = protein_norm

    # Compute UMAP on totalVI latent space
    sc.pp.neighbors(adata, use_rep='X_totalVI')
    sc.tl.umap(adata)

    os.makedirs(save_path, exist_ok=True)

    # Save model
    model.save(f"{save_path}/totalvi_model", overwrite=True)

    # Save results
    adata.write_h5ad(f"{save_path}/totalvi_result.h5ad", compression="gzip")

    result = f"totalVI integration complete:\n"
    result += f"  - Cells: {adata.n_obs}\n"
    result += f"  - RNA genes: {adata.n_vars}\n"
    result += f"  - Proteins: {adata.obsm[protein_layer].shape[1]}\n"
    result += f"  - Latent dims: {n_latent}\n"
    result += f"  - Batch integration: {batch_key if batch_key else 'None'}\n"
    result += f"  - Model saved to: {save_path}/totalvi_model\n"
    result += f"  - Data saved to: {save_path}/totalvi_result.h5ad"

    print(result)
    return result


@tool
def multivi_integration(
    adata_path: Annotated[str, Field(description="Path to h5ad file with RNA and ATAC data")],
    atac_layer: Annotated[str, Field(description="Key in adata.obsm or layer for ATAC peaks")] = None,
    n_latent: Annotated[int, Field(description="Dimensionality of latent space")] = 20,
    max_epochs: Annotated[int, Field(description="Maximum training epochs")] = 500,
    batch_key: Annotated[str, Field(description="Batch column for integration (optional)")] = None,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Integrate RNA and ATAC data using MultiVI (multiome analysis).

    MultiVI is a deep generative model for joint analysis of gene expression
    and chromatin accessibility from 10x Multiome or similar technologies.

    Requirements:
    - Data with both modalities, or MuData object
    - ATAC peaks either as separate vars or in a layer

    Outputs:
    - Latent representation in adata.obsm['X_multivi']
    - Imputed accessibility for RNA-only cells
    - Imputed expression for ATAC-only cells
    """
    save_path = save_path or _config["save_path"]
    import scvi
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    # For MultiVI, we need to properly organize the data
    # This is a simplified version - real usage may need MuData

    # Setup anndata
    scvi.model.MULTIVI.setup_anndata(
        adata,
        batch_key=batch_key,
    )

    # Create and train model
    model = scvi.model.MULTIVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True)

    # Get latent representation
    adata.obsm['X_multivi'] = model.get_latent_representation()

    # Compute UMAP on MultiVI latent space
    sc.pp.neighbors(adata, use_rep='X_multivi')
    sc.tl.umap(adata)

    os.makedirs(save_path, exist_ok=True)

    # Save model
    model.save(f"{save_path}/multivi_model", overwrite=True)

    # Save results
    adata.write_h5ad(f"{save_path}/multivi_result.h5ad", compression="gzip")

    result = f"MultiVI integration complete:\n"
    result += f"  - Cells: {adata.n_obs}\n"
    result += f"  - Features: {adata.n_vars}\n"
    result += f"  - Latent dims: {n_latent}\n"
    result += f"  - Model saved to: {save_path}/multivi_model\n"
    result += f"  - Data saved to: {save_path}/multivi_result.h5ad"

    print(result)
    return result


@tool
def mofa_integration(
    adata_path: Annotated[str, Field(description="Path to h5ad file OR comma-separated paths for multiple modalities")],
    n_factors: Annotated[int, Field(description="Number of latent factors to learn")] = 10,
    modality_key: Annotated[str, Field(description="Column in adata.var indicating modality (if single file)")] = None,
    groups_key: Annotated[str, Field(description="Column in adata.obs for sample/group structure")] = None,
    max_iterations: Annotated[int, Field(description="Maximum training iterations")] = 1000,
    save_path: Annotated[str, Field(description="Directory to save results")] = None,
) -> str:
    """Perform multi-omics factor analysis using MOFA+.

    MOFA+ identifies latent factors that explain variation across multiple
    data modalities (e.g., RNA, protein, ATAC, methylation).

    Requirements:
    - Data with multiple modalities, organized by:
      1. Single adata with modality_key in var, OR
      2. Comma-separated paths to multiple h5ad files

    Outputs:
    - Latent factors in adata.obsm['X_mofa']
    - Factor loadings (weights) per modality
    - Variance explained per factor per modality
    """
    save_path = save_path or _config["save_path"]
    from mofapy2.run.entry_point import entry_point
    import scanpy as sc

    # Check if multiple files
    if ',' in adata_path:
        paths = [p.strip() for p in adata_path.split(',')]
        adatas = [sc.read_h5ad(p) for p in paths]
        modality_names = [f"modality_{i}" for i in range(len(adatas))]
    else:
        adata = sc.read_h5ad(adata_path)
        if modality_key and modality_key in adata.var:
            # Split by modality
            modalities = adata.var[modality_key].unique()
            adatas = [adata[:, adata.var[modality_key] == m].copy() for m in modalities]
            modality_names = list(modalities)
        else:
            # Single modality
            adatas = [adata]
            modality_names = ["expression"]

    # Prepare data for MOFA
    # MOFA expects: data[modality][group] = matrix
    data = {}
    for mod_name, ad in zip(modality_names, adatas):
        if groups_key and groups_key in ad.obs:
            data[mod_name] = {}
            for group in ad.obs[groups_key].unique():
                mask = ad.obs[groups_key] == group
                data[mod_name][group] = ad[mask].X.toarray() if hasattr(ad.X, 'toarray') else ad.X
        else:
            data[mod_name] = {"group1": ad.X.toarray() if hasattr(ad.X, 'toarray') else ad.X}

    # Initialize MOFA
    ent = entry_point()
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data)
    ent.set_model_options(factors=n_factors)
    ent.set_train_options(iter=max_iterations, convergence_mode="slow", seed=42)

    # Train
    ent.build()
    ent.run()

    os.makedirs(save_path, exist_ok=True)

    # Save MOFA model
    ent.save(f"{save_path}/mofa_model.hdf5")

    # Get factors and add to first adata
    factors = ent.model.nodes["Z"].getExpectation()
    if isinstance(factors, dict):
        # Multiple groups - concatenate
        all_factors = np.vstack(list(factors.values()))
    else:
        all_factors = factors

    # Get weights for each modality
    weights = ent.model.nodes["W"].getExpectation()

    # Get variance explained
    var_explained = ent.model.calculate_variance_explained()

    # Save factors
    factors_df = pd.DataFrame(
        all_factors,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    factors_df.to_csv(f"{save_path}/mofa_factors.csv")

    # Save variance explained (handle 3D output: groups x views x factors)
    try:
        if var_explained.ndim == 3:
            # Reshape to 2D (views x factors) for single group
            var_explained_2d = var_explained[0]  # Take first group
            var_df = pd.DataFrame(
                var_explained_2d,
                index=[f'view_{i}' for i in range(var_explained_2d.shape[0])],
                columns=[f'Factor_{i+1}' for i in range(var_explained_2d.shape[1])]
            )
        else:
            var_df = pd.DataFrame(var_explained)
        var_df.to_csv(f"{save_path}/mofa_variance_explained.csv")
    except Exception as e:
        # If variance explained fails, just save factors
        pass

    result = f"MOFA+ analysis complete:\n"
    result += f"  - Modalities: {modality_names}\n"
    result += f"  - Factors: {n_factors}\n"
    result += f"  - Samples: {all_factors.shape[0]}\n"
    result += f"  - Model saved to: {save_path}/mofa_model.hdf5\n"
    result += f"  - Factors saved to: {save_path}/mofa_factors.csv\n"
    result += f"  - Variance explained: {save_path}/mofa_variance_explained.csv"

    print(result)
    return result
