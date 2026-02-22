"""
LLM-Powered Interpretation Tools

Tools that use language models for biological reasoning, annotation,
scientific report generation, figure interpretation, and conclusion verification.

All tools are standalone functions following Biomni pattern.
"""

# Lightweight imports - keep at module level
import os
import re
import json
import pickle
import base64
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

# Module-level config (set via configure_interpretation_tools)
_config = {
    "save_path": "./experiments",
}

def configure_interpretation_tools(save_path: str = "./experiments"):
    """Configure paths for interpretation tools. Call this before using the tools."""
    _config["save_path"] = save_path

# Default paths for data files
DEFAULT_DATA_PATH = "./data"

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
# - matplotlib.pyplot (0.5-1s import time)
# - tqdm (optional progress bars)


# =============================================================================
# Tool 1: Cell Type Annotator (Two-Level Hierarchical Batch Approach)
# =============================================================================

def _load_cell_type_ontology(ontology_path=None):
    """Load hierarchical cell type ontology."""
    # Try provided path first
    if ontology_path and exists(ontology_path):
        with open(ontology_path, "r") as f:
            return json.load(f)

    # Use data directory
    default_path = f"{DEFAULT_DATA_PATH}/cell_type_ontology.json"
    if exists(default_path):
        with open(default_path, "r") as f:
            return json.load(f)

    return None


def _detect_tissue_type(data_info):
    """Detect tissue type from data_info string."""
    data_info_lower = data_info.lower()
    tissue_keywords = {
        "heart": ["heart", "cardiac", "cardiomyocyte", "myocardium", "ventricle", "atrium"],
        "brain": ["brain", "cortex", "hippocampus", "cerebral", "neural", "neuron"],
        "lung": ["lung", "pulmonary", "respiratory", "alveolar", "bronchial"],
        "liver": ["liver", "hepatic", "hepatocyte"],
        "kidney": ["kidney", "renal", "nephron"],
        "intestine": ["intestine", "gut", "colon", "ileum", "jejunum", "duodenum", "gastric"]
    }

    for tissue, keywords in tissue_keywords.items():
        if any(kw in data_info_lower for kw in keywords):
            return tissue

    return None  # Unknown tissue


def _get_tissue_cell_types(ontology, tissue_type):
    """Get relevant cell types for a specific tissue."""
    if not ontology or not tissue_type:
        return list(ontology["categories"].keys()) if ontology else []

    tissue_profile = ontology.get("tissue_profiles", {}).get(tissue_type, {})
    expected = tissue_profile.get("expected_types", [])
    rare = tissue_profile.get("rare_types", [])

    return expected + rare


def _build_cluster_info(adata, cluster_key, cluster_id, markers_dict, composition_df):
    """Build info string for a cluster."""
    info = f"Cluster {cluster_id}: "

    # Add marker genes
    cluster_idx = int(cluster_id)
    if cluster_idx in markers_dict:
        top_markers = markers_dict[cluster_idx][:10]
        info += f"Markers: {', '.join(top_markers)}. "

    # Add transferred cell type composition
    if cluster_id in composition_df.index:
        comp = composition_df.loc[cluster_id]
        top_types = comp[comp > 0.05].sort_values(ascending=False)
        if len(top_types) > 0:
            comp_str = ", ".join([f"{ct} ({pct:.0%})" for ct, pct in top_types.items()])
            info += f"Transferred: {comp_str}."

    return info


def _annotate_level1_batch(cluster_infos, ontology, tissue_type, data_info, llm):
    """Level 1: Assign broad cell type categories to all clusters in one LLM call."""
    # Get valid categories for this tissue
    categories = ontology["categories"]
    tissue_types = _get_tissue_cell_types(ontology, tissue_type)

    # Build category descriptions
    category_info = []
    for cat_name in tissue_types:
        if cat_name in categories:
            cat = categories[cat_name]
            markers = ", ".join(cat.get("markers", [])[:5])
            category_info.append(f"- {cat_name}: {cat.get('description', '')}. Key markers: {markers}")

    category_text = "\n".join(category_info)
    cluster_text = "\n".join([f"- {info}" for info in cluster_infos.values()])

    messages = [
        {"role": "system", "content": f"""You are an expert in {data_info} cell type annotation.
Your task is to assign BROAD cell type categories to clusters based on marker genes and transferred labels.

Available cell type categories for {tissue_type or 'this tissue'}:
{category_text}

Rules:
- Assign exactly ONE category per cluster
- Use the category names exactly as listed above
- Base decisions on marker genes AND transferred cell type composition"""
        },
        {"role": "user", "content": f"""Assign cell type categories to these clusters:

{cluster_text}

OUTPUT FORMAT (one line per cluster, no extra text):
0: [category name]
1: [category name]
..."""}
    ]

    response = llm.invoke(messages).content

    # Parse response
    annotations = {}
    valid_categories = set(categories.keys())

    for line in response.strip().split('\n'):
        match = re.match(r'^(\d+)\s*:\s*(.+)$', line.strip())
        if match:
            cluster_id = match.group(1)
            category = match.group(2).strip()
            # Clean up and validate
            category = re.sub(r'\*+', '', category).strip()
            # Find best match if not exact
            if category not in valid_categories:
                for vc in valid_categories:
                    if vc.lower() in category.lower() or category.lower() in vc.lower():
                        category = vc
                        break
            if category in valid_categories:
                annotations[cluster_id] = category

    return annotations


def _annotate_level2_batch(cluster_ids, category, cluster_infos, ontology, data_info, llm):
    """Level 2: Refine to specific subtypes within a category."""
    cat_info = ontology["categories"].get(category, {})
    subtypes = cat_info.get("subtypes", {})

    if not subtypes or len(cluster_ids) == 0:
        # No subtypes defined - use category name
        return {cid: category for cid in cluster_ids}

    # Build subtype descriptions
    subtype_info = []
    for subtype_name, subtype_data in subtypes.items():
        markers = ", ".join(subtype_data.get("markers", []))
        desc = subtype_data.get("description", "")
        subtype_info.append(f"- {subtype_name}: {desc}. Markers: {markers}")

    subtype_text = "\n".join(subtype_info)
    cluster_text = "\n".join([f"- {cluster_infos[cid]}" for cid in cluster_ids if cid in cluster_infos])

    messages = [
        {"role": "system", "content": f"""You are an expert in {data_info} cell type annotation.
These clusters were identified as {category}. Now refine to specific subtypes.

Available subtypes for {category}:
{subtype_text}

Rules:
- Assign the most specific subtype that matches the markers
- If no subtype matches well, use the general "{category}" """
        },
        {"role": "user", "content": f"""Refine these {category} clusters to specific subtypes:

{cluster_text}

OUTPUT FORMAT (one line per cluster, no extra text):
[cluster_id]: [subtype name]
..."""}
    ]

    response = llm.invoke(messages).content

    # Parse response
    annotations = {}
    valid_subtypes = set(subtypes.keys()) | {category}

    for line in response.strip().split('\n'):
        match = re.match(r'^(\d+)\s*:\s*(.+)$', line.strip())
        if match:
            cluster_id = match.group(1)
            if cluster_id in cluster_ids:
                subtype = match.group(2).strip()
                subtype = re.sub(r'\*+', '', subtype).strip()
                # Find best match
                if subtype not in valid_subtypes:
                    for vs in valid_subtypes:
                        if vs.lower() in subtype.lower() or subtype.lower() in vs.lower():
                            subtype = vs
                            break
                annotations[cluster_id] = subtype if subtype in valid_subtypes else category

    # Fill missing
    for cid in cluster_ids:
        if cid not in annotations:
            annotations[cid] = category

    return annotations


def _select_best_resolution(adata, resolutions=[0.3, 0.5, 0.8, 1.0]):
    """Try multiple resolutions and select the one with best cluster separation."""
    import scanpy as sc

    best_resolution = resolutions[0]
    best_score = -1

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")
        n_clusters = adata.obs[f"leiden_{res}"].nunique()

        # Score: prefer 10-30 clusters, penalize too few or too many
        if 10 <= n_clusters <= 30:
            score = 100 - abs(n_clusters - 20)
        elif n_clusters < 10:
            score = n_clusters * 5
        else:
            score = max(0, 50 - (n_clusters - 30))

        if score > best_score:
            best_score = score
            best_resolution = res

    return best_resolution, f"leiden_{best_resolution}"


@tool
def annotate_cell_types(
    adata_path: Annotated[str, Field(description="Path to preprocessed spatial data")],
    transferred_celltype: Annotated[str, Field(description="Path to transferred cell type CSV")],
    data_info: Annotated[str, Field(description="Dataset description (e.g., 'human heart MERFISH')")],
    save_path: Annotated[str, Field(description="Experiment directory")] = None,
    resolution: Annotated[float, Field(description="Leiden resolution (0=auto)")] = 0,
) -> str:
    """Annotate cell type clusters using hierarchical two-level batch approach.

    This tool uses a HIERARCHICAL approach:
    1. Multi-resolution clustering: tries multiple resolutions, picks best
    2. Level 1 (batch): Assigns BROAD categories to all clusters in ONE LLM call
    3. Level 2 (batch): Refines to specific SUBTYPES within each category
    4. Tissue-aware: filters cell types based on tissue type in data_info

    Benefits:
    - Faster: 2-3 LLM calls instead of N calls
    - More consistent: LLM sees all clusters together
    - Hierarchical: broad category first, then specific subtype
    - Tissue-specific: only shows relevant cell types
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    from ..agent import make_llm

    save_path = save_path or _config["save_path"]
    llm = make_llm(_get_subagent_model(), stop_sequences=[])
    output_path = f"{save_path}/celltype_annotated.h5ad"

    if exists(output_path):
        msg = f"Cell type annotations already exist at {output_path}"
        print(msg)
        return msg

    # Load hierarchical ontology
    ontology = _load_cell_type_ontology()
    if not ontology:
        return "ERROR: Cell type ontology not found. Please ensure data/cell_type_ontology.json exists."

    # Detect tissue type
    tissue_type = _detect_tissue_type(data_info)
    print(f"Detected tissue type: {tissue_type or 'unknown'}")

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Multi-resolution clustering
    if resolution == 0:
        print("Auto-selecting best clustering resolution...")
        best_res, leiden_key = _select_best_resolution(adata)
        print(f"Selected resolution: {best_res}")
    else:
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
        leiden_key = "leiden"

    adata.obs["leiden"] = adata.obs[leiden_key].astype(str)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"Found {n_clusters} clusters")

    # Load transferred labels
    transferred_df = pd.read_csv(transferred_celltype, index_col=0)
    if len(set(adata.obs.index) & set(transferred_df.index)) == 0:
        transferred_df.index = transferred_df.index.str.replace(r'-st$', '', regex=True)

    common_cells = list(set(adata.obs.index) & set(transferred_df.index))
    if len(common_cells) == 0:
        return f"ERROR: No matching cell IDs between adata and transferred labels"

    adata.obs["transferred_celltype"] = transferred_df.loc[adata.obs.index, "predicted_celltype"]

    # Identify marker genes
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=False)
    names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(20)
    scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"]).head(20)

    markers = {}
    for i in range(names.shape[1]):
        genes = names.iloc[:, i].tolist()
        gene_scores = scores.iloc[:, i].tolist()
        markers[i] = list(np.array(genes)[np.array(gene_scores) > 0])

    # Get composition from transferred labels
    composition = (
        adata.obs.groupby("leiden")["transferred_celltype"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    # Build cluster info for all clusters
    cluster_ids = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    cluster_infos = {}
    for cid in cluster_ids:
        cluster_infos[cid] = _build_cluster_info(adata, "leiden", cid, markers, composition)

    # Level 1: Broad category assignment (single LLM call)
    print("\n--- Level 1: Assigning broad categories ---")
    level1_annotations = _annotate_level1_batch(cluster_infos, ontology, tissue_type, data_info, llm)

    for cid in sorted(level1_annotations.keys(), key=lambda x: int(x)):
        print(f"  Cluster {cid}: {level1_annotations[cid]}")

    # Level 2: Refine to subtypes (one LLM call per category)
    print("\n--- Level 2: Refining to subtypes ---")
    final_annotations = {}

    # Group clusters by category
    category_clusters = {}
    for cid, cat in level1_annotations.items():
        if cat not in category_clusters:
            category_clusters[cat] = []
        category_clusters[cat].append(cid)

    for category, cids in category_clusters.items():
        if len(cids) > 0:
            level2_annotations = _annotate_level2_batch(cids, category, cluster_infos, ontology, data_info, llm)
            final_annotations.update(level2_annotations)
            for cid in sorted(cids, key=lambda x: int(x)):
                print(f"  Cluster {cid}: {level2_annotations.get(cid, category)}")

    # Fill any missing clusters
    for cid in cluster_ids:
        if cid not in final_annotations:
            final_annotations[cid] = level1_annotations.get(cid, "Unknown")

    # Apply annotations
    adata.obs["cell_type_broad"] = adata.obs["leiden"].map(level1_annotations)
    adata.obs["cell_type"] = adata.obs["leiden"].map(final_annotations)

    # Generate visualizations
    plt.ioff()

    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
        sc.tl.umap(adata)

    # UMAP plots
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color="leiden", size=3, legend_loc="on data", ax=ax, show=False)
    ax.set_title("UMAP - Leiden Clusters")
    fig.savefig(f"{save_path}/umap_leiden.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color="cell_type_broad", size=3, legend_loc="right margin", ax=ax, show=False)
    ax.set_title("UMAP - Broad Cell Types")
    fig.savefig(f"{save_path}/umap_celltype_broad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color="cell_type", size=3, legend_loc="right margin", ax=ax, show=False)
    ax.set_title("UMAP - Cell Types (Refined)")
    fig.savefig(f"{save_path}/umap_celltype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Spatial plots per sample
    plots_saved = ["umap_leiden.png", "umap_celltype_broad.png", "umap_celltype.png"]

    if "spatial" in adata.obsm:
        batch_col = None
        for col in ["batch", "sample_id", "slide", "sample"]:
            if col in adata.obs.columns:
                batch_col = col
                break

        if batch_col and adata.obs[batch_col].nunique() > 1:
            for sample in adata.obs[batch_col].unique():
                sample_data = adata[adata.obs[batch_col] == sample]
                fig, ax = plt.subplots(figsize=(8, 8))
                sc.pl.embedding(sample_data, basis="spatial", color="cell_type",
                               size=5, legend_loc="right margin", ax=ax, show=False)
                ax.set_title(f"Spatial Cell Types - {sample}")
                fig.savefig(f"{save_path}/spatial_celltype_{sample}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                plots_saved.append(f"spatial_celltype_{sample}.png")
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            sc.pl.embedding(adata, basis="spatial", color="cell_type",
                           size=3, legend_loc="right margin", ax=ax, show=False)
            ax.set_title("Spatial - Cell Types")
            fig.savefig(f"{save_path}/spatial_celltype.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots_saved.append("spatial_celltype.png")

    # Save results
    adata.write_h5ad(output_path, compression="gzip")

    # Save annotation details
    annot_df = pd.DataFrame({
        "cluster": cluster_ids,
        "broad_category": [level1_annotations.get(c, "Unknown") for c in cluster_ids],
        "cell_type": [final_annotations.get(c, "Unknown") for c in cluster_ids],
        "markers": [", ".join(markers.get(int(c), [])[:5]) for c in cluster_ids]
    })
    annot_df.to_csv(f"{save_path}/celltype_annotations.csv", index=False)

    tissue_info = f" (tissue: {tissue_type})" if tissue_type else ""
    msg = f"Successfully annotated {n_clusters} clusters{tissue_info}. Saved to {output_path}. Plots: {', '.join(plots_saved)}"
    print(msg)
    return msg


# =============================================================================
# Tool 3: Tissue Niche Annotator (Batch Approach)
# =============================================================================

def _create_composite_niche_plot(adata_sample, utag_key, save_path, sample_id):
    """Create a composite plot showing all niches with different colors."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.ioff()
    fig, ax = plt.subplots(figsize=(8, 8))

    x, y = adata_sample.obsm["spatial"][:, 0], adata_sample.obsm["spatial"][:, 1]
    niches = sorted(adata_sample.obs[utag_key].unique())

    # Use a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10' if len(niches) <= 10 else 'tab20')
    colors = {niche: cmap(i % 20) for i, niche in enumerate(niches)}

    # Plot each niche with its color
    for niche in niches:
        mask = adata_sample.obs[utag_key] == niche
        ax.scatter(x[mask], y[mask], s=1, c=[colors[niche]], label=f"Niche {niche}")

    # Add legend
    patches = [mpatches.Patch(color=colors[n], label=f"Niche {n}") for n in niches]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.9)
    ax.axis("off")
    ax.set_title(f"Sample: {sample_id}")

    composite_img = f"{save_path}/composite_niches_{sample_id}.png"
    fig.savefig(composite_img, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return composite_img, niches, colors


def _get_niche_info_batch(adata_sample, utag_key, celltype_col):
    """Get cell composition and marker genes for all niches in a sample."""
    import scanpy as sc

    niches = sorted(adata_sample.obs[utag_key].unique())
    niche_info = {}

    # Get marker genes for all niches at once
    marker_genes_all = {}
    if adata_sample.obs[utag_key].nunique() > 1:
        sc.tl.rank_genes_groups(adata_sample, groupby=utag_key, method="wilcoxon", use_raw=False)
        try:
            genes_df = pd.DataFrame(adata_sample.uns["rank_genes_groups"]["names"])
            scores_df = pd.DataFrame(adata_sample.uns["rank_genes_groups"]["scores"])
            for niche in niches:
                if str(niche) in genes_df.columns:
                    genes = genes_df[str(niche)].head(15).tolist()
                    scores = scores_df[str(niche)].head(15).tolist()
                    marker_genes_all[niche] = [g for g, s in zip(genes, scores) if s > 0][:8]
                else:
                    marker_genes_all[niche] = []
        except:
            marker_genes_all = {n: [] for n in niches}
    else:
        marker_genes_all = {n: [] for n in niches}

    # Get cell composition for all niches
    cell_comp = adata_sample.obs.groupby(utag_key)[celltype_col].value_counts(normalize=True).unstack().fillna(0)

    for niche in niches:
        info = f"Niche {niche}: "
        # Cell composition
        if niche in cell_comp.index:
            top_cells = cell_comp.loc[niche].sort_values(ascending=False)
            top_cells = top_cells[top_cells > 0.05]  # Only show > 5%
            for ct, pct in top_cells.items():
                info += f"{ct} ({pct:.0%}), "
        # Marker genes
        if marker_genes_all.get(niche):
            info += f"| Markers: {', '.join(marker_genes_all[niche][:5])}"
        niche_info[niche] = info.rstrip(", ")

    return niche_info


def _annotate_sample_batch(adata_sample, utag_key, celltype_col, data_info,
                           anatomical_path, save_path, sample_id, llm):
    """Annotate all niches in a sample with a single LLM call."""
    import matplotlib.pyplot as plt

    def encode_image(image_path):
        image_bytes = _resize_image_if_needed(image_path)
        return base64.b64encode(image_bytes).decode("utf-8")

    # Create composite plot
    composite_img, niches, colors = _create_composite_niche_plot(
        adata_sample, utag_key, save_path, sample_id
    )

    # Get info for all niches (cell composition + markers)
    niche_info = _get_niche_info_batch(adata_sample, utag_key, celltype_col)

    # Calculate centroid coordinates for each niche (normalized to 0-100 scale)
    coords = adata_sample.obsm["spatial"]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    niche_centroids = {}
    for niche in niches:
        mask = adata_sample.obs[utag_key] == niche
        niche_coords = coords[mask]
        centroid_x = (niche_coords[:, 0].mean() - x_min) / (x_max - x_min) * 100
        centroid_y = (niche_coords[:, 1].mean() - y_min) / (y_max - y_min) * 100
        niche_centroids[niche] = (centroid_x, centroid_y)

    # Build the prompt with niche info AND coordinates
    niche_info_lines = []
    for niche in niches:
        cx, cy = niche_centroids[niche]
        info = niche_info.get(niche, f"Niche {niche}")
        # Add position: x=0 is image left, x=100 is image right; y=0 is image bottom, y=100 is image top
        niche_info_lines.append(f"- {info} | Position: x={cx:.0f}, y={cy:.0f}")

    niche_info_text = "\n".join(niche_info_lines)

    # Prepare images
    composite_b64 = encode_image(composite_img)

    if anatomical_path and exists(anatomical_path):
        # With anatomical reference
        anatomical_b64 = encode_image(anatomical_path)

        messages = [
            {"role": "system", "content": f"""You are an expert in {data_info} spatial biology and anatomy.
Your task is to annotate all tissue niches in a spatial transcriptomics sample.

CRITICAL - Understanding left/right:
- The ANATOMICAL REFERENCE IMAGE (Image 1) shows labels from the PATIENT's perspective
- In patient's perspective: patient's LEFT appears on the RIGHT side of the image
- The SPATIAL DATA IMAGE (Image 2) is from the VIEWER's perspective (what you see is what you get)
- To convert: what appears on the LEFT in Image 2 corresponds to patient's RIGHT anatomically
- Use the centroid coordinates (x,y) provided for each niche to determine precise positions
- x=0 is left side of image, x=100 is right side; y=0 is bottom, y=100 is top

Rules:
- Each niche must have a UNIQUE anatomical name - no duplicates
- Use standard anatomical terminology (e.g., "Left Ventricle", "Right Atrium")
- Match positions using both visual inspection AND centroid coordinates"""
            },
            {"role": "user", "content": [
                {"type": "text", "text": f"""Please annotate all tissue niches for this {data_info} sample.

IMAGE 1 (Anatomical Reference): Shows labeled anatomical regions from PATIENT's perspective.
  - Labels like "Left Ventricle" refer to the patient's left side
  - Patient's left appears on the RIGHT side of this image

IMAGE 2 (Spatial Niches): Shows niches from VIEWER's perspective with colors matching the legend.
  - What you see on the left IS the left side of the image
  - To get anatomical left/right: FLIP the image mentally

Cell composition, marker genes, and centroid positions for each niche:
{niche_info_text}

Using the centroid coordinates and visual comparison with the anatomical reference,
assign the correct anatomical name to each niche.

OUTPUT FORMAT (one line per niche, no markdown):
0: [anatomical name]
1: [anatomical name]
...
"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{anatomical_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{composite_b64}"}}
            ]}
        ]
    else:
        # Without anatomical reference - use cell composition only
        messages = [
            {"role": "system", "content": f"""You are an expert in {data_info} spatial biology.
Your task is to annotate all tissue niches based on their cell type composition and spatial positions."""
            },
            {"role": "user", "content": [
                {"type": "text", "text": f"""Please annotate all tissue niches for this {data_info} sample.

The image shows all niches colored differently. Each color corresponds to a niche number.

Cell composition, marker genes, and centroid positions for each niche:
{niche_info_text}

Based on the cell type composition and spatial organization, assign a descriptive biological name to each niche.

OUTPUT FORMAT (one line per niche, no markdown):
0: [name]
1: [name]
...
"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{composite_b64}"}}
            ]}
        ]

    # Single LLM call for all niches
    response = llm.invoke(messages).content

    # Parse response - only accept valid niche IDs
    annotations = {}
    reasons = {}
    valid_niche_ids = set(str(n) for n in niches)

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        try:
            # Parse "0: Left Ventricle" format - must start with number
            match = re.match(r'^(\d+)\s*:\s*(.+)$', line)
            if match:
                niche_id = match.group(1)
                # Only accept if it's a valid niche ID we're looking for
                if niche_id in valid_niche_ids:
                    name = match.group(2).strip()
                    # Clean up markdown and extra formatting
                    name = re.sub(r'\*+', '', name).strip()
                    name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
                    # Skip if name is empty or looks like more explanation
                    if name and not name.startswith('x=') and len(name) < 100:
                        annotations[niche_id] = name
                        reasons[niche_id] = f"Batch annotation for sample {sample_id}"
        except:
            continue

    # Fill in any missing niches
    for niche in niches:
        niche_str = str(niche)
        if niche_str not in annotations:
            annotations[niche_str] = f"Niche_{niche}"
            reasons[niche_str] = "Parse fallback"

    return annotations, reasons


def _merge_niche_annotations_batch(all_annotations, llm):
    """Merge niche annotations from multiple samples with a single LLM call."""
    from langchain_core.output_parsers import StrOutputParser

    # Collect all niche IDs across samples
    all_niches = set()
    for sample_annot in all_annotations.values():
        all_niches.update(sample_annot.keys())

    # Build info for each niche
    merge_info = []
    for niche_id in sorted(all_niches, key=lambda x: int(x) if x.isdigit() else 0):
        sample_names = []
        for sample_id, annots in all_annotations.items():
            if niche_id in annots:
                sample_names.append(f"{sample_id}: {annots[niche_id]}")
        if sample_names:
            merge_info.append(f"Niche {niche_id}:\n  " + "\n  ".join(sample_names))

    merge_text = "\n\n".join(merge_info)

    messages = [
        {"role": "system", "content": """You are an expert in tissue biology.
Your task is to determine consensus anatomical names for tissue niches based on annotations from multiple samples.
Each niche should have ONE final name that best represents annotations across all samples."""},
        {"role": "user", "content": f"""Based on annotations from multiple samples, provide a consensus name for each niche.

Annotations per sample:
{merge_text}

Rules:
- Choose the most consistent/common annotation across samples
- Use standard anatomical terminology
- Each niche must have a unique name

OUTPUT FORMAT (one line per niche, no markdown, no extra text):
0: [consensus name]
1: [consensus name]
..."""}
    ]

    response = llm.invoke(messages).content

    # Parse response
    merged_names = {}
    merged_reasons = {}

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        try:
            parts = line.split(':', 1)
            niche_id = re.sub(r'[^0-9]', '', parts[0].strip())
            if niche_id:
                name = parts[1].strip()
                name = re.sub(r'\*+', '', name).strip()
                merged_names[niche_id] = name
                merged_reasons[niche_id] = "Consensus from multiple samples"
        except:
            continue

    # Fill in any missing niches with first sample's annotation
    first_sample = list(all_annotations.values())[0] if all_annotations else {}
    for niche_id in all_niches:
        if niche_id not in merged_names:
            merged_names[niche_id] = first_sample.get(niche_id, f"Niche_{niche_id}")
            merged_reasons[niche_id] = "Merge fallback"

    return merged_names, merged_reasons


@tool
def annotate_tissue_niches(
    adata_path: Annotated[str, Field(description="Path to spatial data with cell types")],
    utag_csv: Annotated[str, Field(description="Path to UTAG clustering results CSV")],
    data_info: Annotated[str, Field(description="Dataset description (e.g., 'human heart MERFISH')")],
    save_path: Annotated[str, Field(description="Experiment directory")] = None,
    anatomical_path: Annotated[str, Field(description="Optional path to anatomical tissue image")] = None,
    utag_column: Annotated[str, Field(description="UTAG label column to use")] = "utag",
    celltype_column: Annotated[str, Field(description="Cell type column in adata.obs")] = "cell_type",
    batch_column: Annotated[str, Field(description="Batch/sample column")] = "batch",
) -> str:
    """Annotate tissue niches using spatial domains and cell type composition.

    This tool uses a BATCH approach for efficiency:
    1. Creates composite plot showing all niches per sample (single image)
    2. Annotates ALL niches in one LLM call per sample (with anatomical reference if provided)
    3. Merges annotations across samples with a single LLM call
    4. Generates final per-sample niche plots

    Benefits of batch approach:
    - Fewer LLM calls (1 per sample instead of 1 per niche)
    - Consistent naming (LLM sees all niches together, avoids duplicates)
    - Better anatomical reasoning (can compare relative positions)
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import json
    from ..agent import make_llm

    save_path = save_path or _config["save_path"]
    # Single LLM for all operations (vision-capable)
    llm = make_llm(_get_subagent_model(), stop_sequences=[])

    output_path = f"{save_path}/niche_annotated.h5ad"

    if exists(output_path):
        return f"Niche annotations already exist at {output_path}"

    # Load data
    adata = sc.read_h5ad(adata_path)
    utag_df = pd.read_csv(utag_csv, index_col=0)

    # Find UTAG column
    if utag_column not in utag_df.columns:
        utag_cols = [c for c in utag_df.columns if "utag" in c.lower() or "UTAG" in c]
        if not utag_cols:
            return f"ERROR: No UTAG columns found in {utag_csv}"
        utag_column = utag_cols[0]
    print(f"Using UTAG column: {utag_column}")

    # Align indices
    if len(set(adata.obs.index) & set(utag_df.index)) == 0:
        utag_df.index = utag_df.index.str.replace(r'-st$', '', regex=True)

    adata.obs["utag_main"] = utag_df.loc[adata.obs.index, utag_column].astype(str)

    # Check columns
    if celltype_column not in adata.obs.columns:
        return f"ERROR: Cell type column '{celltype_column}' not found"
    if batch_column not in adata.obs.columns:
        batch_column = None  # Single sample mode

    # Batch annotation per sample
    all_annotations = {}  # {sample: {niche_id: name}}
    all_reasons = {}  # {sample: {niche_id: reason}}

    samples = adata.obs[batch_column].unique() if batch_column else ["all"]

    for sample in samples:
        print(f"\n--- Annotating sample: {sample} (batch mode) ---")
        if batch_column:
            adata_sample = adata[adata.obs[batch_column] == sample].copy()
        else:
            adata_sample = adata.copy()

        # Single LLM call for all niches in this sample
        annotations, reasons = _annotate_sample_batch(
            adata_sample, "utag_main", celltype_column, data_info,
            anatomical_path, save_path, sample, llm
        )
        all_annotations[sample] = annotations
        all_reasons[sample] = reasons

        # Print annotations
        for niche_id in sorted(annotations.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            print(f"  Niche {niche_id}: {annotations[niche_id]}")

    # Save per-sample annotations
    with open(f"{save_path}/niche_annotations_per_sample.json", "w") as f:
        json.dump({"names": all_annotations, "reasons": all_reasons}, f, indent=2)

    # Merge annotations across samples (single LLM call)
    if len(samples) > 1:
        print("\n--- Merging annotations across samples (batch mode) ---")
        merged_names, merged_reasons = _merge_niche_annotations_batch(all_annotations, llm)
    else:
        # Single sample - use directly
        merged_names = all_annotations[samples[0]]
        merged_reasons = all_reasons[samples[0]]

    # Apply merged annotations
    adata.obs["tissue_niche"] = adata.obs["utag_main"].map(merged_names)
    adata.obs["tissue_niche"] = adata.obs["tissue_niche"].fillna("Unknown")

    # Generate final per-sample plots
    plt.ioff()
    for sample in samples:
        if batch_column:
            adata_sample = adata[adata.obs[batch_column] == sample]
        else:
            adata_sample = adata

        fig, ax = plt.subplots(figsize=(8, 8))
        sc.pl.embedding(adata_sample, basis="spatial", color="tissue_niche",
                       size=5, legend_loc="right margin", ax=ax, show=False)
        ax.set_title(f"Tissue Niches - {sample}")
        fig.savefig(f"{save_path}/niche_spatial_{sample}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Save results
    adata.write_h5ad(output_path, compression="gzip")

    niche_df = pd.DataFrame([
        {"niche_id": k, "niche_name": v, "reason": merged_reasons.get(k, "")}
        for k, v in merged_names.items()
    ])
    niche_df.to_csv(f"{save_path}/niche_annotations.csv", index=False)

    msg = f"Successfully annotated {len(merged_names)} niches across {len(samples)} samples. Saved to {output_path}"
    print(msg)
    return msg


# =============================================================================
# Tool 4: Figure Interpreter (Vision LLM)
# =============================================================================

def _resize_image_if_needed(image_path: str, max_size_bytes: int = 4_000_000) -> bytes:
    """Resize image if file size exceeds API limit.

    Claude API has a 5MB per-image limit. We resize to stay under that.
    File size scales roughly with pixel count, so we reduce dimensions proportionally.

    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum file size in bytes (default 4MB for safety margin)

    Returns:
        Image bytes (resized if necessary, original if not)
    """
    from PIL import Image
    import io
    import math

    with open(image_path, "rb") as f:
        original_bytes = f.read()

    if len(original_bytes) <= max_size_bytes:
        return original_bytes

    # Need to resize - scale down proportionally
    with Image.open(image_path) as img:
        scale = math.sqrt(max_size_bytes / len(original_bytes))
        new_size = (max(int(img.width * scale), 100), max(int(img.height * scale), 100))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format=img.format or 'PNG')
        print(f"Resized image: {img.width}x{img.height} -> {new_size[0]}x{new_size[1]}")
        return buffer.getvalue()


@tool
def interpret_figure(
    image_path: Annotated[str, Field(description="Path to the figure image to interpret")],
    context: Annotated[str, Field(description="Context about what was plotted (e.g., 'UMAP colored by cell type', 'spatial distribution of marker genes')")],
    analysis_focus: Annotated[str, Field(description="What aspect to focus on (e.g., 'cluster separation', 'spatial patterns', 'gene expression distribution')")] = "general",
) -> str:
    """Interpret a plotted figure using vision LLM and provide biological insights.

    This tool uses a vision-capable LLM to:
    1. Analyze the visual content of scientific figures
    2. Identify key patterns, clusters, or distributions
    3. Provide biological interpretation based on context
    4. Suggest potential follow-up analyses

    Use this tool after generating plots to get automated interpretation
    that can guide the next steps of analysis.

    Returns:
        A detailed interpretation including:
        - Visual observations
        - Biological interpretation
        - Key findings
        - Suggested follow-up analyses
    """
    from ..agent import make_llm

    # Validate image exists
    if not exists(image_path):
        return f"ERROR: Image file not found at {image_path}"

    # Create vision LLM instance
    llm = make_llm(_get_subagent_model(), stop_sequences=[])

    # Determine image type
    ext = image_path.lower().split('.')[-1]
    mime_type = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'svg': 'image/svg+xml',
        'pdf': 'application/pdf'
    }.get(ext, 'image/png')

    # Resize image if needed and encode to base64
    image_bytes = _resize_image_if_needed(image_path)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Construct vision prompt - keep it concise to avoid bloating context
    messages = [
        {"role": "system", "content": """You are an expert computational biologist. Provide brief, focused figure interpretations in 3-5 lines total. No follow-up suggestions needed."""
        },
        {"role": "user", "content": [
            {"type": "text", "text": f"""Briefly analyze this figure (3-5 lines max).

**Context**: {context}
**Focus**: {analysis_focus}

Format: One line for what you observe, then 2-3 key biological findings as bullet points."""},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
        ]}
    ]

    # Get interpretation
    response = llm.invoke(messages)

    return response.content


