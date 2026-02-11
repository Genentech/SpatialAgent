# Spatial Annotation

Annotate cell types and tissue niches in spatial transcriptomics data.

## Workflow Overview

1. **Explore dataset structure** (always do this first)
2. Preprocess spatial data
3. Find and download scRNA-seq reference from CZI
4. Transfer cell type labels via Harmony integration
5. Annotate cell types using hierarchical approach
6. Run spatial clustering (UTAG) for tissue niches
7. Annotate tissue niches

---

## Step 1: Explore Dataset Structure

**ALWAYS start here.** Use `execute_python` to understand the data:

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/data.h5ad")

# Basic info
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"var columns: {list(adata.var.columns)}")

# Check for spatial coordinates
if 'spatial' in adata.obsm:
    print(f"Spatial coords: {adata.obsm['spatial'].shape}")

# Check data range (normalized?)
print(f"X max: {adata.X.max():.2f}, min: {adata.X.min():.2f}")

# Sample/batch structure
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    if n_unique < 20:
        print(f"{col}: {adata.obs[col].unique().tolist()}")
```

**Identify**:
- Sample/batch column for per-sample processing
- Whether data is already normalized
- Species and tissue type from metadata or user input

---

## Step 2: Preprocess Data

**Tool**: `preprocess_spatial_data`

**Purpose**: Normalize, compute HVGs, PCA, neighbors, and UMAP.

**Key considerations**:
- Skip normalization if data is already normalized (X.max() < 20)
- Adjust `min_genes`, `min_cells` based on data quality

**Inspect**: `inspect_tool_code("preprocess_spatial_data")`

---

## Step 3: Find Reference Dataset

**Tool**: `search_czi_datasets`

**Purpose**: Search CZI CELLxGENE Census for matching scRNA-seq reference.

**Key considerations**:
- Query format: "{species} {tissue} normal"
- Review returned datasets and select best match

**Inspect**: `inspect_tool_code("search_czi_datasets")`

---

## Step 4: Download Reference

**Tool**: `download_czi_reference`

**Purpose**: Download the reference h5ad with cell type annotations.

**Inspect**: `inspect_tool_code("download_czi_reference")`

---

## Step 5: Transfer Labels

**Tool**: `harmony_transfer_labels`

**Purpose**: Integrate spatial data with reference using Harmony, train classifier, transfer labels.

**Key considerations**:
- Ensure gene overlap between spatial and reference data
- Check transferred labels distribution makes biological sense

**Inspect**: `inspect_tool_code("harmony_transfer_labels")`

---

## Step 6: Annotate Cell Types

**Tool**: `annotate_cell_types`

**Purpose**: Refine cell type annotations using two-level hierarchical approach.

**Key considerations**:
- `data_info`: Describe species + tissue + technology (e.g., "Human heart MERFISH")
- `resolution`: Set to 0 for automatic selection, or specify (0.3-1.0)
- Level 1 assigns broad categories, Level 2 refines to subtypes

**Inspect**: `inspect_tool_code("annotate_cell_types")`

---

## Step 7: Spatial Clustering (UTAG)

**Tool**: `run_utag_clustering`

**Purpose**: Identify tissue niches based on spatial proximity and gene expression.

**Key considerations**:
- `slide_key`: Column for sample IDs (from Step 1) - runs UTAG per sample
- `resolutions`: List of clustering resolutions (e.g., [0.1, 0.3])

**Inspect**: `inspect_tool_code("run_utag_clustering")`

---

## Step 8: Annotate Niches

**Tool**: `annotate_tissue_niches`

**Purpose**: Annotate spatial niches based on cell type composition and location.

**Key considerations**:
- `anatomical_reference`: Path to anatomical image helps with left/right orientation
- Uses batch approach (all niches per sample in one LLM call)

**Inspect**: `inspect_tool_code("annotate_tissue_niches")`

---

## Step 9: Summarize Results

**Tool**: `summarize_celltypes`

**Purpose**: Report cell type distributions and key findings.

---

## Output Files

- `preprocessed.h5ad` - Normalized data with PCA/UMAP
- `celltype_transferred.csv` - Reference-transferred labels
- `celltype_annotated.h5ad` - Data with `cell_type`, `cell_type_broad` columns
- `celltype_annotations.csv` - Cluster-level annotations
- `utag_main_result.csv` - Spatial niche clusters
- `niche_annotated.h5ad` - Data with `tissue_niche` column
- `niche_annotations.csv` - Niche annotations
- Visualization plots (UMAP, spatial maps)
