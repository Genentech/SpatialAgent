# Spatial Deconvolution Analysis

Estimate cell type compositions in spatial transcriptomics spots using deep learning methods from scvi-tools and cell2location.

## Platform Applicability

**This is the correct workflow for cell type identification on spot-based platforms** (Visium, Slide-seq, ST). Each spot contains multiple cells, so deconvolution estimates cell type **proportions** rather than assigning a single label. If a user asks to "annotate cell types" on Visium or other spot-based data, this is the appropriate skill.

For single-cell resolution platforms (MERFISH, Xenium, CosMx, SeqFISH), use the `annotation` skill instead.

---

## Prerequisites

**Required**:
- Single-cell RNA-seq reference (h5ad) with cell type annotations
- Spatial transcriptomics data (h5ad) - Visium, Slide-seq, or similar
- Common genes between datasets (>100 recommended)
- GPU recommended for faster training

**Optional**:
- Batch information for multi-sample integration
- Raw counts (preferred over normalized data)

---

## Available Methods

| Method | Use Case | Speed | Outputs |
|--------|----------|-------|---------|
| **DestVI** | Multi-resolution analysis with cell state variation | Medium | Proportions + gamma latent space |
| **Cell2location** | Bayesian mapping with uncertainty quantification | Slow | Abundance with confidence intervals |
| **Stereoscope** | Simple cell type proportions | Fast | Proportions |
| **gimVI** | Gene imputation for limited panels | Medium | Imputed gene expression |

---

## Workflow Overview

1. **Explore datasets** (check cell type annotations, gene overlap)
2. Choose deconvolution method based on use case
3. Run deconvolution
4. Visualize and interpret results

---

## Step 1: Explore Datasets

Use `execute_python` to check data compatibility:

```python
import scanpy as sc

# Load data
sc_adata = sc.read_h5ad("path/to/reference.h5ad")
st_adata = sc.read_h5ad("path/to/spatial.h5ad")

# Check cell type annotations
print(f"scRNA-seq: {sc_adata.shape}")
print(f"Cell types: {sc_adata.obs['cell_type'].nunique()}")
print(sc_adata.obs['cell_type'].value_counts())

# Check gene overlap
common = set(sc_adata.var_names) & set(st_adata.var_names)
print(f"\nSpatial: {st_adata.shape}")
print(f"Common genes: {len(common)}")
```

---

## Step 2: Choose Method

### Option A: DestVI (Recommended for detailed analysis)

Best for: Understanding cell state variation within cell types

**Tool**: `destvi_deconvolution`
- `sc_adata_path`: Path to scRNA-seq reference
- `st_adata_path`: Path to spatial data
- `cell_type_key`: Cell type column name
- `sc_max_epochs`: 300 (single-cell model)
- `st_max_epochs`: 2500 (spatial model)

**Outputs**:
- `destvi_proportions.csv`: Cell type proportions per spot
- `destvi_spatial.h5ad`: Spatial data with proportions in obsm
- Trained models for further analysis (gamma space)

### Option B: Cell2location (Best uncertainty quantification)

Best for: Bayesian inference with tissue-specific priors

**Tool**: `cell2location_mapping`
- `sc_adata_path`: Path to scRNA-seq reference
- `st_adata_path`: Path to spatial data
- `cell_type_key`: Cell type column name
- `batch_key`: Optional batch column for multi-sample reference
- `n_cells_per_location`: Expected cells per spot (tissue-dependent)
  - ~30 for lymph node
  - ~8 for brain
  - ~20 for most tissues (default)
- `detection_alpha`: 200 (default), use 20 for high batch variation

**Outputs**:
- `cell2location_abundance.csv`: 5% quantile of cell abundance (confident estimates)
- `cell2location_spatial.h5ad`: Full posterior results
- Trained models for visualization

### Option C: Stereoscope (Fast and simple)

Best for: Quick deconvolution with straightforward proportions

**Tool**: `stereoscope_deconvolution`
- `sc_adata_path`: Path to scRNA-seq reference
- `st_adata_path`: Path to spatial data
- `cell_type_key`: Cell type column name
- `sc_max_epochs`: 100
- `st_max_epochs`: 2000

**Outputs**:
- `stereoscope_proportions.csv`: Cell type proportions per spot
- `stereoscope_spatial.h5ad`: Spatial data with proportions

### Option D: gimVI (Gene imputation)

Best for: Imputing missing genes in limited spatial panels (FISH-based)

**Tool**: `gimvi_imputation`
- `sc_adata_path`: Path to scRNA-seq reference
- `st_adata_path`: Path to spatial data
- `genes_to_impute`: Comma-separated list (empty for all missing)
- `max_epochs`: 200

**Outputs**:
- `gimvi_imputed.csv`: Imputed expression matrix
- `gimvi_spatial.h5ad`: Spatial data with imputed values in obsm

---

## Step 3: Visualize Results

```python
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# Load results
st_adata = sc.read_h5ad("experiments/destvi_spatial.h5ad")
proportions = pd.read_csv("experiments/destvi_proportions.csv", index_col=0)

# Plot cell type distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ct in enumerate(proportions.columns[:6]):
    ax = axes.flat[i]
    st_adata.obs[ct] = proportions[ct]
    sc.pl.spatial(st_adata, color=ct, ax=ax, show=False, title=ct)
plt.tight_layout()
plt.savefig("celltype_distribution.png", dpi=150)
```

---

## Tips

1. **Data preparation**: Use raw counts when possible. Normalized data may affect model training.

2. **Gene filtering**: Filter to common genes before running. Low-quality genes (few cells expressing) may add noise.

3. **Training time**: DestVI and Cell2location require longer training. Use GPU for faster convergence.

4. **Cell type balance**: Ensure reference has sufficient cells for each cell type (>50 recommended).

5. **Tissue-specific parameters**: Adjust `n_cells_per_location` based on expected tissue density.

6. **Quality assessment**:
   - Check reconstruction loss converges
   - Verify proportions sum to ~1 per spot
   - Compare known marker genes with deconvolution results

---

## Method Comparison

| Aspect | DestVI | Cell2location | Stereoscope | gimVI |
|--------|--------|---------------|-------------|-------|
| Speed | Medium | Slow | Fast | Medium |
| Memory | High | High | Medium | Medium |
| Cell states | Yes (gamma) | No | No | N/A |
| Uncertainty | No | Yes (posterior) | No | No |
| Batch effects | Limited | Yes | No | No |
| GPU required | Recommended | Recommended | Optional | Optional |
