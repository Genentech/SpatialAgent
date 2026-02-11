# Spatial Mapping with Tangram

Map single-cell RNA-seq data onto spatial transcriptomics to enable cell type deconvolution and gene expression imputation.

## Workflow Overview

1. **Explore datasets** (always do this first)
2. Preprocess data for Tangram
3. Map cells to spatial locations
4. Project cell type annotations
5. Project gene expression
6. Evaluate mapping quality

---

## Step 1: Explore Datasets

**ALWAYS start here.** Understand both datasets:

```python
import scanpy as sc

# Load scRNA-seq reference
adata_sc = sc.read_h5ad("path/to/scrna.h5ad")
print(f"scRNA-seq: {adata_sc.shape}")
print(f"Cell types: {adata_sc.obs['cell_type'].nunique()}")
print(f"Obs columns: {list(adata_sc.obs.columns)}")

# Load spatial data
adata_sp = sc.read_h5ad("path/to/spatial.h5ad")
print(f"Spatial: {adata_sp.shape}")
print(f"Has spatial coords: {'spatial' in adata_sp.obsm}")
```

**Identify**:
- Cell type annotation column in scRNA-seq (e.g., 'cell_type', 'cell_subclass')
- Whether datasets are from same tissue/species
- Data normalization status

---

## Step 2: Preprocess for Tangram

**Tool**: `tangram_preprocess`

**Purpose**: Find shared genes and compute density priors.

**Parameters**:
- `marker_genes="auto"`: Auto-compute from differential expression
- `cell_type_key`: Column with cell type annotations
- `n_markers`: Top markers per cell type (default 100)

**Outputs**:
- `tangram_sc_prep.h5ad`: Preprocessed scRNA-seq
- `tangram_sp_prep.h5ad`: Preprocessed spatial with density priors

---

## Step 3: Map Cells to Space

**Tool**: `tangram_map_cells`

**Purpose**: Create cell-to-spot probability matrix.

**Modes**:
- `mode="cells"`: Single-cell resolution (GPU recommended, slower)
- `mode="clusters"`: Cluster averages (faster, good for cross-sample)

**Parameters**:
- `device="cuda:0"` for GPU, `"cpu"` for CPU
- `num_epochs`: 500-1000 typical

**Output**: `tangram_mapping.h5ad` (cells × spots probability matrix)

---

## Step 4: Project Cell Annotations

**Tool**: `tangram_project_annotations`

**Purpose**: Transfer cell type probabilities to spatial spots.

**Parameters**:
- `annotation`: Cell type column to project

**Outputs**:
- `tangram_annotated.h5ad`: Spatial data with predictions
- `tangram_celltype_probs.csv`: Probability matrix

---

## Step 5: Project Gene Expression

**Tool**: `tangram_project_genes`

**Purpose**: Impute full transcriptome onto spatial locations.

**Use cases**:
- Visualize genes not in spatial panel
- Correct dropout in spatial measurements
- Analyze any gene spatially

**Output**: `tangram_projected.h5ad` (spots × genes)

---

## Step 6: Evaluate Mapping

**Tool**: `tangram_evaluate`

**Purpose**: Validate mapping by comparing predicted vs measured expression.

**Metrics**:
- **Training score > 0.7**: Good mapping
- **AUC > 0.5**: Reasonable test predictions
- Low scores often from sparsity mismatch (dropout)

**Outputs**:
- `tangram_scores.csv`: Per-gene scores
- `tangram_auc.png`: Validation curve

---

## Visualization Examples

After mapping, visualize results:

```python
import scanpy as sc
import matplotlib.pyplot as plt

# Load annotated spatial data
adata_sp = sc.read_h5ad("experiments/tangram_annotated.h5ad")

# Plot cell type probabilities
sc.pl.spatial(adata_sp, color="tangram_celltype", title="Predicted Cell Types")

# Load projected genes
adata_ge = sc.read_h5ad("experiments/tangram_projected.h5ad")

# Compare measured vs predicted for specific gene
gene = "CD3E"
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sc.pl.spatial(adata_sp, color=gene, ax=axes[0], title=f"{gene} (measured)")
sc.pl.spatial(adata_ge, color=gene, ax=axes[1], title=f"{gene} (predicted)")
```

---

## Output Files

- `tangram_sc_prep.h5ad` - Preprocessed scRNA-seq
- `tangram_sp_prep.h5ad` - Preprocessed spatial
- `tangram_mapping.h5ad` - Cell-to-spot probability matrix
- `tangram_annotated.h5ad` - Spatial with cell type predictions
- `tangram_celltype_probs.csv` - Cell type probability matrix
- `tangram_projected.h5ad` - Imputed gene expression
- `tangram_scores.csv` - Per-gene mapping scores
- `tangram_auc.png` - Validation AUC curve

---

## Tips

1. **GPU recommended** for `mode="cells"` with large datasets
2. Use `mode="clusters"` when scRNA-seq and spatial are from different samples
3. More training genes = better mapping, but needs shared expression
4. Low scores for sparse genes are expected (dropout in spatial data)
5. Check that tissue types match between scRNA-seq and spatial
