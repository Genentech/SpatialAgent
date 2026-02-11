# Cell-Cell Communication Analysis

Analyze cell-cell interactions and communication dynamics in spatial transcriptomics data.

## Prerequisites

**Required**: Data must have cell type annotations in `obs`.
- If data is not annotated, run the **annotation** skill first

**Optional**: Spatial niche annotations for region-specific analysis.

---

## Workflow Overview

1. **Explore dataset structure** (always do this first)
2. Characterize dataset (cell types, conditions)
3. Infer cell-cell interactions (LIANA + Cell2Cell + TensorLy)
4. Analyze spatial context (if niches annotated)
5. Compare across conditions (if applicable)
6. Generate summary report

---

## Step 1: Explore Dataset Structure

**ALWAYS start here.** Use `execute_python` to understand the data:

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/data.h5ad")

# Basic info
print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")

# Check for required columns
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    print(f"  {col}: {n_unique} unique values")
    if n_unique < 20:
        print(f"    values: {adata.obs[col].unique().tolist()}")
```

**Identify**:
- Cell type column name (e.g., `cell_type`, `celltype`, `annotation`)
- Sample/batch column name (e.g., `sample`, `batch`, `sample_id`)
- Condition column name (e.g., `condition`, `group`, `treatment`)

---

## Step 2: Characterize Dataset

**Tool**: `summarize_celltypes`
- List cell types and their frequencies

**Tool**: `summarize_conditions` (if conditions present)
- Compare cell type distributions across conditions

---

## Step 3: Infer Cell-Cell Interactions

**Tool**: `liana_tensor`
- `sample_key`: column for sample/batch IDs (from Step 1)
- `condition_key`: column for conditions (from Step 1)
- `cell_type_key`: column with cell type annotations (from Step 1)
- Pipeline: LIANA → Cell2Cell → TensorLy tensor factorization

---

## Step 4: Analyze Spatial Context

**Tool**: `summarize_tissue_regions`
- Requires spatial niche annotations (from annotation skill)
- Analyze cell type composition per region

### Spatial Colocalization Analysis

**Tool**: `squidpy_spatial_neighbors`
- Build spatial neighbors graph (required first)
- Use `coord_type="visium"` for Visium, `"generic"` for others

**Tool**: `squidpy_nhood_enrichment`
- Test if interacting cell types are physically neighbors
- Positive z-score = colocalized, negative = avoid each other

**Tool**: `squidpy_co_occurrence`
- Measure co-occurrence at different spatial distances
- Identifies spatial range of cell type interactions

**Tool**: `squidpy_ligrec`
- Permutation-based ligand-receptor analysis with spatial context

---

## Step 5: Cross-Condition Dynamics

**Tool**: `infer_dynamics` (if multiple conditions)
- Compare interactions between conditions
- Identify condition-specific communication changes

---

## Step 6: Generate Report

**Tool**: `report_subagent`
- Auto-discovers all figures and CSV outputs
- Uses vision LLM to interpret figures
- Generates publication-quality report with biological interpretation

---

## Output Files

- `interactions/` folder with interaction results
- Ligand-receptor pairs ranked by significance
- Summary report with biological interpretation
