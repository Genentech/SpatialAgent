# CellPhoneDB Cell-Cell Communication Analysis

Predict ligand-receptor interactions between cell types using CellPhoneDB's curated database and statistical framework.

## Prerequisites

**Required**:
- Single-cell or spatial transcriptomics data (h5ad format)
- Cell type annotations in `obs`
- Human gene symbols (convert orthologs if using mouse/other species)

**Optional**:
- Differentially expressed genes (for DEG-based analysis)
- Microenvironment annotations (for spatial context)

---

## Workflow Overview

1. **Explore dataset structure** (identify cell type column)
2. Prepare data for CellPhoneDB
3. Choose analysis method (Simple, Statistical, or DEG-based)
4. Run interaction analysis
5. Filter and search results
6. Visualize interactions

---

## Step 1: Explore Dataset

Use `execute_python` to identify the correct column names:

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/data.h5ad")

print(f"Shape: {adata.shape}")
print(f"obs columns: {list(adata.obs.columns)}")

# Find cell type column
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    if 5 < n_unique < 50:  # Likely cell type column
        print(f"{col}: {adata.obs[col].unique().tolist()}")
```

---

## Step 2: Prepare Data

**Tool**: `cellphonedb_prepare`
- `adata_path`: Path to h5ad file
- `cell_type_key`: Cell type column identified in Step 1
- `layer`: Optional - specific layer for counts (empty for .X)
- `save_path`: Output directory

Creates:
- `cellphonedb_meta.txt`: Cell-to-cell type mapping
- `cellphonedb_counts.h5ad`: Expression matrix

---

## Step 3: Choose Analysis Method

CellPhoneDB offers three analysis methods:

### METHOD 1: Simple Analysis (No statistics)
Fast analysis returning mean ligand-receptor expression without statistical testing.

**Tool**: `cellphonedb_analysis` with `iterations=0`
- `iterations`: Set to 0 for simple analysis
- `threshold`: 0.1 (min fraction of cells expressing gene)
- `score_interactions`: True to score interactions by specificity (0-10 scale)

Output: `means.csv`, `deconvoluted.csv`, `interaction_scores.csv` (if scoring enabled)

### METHOD 2: Statistical Analysis (Recommended for discovery)
Uses permutation testing to identify significantly enriched interactions.

**Tool**: `cellphonedb_analysis`
- `iterations`: 1000 (default, increase for more precision)
- `threshold`: 0.1 (min fraction of cells expressing gene)
- `microenvs_path`: Optional - restrict analysis to colocalized cells
- `score_interactions`: True to score interactions by specificity
- `threads`: 4 (increase for faster analysis)

Output: `means.csv`, `pvalues.csv`, `significant_means.csv`, `deconvoluted.csv`

### METHOD 3: DEG-Based Analysis (For targeted studies)
Focuses on interactions involving differentially expressed genes. Returns only relevant interactions where at least one partner is a DEG.

First, create DEG file (format: two columns - `Cell`, `Gene`):
```python
import pandas as pd
import scanpy as sc

# Get DEGs for each cell type
degs = []
for ct in adata.obs['cell_type'].unique():
    markers = sc.get.rank_genes_groups_df(adata, group=ct)
    top_genes = markers[markers['pvals_adj'] < 0.05].head(50)['names']
    for gene in top_genes:
        degs.append({'Cell': ct, 'Gene': gene})

pd.DataFrame(degs).to_csv('degs.txt', sep='\t', index=False)
```

**Tool**: `cellphonedb_degs_analysis`
- `degs_path`: Path to DEG file (two columns: Cell, Gene)
- `threshold`: 0.1 (min fraction of cells expressing gene)
- `microenvs_path`: Optional - restrict analysis to colocalized cells
- `score_interactions`: True to score interactions by specificity
- `threads`: 4

Output: `relevant_interactions.txt`, `significant_means.csv`, `means.csv`, `deconvoluted.csv`

---

## Microenvironments (Spatial Context)

To restrict analysis to spatially colocalized cell types, create a microenvironment file:

```python
# Format: two columns - Cell (barcode), Microenvironment (region name)
microenvs = pd.DataFrame({
    'Cell': adata.obs_names,
    'Microenvironment': adata.obs['spatial_region']  # your region column
})
microenvs.to_csv('microenvs.txt', sep='\t', index=False)
```

Then pass `microenvs_path` to analysis tools. Only interactions between cell types in the same microenvironment will be considered.

---

## Step 4: Search and Filter Results

**Tool**: `cellphonedb_filter`
- `cell_types`: Filter by specific cell types (e.g., "T cell,Macrophage")
- `genes`: Filter by specific genes (e.g., "CD40,CD40LG")
- `min_mean`: Minimum expression threshold

---

## Step 5: Visualize Results

**Tool**: `cellphonedb_plot`
- `plot_type`: "dotplot" (expression heatmap), "heatmap" (interaction counts), or "chord" (network)
- `cell_types`: Focus on specific cell types
- `top_n`: Number of top interactions to show

---

## Output Files

**Simple analysis (METHOD 1)**:
- `means.csv`: Mean expression for all interactions
- `deconvoluted.csv`: Subunit details for complexes
- `interaction_scores.csv`: Specificity scores (if enabled)

**Statistical analysis (METHOD 2)**:
- `means.csv`: Mean expression for all interactions
- `pvalues.csv`: P-values from permutation testing
- `significant_means.csv`: Filtered significant interactions
- `deconvoluted.csv`: Subunit details for complexes
- `interaction_scores.csv`: Specificity scores (if enabled)

**DEG analysis (METHOD 3)**:
- `relevant_interactions.txt`: Binary matrix (1 = DEG-related interaction)
- `significant_means.csv`: Mean values for relevant interactions
- `means.csv`: Mean expression for all tested interactions
- `deconvoluted.csv`: Subunit details for complexes

---

## Tips

1. **Gene symbols**: CellPhoneDB requires human gene symbols. Convert mouse genes using gget or biomart.

2. **Cell type naming**: Avoid numeric names and dashes in cell type labels.

3. **Directionality**: Interactions are directional (A->B != B->A). The format is "source|target".

4. **Interpretation**: High mean + low p-value = strong, specific interaction.

5. **Score interactions**: Use `score_interactions=True` to rank interactions by specificity (0-10 scale).

6. **Spatial context**: Use `microenvs_path` to focus on spatially relevant interactions.
