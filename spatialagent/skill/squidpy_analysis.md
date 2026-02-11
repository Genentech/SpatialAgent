# Squidpy Spatial Analysis

Comprehensive spatial analysis tools for single-cell and spatial transcriptomics data using Squidpy.

## When to Use Squidpy

| Analysis Type | Tool |
|---------------|------|
| Build spatial graph | `squidpy_spatial_neighbors` |
| Cell type colocalization | `squidpy_nhood_enrichment` |
| Co-occurrence patterns | `squidpy_co_occurrence` |
| Spatially variable genes | `squidpy_spatial_autocorr` |
| Point pattern analysis | `squidpy_ripley` |
| Network topology | `squidpy_centrality` |
| Cell-cell interactions | `squidpy_ligrec` |

---

## Workflow Overview

1. **Build spatial neighbors graph** (required first step)
2. **Analyze spatial patterns** (choose based on question)
3. **Interpret results**

---

## Step 1: Build Spatial Neighbors Graph

**Tool**: `squidpy_spatial_neighbors`

This is required before any other Squidpy analysis.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coord_type` | `"visium"` (hex), `"grid"` (square), or `"generic"` (any) | `"generic"` |
| `n_neighs` | Number of neighbors for generic/KNN | 6 |
| `n_rings` | Number of hex/grid rings for visium/grid | 1 |
| `delaunay` | Use Delaunay triangulation | False |
| `radius` | Radius cutoff (0 = disabled) | 0 |

### Tips

- For **Visium data**: Use `coord_type="visium"`, `n_rings=1` (6 neighbors)
- For **generic spatial data**: Use `coord_type="generic"`, `n_neighs=6-10`
- For **Delaunay triangulation**: Set `delaunay=True` for natural neighbor connections
- For **radius-based neighbors**: Set `radius` to distance threshold (e.g., 100 microns)

---

## Step 2: Choose Analysis Based on Question

### A. Are certain cell types colocalized?

**Tool**: `squidpy_nhood_enrichment`

Tests whether cells of one type are more/less likely to neighbor cells of another type.

**Output**:
- Z-scores: positive = enriched, negative = depleted
- Heatmap showing cell type interactions

**Example questions**:
- "Are T cells enriched near tumor cells?"
- "Which cell types cluster together?"

---

### B. Do cell types co-occur spatially?

**Tool**: `squidpy_co_occurrence`

Measures co-occurrence probability at different spatial distances.

**Parameters**:
- `interval`: Number of distance bins (default 50)
- `n_splits`: Split for computation (default 2)

**Output**:
- Co-occurrence scores per distance
- Shows spatial range of interactions

**Example questions**:
- "At what distance do T cells and tumor cells interact?"
- "How does co-occurrence change with distance?"

---

### C. Which genes are spatially variable?

**Tool**: `squidpy_spatial_autocorr`

Identifies genes with spatial patterns using Moran's I or Geary's C statistics.

**Parameters**:
- `mode`: `"moran"` (default) or `"geary"`
- `genes`: List of genes or "highly_variable" or None (all)
- `n_perms`: Permutations for p-value (default 100)
- `n_jobs`: Parallel jobs (default 1)

**Output**:
- Moran's I: 1 = clustered, 0 = random, -1 = dispersed
- Geary's C: 0 = clustered, 1 = random
- p-values from permutation test

**Example questions**:
- "Which genes show spatial clustering?"
- "Are marker genes spatially organized?"

---

### D. What are the spatial patterns? (Point Process)

**Tool**: `squidpy_ripley`

Characterizes point patterns using Ripley's statistics.

**Parameters**:
- `mode`: `"F"`, `"G"`, or `"L"`
  - F: Empty space function
  - G: Nearest neighbor distribution
  - L: Ripley's L (cluster detection)
- `n_simulations`: Bootstrap simulations (default 100)

**Output**:
- L > 0: Clustering at that distance
- L < 0: Dispersion/regularity
- L = 0: Complete spatial randomness

**Example questions**:
- "Are tumor cells clustered or dispersed?"
- "At what scale do clusters form?"

---

### E. What is the network structure?

**Tool**: `squidpy_centrality`

Computes graph centrality metrics per cell type.

**Parameters**:
- `mode`: `"closeness"` or `"degree"`
  - closeness: How central in the network
  - degree: Number of connections

**Output**:
- Centrality scores per cell type
- Identifies spatially central vs peripheral populations

**Example questions**:
- "Which cell types are most central?"
- "Are tumor cells at the network periphery?"

---

### F. How do cell types interact physically?

**Tool**: `squidpy_interaction_matrix`

Computes cell-cell contact/interaction matrix.

**Parameters**:
- `normalized`: Normalize by cell counts (default True)

**Output**:
- Interaction counts/frequencies between all cell type pairs

---

### G. What ligand-receptor interactions occur?

**Tool**: `squidpy_ligrec`

Permutation-based ligand-receptor analysis with spatial context.

**Parameters**:
- `n_perms`: Number of permutations (default 1000)
- `threshold`: Expression threshold (default 0.01)
- `corr_method`: Multiple testing correction (default "fdr_bh")
- `gene_symbols`: Column with gene symbols (optional)

**Output**:
- Significant LR pairs between cell types
- P-values from permutation test
- Mean expression values

---

## Complete Workflow Examples

### Example 1: Tumor Microenvironment Analysis

```
1. squidpy_spatial_neighbors(coord_type="generic", n_neighs=6)
2. squidpy_nhood_enrichment(cluster_key="cell_type")
   → Find which immune cells infiltrate tumor
3. squidpy_co_occurrence(cluster_key="cell_type")
   → Measure distance-dependent interactions
4. squidpy_ligrec(cluster_key="cell_type")
   → Identify signaling between tumor and immune cells
```

### Example 2: Spatial Gene Expression Analysis

```
1. squidpy_spatial_neighbors(coord_type="visium", n_rings=1)
2. squidpy_spatial_autocorr(mode="moran", genes="highly_variable")
   → Find spatially variable genes
3. squidpy_ripley(cluster_key="cell_type", mode="L")
   → Characterize spatial clustering
```

### Example 3: Tissue Organization Analysis

```
1. squidpy_spatial_neighbors(coord_type="generic", delaunay=True)
2. squidpy_centrality(cluster_key="cell_type", mode="closeness")
   → Find central cell populations
3. squidpy_nhood_enrichment(cluster_key="cell_type")
   → Identify neighborhood preferences
```

---

## Output Files

All tools save results to the specified `save_path`:

| Tool | Outputs |
|------|---------|
| `squidpy_spatial_neighbors` | Updated AnnData with spatial graph |
| `squidpy_nhood_enrichment` | Heatmap PNG, Z-scores CSV |
| `squidpy_co_occurrence` | Co-occurrence plot PNG, scores CSV |
| `squidpy_spatial_autocorr` | Autocorrelation results CSV |
| `squidpy_ripley` | Ripley's statistics plot PNG, CSV |
| `squidpy_centrality` | Centrality scores CSV |
| `squidpy_interaction_matrix` | Interaction matrix PNG, CSV |
| `squidpy_ligrec` | LR results CSV, dotplot PNG |

---

## Tips

1. **Always run `squidpy_spatial_neighbors` first** - Other tools depend on the spatial graph.

2. **Choose coord_type carefully**:
   - Visium: Use `"visium"` for hex grid
   - Slide-seq, MERFISH: Use `"generic"`

3. **Moran's I interpretation**:
   - High positive I → Gene is spatially clustered
   - Near zero → Random distribution
   - High negative I → Gene is spatially dispersed (rare)

4. **Neighborhood enrichment interpretation**:
   - Positive z-score → Cell types colocalize more than expected
   - Negative z-score → Cell types avoid each other
   - Near zero → Random spatial distribution

5. **For large datasets**: Use `n_jobs > 1` for parallel computation in spatial_autocorr.
