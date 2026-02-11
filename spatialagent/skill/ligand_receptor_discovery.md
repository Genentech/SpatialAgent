# Ligand-Receptor Interaction Discovery

Comprehensive workflow for discovering and validating ligand-receptor interactions in single-cell and spatial data.

## When to Use

- Identify communication pathways between cell types
- Find immune checkpoint interactions
- Discover growth factor signaling networks
- Validate known pathways in new datasets

---

## Workflow Overview

1. **Data preparation** - Ensure proper gene symbols and cell annotations
2. **Run CellPhoneDB** - Statistical interaction prediction
3. **Filter by biology** - Focus on pathways of interest
4. **Cross-validate** - Compare with literature/databases
5. **Spatial validation** - Check colocalization (if spatial data)

---

## Step 1: Data Preparation

### Check gene symbols
```python
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")

# Check if genes are human symbols
sample_genes = adata.var_names[:10].tolist()
print(f"Sample genes: {sample_genes}")

# Convert mouse to human if needed (using gget)
import gget
# gget.orthologs() for conversion
```

### Verify cell annotations
```python
print(f"Cell types: {adata.obs['cell_type'].value_counts()}")
# Should have clear, meaningful cell type labels
```

---

## Step 2: Run CellPhoneDB Analysis

**Tool**: `cellphonedb_prepare`
- Prepare metadata and counts files

**Tool**: `cellphonedb_analysis`
- Run permutation-based analysis
- Default 1000 iterations recommended

---

## Step 3: Filter by Biological Interest

### Option A: Filter by cell types
**Tool**: `cellphonedb_filter`
```
cell_types: "T cell,Tumor cell"
```

### Option B: Filter by specific genes/pathways
**Tool**: `cellphonedb_filter`
```
genes: "PD1,PDL1,CTLA4,CD28,CD80,CD86"  # Immune checkpoints
```

### Option C: Filter by expression level
**Tool**: `cellphonedb_filter`
```
min_mean: 0.5  # Only highly expressed interactions
```

---

## Step 4: Common Pathway Queries

### Immune Checkpoints
```
genes: "PDCD1,CD274,CTLA4,CD28,CD80,CD86,LAG3,HAVCR2,TIGIT"
```

### Growth Factors
```
genes: "EGF,EGFR,VEGFA,VEGFR,FGF,FGFR,PDGF,PDGFR,HGF,MET"
```

### Chemokines
```
genes: "CXCL,CCL,CXCR,CCR"  # Partial match works
```

### Notch Signaling
```
genes: "NOTCH1,NOTCH2,JAG1,JAG2,DLL1,DLL4"
```

### Wnt Signaling
```
genes: "WNT,FZD,LRP5,LRP6"
```

---

## Step 5: Visualize Results

**Tool**: `cellphonedb_plot`
- `plot_type: "dotplot"` - See expression levels across cell pairs
- `plot_type: "heatmap"` - Overview of interaction counts
- `plot_type: "chord"` - Network view of interactions

---

## Step 6: Spatial Validation (if applicable)

If you have spatial data, validate colocalization:

**Tool**: `squidpy_spatial_neighbors`
- Build spatial neighbors graph first
- Use `coord_type="generic"` for most spatial data

**Tool**: `squidpy_nhood_enrichment`
- Check if interacting cell types are colocalized
- Positive z-scores indicate cell types neighbor more than expected

---

## Output Interpretation

### significant_means.csv columns:
- `interacting_pair`: Ligand_Receptor format
- `partner_a/b`: Gene partners in interaction
- `gene_a/b`: Specific genes involved
- `secreted`: Whether ligand is secreted
- `receptor_a/b`: Which partner is receptor
- `CellTypeA|CellTypeB`: Mean expression (source|target)

### Key metrics:
- **Mean > 0**: Interaction is expressed
- **P-value < 0.05**: Interaction is specific to this cell pair
- **Higher mean + lower p-value = stronger evidence**

---

## Tips

1. **Secreted vs membrane-bound**: Secreted ligands can act at distance; membrane-bound require contact.

2. **Directionality matters**: Source cell expresses ligand, target expresses receptor.

3. **Complex interactions**: Some interactions involve multi-subunit complexes (check deconvoluted.csv).

4. **False positives**: High expression doesn't guarantee functional signaling. Consider downstream validation.
