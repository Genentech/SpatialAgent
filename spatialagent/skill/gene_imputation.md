# Gene Expression Imputation with Tangram

Use Tangram to impute and correct gene expression in spatial transcriptomics data by leveraging scRNA-seq profiles.

## Use Cases

- **Visualize genes not in spatial panel**: Project any gene from scRNA-seq
- **Correct dropout artifacts**: Fix missing values in spatial measurements
- **Expand gene coverage**: Go from limited spatial panel to full transcriptome

---

## Workflow Overview

1. Explore datasets and identify target genes
2. Preprocess data for Tangram
3. Map cells to space
4. Project gene expression
5. Compare measured vs predicted
6. Visualize imputed genes

---

## Step 1: Explore and Plan

Identify genes of interest not well-measured in spatial data:

```python
import scanpy as sc
import numpy as np

adata_sc = sc.read_h5ad("scrna.h5ad")
adata_sp = sc.read_h5ad("spatial.h5ad")

# Find genes in scRNA-seq but sparse in spatial
sc_genes = set(adata_sc.var_names)
sp_genes = set(adata_sp.var_names)

# Genes only in scRNA-seq (can be imputed)
imputable = sc_genes - sp_genes
print(f"Genes only in scRNA-seq: {len(imputable)}")

# Genes in both but sparse in spatial (can be corrected)
shared = sc_genes & sp_genes
for gene in ['CD3E', 'CD8A', 'FOXP3']:  # example genes
    if gene in shared:
        sp_expr = adata_sp[:, gene].X
        sparsity = (sp_expr == 0).sum() / len(sp_expr)
        print(f"{gene}: {sparsity:.1%} zero in spatial")
```

---

## Step 2: Preprocess

**Tool**: `tangram_preprocess`

Use cell type markers as training genes:

```
tangram_preprocess(
    adata_sc_path="scrna.h5ad",
    adata_sp_path="spatial.h5ad",
    marker_genes="auto",
    cell_type_key="cell_type"
)
```

---

## Step 3: Map Cells

**Tool**: `tangram_map_cells`

For gene imputation, `mode="cells"` gives higher resolution:

```
tangram_map_cells(
    adata_sc_path="experiments/tangram_sc_prep.h5ad",
    adata_sp_path="experiments/tangram_sp_prep.h5ad",
    mode="cells",
    device="cuda:0",  # GPU recommended
    num_epochs=500
)
```

---

## Step 4: Project Genes

**Tool**: `tangram_project_genes`

This creates imputed expression for ALL genes in scRNA-seq:

```
tangram_project_genes(
    adata_map_path="experiments/tangram_mapping.h5ad",
    adata_sc_path="scrna.h5ad"
)
```

---

## Step 5: Evaluate Imputation Quality

**Tool**: `tangram_evaluate`

Compare predicted vs measured for shared genes:

```
tangram_evaluate(
    adata_ge_path="experiments/tangram_projected.h5ad",
    adata_sp_path="spatial.h5ad",
    adata_sc_path="scrna.h5ad"
)
```

**Interpretation**:
- High scores for shared genes = reliable imputation for missing genes
- Check sparsity: genes sparse in spatial but not in scRNA-seq benefit most

---

## Step 6: Visualize Imputed Genes

```python
import scanpy as sc
import matplotlib.pyplot as plt

# Load projected data
adata_ge = sc.read_h5ad("experiments/tangram_projected.h5ad")
adata_sp = sc.read_h5ad("spatial.h5ad")

# Gene not in spatial panel
gene = "FOXP3"  # Treg marker, often missing in spatial

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Measured (may be all zeros)
if gene in adata_sp.var_names:
    sc.pl.spatial(adata_sp, color=gene, ax=axes[0], show=False, title=f"{gene} (measured)")
else:
    axes[0].text(0.5, 0.5, "Not in panel", ha='center')
    axes[0].set_title(f"{gene} (measured)")

# Imputed from scRNA-seq
sc.pl.spatial(adata_ge, color=gene.lower(), ax=axes[1], show=False, title=f"{gene} (imputed)")
plt.savefig("imputed_gene.png", dpi=150)
```

---

## Comparing Measured vs Imputed

For genes in both datasets, compare quality:

```python
import tangram as tg

# Load data
adata_ge = sc.read_h5ad("experiments/tangram_projected.h5ad")
adata_sp = sc.read_h5ad("spatial.h5ad")

# Compare specific genes
genes = ['cd3e', 'cd8a', 'cd4']  # lowercase for tangram
tg.plot_genes_sc(genes, adata_measured=adata_sp, adata_predicted=adata_ge, perc=0.02)
plt.savefig("gene_comparison.png", dpi=150)
```

---

## Output Files

- `tangram_projected.h5ad` - Full imputed transcriptome
- `tangram_scores.csv` - Per-gene imputation quality
- Visualization plots comparing measured vs imputed

---

## Tips

1. **Training genes matter**: Use cell type markers for best mapping
2. **Check sparsity**: Imputation works best when gene is sparse in spatial but not in scRNA-seq
3. **Validate with known patterns**: Check that imputed genes show expected spatial patterns
4. **Use for discovery**: Impute pathway genes, transcription factors not in spatial panel
