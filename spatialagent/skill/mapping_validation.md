# Tangram Mapping Validation

Validate and troubleshoot Tangram spatial mappings using cross-validation, diagnostic plots, and quality metrics.

## Why Validate?

- Ensure mapping is biologically meaningful
- Identify problematic genes or cell types
- Compare different mapping strategies
- Build confidence before downstream analysis

---

## Validation Approaches

1. **Training score analysis**: How well do training genes match?
2. **Test gene prediction**: Can we predict held-out genes?
3. **Cross-validation**: Systematic holdout testing
4. **Biological validation**: Do patterns match known biology?

---

## Step 1: Basic Quality Check

After running `tangram_map_cells`, check training scores:

```python
import scanpy as sc
import tangram as tg

ad_map = sc.read_h5ad("experiments/tangram_mapping.h5ad")

# Training gene scores
train_df = ad_map.uns["train_genes_df"]
print(f"Average training score: {train_df['train_score'].mean():.3f}")
print(f"Median training score: {train_df['train_score'].median():.3f}")

# Score distribution
print("\nScore distribution:")
print(train_df['train_score'].describe())

# Top and bottom genes
print("\nTop 10 genes:")
print(train_df.head(10))
print("\nBottom 10 genes:")
print(train_df.tail(10))
```

**Interpretation**:
- Average > 0.7: Good mapping
- Average 0.5-0.7: Moderate, may need tuning
- Average < 0.5: Poor, check data compatibility

---

## Step 2: Training Diagnostic Plots

**Tool**: `tangram_evaluate`

Or manually generate plots:

```python
import tangram as tg
import matplotlib.pyplot as plt

# 4-panel diagnostic plot
tg.plot_training_scores(ad_map, bins=20, alpha=0.5)
plt.savefig("training_diagnostics.png", dpi=150, bbox_inches="tight")
```

**Panels explained**:
1. **Score histogram**: Distribution of gene scores
2. **Score vs SC sparsity**: Low sparsity genes should score high
3. **Score vs SP sparsity**: Sparse spatial genes often score low
4. **Score vs sparsity diff**: Large diff = dropout mismatch

---

## Step 3: Test Gene Evaluation

Project genes and evaluate predictions:

```python
# Project genes
ad_ge = tg.project_genes(ad_map, adata_sc)
ad_ge.write_h5ad("experiments/tangram_projected.h5ad")

# Compare with spatial data
df_genes = tg.compare_spatial_geneexp(ad_ge, adata_sp, adata_sc)

# Separate training and test
train_genes = df_genes[df_genes["is_training"] == True]
test_genes = df_genes[df_genes["is_training"] == False]

print(f"Training genes: {len(train_genes)}, avg score: {train_genes['score'].mean():.3f}")
print(f"Test genes: {len(test_genes)}, avg score: {test_genes['score'].mean():.3f}")
```

---

## Step 4: AUC Evaluation

The AUC metric captures overall prediction quality:

```python
# Compute AUC
metrics, coords = tg.eval_metric(df_genes)
print(f"AUC score: {metrics['auc_score']:.3f}")

# Plot AUC curve
tg.plot_auc(df_genes)
plt.savefig("auc_curve.png", dpi=150)
```

**Interpretation**:
- AUC > 0.6: Good prediction on test genes
- AUC 0.4-0.6: Moderate
- AUC < 0.4: Poor, mapping may not generalize

---

## Step 5: Cross-Validation (Optional)

For rigorous validation, use leave-one-out or k-fold CV:

```python
# Leave-one-out CV (slow but thorough)
cv_dict = tg.cross_val(
    adata_sc,
    adata_sp,
    cluster_label="cell_type",
    mode="clusters",
    cv_mode="loo",  # or "10fold"
    num_epochs=500,
    device="cuda:0",
    verbose=True,
)

print(f"CV avg train score: {cv_dict['avg_train_score']:.3f}")
print(f"CV avg test score: {cv_dict['avg_test_score']:.3f}")
```

---

## Step 6: Biological Validation

Check that mapped cell types match expected patterns:

```python
import scanpy as sc
import matplotlib.pyplot as plt

adata_sp = sc.read_h5ad("experiments/tangram_annotated.h5ad")

# Known spatial patterns to verify
# Example: T cells should be in lymphoid regions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

cell_types = ['T cells', 'B cells', 'Macrophages',
              'Epithelial', 'Fibroblasts', 'Endothelial']

for ax, ct in zip(axes.flat, cell_types):
    if ct in adata_sp.obsm["tangram_ct_pred"].columns:
        adata_sp.obs["_tmp"] = adata_sp.obsm["tangram_ct_pred"][ct]
        sc.pl.spatial(adata_sp, color="_tmp", ax=ax, show=False, title=ct)
        del adata_sp.obs["_tmp"]

plt.tight_layout()
plt.savefig("celltype_validation.png", dpi=150)
```

---

## Troubleshooting Low Scores

### Problem: Low training scores

**Causes**:
- Few shared genes between datasets
- Different normalization
- Batch effects

**Solutions**:
```python
# Check gene overlap
sc_genes = set(adata_sc.var_names.str.lower())
sp_genes = set(adata_sp.var_names.str.lower())
overlap = sc_genes & sp_genes
print(f"Shared genes: {len(overlap)}")

# Ensure same normalization
print(f"SC max: {adata_sc.X.max():.2f}")
print(f"SP max: {adata_sp.X.max():.2f}")
```

### Problem: Good training but poor test scores

**Causes**:
- Overfitting to training genes
- Training genes not representative

**Solutions**:
- Use more diverse training genes
- Increase regularization (lambda_r)
- Try `mode="clusters"` for smoother mapping

### Problem: Some cell types poorly mapped

**Causes**:
- Cell type not in spatial region
- Few marker genes for that type

**Solutions**:
- Check cell type abundance in spatial data
- Add more markers for underperforming types

---

## Comparing Mapping Strategies

Test different approaches:

```python
results = []

# Strategy 1: Cells mode
ad_map1 = tg.map_cells_to_space(adata_sc, adata_sp, mode="cells", num_epochs=500)
score1 = ad_map1.uns["train_genes_df"]["train_score"].mean()
results.append(("cells", score1))

# Strategy 2: Clusters mode
ad_map2 = tg.map_cells_to_space(adata_sc, adata_sp, mode="clusters",
                                 cluster_label="cell_type", num_epochs=500)
score2 = ad_map2.uns["train_genes_df"]["train_score"].mean()
results.append(("clusters", score2))

# Compare
for mode, score in results:
    print(f"{mode}: {score:.3f}")
```

---

## Quality Checklist

Before using mapping results:

- [ ] Average training score > 0.7
- [ ] AUC score > 0.5
- [ ] Top genes include expected markers
- [ ] Cell type patterns match known biology
- [ ] No obvious artifacts in spatial plots
- [ ] Sparsity analysis shows expected dropout pattern
