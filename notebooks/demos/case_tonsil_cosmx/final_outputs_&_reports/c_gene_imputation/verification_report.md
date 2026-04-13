# Verification & Robustness Analysis

## Overview

This report provides structured verification of key biological claims from the gene imputation analysis.

## Claim 1

**Statement:** Imputed genes show strong spatial structure: Mean Moran's I = 0.022 with 87.7% of gene-FOV pairs significant (p<0.05)

**Evidence:**
imputed_unmeasured_spatial.tsv: 1000 gene-FOV pairs, mean I=0.022, 877 significant

**Potential Confounders:**
Spatial smoothing inherent to imputation; cluster-based mapping may inflate autocorrelation

**Verdict:** **MODERATE**

---

## Claim 2

**Statement:** Top imputed genes (RGS13, IRAG2, H2AZ2, DUT, DEK) are enriched in lymphoid follicles with log2FC > 0.3 and q<0.001

**Evidence:**
imputed_unmeasured_follicle_enrichment.tsv: 5/5 genes with log2FC>0.3, all q<0.001

**Potential Confounders:**
Follicle regions may have higher cell density affecting imputation; binary classification (follicle/none) oversimplifies spatial gradients

**Verdict:** **STRONG**

---

## Claim 3

**Statement:** Imputed genes reveal GC B cell-specific programs: 13/15 top cell-type-specific genes are B cell GC markers

**Evidence:**
imputed_unmeasured_celltype_specificity.tsv: 14/15 top genes are B cell GC-specific

**Potential Confounders:**
B cell GC is largest cell type (27K cells); reference may be biased toward GC signatures; specificity score depends on denominator choice

**Verdict:** **STRONG**

---

## Claim 4

**Statement:** Measured gene reconstruction quality: mean cosine similarity 0.303, consistent across cell types (0.48-0.58)

**Evidence:**
qc_holdout_reconstruction.tsv: 127 genes, mean=0.303; qc_celltype_retention.tsv: range 0.33-0.58

**Potential Confounders:**
Modest reconstruction accuracy (0.3) indicates directional but not quantitative precision; cluster-based mapping loses single-cell resolution

**Verdict:** **WEAK**

---

## Claim 5

**Statement:** Imputed genes identify metabolic specialization in GC B cells: mitochondrial genes (ATP5MC3, COX5A, COX8A) show 2-fold enrichment

**Evidence:**
imputed_unmeasured_celltype_specificity.tsv: 4/4 mito genes with specificity>1.8 in B cell GC

**Potential Confounders:**
Mitochondrial genes are housekeeping; enrichment may reflect cell size/metabolic rate rather than GC-specific program; need validation with metabolic assays

**Verdict:** **MODERATE**

---

## Summary

- **STRONG** claims: 2
- **MODERATE** claims: 2
- **WEAK** claims: 1

## Recommendations

1. **STRONG claims** are well-supported and can be reported with confidence
2. **MODERATE claims** should be reported with caveats and confounders acknowledged
3. **WEAK claims** require additional validation before publication
