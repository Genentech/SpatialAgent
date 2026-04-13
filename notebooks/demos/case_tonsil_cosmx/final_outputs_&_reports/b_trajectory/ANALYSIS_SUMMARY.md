# Tonsil CosMx Lineage Trajectory Analysis - Final Summary

**Date:** 2026-02-11 02:38  
**Dataset:** CosMx Tonsil (720,497 cells, 990 genes, 25 FOVs)  
**Analysis:** Lineage-focused trajectory inference with spatial organization testing

---

## Executive Summary

This analysis performed rigorous trajectory inference for B-cell and T-cell lineages in healthy human tonsil tissue, followed by within-FOV spatial analysis to test whether differentiation trajectories exhibit spatial organization.

### Key Findings

1. **B-cell Differentiation Trajectory**
   - 300,489 cells analyzed across 7 distinct differentiation states
   - Trajectory: Naive → GC (Early/Late/Exit) → Memory/Plasma
   - Pseudotime validated by canonical markers (TCL1A, BCL6, PRDM1, etc.)
   - Root stability: HIGH (r > 0.9 across alternative roots)

2. **T-cell Differentiation Trajectory**
   - 106,189 cells analyzed across 10 functional states
   - Includes Naive, CD4/CD8 activated, Tfh, Treg, Cytotoxic, Effector Memory
   - Pseudotime validated by T-cell markers (CCR7, CD69, GZMB, PDCD1, etc.)
   - Root stability: MODERATE (r = 0.54-0.84 across alternative roots)

3. **Spatial Organization of Trajectories**
   - **B-cells**: Median spatial correlation r = 0.189 (9/10 FOVs significant, p < 0.05)
   - **T-cells**: Median spatial correlation r = 0.223 (9/10 FOVs significant, p < 0.05)
   - Both lineages show cells at similar differentiation stages cluster spatially
   - Spatial organization reproducible across FOVs

4. **Technical Validation**
   - Minimal QC confounding: B-cell r=0.103, T-cell r=0.020 with n_transcripts
   - Marker expression coherent with known biology
   - Trajectory states align with L2_CT annotations

---

## Analysis Workflow

### 1. Pre-flight Checks ✓
- Coordinates: `obsm['spatial']` (X: 12.6-5449.1, Y: 13.5-3634.4)
- FOVs: 25 fields (mean 28,820 cells/FOV)
- Cell types: L0_CT → L1_CT → L1.5_CT → L2_CT hierarchy
- Expression: `norm_1e4_log1p` layer (log-normalized)
- QC: n_transcripts (mean=124.6), n_genes (mean=73.3)

### 2. Lineage Definition ✓
- **B-cell**: 300,489 cells (41.7% of dataset)
  - B cell Naive (28.7%), B cell GC (43.9%), B cell Memory (13.1%)
  - Plasma cells (13.1%): IGHG, IGHM, IGHA
  
- **T-cell**: 106,189 cells (14.7% of dataset)
  - CD8 Naive (25.8%), CD4 (22.3%), CD4 Fh (15.9%)
  - CD4 Naive (8.7%), CD8 Eff. Mem. (8.2%), CD4 Reg (6.7%)

### 3. Trajectory Inference ✓

**Method**: PAGA + Diffusion Pseudotime (DPT)
- Neighbors: scANVI embedding, k=30
- Leiden clustering: resolution=0.8
- Root selection: Marker-anchored naive populations
- Sensitivity testing: 2-3 alternative roots per lineage

**B-cell Results**:
- 15 Leiden clusters → 7 trajectory states
- Root: Cluster 11 (75.9% naive B cells)
- Pseudotime range: [0.000, 1.000]
- States: Naive, GC_Early, GC_Late, GC_Exit, Memory, Plasmablast, Plasma

**T-cell Results**:
- 14 Leiden clusters → 10 trajectory states
- Root: Cluster 1 (53.3% naive T cells)
- Pseudotime range: [0.000, 1.000]
- States: Naive, CD4_Early, CD4_Activated, CD8_Early, CD8_Activated, Cycling, Regulatory, Tfh, Cytotoxic, Effector_Memory

### 4. Marker Validation ✓

**B-cell markers validated** (18 genes):
- Naive: TCL1A (early peak), IL4R, CD27
- GC: BCL6, AICDA, CD38, CXCR4, CXCR5 (mid-pseudotime)
- Memory: CD27, CD80, CD86
- Plasma: PRDM1, XBP1, MZB1, JCHAIN (late peak)
- Immunoglobulin: IGHG1/2, IGHA1, IGHM (class-switching dynamics)

**T-cell markers validated** (28 genes):
- Naive: CCR7, SELL, TCF7, IL7R (early)
- Activated: CD69, CD44, CD38
- Tfh: CXCR5, PDCD1, ICOS, BCL6, IL21
- Treg: FOXP3, IL2RA, CTLA4
- Effector CD8: GZMA, GZMB, PRF1, NKG7, GNLY
- Exhaustion: PDCD1, CTLA4, HAVCR2, LAG3, TIGIT

### 5. Spatial Analysis ✓

**Method**: Local spatial correlation (k=30 neighbors, 500-cell sample/FOV)
- Computed within-FOV only (no cross-FOV analysis)
- Spearman correlation between cell pseudotime and neighbor mean pseudotime
- Tested on 10 representative FOVs

**Results**:
- B-cell spatial correlation: range [0.041, 0.353], median 0.189
- T-cell spatial correlation: range [0.021, 0.307], median 0.223
- Significant spatial organization in 9/10 FOVs for both lineages
- Indicates cells at similar differentiation stages cluster spatially

---

## Output Files

### Core Data
- `tonsil_with_trajectories.h5ad` - Full dataset with trajectory annotations
- `per_cell_summary.csv` - All cells (720,497 rows)
- `bcell_per_cell_summary.csv` - B-cell subset (300,489 rows)
- `tcell_per_cell_summary.csv` - T-cell subset (106,189 rows)

### Lineage Composition
- `bcell_lineage_composition.csv` - B-cell L2_CT breakdown
- `tcell_lineage_composition.csv` - T-cell L2_CT breakdown
- `bcell_cluster_composition.csv` - Leiden cluster × L2_CT matrix
- `tcell_cluster_composition.csv` - Leiden cluster × L2_CT matrix

### Spatial Analysis
- `bcell_spatial_correlation_sample.csv` - Per-FOV spatial correlation (B-cells)
- `tcell_spatial_correlation_sample.csv` - Per-FOV spatial correlation (T-cells)
- `fov_qc_summary.csv` - QC metrics per FOV

### Sensitivity Analysis
- `bcell_root_sensitivity.csv` - Alternative root testing (B-cells)
- `tcell_root_sensitivity.csv` - Alternative root testing (T-cells)

### Figures (16 total)
- QC: `qc_distributions.png`
- B-cell trajectory: `bcell_paga_graph.png`, `bcell_umap_celltype_clusters.png`, `bcell_pseudotime_overview.png`, `bcell_markers_pseudotime.png`, `bcell_key_markers_trends.png`, `bcell_trajectory_states.png`, `bcell_spatial_organization.png`
- T-cell trajectory: `tcell_paga_graph.png`, `tcell_umap_celltype_clusters.png`, `tcell_pseudotime_overview.png`, `tcell_markers_pseudotime.png`, `tcell_trajectory_states.png`, `tcell_spatial_organization.png`
- Summary: `comprehensive_summary.png`

### Reports
- `deep_research_report.md` / `.html` - Full biological interpretation
- `conclusion_verification.md` - Evidence-based verification

---

## Key Conclusions

### Biological Insights

1. **Tonsil B-cell differentiation follows canonical germinal center dynamics**
   - Naive B cells enter germinal centers (GC_Early)
   - Undergo somatic hypermutation and selection (GC_Late)
   - Exit as memory B cells or differentiate to plasma cells
   - Class-switching evident from Ig heavy chain expression patterns

2. **T-cell compartment shows diverse functional specialization**
   - Naive T cells activate and differentiate along multiple paths
   - Tfh cells likely provide B-cell help in germinal centers
   - Regulatory T cells maintain immune homeostasis
   - Cytotoxic and effector memory populations present

3. **Spatial organization reflects functional tissue architecture**
   - Differentiation stage-specific spatial clustering
   - Consistent across FOVs, suggesting stereotyped tonsil organization
   - Likely reflects germinal center zones, T-cell zones, and interfollicular regions

### Technical Validation

1. **Trajectory inference is robust**
   - B-cell rooting highly stable (r > 0.9)
   - T-cell rooting moderately stable (r = 0.54-0.84)
   - Marker expression validates biological coherence
   - Minimal QC confounding

2. **Spatial analysis is rigorous**
   - Within-FOV analysis avoids cross-FOV artifacts
   - Sampling approach enables analysis of large dataset
   - Reproducible patterns across multiple FOVs
   - Weak-to-moderate correlation (r ~ 0.2) biologically plausible

---

## Limitations

1. **Trajectory inference**
   - T-cell root sensitivity moderate (may benefit from additional validation)
   - Cycling cells treated as separate state (not regressed out)
   - Branching complexity may be underestimated

2. **Spatial analysis**
   - Only 10/25 FOVs analyzed (computational constraints)
   - Distance-dependent analysis not performed
   - Anatomical structures (GC, mantle zones) not explicitly identified
   - Cross-lineage spatial relationships not analyzed

3. **Technical**
   - Limited gene panel (990 genes) may miss some markers
   - Cell segmentation quality not independently validated
   - Single tonsil sample (biological replication needed)

---

## Future Directions

1. **Extended spatial analysis**
   - Analyze all 25 FOVs
   - Distance-dependent correlation (multi-scale analysis)
   - Identify anatomical structures (GC centers, zones)
   - Cross-lineage spatial co-localization

2. **Trajectory refinement**
   - Test additional trajectory methods (RNA velocity, CellRank)
   - Explicit branching analysis
   - Cell cycle regression
   - Trajectory confidence metrics

3. **Biological validation**
   - Compare to additional tonsil samples
   - Integrate with scRNA-seq reference
   - Validate key markers by immunofluorescence
   - Functional enrichment analysis per state

4. **Mechanistic insights**
   - Cell-cell interaction analysis (LIANA)
   - Ligand-receptor pairs along trajectories
   - Niche factor identification
   - Comparison to disease states

---

## Software & Methods

- **Scanpy** 1.11.5: Preprocessing, clustering, UMAP
- **PAGA**: Trajectory graph abstraction
- **Diffusion Pseudotime (DPT)**: Continuous trajectory ordering
- **Spatial analysis**: Custom k-NN correlation (Spearman)
- **Visualization**: Matplotlib, Seaborn

---

## Contact & Reproducibility

All analysis code, intermediate results, and figures are saved in:
`/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5`

For questions or collaboration:
- Dataset: `/scratch/site/u/coelhop4/TLS_tonsils/hs_tonsil_cosmx/CosMx_Tonsil_scANVI_harmonized_final.h5ad`
- Output: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5`

---

*Analysis completed: 2026-02-11 02:38*
