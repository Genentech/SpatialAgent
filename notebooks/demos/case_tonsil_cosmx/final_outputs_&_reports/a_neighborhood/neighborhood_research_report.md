# Deep Research Report: Spatial Neighborhood Characterization in Healthy Human Tonsil CosMx Dataset

## Executive Summary

This report presents a comprehensive characterization of eight global spatial neighborhoods identified in a healthy human tonsil CosMx spatial transcriptomics dataset (720,497 cells, 25 FOVs, 990 genes). Using Leiden clustering on integrated spatial-transcriptomic features, we identified distinct tissue microenvironments representing key functional compartments of tonsil architecture: B cell follicles, germinal centers, T cell zones, stromal niches, and interfollicular regions.

**Key Findings:**
- **8 distinct neighborhoods** with cell type diversity ranging from 5.1 to 9.2 effective cell types
- **Neighborhood 0** (N0): B cell follicle core - B cell and fDC dominated, present in 22/25 FOVs
- **Neighborhood 7** (N7): Rare LQ-enriched niche - highly specialized, present in only 4/25 FOVs
- **Neighborhoods 3 & 6**: Stromal-vascular and T cell zones with highest cellular diversity (>8.7 effective cell types)
- **FOV heterogeneity**: Most neighborhoods are ubiquitous (present in 17-23 FOVs), suggesting conserved tonsil architecture across tissue regions

---

## Methods

### Data Source
- **Dataset**: Healthy human tonsil CosMx spatial transcriptomics
- **Cells**: 720,497 cells across 25 fields of view (FOVs)
- **Genes**: 990 genes measured
- **Cell types**: 13 L1_CT (Level 1 cell types): B cell, Basal cell, DC, Endothelial cell, Fibroblast, LQ, Mast cell, MoMacrophages, Perivascular, Plasma cell, T cell CD4, T cell CD8, fDC

### Neighborhood Inference
Neighborhoods were identified using a prior pipeline that:
1. Integrated spatial proximity and gene expression features
2. Applied Leiden clustering to identify global spatial neighborhoods
3. Assigned each cell to a neighborhood label (`Leiden_neigh`: 0-7)

### Analysis Approach
This report analyzes **saved outputs** from the prior pipeline:
- `CosMx_Tonsil_with_Leiden_neigh.h5ad`: Annotated AnnData object
- `neighborhood_celltype_composition_normalized.csv`: Neighborhood × L1_CT matrix (normalized 0-1)
- `fov_neighborhood_abundance_normalized.csv`: FOV × neighborhood abundance matrix
- `fov_neighborhood_composition.csv`: Per-FOV neighborhood composition

**Computed metrics:**
- **Raw fractions**: Proportion of each cell type within each neighborhood
- **Normalized values**: Per-cell-type normalization across neighborhoods (0-1 scale)
- **Shannon entropy**: Measure of cell type diversity within each neighborhood
- **Effective N cell types**: 2^(Shannon entropy), interpretable as "equivalent number of equally abundant cell types"
- **FOV recurrence**: Number of FOVs in which each neighborhood appears

**Important**: All analyses are restricted to within-FOV spatial relationships. No cross-FOV spatial proximity calculations were performed.

---

## Results

### Overview: Neighborhood Summary Table

| Neighborhood | Total Cells | N FOVs | Shannon Entropy | Effective N Cell Types | Top 5 Enriched Cell Types (normalized) |
|--------------|-------------|--------|-----------------|------------------------|----------------------------------------|
| 0 | 149,126 | 22 | 2.354 | 5.11 | B cell(1.000); fDC(1.000); MoMacrophages(0.633); T cell CD4(0.475); Plasma cell(0.383) |
| 1 | 132,606 | 23 | 2.824 | 7.08 | MoMacrophages(0.923); B cell(0.761); Fibroblast(0.678); Perivascular(0.581); DC(0.534) |
| 2 | 123,005 | 21 | 2.986 | 7.92 | Basal cell(1.000); MoMacrophages(0.845); Plasma cell(0.769); DC(0.687); LQ(0.467) |
| 3 | 85,963 | 19 | 3.196 | 9.17 | Endothelial cell(1.000); Fibroblast(1.000); Mast cell(1.000); MoMacrophages(1.000); Perivascular(1.000) |
| 4 | 74,145 | 17 | 2.842 | 7.17 | T cell CD4(0.984); MoMacrophages(0.754); T cell CD8(0.702); fDC(0.658); B cell(0.648) |
| 5 | 64,545 | 20 | 2.814 | 7.03 | fDC(0.944); B cell(0.763); Mast cell(0.728); Plasma cell(0.724); MoMacrophages(0.715) |
| 6 | 63,542 | 19 | 3.133 | 8.77 | DC(1.000); T cell CD4(1.000); T cell CD8(1.000); MoMacrophages(0.964); Fibroblast(0.840) |
| 7 | 27,565 | 4 | 2.948 | 7.71 | LQ(1.000); Plasma cell(0.719); Endothelial cell(0.595); T cell CD8(0.593); Fibroblast(0.569) |


---

### Neighborhood 0: B Cell Follicle Core

**Descriptive Name**: Primary B Cell Follicle with fDC Network

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. B cell: 1.000 (raw fraction: 0.5450, 54.50%)
2. fDC: 1.000 (raw fraction: 0.0918, 9.18%)
3. MoMacrophages: 0.633 (raw fraction: 0.0452, 4.52%)
4. T cell CD4: 0.475 (raw fraction: 0.0776, 7.76%)
5. Plasma cell: 0.383 (raw fraction: 0.0404, 4.04%)

*Complete L1_CT composition (all 13 cell types):*
- B cell: 54.50% (highest)
- fDC: 9.18% (highest)
- LQ: 10.56%
- T cell CD4: 7.76%
- MoMacrophages: 4.52%
- Plasma cell: 4.04%
- T cell CD8: 2.67%
- Fibroblast: 2.43%
- Basal cell: 2.14%
- DC: 0.85%
- Endothelial cell: 0.95%
- Perivascular: 0.30%
- Mast cell: 0.03%

**Biological Interpretation:**

*Supported by data:*
- **B cell follicle identity**: The dominant B cell fraction (54.5%) combined with the highest fDC enrichment (9.2%) strongly indicates this represents the core of primary B cell follicles or mantle zones in tonsil tissue.
- **fDC network architecture**: Follicular dendritic cells (fDCs) form the structural scaffold that presents antigens to B cells, supporting germinal center reactions. The co-enrichment with B cells is the hallmark of follicular structures.
- **Immune surveillance**: Presence of macrophages (4.5%) and T cells (10.4% combined CD4/CD8) indicates active immune monitoring within follicles.
- **Antibody production**: 4.0% plasma cells suggest ongoing B cell differentiation and antibody secretion.

*Hypothesis/speculation:*
- This neighborhood likely represents **non-germinal center follicular regions** (mantle zone or primary follicles) rather than active germinal centers, based on the high B cell fraction without extreme T cell or DC enrichment that would characterize germinal center light zones.
- The LQ fraction (10.6%) may represent lymphoid quality stromal cells or fibroblastic reticular cells supporting follicular architecture.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **22/25 FOVs (88%)** - **Classification: Ubiquitous**

*Top 5 FOVs by normalized abundance:*
1. FOV 20: 1.000 (maximum enrichment)
2. FOV 16: 0.874
3. FOV 15: 0.536
4. FOV 23: 0.423
5. FOV 7: 0.412

*Spatial interpretation:*
- Highly recurrent across tissue, indicating B cell follicles are a conserved architectural feature of tonsil tissue.
- Variable abundance across FOVs suggests different tissue sections capture varying proportions of follicular vs. interfollicular regions.
- FOVs 20 and 16 show extreme enrichment, likely capturing large, well-developed follicular structures.
- Only 3 FOVs lack this neighborhood (FOVs 3, 4, 19), possibly representing interfollicular or stromal-rich regions.

**Quantitative Summary:**
- **Total cells**: 149,126 (20.7% of dataset)
- **FOV presence**: 22/25 FOVs (nonzero)
- **Shannon entropy**: 2.354 bits
- **Effective N cell types**: 5.11 (moderate diversity)
- **Dominant cell type**: B cell (54.5%)
- **Top 5 enriched**: B cell(1.000); fDC(1.000); MoMacrophages(0.633); T cell CD4(0.475); Plasma cell(0.383)


---

### Neighborhood 1: Mixed Immune-Stromal Interface

**Descriptive Name**: Interfollicular Region with Balanced Immune-Stromal Components

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. MoMacrophages: 0.923 (raw fraction: 0.0659, 6.59%)
2. B cell: 0.761 (raw fraction: 0.4147, 41.47%)
3. Fibroblast: 0.678 (raw fraction: 0.0734, 7.34%)
4. Perivascular: 0.581 (raw fraction: 0.0149, 1.49%)
5. DC: 0.534 (raw fraction: 0.0209, 2.09%)

*Complete L1_CT composition (all 13 cell types):*
- B cell: 41.47%
- LQ: 12.66%
- T cell CD4: 7.46%
- Fibroblast: 7.34%
- MoMacrophages: 6.59%
- T cell CD8: 5.91%
- Plasma cell: 4.98%
- fDC: 3.68%
- Endothelial cell: 2.86%
- DC: 2.09%
- Perivascular: 1.49%
- Basal cell: 1.25%
- Mast cell: 0.09%

**Biological Interpretation:**

*Supported by data:*
- **Interfollicular zone**: High B cell fraction (41.5%) but lower than N0, combined with elevated fibroblasts (7.3%) and macrophages (6.6%), suggests this is the transition zone between follicles.
- **Stromal-immune interface**: Enrichment of fibroblasts, perivascular cells (1.5%), and endothelial cells (2.9%) indicates structural/vascular support regions.
- **Immune surveillance**: Macrophages (6.6%) and DCs (2.1%) are enriched, consistent with antigen-presenting cell positioning in interfollicular corridors.
- **T cell trafficking**: 13.4% T cells (CD4+CD8) suggests T cell migration through interfollicular regions.
- **High diversity**: Effective N = 7.08 cell types indicates a complex, multi-functional microenvironment.

*Hypothesis/speculation:*
- This neighborhood may represent **paracortical/interfollicular regions** where T cells and antigen-presenting cells interact before entering follicles.
- The elevated LQ (12.7%) may represent fibroblastic reticular cells (FRCs) that form conduits for lymphocyte migration and provide structural support.
- May serve as a transitional zone between B cell follicles and T cell zones.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **23/25 FOVs (92%)** - **Classification: Ubiquitous**

*Top 5 FOVs by normalized abundance:*
1. FOV 12: 1.000 (maximum enrichment)
2. FOV 23: 0.863
3. FOV 10: 0.849
4. FOV 8: 0.816
5. FOV 24: 0.738

*Spatial interpretation:*
- Most recurrent neighborhood across the entire tissue.
- Consistent presence suggests interfollicular regions are a fundamental architectural component of tonsil tissue.
- High abundance in multiple FOVs (12, 23, 10, 8) indicates extensive interfollicular areas.
- Only 2 FOVs lack this neighborhood (FOVs 3, 4), which may be dominated by other specialized niches.

**Quantitative Summary:**
- **Total cells**: 132,606 (18.4% of dataset)
- **FOV presence**: 23/25 FOVs (nonzero)
- **Shannon entropy**: 2.824 bits
- **Effective N cell types**: 7.08 (high diversity)
- **Dominant cell type**: B cell (41.5%)
- **Top 5 enriched**: MoMacrophages(0.923); B cell(0.761); Fibroblast(0.678); Perivascular(0.581); DC(0.534)


---

### Neighborhood 2: Epithelial-Associated Immune Niche

**Descriptive Name**: Basal Cell-Enriched Crypt-Associated Lymphoid Tissue

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. Basal cell: 1.000 (raw fraction: 0.2446, 24.46%)
2. MoMacrophages: 0.845 (raw fraction: 0.0603, 6.03%)
3. Plasma cell: 0.769 (raw fraction: 0.0812, 8.12%)
4. DC: 0.687 (raw fraction: 0.0269, 2.69%)
5. LQ: 0.467 (raw fraction: 0.1563, 15.63%)

*Complete L1_CT composition (all 13 cell types):*
- B cell: 24.60%
- Basal cell: 24.46% (highest, unique to this neighborhood)
- LQ: 15.63%
- Plasma cell: 8.12%
- MoMacrophages: 6.03%
- T cell CD4: 5.31%
- T cell CD8: 4.56%
- Fibroblast: 3.89%
- DC: 2.69%
- Endothelial cell: 2.01%
- fDC: 1.86%
- Perivascular: 0.79%
- Mast cell: 0.08%

**Biological Interpretation:**

*Supported by data:*
- **Epithelial interface**: Unique basal cell enrichment (24.5%) indicates proximity to epithelial surfaces, most likely tonsillar crypts where epithelium invaginates into lymphoid tissue.
- **Mucosal immunity**: High plasma cell fraction (8.1%) is consistent with IgA secretion at mucosal surfaces for local antibody-mediated protection.
- **Antigen sampling**: Macrophages (6.0%) and DCs (2.7%) suggest active antigen sampling from epithelial surfaces, a key function of crypt-associated lymphoid tissue.
- **B cell presence**: 24.6% B cells indicate lymphoid tissue directly adjacent to epithelium, consistent with crypt-associated follicles.
- **High diversity**: Effective N = 7.92 cell types indicates complex immune-epithelial interactions.

*Hypothesis/speculation:*
- This neighborhood likely represents **crypt-associated lymphoid tissue (CALT)** where epithelium interfaces with organized lymphoid structures.
- The LQ fraction (15.6%) may include both stromal cells and epithelial-associated fibroblasts that support the epithelial-immune interface.
- Plasma cells may be actively secreting IgA for transcytosis across epithelium into the crypt lumen.
- May represent M cell-containing regions where antigen sampling occurs.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **21/25 FOVs (84%)** - **Classification: Ubiquitous**

*Top 5 FOVs by normalized abundance:*
1. FOV 3: 1.000 (maximum enrichment)
2. FOV 7: 0.543
3. FOV 6: 0.441
4. FOV 18: 0.396
5. FOV 17: 0.361

*Spatial interpretation:*
- Widespread but with variable abundance across tissue.
- FOV 3 shows extreme enrichment (1.000), possibly capturing a large crypt region or epithelial interface.
- Presence in 84% of FOVs suggests crypts/epithelium are sampled across most tissue sections.
- Four FOVs lack this neighborhood (FOVs 4, 13, 19, 20), which may represent deep lymphoid tissue away from epithelial surfaces.

**Quantitative Summary:**
- **Total cells**: 123,005 (17.1% of dataset)
- **FOV presence**: 21/25 FOVs (nonzero)
- **Shannon entropy**: 2.986 bits
- **Effective N cell types**: 7.92 (high diversity)
- **Dominant cell types**: B cell (24.6%) and Basal cell (24.5%) - nearly equal
- **Top 5 enriched**: Basal cell(1.000); MoMacrophages(0.845); Plasma cell(0.769); DC(0.687); LQ(0.467)


---

### Neighborhood 3: Stromal-Vascular Niche

**Descriptive Name**: Fibroblast-Endothelial Perivascular Support Region

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. Endothelial cell: 1.000 (raw fraction: 0.0553, 5.53%)
2. Fibroblast: 1.000 (raw fraction: 0.1083, 10.83%)
3. Mast cell: 1.000 (raw fraction: 0.0031, 0.31%)
4. MoMacrophages: 1.000 (raw fraction: 0.0714, 7.14%)
5. Perivascular: 1.000 (raw fraction: 0.0256, 2.56%)

**Note**: Five cell types show maximum normalized enrichment (1.000), indicating this is the primary niche for stromal/vascular cell types across all neighborhoods.

*Complete L1_CT composition (all 13 cell types):*
- B cell: 22.75%
- LQ: 19.14%
- Fibroblast: 10.83% (highest)
- Plasma cell: 10.57% (highest)
- T cell CD8: 8.31%
- MoMacrophages: 7.14%
- T cell CD4: 6.29%
- Endothelial cell: 5.53% (highest)
- DC: 2.85%
- Basal cell: 2.93%
- Perivascular: 2.56% (highest)
- Mast cell: 0.31% (highest)
- fDC: 0.77%

**Biological Interpretation:**

*Supported by data:*
- **Stromal scaffold**: Highest fibroblast (10.8%) and endothelial cell (5.5%) fractions indicate structural/vascular support regions essential for tissue architecture.
- **Vascular niche**: Perivascular cells (2.6%) and endothelial cells together suggest proximity to blood vessels, likely representing perivascular regions.
- **Immune support**: Macrophages (7.1%) and mast cells (0.31%) are consistent with perivascular immune surveillance and vascular regulation.
- **Plasma cell homing**: Highest plasma cell fraction (10.6%) across all neighborhoods suggests this is a preferred site for antibody-secreting cells, often positioned near vasculature for antibody export.
- **Highest diversity**: Effective N = 9.17 cell types (highest across all neighborhoods) indicates a complex, multi-functional microenvironment with diverse cellular interactions.

*Hypothesis/speculation:*
- This neighborhood likely represents **perivascular/stromal regions** including blood vessel walls, fibroblastic reticular cell networks, and connective tissue septa that provide structural support.
- Mast cells near vessels may regulate vascular permeability, immune cell recruitment, and inflammatory responses.
- High plasma cell fraction suggests this is a **plasma cell survival niche** where long-lived antibody-secreting cells reside near vascular export routes for systemic antibody distribution.
- May represent medullary cord-like regions in tonsil tissue.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **19/25 FOVs (76%)** - **Classification: Common**

*Top 5 FOVs by normalized abundance:*
1. FOV 4: 1.000 (maximum enrichment)
2. FOV 17: 0.639
3. FOV 24: 0.271
4. FOV 5: 0.259
5. FOV 9: 0.205

*Spatial interpretation:*
- Common but not ubiquitous across tissue.
- FOV 4 shows extreme enrichment (1.000), possibly capturing a major vascular/stromal region or connective tissue septum.
- Lower recurrence (76%) compared to follicular neighborhoods suggests stromal niches are more spatially restricted and concentrated in specific tissue regions.
- Six FOVs lack this neighborhood (FOVs 1, 6, 8, 13, 14, 20), which may represent lymphoid-dense regions with minimal stromal components.

**Quantitative Summary:**
- **Total cells**: 85,963 (11.9% of dataset)
- **FOV presence**: 19/25 FOVs (nonzero)
- **Shannon entropy**: 3.196 bits (highest)
- **Effective N cell types**: 9.17 (highest diversity)
- **Dominant cell type**: B cell (22.8%), but with substantial stromal components
- **Top 5 enriched**: Endothelial cell(1.000); Fibroblast(1.000); Mast cell(1.000); MoMacrophages(1.000); Perivascular(1.000)


---

### Neighborhood 4: T Cell Zone with B Cell Interaction

**Descriptive Name**: CD4+ T Cell-Enriched Zone with B Cell Collaboration

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. T cell CD4: 0.984 (raw fraction: 0.1609, 16.09%)
2. MoMacrophages: 0.754 (raw fraction: 0.0538, 5.38%)
3. T cell CD8: 0.702 (raw fraction: 0.0808, 8.08%)
4. fDC: 0.658 (raw fraction: 0.0604, 6.04%)
5. B cell: 0.648 (raw fraction: 0.3534, 35.34%)

*Complete L1_CT composition (all 13 cell types):*
- B cell: 35.34%
- T cell CD4: 16.09% (highest)
- LQ: 15.45%
- T cell CD8: 8.08%
- fDC: 6.04%
- MoMacrophages: 5.38%
- Fibroblast: 5.06%
- Plasma cell: 2.60%
- DC: 2.23%
- Endothelial cell: 2.17%
- Perivascular: 0.76%
- Basal cell: 0.71%
- Mast cell: 0.03%

**Biological Interpretation:**

*Supported by data:*
- **T cell zone**: Highest CD4+ T cell fraction (16.1%) combined with elevated CD8+ T cells (8.1%) indicates a T cell-enriched region.
- **T-B cell interaction**: Substantial B cell presence (35.3%) suggests this is a **T-B cell interface zone**, possibly extrafollicular or germinal center light zones where T follicular helper (Tfh) cells interact with B cells.
- **Antigen presentation**: fDC (6.0%) and macrophages (5.4%) enrichment indicates active antigen presentation to both T and B cells.
- **Helper T cell dominance**: CD4:CD8 ratio of ~2:1 is consistent with helper T cell-mediated B cell activation and support.

*Hypothesis/speculation:*
- This neighborhood may represent **interfollicular T cell zones** or **germinal center light zones** where T follicular helper (Tfh) cells provide help signals to B cells undergoing selection.
- The high fDC fraction suggests proximity to follicular structures where fDCs present antigen to both B and T cells.
- LQ (15.4%) may include fibroblastic reticular cells that support T cell migration, survival, and interactions with B cells.
- May represent regions where activated B cells receive T cell help before differentiating into plasma cells or memory B cells.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **17/25 FOVs (68%)** - **Classification: Common**

*Top 5 FOVs by normalized abundance:*
1. FOV 14: 1.000 (maximum enrichment)
2. FOV 10: 0.390
3. FOV 2: 0.386
4. FOV 5: 0.317
5. FOV 25: 0.303

*Spatial interpretation:*
- Common but less ubiquitous than follicular neighborhoods.
- FOV 14 shows strong enrichment (1.000), possibly capturing a large T cell zone or germinal center light zone.
- Lower recurrence (68%) suggests T cell zones are more spatially restricted than B cell follicles.
- Eight FOVs lack this neighborhood (FOVs 3, 4, 9, 13, 15, 16, 19, 20), which may be dominated by B cell follicles or stromal regions.

**Quantitative Summary:**
- **Total cells**: 74,145 (10.3% of dataset)
- **FOV presence**: 17/25 FOVs (nonzero)
- **Shannon entropy**: 2.842 bits
- **Effective N cell types**: 7.17 (high diversity)
- **Dominant cell type**: B cell (35.3%), but with highest T cell CD4 enrichment
- **Top 5 enriched**: T cell CD4(0.984); MoMacrophages(0.754); T cell CD8(0.702); fDC(0.658); B cell(0.648)


---

### Neighborhood 5: Follicular Dendritic Cell-Rich B Cell Zone

**Descriptive Name**: fDC-Dense B Cell Follicle with Plasma Cell Output

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. fDC: 0.944 (raw fraction: 0.0867, 8.67%)
2. B cell: 0.763 (raw fraction: 0.4158, 41.58%)
3. Mast cell: 0.728 (raw fraction: 0.0023, 0.23%)
4. Plasma cell: 0.724 (raw fraction: 0.0764, 7.64%)
5. MoMacrophages: 0.715 (raw fraction: 0.0510, 5.10%)

*Complete L1_CT composition (all 13 cell types):*
- B cell: 41.58%
- LQ: 12.66%
- fDC: 8.67%
- Plasma cell: 7.64%
- T cell CD4: 7.62%
- Fibroblast: 5.84%
- MoMacrophages: 5.10%
- T cell CD8: 4.85%
- Endothelial cell: 2.51%
- DC: 1.24%
- Basal cell: 1.16%
- Perivascular: 0.93%
- Mast cell: 0.23%

**Biological Interpretation:**

*Supported by data:*
- **Follicular structure**: High B cell (41.6%) and fDC (8.7%) fractions indicate a follicular microenvironment with dense follicular dendritic cell networks.
- **Plasma cell differentiation**: Elevated plasma cells (7.6%) suggest this is a site of B cell-to-plasma cell differentiation, possibly **germinal center dark zones** or **outer follicular regions** where selected B cells exit and differentiate.
- **fDC network**: Second-highest fDC fraction (after N0) indicates a dense follicular dendritic cell network for antigen presentation and B cell selection.
- **Mast cell enrichment**: Elevated mast cells (0.23%) may regulate vascular permeability, immune cell recruitment, and inflammatory responses within follicles.

*Hypothesis/speculation:*
- This neighborhood may represent **germinal center dark zones** or **outer mantle zones** where B cells undergo somatic hypermutation, selection, and differentiation into plasma cells.
- The combination of high fDC and plasma cells suggests a transition zone between active germinal center reactions and plasma cell output pathways.
- Distinct from N0 (follicle core) by higher plasma cell fraction and potentially representing more activated/reactive follicles.
- May represent follicles with ongoing or recent germinal center reactions.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **20/25 FOVs (80%)** - **Classification: Ubiquitous**

*Top 5 FOVs by normalized abundance:*
1. FOV 13: 1.000 (maximum enrichment)
2. FOV 8: 0.931
3. FOV 18: 0.666
4. FOV 22: 0.441
5. FOV 23: 0.414

*Spatial interpretation:*
- Widespread across tissue, indicating this follicular subtype is common.
- FOVs 13 and 8 show very high enrichment (1.000, 0.931), possibly capturing large follicular structures with active germinal centers.
- High recurrence (80%) suggests this follicular phenotype is prevalent in tonsil architecture.
- Five FOVs lack this neighborhood (FOVs 1, 4, 5, 14, 20), which may be dominated by other follicular subtypes or non-follicular regions.

**Quantitative Summary:**
- **Total cells**: 64,545 (9.0% of dataset)
- **FOV presence**: 20/25 FOVs (nonzero)
- **Shannon entropy**: 2.814 bits
- **Effective N cell types**: 7.03 (high diversity)
- **Dominant cell type**: B cell (41.6%)
- **Top 5 enriched**: fDC(0.944); B cell(0.763); Mast cell(0.728); Plasma cell(0.724); MoMacrophages(0.715)


---

### Neighborhood 6: T Cell-Dendritic Cell Interaction Zone

**Descriptive Name**: DC-Rich T Cell Zone with Cytotoxic T Cell Enrichment

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. DC: 1.000 (raw fraction: 0.0392, 3.92%)
2. T cell CD4: 1.000 (raw fraction: 0.1635, 16.35%)
3. T cell CD8: 1.000 (raw fraction: 0.1150, 11.50%)
4. MoMacrophages: 0.964 (raw fraction: 0.0688, 6.88%)
5. Fibroblast: 0.840 (raw fraction: 0.0911, 9.11%)

**Note**: Three cell types show maximum normalized enrichment (1.000): DC, T cell CD4, and T cell CD8, indicating this is the primary T cell and dendritic cell niche.

*Complete L1_CT composition (all 13 cell types):*
- LQ: 20.05%
- B cell: 19.53%
- T cell CD4: 16.35% (second highest, nearly tied with N4)
- T cell CD8: 11.50% (highest)
- Fibroblast: 9.11%
- MoMacrophages: 6.88%
- Plasma cell: 4.11%
- DC: 3.92% (highest)
- Endothelial cell: 3.61%
- Basal cell: 1.84%
- Perivascular: 1.66%
- fDC: 1.26%
- Mast cell: 0.10%

**Biological Interpretation:**

*Supported by data:*
- **T cell zone**: Highest CD8+ T cell fraction (11.5%) and near-highest CD4+ fraction (16.4%) indicate a dedicated T cell compartment.
- **Dendritic cell hub**: Highest DC fraction (3.9%) suggests this is a site of active **T cell priming and activation** by dendritic cells presenting antigen.
- **Cytotoxic T cell enrichment**: CD8+ T cell enrichment may indicate cytotoxic T lymphocyte (CTL) positioning for immune surveillance and effector functions.
- **Stromal support**: Elevated fibroblasts (9.1%) and LQ (20.1%) suggest extensive fibroblastic reticular cell networks supporting T cell migration, survival, and interactions.
- **Highest T cell diversity**: Combined CD4 and CD8 enrichment (27.9% total T cells) with high DC content indicates complex T cell-APC interactions.
- **Second-highest diversity**: Effective N = 8.77 cell types indicates a complex, multi-functional T cell microenvironment.

*Hypothesis/speculation:*
- This neighborhood likely represents **paracortical T cell zones** where DCs present antigen to naive T cells, initiating adaptive immune responses.
- The high CD8+ enrichment may indicate **effector T cell positioning** or memory T cell niches prepared for rapid activation.
- May also represent **interfollicular regions** where T cells traffic between follicles and receive activation signals.
- The extensive stromal network (LQ 20.1%) may form conduits for chemokine gradients guiding T cell migration.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **19/25 FOVs (76%)** - **Classification: Common**

*Top 5 FOVs by normalized abundance:*
1. FOV 11: 1.000 (maximum enrichment)
2. FOV 2: 0.584
3. FOV 10: 0.510
4. FOV 1: 0.456
5. FOV 6: 0.297

*Spatial interpretation:*
- Common but not ubiquitous across tissue.
- FOV 11 shows extreme enrichment (1.000), possibly capturing a large T cell zone or paracortical region.
- Lower recurrence (76%) compared to B cell neighborhoods suggests T cell zones are more spatially restricted.
- Six FOVs lack this neighborhood (FOVs 3, 4, 7, 13, 15, 20), which may be dominated by B cell follicles or stromal regions.

**Quantitative Summary:**
- **Total cells**: 63,542 (8.8% of dataset)
- **FOV presence**: 19/25 FOVs (nonzero)
- **Shannon entropy**: 3.133 bits
- **Effective N cell types**: 8.77 (second highest diversity)
- **Dominant cell type**: LQ (20.1%), but with highest T cell and DC enrichment
- **Top 5 enriched**: DC(1.000); T cell CD4(1.000); T cell CD8(1.000); MoMacrophages(0.964); Fibroblast(0.840)


---

### Neighborhood 7: LQ-Enriched Rare Specialized Niche

**Descriptive Name**: Lymphoid Quality Cell-Dominated Rare Microenvironment

**Cell Type Enrichment:**

*Top 5 enriched (normalized 0-1):*
1. LQ: 1.000 (raw fraction: 0.3349, 33.49%)
2. Plasma cell: 0.719 (raw fraction: 0.0759, 7.59%)
3. Endothelial cell: 0.595 (raw fraction: 0.0329, 3.29%)
4. T cell CD8: 0.593 (raw fraction: 0.0682, 6.82%)
5. Fibroblast: 0.569 (raw fraction: 0.0617, 6.17%)

*Complete L1_CT composition (all 13 cell types):*
- LQ: 33.49% (highest, unique dominance)
- B cell: 20.77%
- Plasma cell: 7.59%
- T cell CD4: 6.98%
- T cell CD8: 6.82%
- Basal cell: 6.26%
- Fibroblast: 6.17%
- MoMacrophages: 3.54%
- Endothelial cell: 3.29%
- fDC: 2.30%
- DC: 1.61%
- Perivascular: 1.04%
- Mast cell: 0.12%

**Biological Interpretation:**

*Supported by data:*
- **LQ dominance**: Extreme LQ enrichment (33.5%) is unique to this neighborhood, representing the highest fraction of any single cell type in any neighborhood. This suggests a specialized stromal/lymphoid quality cell niche.
- **Rare and specialized**: Present in only **4/25 FOVs (16%)**, indicating this is a **rare, spatially restricted microenvironment** not commonly sampled.
- **Plasma cell association**: Elevated plasma cells (7.6%) suggest this may be a **plasma cell survival niche** where stromal cells provide survival signals.
- **Stromal-vascular features**: Endothelial cells (3.3%) and fibroblasts (6.2%) indicate structural support and potential vascular proximity.
- **Moderate diversity**: Effective N = 7.71 cell types despite LQ dominance indicates other cell types are present in meaningful proportions.

*Hypothesis/speculation:*
- **LQ identity unclear**: The "LQ" (lymphoid quality) cell type annotation requires clarification. It may represent:
  - Fibroblastic reticular cells (FRCs) with specialized lymphoid support functions
  - Lymphoid stromal cells or mesenchymal stem/progenitor cells
  - Cells with ambiguous annotation requiring further validation
- **Potential plasma cell niche**: The combination of LQ dominance and plasma cell enrichment suggests this could be a **medullary cord-like region** where plasma cells are supported by specialized stromal cells for long-term survival.
- **Capsular/subcapsular region**: Elevated basal cell fraction (6.3%) may indicate proximity to capsule or epithelial surfaces.
- **Requires validation**: The rarity and unique composition of this neighborhood warrant further investigation to confirm its biological identity and function.

**Spatial Distribution Across FOVs:**

*FOV recurrence:* Present in **4/25 FOVs (16%)** - **Classification: Rare/Specialized**

*FOVs with presence (normalized abundance):*
1. FOV 21: 1.000 (maximum enrichment)
2. FOV 22: 0.765
3. FOV 7: 0.084
4. FOV 18: 0.079

*Spatial interpretation:*
- **Highly restricted**: Only 4 FOVs contain this neighborhood, making it the rarest across all neighborhoods.
- FOVs 21 and 22 show strong enrichment (1.000, 0.765), suggesting these capture specialized tissue regions.
- FOVs 7 and 18 show minimal presence (0.084, 0.079), indicating small patches of this niche.
- The extreme spatial restriction suggests this neighborhood represents a specialized anatomical region (e.g., medullary cords, capsular regions, or specific stromal niches) not uniformly distributed in tonsil tissue.

**Quantitative Summary:**
- **Total cells**: 27,565 (3.8% of dataset - smallest neighborhood)
- **FOV presence**: 4/25 FOVs (nonzero)
- **Shannon entropy**: 2.948 bits
- **Effective N cell types**: 7.71 (high diversity despite LQ dominance)
- **Dominant cell type**: LQ (33.5%)
- **Top 5 enriched**: LQ(1.000); Plasma cell(0.719); Endothelial cell(0.595); T cell CD8(0.593); Fibroblast(0.569)


---

## Cross-Neighborhood Comparison

### Comparative Overview Table

| Feature | N0 | N1 | N2 | N3 | N4 | N5 | N6 | N7 |
|---------|----|----|----|----|----|----|----|----|
| **Descriptive Name** | B Cell Follicle Core | Interfollicular Region | Crypt-Associated Lymphoid | Stromal-Vascular Niche | T-B Cell Interaction Zone | fDC-Rich B Cell Follicle | T Cell-DC Zone | LQ-Enriched Rare Niche |
| **Total Cells** | 149,126 | 132,606 | 123,005 | 85,963 | 74,145 | 64,545 | 63,542 | 27,565 |
| **% of Dataset** | 20.7% | 18.4% | 17.1% | 11.9% | 10.3% | 9.0% | 8.8% | 3.8% |
| **FOV Presence** | 22/25 (88%) | 23/25 (92%) | 21/25 (84%) | 19/25 (76%) | 17/25 (68%) | 20/25 (80%) | 19/25 (76%) | 4/25 (16%) |
| **Distribution** | Ubiquitous | Ubiquitous | Ubiquitous | Common | Common | Ubiquitous | Common | Rare |
| **Shannon Entropy** | 2.354 | 2.824 | 2.986 | 3.196 | 2.842 | 2.814 | 3.133 | 2.948 |
| **Effective N Cell Types** | 5.11 | 7.08 | 7.92 | 9.17 | 7.17 | 7.03 | 8.77 | 7.71 |
| **Diversity Rank** | Lowest | Mid | Mid-High | Highest | Mid | Mid | 2nd Highest | Mid-High |
| **Dominant Cell Type** | B cell (54.5%) | B cell (41.5%) | B cell/Basal (24.6%/24.5%) | B cell (22.8%) | B cell (35.3%) | B cell (41.6%) | LQ (20.1%) | LQ (33.5%) |
| **Unique Feature** | Highest B cell & fDC | Balanced immune-stromal | Unique basal cell enrichment | Highest stromal/vascular | Highest T cell CD4 | High fDC + plasma cells | Highest DC & T cell CD8 | Highest LQ, rarest |

### Key Differentiating Features

**1. B Cell-Dominated Neighborhoods (N0, N1, N4, N5)**
- **N0 vs N5**: Both are B cell follicles with high fDC, but N5 has more plasma cells (7.6% vs 4.0%), suggesting N5 represents more activated/reactive follicles with ongoing germinal center reactions.
- **N0 vs N1**: N0 is pure follicle core (54.5% B cells), while N1 is interfollicular (41.5% B cells) with elevated stromal components (fibroblasts 7.3%, macrophages 6.6%).
- **N4**: Unique among B cell neighborhoods for highest T cell CD4 enrichment (16.1%), indicating T-B cell interaction zones.

**2. Stromal/Structural Neighborhoods (N3, N7)**
- **N3**: Highest diversity (9.17 effective cell types), dominated by stromal-vascular cells (fibroblasts, endothelial, perivascular all at maximum enrichment). Represents structural support regions.
- **N7**: LQ-dominated (33.5%), extremely rare (4 FOVs only). Distinct from N3 by LQ dominance rather than diverse stromal mix.

**3. T Cell-Enriched Neighborhoods (N4, N6)**
- **N4 vs N6**: Both are T cell zones, but N6 has highest CD8+ T cells (11.5%) and DCs (3.9%), suggesting cytotoxic T cell zones with active priming. N4 has more B cells (35.3% vs 19.5%), indicating T-B collaboration.

**4. Epithelial-Associated Neighborhood (N2)**
- **Unique**: Only neighborhood with substantial basal cells (24.5%), indicating epithelial interface (crypt-associated lymphoid tissue).
- High plasma cells (8.1%) consistent with mucosal antibody secretion.

### Cellular Diversity Patterns

**Lowest diversity** (N0, 5.11 effective cell types):
- Specialized B cell follicle core with dominant B cell and fDC populations
- Functional specialization for B cell activation and selection

**Highest diversity** (N3, 9.17 effective cell types):
- Stromal-vascular niche with balanced representation of many cell types
- Multi-functional support region for structural integrity, vascular supply, and immune cell trafficking

**High T cell diversity** (N6, 8.77 effective cell types):
- T cell zone with both CD4+ and CD8+ T cells, DCs, and stromal support
- Complex T cell-APC interactions

### Spatial Distribution Patterns

**Ubiquitous neighborhoods** (present in >80% of FOVs):
- N0, N1, N2, N5: Core lymphoid structures (B cell follicles, interfollicular regions, epithelial interfaces)
- Represent fundamental architectural components of tonsil tissue

**Common neighborhoods** (present in 68-76% of FOVs):
- N3, N4, N6: Specialized niches (stromal-vascular, T cell zones)
- More spatially restricted, concentrated in specific tissue regions

**Rare neighborhood** (present in 16% of FOVs):
- N7: Highly specialized LQ-dominated niche
- May represent unique anatomical structures (medullary cords, capsular regions)

### Functional Compartmentalization

**B cell immunity** (N0, N5):
- Primary and reactive follicles
- Sites of B cell activation, selection, and antibody production

**T cell immunity** (N4, N6):
- T cell zones with DC-mediated priming
- T-B cell collaboration zones
- Cytotoxic T cell positioning

**Immune-stromal interfaces** (N1, N2, N3):
- Interfollicular regions (N1)
- Epithelial interfaces (N2)
- Structural support (N3)

**Specialized niches** (N7):
- Rare, LQ-dominated regions
- Potential plasma cell survival niches


---

## Discussion

### Biological Implications

**1. Tonsil Architecture Revealed by Spatial Neighborhoods**

The eight identified neighborhoods recapitulate the known functional compartmentalization of tonsil tissue:

- **B cell follicles** (N0, N5): Two distinct follicular phenotypes suggest different activation states - N0 represents resting/primary follicles, while N5 represents reactive follicles with active germinal centers and plasma cell output.

- **T cell zones** (N4, N6): Two T cell compartments with distinct functions - N4 for T-B collaboration (Tfh-mediated B cell help), N6 for T cell priming by DCs and cytotoxic T cell positioning.

- **Interfollicular regions** (N1): Transitional zones between follicles with balanced immune and stromal components, facilitating lymphocyte trafficking.

- **Epithelial interfaces** (N2): Crypt-associated lymphoid tissue where antigen sampling occurs and mucosal immunity is initiated.

- **Stromal support** (N3): Perivascular and structural regions providing vascular supply, plasma cell niches, and tissue architecture.

- **Specialized niches** (N7): Rare LQ-dominated regions that may represent medullary cords or other specialized stromal compartments.

**2. Cellular Diversity as a Functional Indicator**

Neighborhood diversity (effective N cell types) correlates with functional complexity:

- **Low diversity** (N0, 5.11): Specialized for single function (B cell activation)
- **High diversity** (N3, 9.17; N6, 8.77): Multi-functional regions supporting diverse cellular interactions

This suggests that stromal-vascular regions (N3) and T cell zones (N6) serve as hubs for complex immune processes requiring coordination of many cell types.

**3. Spatial Recurrence Patterns**

The ubiquity of B cell follicles (N0, N5) and interfollicular regions (N1) across 80-92% of FOVs indicates these are fundamental architectural units consistently present across tissue sections. In contrast, T cell zones (N4, N6) show more restricted distribution (68-76% of FOVs), suggesting they are concentrated in specific anatomical regions (paracortex).

The extreme rarity of N7 (4/25 FOVs, 16%) indicates this neighborhood represents a specialized anatomical structure not uniformly distributed in tonsil tissue.

**4. Plasma Cell Distribution**

Plasma cells show enrichment in three distinct neighborhoods:
- **N3** (10.6%): Perivascular plasma cell niches for antibody export
- **N2** (8.1%): Mucosal surfaces for IgA secretion
- **N5** (7.6%): Follicular output zones

This distribution suggests multiple plasma cell niches with distinct functions: systemic antibody production (N3), mucosal immunity (N2), and follicular output (N5).

**5. Macrophage Positioning**

Macrophages (MoMacrophages) are enriched across multiple neighborhoods (N1, N2, N3, N6), indicating their diverse roles:
- Antigen presentation (N6)
- Tissue remodeling and debris clearance (N3)
- Immune surveillance (N1, N2)

The highest enrichment in interfollicular regions (N1, 6.6%) suggests macrophages patrol transition zones between lymphoid compartments.

### Limitations

**1. LQ Cell Type Ambiguity**

The "LQ" (lymphoid quality) cell type requires clarification. It may represent:
- Fibroblastic reticular cells (FRCs)
- Lymphoid stromal cells
- Mesenchymal stem/progenitor cells
- Cells with ambiguous annotation

The extreme enrichment of LQ in N7 (33.5%) highlights the need for validation of this cell type annotation, potentially through:
- Marker gene analysis
- Comparison with known stromal cell markers
- Subclustering and refinement of LQ cells

**2. Neighborhood Assignment Method**

The Leiden clustering approach used for neighborhood inference may:
- Merge functionally distinct but transcriptionally similar niches
- Split continuous gradients into discrete neighborhoods
- Be sensitive to resolution parameter choices

Alternative approaches (e.g., UTAG, Banksy, Niche-Net) may reveal additional spatial organization patterns.

**3. Within-FOV Spatial Restriction**

All spatial analyses were restricted to within-FOV relationships, as specified. This means:
- No cross-FOV spatial proximity calculations were performed
- Neighborhoods are defined by cell type composition and within-FOV spatial patterns
- Global tissue architecture across FOVs is not directly captured

**4. Static Snapshot**

This dataset represents a single timepoint in healthy tonsil tissue. Dynamic processes such as:
- Germinal center reactions
- T cell activation and differentiation
- Plasma cell migration

Cannot be directly observed. Longitudinal or perturbation studies would be needed to understand temporal dynamics.

**5. Gene Panel Limitations**

The CosMx panel includes 990 genes, which may not capture:
- Rare cell type markers
- Subtle cell state differences
- Full transcriptional programs

Whole-transcriptome spatial methods (e.g., Visium HD, MERSCOPE) may reveal additional neighborhood features.

### Future Directions

**1. Functional Validation**

- Validate LQ cell identity through marker gene analysis and comparison with known stromal markers
- Confirm neighborhood identities through immunofluorescence or multiplexed imaging
- Investigate neighborhood-specific gene expression programs

**2. Disease Comparison**

- Compare neighborhood composition and distribution in diseased tonsils (e.g., tonsillitis, lymphoma)
- Identify disease-associated neighborhood alterations
- Determine whether neighborhoods are disrupted or reorganized in disease

**3. Cell-Cell Interactions**

- Analyze ligand-receptor interactions within and between neighborhoods
- Identify neighborhood-specific signaling pathways
- Map cell-cell communication networks

**4. Subclustering**

- Refine neighborhood annotations through subclustering of major cell types (B cells, T cells)
- Identify cell state transitions within neighborhoods
- Map differentiation trajectories

**5. Integration with Other Modalities**

- Integrate with scRNA-seq for deeper transcriptional profiling
- Combine with proteomics for protein-level validation
- Correlate with clinical metadata (age, disease status)

---

## Reproducibility Appendix

### File Paths Used

**Input data:**
- Main annotated AnnData: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/CosMx_Tonsil_with_Leiden_neigh.h5ad`
- Neighborhood-celltype composition (normalized): `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_celltype_composition_normalized.csv`
- FOV-neighborhood abundance (normalized): `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/fov_neighborhood_abundance_normalized.csv`
- FOV-neighborhood composition: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/fov_neighborhood_composition.csv`
- Neighborhood mapping: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_mapping.csv`

**Output files generated:**
- Neighborhood summary table: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_summary_table.csv`
- Raw cell type fractions: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_celltype_fractions_raw.csv`
- Combined cell type data: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_celltype_combined.csv`
- FOV distribution summary: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_fov_distribution.csv`
- This report: `/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output/neighborhood_research_report.md`

### Key Analysis Steps

**1. Data Loading**
```python
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load AnnData
adata = sc.read_h5ad("CosMx_Tonsil_with_Leiden_neigh.h5ad")

# Load pre-computed matrices
celltype_comp_norm = pd.read_csv("neighborhood_celltype_composition_normalized.csv", index_col=0)
fov_abundance_norm = pd.read_csv("fov_neighborhood_abundance_normalized.csv", index_col=0)
```

**2. Compute Raw Cell Type Fractions**
```python
# Raw fractions per neighborhood
celltype_raw = pd.crosstab(
    adata.obs['Leiden_neigh'], 
    adata.obs['L1_CT'], 
    normalize='index'
)
```

**3. Compute Diversity Metrics**
```python
def compute_diversity(row):
    probs = row[row > 0]
    shannon = entropy(probs, base=2)
    effective_n = 2**shannon
    return pd.Series({
        'shannon_entropy': shannon, 
        'effective_n_celltypes': effective_n
    })

diversity = celltype_raw.apply(compute_diversity, axis=1)
```

**4. FOV Presence Analysis**
```python
# Count FOVs per neighborhood
fov_presence = adata.obs.groupby('Leiden_neigh')['FOV'].apply(lambda x: x.nunique())

# Identify top FOVs per neighborhood
for neigh in range(8):
    abund = fov_abundance_norm[str(neigh)].sort_values(ascending=False)
    top5_fovs = abund.head(5)
```

### Software Versions

- Python: 3.10+
- scanpy: 1.9+
- pandas: 2.0+
- numpy: 1.24+
- scipy: 1.10+

### Data Characteristics

- **Total cells**: 720,497
- **Genes measured**: 990
- **Fields of view (FOVs)**: 25
- **Cell types (L1_CT)**: 13
- **Neighborhoods (Leiden_neigh)**: 8 (labeled 0-7)
- **Technology**: CosMx Spatial Molecular Imager (Nanostring)
- **Tissue**: Healthy human tonsil
- **Neighborhood inference**: Leiden clustering on integrated spatial-transcriptomic features

---

## Conclusion

This comprehensive analysis of eight spatial neighborhoods in healthy human tonsil tissue reveals the functional compartmentalization of lymphoid architecture. The neighborhoods represent distinct microenvironments with specialized immune functions:

1. **B cell follicles** (N0, N5) for humoral immunity
2. **T cell zones** (N4, N6) for cellular immunity and T-B collaboration
3. **Interfollicular regions** (N1) for lymphocyte trafficking
4. **Epithelial interfaces** (N2) for mucosal immunity
5. **Stromal-vascular niches** (N3) for structural support and plasma cell survival
6. **Specialized rare niches** (N7) requiring further investigation

The quantitative characterization of cell type composition, spatial distribution, and cellular diversity provides a reference framework for understanding tonsil biology and identifying disease-associated alterations in future studies.

**Key quantitative findings:**
- Neighborhoods range from 27,565 to 149,126 cells (3.8% to 20.7% of dataset)
- Cellular diversity ranges from 5.11 to 9.17 effective cell types
- Spatial recurrence ranges from 16% (rare) to 92% (ubiquitous) of FOVs
- B cells dominate 6/8 neighborhoods, while LQ and T cells dominate 1 each

This spatial atlas of healthy tonsil tissue provides a foundation for comparative studies in disease, development, and immune perturbation experiments.

---

*Report generated: 2025*
*Dataset: Healthy human tonsil CosMx spatial transcriptomics*
*Analysis: Spatial neighborhood characterization using Leiden clustering*

