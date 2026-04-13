# Spatial Neighborhood Analysis Summary
## Human Tonsil CosMx Data

### Dataset
- Total cells: 720,497
- FOVs: 25
- Cell types (L1_CT): 13

### Spatial Neighborhoods
- Global neighborhoods identified: 8
- Mean FOVs per neighborhood: 18.1
- All neighborhoods recur across multiple FOVs: True

### Neighborhood Characteristics

**Neighborhood 0**
- Cells: 149,126 (20.7%)
- Present in 22 FOVs
- Top cell types: B cell (54.5%), LQ (10.6%), fDC (9.2%)

**Neighborhood 1**
- Cells: 132,606 (18.4%)
- Present in 23 FOVs
- Top cell types: B cell (41.5%), LQ (14.8%), T cell CD4 (7.5%)

**Neighborhood 2**
- Cells: 123,005 (17.1%)
- Present in 21 FOVs
- Top cell types: B cell (24.6%), Basal cell (24.5%), LQ (15.6%)

**Neighborhood 3**
- Cells: 85,963 (11.9%)
- Present in 19 FOVs
- Top cell types: B cell (22.8%), LQ (19.1%), Fibroblast (10.8%)

**Neighborhood 4**
- Cells: 74,145 (10.3%)
- Present in 17 FOVs
- Top cell types: B cell (35.3%), T cell CD4 (16.1%), LQ (15.5%)

**Neighborhood 5**
- Cells: 64,545 (9.0%)
- Present in 20 FOVs
- Top cell types: B cell (41.6%), LQ (12.7%), fDC (8.7%)

**Neighborhood 6**
- Cells: 63,542 (8.8%)
- Present in 19 FOVs
- Top cell types: LQ (20.1%), B cell (19.5%), T cell CD4 (16.3%)

**Neighborhood 7**
- Cells: 27,565 (3.8%)
- Present in 4 FOVs
- Top cell types: LQ (33.5%), B cell (20.8%), Plasma cell (7.6%)


### Files Generated
1. CosMx_Tonsil_with_Leiden_neigh.h5ad - Updated AnnData
2. per_cell_data.csv - Per-cell table with FOV, L1_CT, Leiden_neigh, x, y
3. neighborhood_celltype_composition_normalized.csv - Neighborhood × Cell type matrix
4. fov_neighborhood_abundance_normalized.csv - FOV × Neighborhood matrix
5. neighborhood_mapping.csv - FOV-neighborhood to global neighborhood mapping
6. neighborhood_recurrence_matrix.csv - FOV × Global neighborhood presence matrix
7. Figures (8 total) - Heatmaps, dotplots, and spatial visualizations
