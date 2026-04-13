# Cell-Cell Interaction Analysis: MU3 → MU4 → MU6 → MU7 → MU9 Trajectory

## Executive Summary

This report analyzes cell-cell interactions along the MU3 → MU4 → MU6 → MU7 → MU9 neighborhood trajectory in a mouse colitis model using CellPhoneDB results. The trajectory reveals a progressive inflammatory cascade characterized by sequential waves of inflammation-associated epithelial (IAE) and fibroblast (IAF) populations, accompanied by dynamic immune cell infiltration.

### Key Findings:

1. **Sequential IAE Dominance**: IAE1 peaks at MU4 (32.7%), transitions to IAE2 at MU6 (45.9%), then IAE3 at MU7 (11.8%), suggesting progressive epithelial remodeling states
2. **Peak Inflammation at MU7**: Maximum cellular complexity (84,095 cells, 71 cell types) and interaction diversity (3,214 interactions) with coordinated IAE3, IAF2, and IAF3 expansion
3. **Fibroblast-Epithelial Signaling**: Dominant Col1a2-Sdc1 and Thbs1-Sdc1 interactions throughout, indicating persistent ECM remodeling and tissue repair signaling
4. **Immune Infiltration Dynamics**: Neutrophil and monocyte infiltration peaks at MU6-MU7 with prominent Mmp9-Itgam interactions, suggesting active tissue remodeling
5. **Resolution Phase at MU9**: Emergence of Epithelial (Clu+) cells (33%) with reduced inflammatory signatures but persistent stromal-epithelial interactions

---

## Trajectory Overview

| Neighborhood | Total Cells | Cell Types | Total Interactions | IA Interactions | Immune Interactions |
|--------------|-------------|------------|-------------------|-----------------|--------------------|
| MU3 | 90,604 | 71 | 2108 | 105 | 689 |
| MU4 | 61,902 | 67 | 858 | 179 | 240 |
| MU6 | 50,683 | 68 | 449 | 149 | 166 |
| MU7 | 84,095 | 71 | 3214 | 938 | 1069 |
| MU9 | 8,295 | 65 | 956 | 301 | 331 |

---

## MU3: Detailed Analysis

### Cell Composition

**Total Cells**: 90,604  
**Unique Cell Types**: 71  

**Top 10 Cell Types**:

1. **Colonocytes**: 44.9% (n=40,685)
2. **Fibro 2**: 11.7% (n=10,588)
3. **Goblet 2**: 7.5% (n=6,753)
4. **TA**: 5.6% (n=5,084)
5. **Capillarial EC**: 4.3% (n=3,939)
6. **Macrophage (Itgax+)**: 3.1% (n=2,776)
7. **SMC 1**: 2.3% (n=2,121)
8. **Plasma cell**: 2.3% (n=2,058)
9. **IAE 1**: 2.2% (n=1,987)
10. **Macrophage (Mrc1+)**: 1.7% (n=1,507)

**Inflammation-Associated Cell Types**:

- **IAE 1**: 2.2% (n=1,987)
- **IAE 2**: 1.0% (n=883)
- **IAE 3**: 0.1% (n=70)
- **IAF 1**: 0.1% (n=79)
- **IAF 2**: 0.4% (n=334)
- **IAF 3**: 0.2% (n=201)
- **IASMC 1**: 0.3% (n=238)

### Top 15 Cell-Cell Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | Plasma cell | Col1a2 | Sdc1 | 0.790 | 0.0000 |
| 2 | Fibro 5 | TA | Col1a2 | Sdc1 | 0.710 | 0.0000 |
| 3 | Fibro 2 | Plasma cell | Col1a2 | Sdc1 | 0.634 | 0.0007 |
| 4 | Fibro 5 | Colonocytes | Col1a2 | Sdc1 | 0.618 | 0.0000 |
| 5 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.614 | 0.0000 |
| 6 | Fibro 6 | Plasma cell | Col1a2 | Sdc1 | 0.596 | 0.0000 |
| 7 | Fibro 13 | Plasma cell | Col1a2 | Sdc1 | 0.575 | 0.0000 |
| 8 | Fibro 5 | Goblet 1 | Col1a2 | Sdc1 | 0.566 | 0.0001 |
| 9 | Fibro 2 | Plasma cell | Col6a1 | Sdc1 | 0.566 | 0.0000 |
| 10 | Fibro 5 | Goblet 2 | Col1a2 | Sdc1 | 0.559 | 0.0000 |
| 11 | Fibro 2 | TA | Col1a2 | Sdc1 | 0.553 | 0.0000 |
| 12 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.549 | 0.0000 |
| 13 | Fibro 5 | Fibro 6 | Col1a2 | Sdc1 | 0.543 | 0.0000 |
| 14 | Glia (Gfra3+) | Fibro 6 | Sparc | Fgfr1 | 0.542 | 0.0000 |
| 15 | Fibro 5 | Fibro 5 | Col1a2 | Sdc1 | 0.535 | 0.0000 |

### Top Inflammation-Associated Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.614 | 0.0000 |
| 2 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.549 | 0.0000 |
| 3 | Fibro 2 | IAE 1 | Col1a2 | Sdc1 | 0.460 | 0.0000 |
| 4 | Fibro 6 | IAE 1 | Col1a2 | Sdc1 | 0.400 | 0.0012 |
| 5 | Fibro 2 | IAE 2 | Col1a2 | Sdc1 | 0.393 | 0.0019 |
| 6 | Fibro 13 | IAE 1 | Col1a2 | Sdc1 | 0.385 | 0.0008 |
| 7 | Fibro 2 | IAE 1 | Col6a1 | Sdc1 | 0.380 | 0.0018 |
| 8 | Fibro 6 | IAE 2 | Col1a2 | Sdc1 | 0.358 | 0.0042 |
| 9 | Fibro 13 | IAE 2 | Col1a2 | Sdc1 | 0.343 | 0.0062 |
| 10 | Fibro 13 | IAE 1 | Col6a1 | Sdc1 | 0.341 | 0.0010 |

### Key Immune-Epithelial Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Goblet 2 | Plasma cell | Thbs1 | Sdc1 | 0.371 | 0.0000 |
| 2 | Colonocytes | Macrophage (Itgax+) | Vegfa | Sirpa | 0.323 | 0.0000 |
| 3 | Macrophage (Mrc1+) | TA | Col1a2 | Sdc1 | 0.319 | 0.0033 |
| 4 | Colonocytes | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.315 | 0.0000 |
| 5 | Colonocytes | Macrophage (Mrc1+) | Vegfa | Sirpa | 0.311 | 0.0002 |
| 6 | Macrophage (Itgax+) | TA | Col1a2 | Sdc1 | 0.304 | 0.0019 |
| 7 | Goblet 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.304 | 0.0003 |
| 8 | Macrophage (Mrc1+) | TA | Col6a1 | Sdc1 | 0.294 | 0.0028 |
| 9 | IAE 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.293 | 0.0006 |
| 10 | Goblet 2 | Macrophage (Mrc1+) | Vegfa | Sirpa | 0.291 | 0.0002 |

---

## MU4: Detailed Analysis

### Cell Composition

**Total Cells**: 61,902  
**Unique Cell Types**: 67  

**Top 10 Cell Types**:

1. **IAE 1**: 32.7% (n=20,237)
2. **Colonocytes**: 23.2% (n=14,352)
3. **TA**: 9.2% (n=5,678)
4. **Goblet 2**: 5.1% (n=3,129)
5. **Fibro 2**: 4.8% (n=2,985)
6. **Capillarial EC**: 2.2% (n=1,385)
7. **Fibro 6**: 1.9% (n=1,146)
8. **Macrophage (Itgax+)**: 1.8% (n=1,119)
9. **Macrophage (Mrc1+)**: 1.8% (n=1,087)
10. **IAF 1**: 1.7% (n=1,062)

**Inflammation-Associated Cell Types**:

- **IAE 1**: 32.7% (n=20,237)
- **IAE 2**: 1.2% (n=729)
- **IAE 3**: 0.2% (n=132)
- **IAF 1**: 1.7% (n=1,062)
- **IAF 2**: 0.9% (n=567)
- **IAF 3**: 0.7% (n=427)
- **IASMC 1**: 0.5% (n=338)

### Top 15 Cell-Cell Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | TA | Col1a2 | Sdc1 | 0.737 | 0.0000 |
| 2 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.675 | 0.0000 |
| 3 | Fibro 5 | Colonocytes | Col1a2 | Sdc1 | 0.663 | 0.0000 |
| 4 | Fibro 5 | Stem cells | Col1a2 | Sdc1 | 0.618 | 0.0000 |
| 5 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.608 | 0.0000 |
| 6 | Fibro 5 | Goblet 1 | Col1a2 | Sdc1 | 0.604 | 0.0000 |
| 7 | Fibro 5 | Goblet 2 | Col1a2 | Sdc1 | 0.595 | 0.0000 |
| 8 | Fibro 5 | Fibro 13 | Col1a2 | Sdc1 | 0.552 | 0.0001 |
| 9 | Fibro 2 | TA | Col1a2 | Sdc1 | 0.544 | 0.0000 |
| 10 | Fibro 5 | Fibro 2 | Col1a2 | Sdc1 | 0.544 | 0.0000 |
| 11 | Fibro 5 | Capillarial EC | Col1a2 | Flt4 | 0.543 | 0.0000 |
| 12 | Fibro 6 | TA | Col1a2 | Sdc1 | 0.528 | 0.0000 |
| 13 | Fibro 13 | TA | Col1a2 | Sdc1 | 0.498 | 0.0000 |
| 14 | Fibro 2 | IAE 1 | Col1a2 | Sdc1 | 0.486 | 0.0000 |
| 15 | Fibro 2 | Colonocytes | Col1a2 | Sdc1 | 0.477 | 0.0000 |

### Top Inflammation-Associated Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.675 | 0.0000 |
| 2 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.608 | 0.0000 |
| 3 | Fibro 2 | IAE 1 | Col1a2 | Sdc1 | 0.486 | 0.0000 |
| 4 | Fibro 6 | IAE 1 | Col1a2 | Sdc1 | 0.469 | 0.0000 |
| 5 | IAF 1 | TA | Col1a2 | Sdc1 | 0.466 | 0.0019 |
| 6 | Fibro 13 | IAE 1 | Col1a2 | Sdc1 | 0.437 | 0.0003 |
| 7 | Fibro 2 | IAE 2 | Col1a2 | Sdc1 | 0.429 | 0.0000 |
| 8 | IAF 1 | IAE 1 | Col1a2 | Sdc1 | 0.426 | 0.0003 |
| 9 | Fibro 2 | IAE 1 | Col6a1 | Sdc1 | 0.412 | 0.0000 |
| 10 | IAF 1 | Colonocytes | Col1a2 | Sdc1 | 0.409 | 0.0014 |

### Key Immune-Epithelial Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Macrophage (Itgax+) | TA | Col1a2 | Sdc1 | 0.321 | 0.0011 |
| 2 | Colonocytes | Macrophage (Itgax+) | Vegfa | Sirpa | 0.303 | 0.0001 |
| 3 | Macrophage (Itgax+) | TA | Col6a1 | Sdc1 | 0.300 | 0.0002 |
| 4 | Colonocytes | Macrophage (Mrc1+) | Vegfa | Sirpa | 0.296 | 0.0004 |
| 5 | Colonocytes | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.294 | 0.0001 |
| 6 | Goblet 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.292 | 0.0000 |
| 7 | Goblet 2 | Macrophage (Mrc1+) | Vegfa | Sirpa | 0.281 | 0.0003 |
| 8 | Goblet 1 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.277 | 0.0009 |
| 9 | Goblet 1 | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.271 | 0.0007 |
| 10 | IAE 1 | Macrophage (Mrc1+) | Vegfa | Sirpa | 0.270 | 0.0000 |

---

## MU6: Detailed Analysis

### Cell Composition

**Total Cells**: 50,683  
**Unique Cell Types**: 68  

**Top 10 Cell Types**:

1. **IAE 2**: 45.9% (n=23,246)
2. **Colonocytes**: 11.6% (n=5,883)
3. **TA**: 7.3% (n=3,695)
4. **Goblet 2**: 4.4% (n=2,226)
5. **Stem cells**: 3.7% (n=1,882)
6. **IAE 1**: 3.5% (n=1,789)
7. **Fibro 2**: 2.6% (n=1,340)
8. **Capillarial EC**: 1.7% (n=843)
9. **Macrophage (Itgax+)**: 1.6% (n=786)
10. **Goblet 1**: 1.5% (n=757)

**Inflammation-Associated Cell Types**:

- **IAE 1**: 3.5% (n=1,789)
- **IAE 2**: 45.9% (n=23,246)
- **IAE 3**: 1.0% (n=508)
- **IAF 1**: 0.2% (n=86)
- **IAF 2**: 0.9% (n=461)
- **IAF 3**: 1.0% (n=521)
- **IASMC 1**: 0.6% (n=314)

### Top 15 Cell-Cell Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | TA | Col1a2 | Sdc1 | 0.656 | 0.0000 |
| 2 | Fibro 5 | Repair associated  (Arg1+) | Col1a2 | Sdc1 | 0.614 | 0.0000 |
| 3 | Fibro 5 | IAE 3 | Col1a2 | Sdc1 | 0.598 | 0.0000 |
| 4 | Neutrophil 1 | Monocyte | Mmp9 | Itgam | 0.574 | 0.0000 |
| 5 | Fibro 5 | Colonocytes | Col1a2 | Sdc1 | 0.573 | 0.0000 |
| 6 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.573 | 0.0000 |
| 7 | IASMC 1 | TA | Thbs1 | Sdc1 | 0.572 | 0.0000 |
| 8 | Fibro 5 | Stem cells | Col1a2 | Sdc1 | 0.569 | 0.0000 |
| 9 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.556 | 0.0000 |
| 10 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.546 | 0.0000 |
| 11 | IASMC 1 | Repair associated  (Arg1+) | Thbs1 | Sdc1 | 0.542 | 0.0000 |
| 12 | Neutrophil 1 | Neutrophil 1 | Mmp9 | Itgam | 0.538 | 0.0000 |
| 13 | Neutrophil 1 | Macrophage (Itgax+) | Mmp9 | Itgam | 0.533 | 0.0000 |
| 14 | Fibro 5 | Goblet 2 | Col1a2 | Sdc1 | 0.516 | 0.0000 |
| 15 | IASMC 1 | IAE 3 | Thbs1 | Sdc1 | 0.513 | 0.0000 |

### Top Inflammation-Associated Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | IAE 3 | Col1a2 | Sdc1 | 0.598 | 0.0000 |
| 2 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.573 | 0.0000 |
| 3 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.556 | 0.0000 |
| 4 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.546 | 0.0000 |
| 5 | IASMC 1 | IAE 3 | Thbs1 | Sdc1 | 0.513 | 0.0000 |
| 6 | IASMC 1 | IAF 3 | Thbs1 | Sdc1 | 0.485 | 0.0000 |
| 7 | IASMC 1 | IAE 1 | Thbs1 | Sdc1 | 0.485 | 0.0000 |
| 8 | IAF 3 | Capillarial EC | Col18a1 | Kdr | 0.471 | 0.0000 |
| 9 | IASMC 1 | IAE 2 | Thbs1 | Sdc1 | 0.464 | 0.0000 |
| 10 | IAF 3 | TA | Col1a2 | Sdc1 | 0.438 | 0.0000 |

### Key Immune-Epithelial Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Monocyte | TA | Thbs1 | Sdc1 | 0.384 | 0.0001 |
| 2 | Monocyte | IAE 3 | Thbs1 | Sdc1 | 0.319 | 0.0013 |
| 3 | Monocyte | Colonocytes | Thbs1 | Sdc1 | 0.304 | 0.0001 |
| 4 | Monocyte | IAE 1 | Thbs1 | Sdc1 | 0.300 | 0.0001 |
| 5 | Colonocytes | Macrophage (Itgax+) | Vegfa | Sirpa | 0.276 | 0.0000 |
| 6 | Monocyte | IAE 2 | Thbs1 | Sdc1 | 0.271 | 0.0031 |
| 7 | Monocyte | TA | Thbs1 | Lrp5 | 0.270 | 0.0028 |
| 8 | Goblet 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.264 | 0.0000 |
| 9 | Colonocytes | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.259 | 0.0000 |
| 10 | IAE 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.256 | 0.0007 |

---

## MU7: Detailed Analysis

### Cell Composition

**Total Cells**: 84,095  
**Unique Cell Types**: 71  

**Top 10 Cell Types**:

1. **IAE 3**: 11.8% (n=9,897)
2. **Stem cells**: 10.1% (n=8,478)
3. **IAF 3**: 9.4% (n=7,902)
4. **IAE 2**: 8.3% (n=6,964)
5. **IAF 2**: 7.6% (n=6,412)
6. **Macrophage (Itgax+)**: 3.5% (n=2,944)
7. **IAE 1**: 3.4% (n=2,901)
8. **SMC 1**: 3.3% (n=2,815)
9. **Monocyte**: 3.0% (n=2,542)
10. **Neutrophil 1**: 2.8% (n=2,331)

**Inflammation-Associated Cell Types**:

- **IAE 1**: 3.4% (n=2,901)
- **IAE 2**: 8.3% (n=6,964)
- **IAE 3**: 11.8% (n=9,897)
- **IAF 1**: 0.8% (n=710)
- **IAF 2**: 7.6% (n=6,412)
- **IAF 3**: 9.4% (n=7,902)
- **IASMC 1**: 2.2% (n=1,853)

### Top 15 Cell-Cell Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | Repair associated  (Arg1+) | Col1a2 | Sdc1 | 0.654 | 0.0000 |
| 2 | Fibro 5 | TA | Col1a2 | Sdc1 | 0.648 | 0.0000 |
| 3 | IASMC 1 | Repair associated  (Arg1+) | Thbs1 | Sdc1 | 0.636 | 0.0000 |
| 4 | IASMC 1 | TA | Thbs1 | Sdc1 | 0.635 | 0.0000 |
| 5 | Fibro 5 | Plasma cell | Col1a2 | Sdc1 | 0.613 | 0.0000 |
| 6 | Fibro 5 | IAE 3 | Col1a2 | Sdc1 | 0.593 | 0.0000 |
| 7 | IASMC 1 | Plasma cell | Thbs1 | Sdc1 | 0.593 | 0.0000 |
| 8 | Fibro 4 | Repair associated  (Arg1+) | Col1a2 | Sdc1 | 0.593 | 0.0000 |
| 9 | Fibro 5 | Colonocytes | Col1a2 | Sdc1 | 0.588 | 0.0000 |
| 10 | Fibro 4 | TA | Col1a2 | Sdc1 | 0.586 | 0.0000 |
| 11 | IASMC 1 | IAE 3 | Thbs1 | Sdc1 | 0.579 | 0.0000 |
| 12 | IASMC 1 | Colonocytes | Thbs1 | Sdc1 | 0.574 | 0.0000 |
| 13 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.569 | 0.0000 |
| 14 | Fibro 5 | EEC | Col1a2 | Sdc1 | 0.562 | 0.0000 |
| 15 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.561 | 0.0000 |

### Top Inflammation-Associated Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | IAE 3 | Col1a2 | Sdc1 | 0.593 | 0.0000 |
| 2 | IASMC 1 | IAE 3 | Thbs1 | Sdc1 | 0.579 | 0.0000 |
| 3 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.569 | 0.0000 |
| 4 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.561 | 0.0000 |
| 5 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.560 | 0.0000 |
| 6 | IASMC 1 | IAE 1 | Thbs1 | Sdc1 | 0.556 | 0.0000 |
| 7 | IASMC 1 | IAF 3 | Thbs1 | Sdc1 | 0.550 | 0.0000 |
| 8 | Fibro 5 | IAF 1 | Col1a2 | Sdc1 | 0.541 | 0.0000 |
| 9 | IASMC 1 | IAE 2 | Thbs1 | Sdc1 | 0.541 | 0.0000 |
| 10 | Fibro 4 | IAE 3 | Col1a2 | Sdc1 | 0.538 | 0.0000 |

### Key Immune-Epithelial Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Neutrophil 1 | IAE 3 | Mmp9 | Ifnar1 | 0.458 | 0.0000 |
| 2 | Neutrophil 1 | IAE 3 | Mmp9 | Cd44 | 0.431 | 0.0000 |
| 3 | Monocyte | TA | Thbs1 | Sdc1 | 0.393 | 0.0000 |
| 4 | Monocyte | IAE 3 | Thbs1 | Sdc1 | 0.345 | 0.0001 |
| 5 | Neutrophil 2 | Goblet 1 | Cd14 | Tlr1 | 0.336 | 0.0028 |
| 6 | Monocyte | Colonocytes | Thbs1 | Sdc1 | 0.332 | 0.0002 |
| 7 | Neutrophil 2 | TA | Cd14 | Tlr1 | 0.321 | 0.0014 |
| 8 | Monocyte | IAE 1 | Thbs1 | Sdc1 | 0.320 | 0.0001 |
| 9 | Neutrophil 2 | TA | Thbs1 | Sdc1 | 0.318 | 0.0002 |
| 10 | Monocyte | IAE 2 | Thbs1 | Sdc1 | 0.301 | 0.0002 |

---

## MU9: Detailed Analysis

### Cell Composition

**Total Cells**: 8,295  
**Unique Cell Types**: 65  

**Top 10 Cell Types**:

1. **Epithelial (Clu+)**: 33.0% (n=2,741)
2. **Colonocytes**: 12.1% (n=1,006)
3. **Fibro 2**: 5.2% (n=435)
4. **TA**: 5.2% (n=432)
5. **SMC 1**: 5.2% (n=428)
6. **IAE 2**: 5.0% (n=412)
7. **Fibro 5**: 3.2% (n=267)
8. **Goblet 2**: 2.9% (n=242)
9. **Goblet 1**: 2.2% (n=180)
10. **Stem cells**: 2.1% (n=177)

**Inflammation-Associated Cell Types**:

- **IAE 1**: 1.4% (n=115)
- **IAE 2**: 5.0% (n=412)
- **IAE 3**: 0.9% (n=71)
- **IAF 1**: 0.0% (n=3)
- **IAF 2**: 1.0% (n=82)
- **IAF 3**: 0.5% (n=42)
- **IASMC 1**: 1.1% (n=89)

### Top 15 Cell-Cell Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | Plasma cell | Col1a2 | Sdc1 | 0.675 | 0.0000 |
| 2 | Fibro 5 | TA | Col1a2 | Sdc1 | 0.638 | 0.0000 |
| 3 | Fibro 5 | Epithelial (Clu+) | Col1a2 | Sdc1 | 0.586 | 0.0000 |
| 4 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.586 | 0.0000 |
| 5 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.578 | 0.0000 |
| 6 | Fibro 5 | Colonocytes | Col1a2 | Sdc1 | 0.558 | 0.0000 |
| 7 | Fibro 5 | Stem cells | Col1a2 | Sdc1 | 0.552 | 0.0000 |
| 8 | Fibro 5 | Goblet 2 | Col1a2 | Sdc1 | 0.546 | 0.0000 |
| 9 | Fibro 2 | Plasma cell | Col1a2 | Sdc1 | 0.544 | 0.0000 |
| 10 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.542 | 0.0000 |
| 11 | Fibro 5 | Goblet 1 | Col1a2 | Sdc1 | 0.532 | 0.0000 |
| 12 | Fibro 2 | TA | Col1a2 | Sdc1 | 0.507 | 0.0000 |
| 13 | Fibro 5 | Treg | Col1a2 | Itgb3 | 0.505 | 0.0000 |
| 14 | Fibro 5 | Fibro 5 | Col1a2 | Sdc1 | 0.498 | 0.0000 |
| 15 | Fibro 5 | Treg | Col1a2 | Cd44 | 0.493 | 0.0000 |

### Top Inflammation-Associated Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Fibro 5 | IAE 1 | Col1a2 | Sdc1 | 0.586 | 0.0000 |
| 2 | Fibro 5 | IAE 2 | Col1a2 | Sdc1 | 0.578 | 0.0000 |
| 3 | Fibro 5 | IAF 3 | Col1a2 | Sdc1 | 0.542 | 0.0000 |
| 4 | Fibro 5 | IAF 3 | Col1a2 | Cd44 | 0.457 | 0.0000 |
| 5 | Fibro 2 | IAE 1 | Col1a2 | Sdc1 | 0.455 | 0.0000 |
| 6 | Fibro 2 | IAE 2 | Col1a2 | Sdc1 | 0.447 | 0.0000 |
| 7 | IAF 3 | Plasma cell | Col1a2 | Sdc1 | 0.429 | 0.0000 |
| 8 | Fibro 2 | IAF 3 | Col1a2 | Sdc1 | 0.410 | 0.0003 |
| 9 | Fibro 5 | IAF 3 | Sparc | Fgfr1 | 0.406 | 0.0007 |
| 10 | IAF 3 | TA | Col1a2 | Sdc1 | 0.392 | 0.0005 |

### Key Immune-Epithelial Interactions

| Rank | Source | Target | Ligand | Receptor | Mean Expression | P-value |
|------|--------|--------|--------|----------|-----------------|--------|
| 1 | Colonocytes | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.275 | 0.0000 |
| 2 | Plasma cell | TA | Col1a2 | Sdc1 | 0.263 | 0.0010 |
| 3 | Colonocytes | Macrophage (Itgax+) | Vegfa | Sirpa | 0.263 | 0.0000 |
| 4 | Goblet 1 | Plasma cell | Thbs1 | Sdc1 | 0.262 | 0.0000 |
| 5 | Goblet 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.250 | 0.0000 |
| 6 | IAE 3 | Plasma cell | Thbs1 | Sdc1 | 0.248 | 0.0185 |
| 7 | Goblet 1 | Plasma cell | Tnfsf13 | Tnfrsf13b | 0.244 | 0.0000 |
| 8 | IAE 2 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.241 | 0.0000 |
| 9 | Goblet 2 | Macrophage (Itgax+) | Cx3cl1 | Cx3cr1 | 0.235 | 0.0000 |
| 10 | Goblet 1 | Macrophage (Itgax+) | Vegfa | Sirpa | 0.233 | 0.0000 |

---

## Biological Interpretation & Key Insights

### 1. Progressive Epithelial Remodeling Cascade

The trajectory reveals a sequential activation of inflammation-associated epithelial states:

- **MU3 (Baseline)**: Predominantly healthy colonocytes (44.9%) with minimal IAE presence (2.2% IAE1)
- **MU4 (Early Inflammation)**: Dramatic IAE1 expansion to 32.7%, suggesting initial epithelial stress response
- **MU6 (Peak IAE2)**: IAE2 dominates at 45.9%, indicating transition to a more severe inflammatory epithelial state
- **MU7 (Maximal Complexity)**: Balanced IAE1/IAE2/IAE3 with peak IAE3 (11.8%), representing the most severe inflammatory epithelial phenotype
- **MU9 (Resolution/Repair)**: Emergence of Epithelial (Clu+) cells (33%), potentially representing a repair-associated epithelial state

**Biological Significance**: This sequential IAE progression (IAE1→IAE2→IAE3) mirrors the temporal progression of colitis severity, with each IAE subtype potentially representing distinct epithelial damage/repair states.

### 2. Coordinated Fibroblast Activation

Inflammation-associated fibroblasts (IAF) show parallel expansion:

- **MU4**: IAF1 emergence (1.7%), early stromal response
- **MU6**: IAF3 expansion (1.0%), coordinated with IAE2 peak
- **MU7**: Peak IAF2 (7.6%) and IAF3 (9.4%), maximal fibroblast activation

**Key Interactions**:
- **Col1a2-Sdc1**: Dominant fibroblast→epithelial interaction throughout (mean expression 0.3-0.7)
  - Indicates persistent ECM deposition and epithelial-stromal crosstalk
  - Fibro5 consistently shows highest expression levels
- **Thbs1-Sdc1**: Emerges strongly at MU6-MU7 via IASMC1 (mean ~0.5-0.6)
  - Thrombospondin-1 signaling suggests TGF-β pathway activation and matrix remodeling
- **IAF→IAE interactions**: Direct IAF3→IAE crosstalk peaks at MU7 (47 interactions)
  - Includes Col1a2-Sdc1, Col18a1-Itgb5, Cxcl12-Itgav, Hgf-Sdc1
  - HGF signaling (Hgf-Sdc1) suggests hepatocyte growth factor-mediated epithelial repair attempts

**Biological Significance**: IAF activation and ECM remodeling are hallmarks of tissue fibrosis. The persistent Col1a2-Sdc1 interactions suggest ongoing collagen deposition, while HGF signaling indicates compensatory repair mechanisms.

### 3. Immune Cell Infiltration & Inflammatory Signaling

**Neutrophil Infiltration (MU6-MU7)**:
- Neutrophil 1 peaks at MU7 (2.8%) with dominant **Mmp9-Itgam** interactions
  - Mmp9-Itgam (mean 0.52-0.57): Neutrophil→Monocyte, Neutrophil→Macrophage
  - Matrix metalloproteinase-9 signaling indicates active tissue degradation
  - Mmp9-Cd44 and Mmp9-Ifnar1 to IAE3 (mean ~0.43-0.46) suggests direct neutrophil-epithelial damage

**Macrophage Dynamics**:
- **MU3-MU4**: Macrophage (Itgax+) interactions with endothelium via **Tgfb1-Acvrl1** (mean 0.37-0.40)
  - TGF-β signaling to endothelial cells suggests vascular remodeling
- **MU4-MU7**: Macrophage→Fibroblast **Tgfb1-Tgfbr3** interactions (mean 0.30-0.38)
  - Direct macrophage-to-fibroblast TGF-β signaling drives fibrotic responses
- Epithelial→Macrophage **Vegfa-Sirpa** and **Cx3cl1-Cx3cr1** persist throughout
  - VEGF and fractalkine signaling maintains macrophage recruitment and activation

**Monocyte Infiltration (MU6-MU7)**:
- Monocyte expansion from 0.7% (MU6) to 3.0% (MU7)
- **Monocyte→Epithelial Thbs1-Sdc1** interactions (mean 0.30-0.39)
  - Thrombospondin-1 from monocytes to TA, IAE1, IAE2, IAE3
  - Suggests monocyte-mediated epithelial remodeling

**Biological Significance**: The coordinated neutrophil-monocyte-macrophage infiltration at MU6-MU7 represents the peak inflammatory phase. Mmp9 signaling indicates active tissue destruction, while TGF-β pathways drive fibrotic responses. The persistence of these interactions suggests chronic inflammation rather than acute injury.

### 4. Epithelial-Immune Crosstalk

**Epithelial Signals to Immune Cells**:
- **Vegfa-Sirpa**: Colonocytes/Goblet cells → Macrophages (mean 0.26-0.32)
  - VEGF signaling recruits and activates macrophages
- **Cx3cl1-Cx3cr1**: Colonocytes → Macrophages (mean 0.26-0.32)
  - Fractalkine-CX3CR1 axis maintains tissue-resident macrophage populations
- **C3-Itgax**: IAE2 → cDC1 at MU9 (mean 0.34)
  - Complement C3 signaling to dendritic cells suggests antigen presentation

**Immune Signals to Epithelium**:
- **Mmp9-Cd44/Ifnar1**: Neutrophils → IAE3 at MU7 (mean 0.43-0.46)
  - Direct neutrophil-mediated epithelial damage
- **Thbs1-Sdc1**: Monocytes → TA/IAE (mean 0.30-0.39)
  - Monocyte-derived thrombospondin-1 modulates epithelial behavior

**Biological Significance**: Bidirectional epithelial-immune communication drives the inflammatory cascade. Epithelial VEGF and fractalkine maintain immune infiltration, while immune-derived MMPs and thrombospondin-1 directly damage and remodel epithelium.

### 5. Wnt Signaling in Epithelial Homeostasis

**Wnt5a-Fzd5** interactions persist throughout the trajectory:
- Fibro2 → Colonocytes/TA (mean 0.19-0.28)
- IAF3 → IAE1/IAE2 at MU6-MU9 (mean 0.09-0.24)

**Wnt4 signaling** emerges in IAF→IAE interactions:
- IAF1 → IAE1 at MU4: Wnt4-Lrp6, Wnt4-Lrp5
- IAF3 → IAE2/IAE3 at MU6-MU7: Wnt4-Fzd5, Wnt4-Fzd1

**Biological Significance**: Wnt signaling from fibroblasts to epithelium is critical for intestinal stem cell maintenance and epithelial regeneration. The persistence of Wnt5a and emergence of Wnt4 suggest ongoing attempts at epithelial repair despite inflammatory damage.

### 6. Vascular Remodeling

**Endothelial Interactions**:
- **Macrophage→Endothelium Tgfb1-Acvrl1** (mean 0.37-0.40)
  - TGF-β signaling to capillary endothelium throughout trajectory
  - ALK1 (Acvrl1) receptor activation promotes angiogenesis
- **IAF3→Capillarial EC Col18a1-Kdr** at MU6 (mean 0.47)
  - Collagen XVIII (endostatin precursor) to VEGFR2 signaling
  - Suggests anti-angiogenic signals from activated fibroblasts

**Biological Significance**: Vascular remodeling is a key feature of IBD. The balance between pro-angiogenic (TGF-β/ALK1) and anti-angiogenic (Col18a1/endostatin) signals reflects the complex vascular changes during inflammation.

## Therapeutic Implications

### 1. Targeting ECM Remodeling

**Target**: Col1a2-Sdc1 and Thbs1-Sdc1 interactions

**Rationale**:
- Dominant interactions throughout trajectory (mean 0.3-0.7)
- Persistent collagen deposition indicates ongoing fibrosis
- Syndecan-1 (Sdc1) is a key ECM receptor on epithelial cells

**Potential Strategies**:
- Anti-fibrotic agents targeting collagen synthesis (e.g., pirfenidone analogs)
- Syndecan-1 modulation to reduce ECM-epithelial interactions
- Thrombospondin-1 inhibitors to block TGF-β activation

### 2. Neutrophil-Targeted Therapies

**Target**: Mmp9-Itgam and Mmp9-Cd44 interactions

**Rationale**:
- Peak at MU6-MU7 during maximal inflammation
- Direct neutrophil-mediated epithelial damage
- MMP9 drives tissue destruction

**Potential Strategies**:
- MMP9 inhibitors (e.g., andecaliximab, currently in clinical trials)
- Neutrophil elastase inhibitors
- CXCR2 antagonists to block neutrophil recruitment
- Anti-integrin therapies targeting Mac-1 (CD11b/CD18, Itgam/Itgb2)

### 3. TGF-β Pathway Modulation

**Target**: Tgfb1-Acvrl1, Tgfb1-Tgfbr3 interactions

**Rationale**:
- Macrophage→Endothelium and Macrophage→Fibroblast TGF-β signaling
- Drives both vascular remodeling and fibrosis
- Persistent throughout trajectory

**Potential Strategies**:
- Pan-TGF-β inhibitors (caution: may impair epithelial repair)
- Selective ALK1 inhibitors to target pathological angiogenesis
- TGFβR3 (betaglycan) modulation to reduce fibroblast activation

### 4. Epithelial Repair Enhancement

**Target**: Wnt signaling pathways and HGF-Sdc1

**Rationale**:
- Wnt5a-Fzd5 and Wnt4-Lrp5/Lrp6 maintain epithelial regeneration
- HGF-Sdc1 (IAF3→IAE) emerges at MU7 as repair signal
- Endogenous repair mechanisms are active but insufficient

**Potential Strategies**:
- Wnt agonists to enhance epithelial stem cell activity
- Recombinant HGF or HGF mimetics to boost epithelial repair
- R-spondin proteins to amplify Wnt signaling

### 5. Macrophage Repolarization

**Target**: Vegfa-Sirpa and Cx3cl1-Cx3cr1 axes

**Rationale**:
- Epithelial VEGF and fractalkine maintain pro-inflammatory macrophages
- Persistent throughout trajectory
- Blocking recruitment may reduce chronic inflammation

**Potential Strategies**:
- Anti-VEGF therapies (e.g., bevacizumab, already used in some IBD cases)
- CX3CR1 antagonists to disrupt fractalkine signaling
- Macrophage repolarization agents (e.g., IL-10, CSF-1R inhibitors)

### 6. Stage-Specific Interventions

**Early Stage (MU3-MU4)**:
- Target IAE1 expansion and early fibroblast activation
- Anti-inflammatory biologics (anti-TNF, anti-IL-12/23)
- Early anti-fibrotic intervention

**Peak Inflammation (MU6-MU7)**:
- Neutrophil-targeted therapies (MMP9 inhibitors)
- Anti-integrin therapies (vedolizumab)
- TGF-β pathway inhibition

**Resolution Phase (MU9)**:
- Epithelial repair enhancement (Wnt agonists, HGF)
- Anti-fibrotic maintenance therapy
- Macrophage repolarization to promote tissue healing

---

## Unexpected Findings & Novel Insights

### 1. IASMC1 Emerges as Key Player at MU6-MU7

**Observation**: Inflammation-associated smooth muscle cells (IASMC1) show dramatic Thbs1-Sdc1 interactions with epithelium (mean 0.46-0.64)

**Significance**: Smooth muscle cells are not typically considered primary inflammatory mediators. Their activation and thrombospondin-1 secretion suggests a broader role in tissue remodeling than previously appreciated. This may represent a novel therapeutic target.

### 2. Sequential IAE Subtypes Suggest Distinct Epithelial States

**Observation**: IAE1→IAE2→IAE3 progression with distinct interaction profiles

**Significance**: Rather than a single "damaged epithelium" state, the data suggest at least three distinct inflammation-associated epithelial phenotypes with different stromal and immune interactions. This challenges the binary "healthy vs. damaged" epithelium model and suggests more nuanced epithelial responses to inflammation.

### 3. Epithelial (Clu+) Cells at MU9: Resolution or Metaplasia?

**Observation**: 33% of MU9 cells are Epithelial (Clu+), a distinct population not prominent in earlier neighborhoods

**Significance**: Clusterin (Clu) expression is associated with cellular stress, apoptosis, and tissue remodeling. This population may represent:
- A repair-associated epithelial state
- A metaplastic transformation
- A transitional state between inflammation and resolution

Further characterization of this population is warranted.

### 4. Persistent Col1a2-Sdc1 Despite Changing Cellular Composition

**Observation**: Col1a2-Sdc1 remains the dominant interaction across all neighborhoods despite dramatic shifts in cell composition

**Significance**: This suggests that ECM-epithelial interactions are a fundamental feature of the inflammatory microenvironment, independent of specific cell type composition. Targeting this interaction may have broad therapeutic efficacy across disease stages.

### 5. Lack of Canonical Inflammatory Cytokine Interactions

**Observation**: Limited detection of TNF-TNFR, IL1B-IL1R1, IL6-IL6RA interactions

**Significance**: This may reflect:
- Spatial segregation of cytokine signaling (not captured in these neighborhoods)
- Paracrine vs. autocrine signaling patterns
- Limitations of the MERFISH gene panel
- Dominance of ECM and growth factor signaling over cytokine signaling in spatial context

---

## Conclusions

This comprehensive analysis of the MU3→MU4→MU6→MU7→MU9 trajectory reveals:

1. **Progressive Inflammatory Cascade**: Sequential activation of IAE1→IAE2→IAE3 epithelial states with coordinated IAF expansion represents a spatial-temporal map of colitis progression

2. **ECM Remodeling Dominates**: Col1a2-Sdc1 and Thbs1-Sdc1 interactions are the most abundant and persistent, highlighting ECM-epithelial crosstalk as a central feature of IBD pathology

3. **Peak Inflammation at MU7**: Maximal cellular complexity, immune infiltration, and interaction diversity characterize the most severe inflammatory state, with coordinated neutrophil, monocyte, and macrophage activity

4. **Therapeutic Opportunities**: Multiple targetable pathways identified:
   - ECM remodeling (Col1a2, Thbs1)
   - Neutrophil-mediated damage (Mmp9)
   - TGF-β signaling (fibrosis and angiogenesis)
   - Epithelial repair (Wnt, HGF)
   - Macrophage activation (VEGF, CX3CL1)

5. **Novel Insights**:
   - IASMC1 as unexpected inflammatory mediator
   - Distinct IAE subtypes with unique interaction profiles
   - Epithelial (Clu+) as potential resolution/metaplasia marker

6. **Clinical Implications**: The trajectory suggests that IBD treatment should be stage-specific, with early anti-inflammatory intervention, peak-stage neutrophil targeting, and resolution-phase epithelial repair enhancement

---

## Recommendations for Future Analysis

1. **Validate IAE/IAF subtypes**: Perform differential expression analysis to characterize molecular signatures of IAE1, IAE2, IAE3 and IAF1, IAF2, IAF3

2. **Characterize Epithelial (Clu+)**: Determine if this represents a repair state, metaplasia, or transitional phenotype through marker gene analysis

3. **IASMC1 functional studies**: Investigate the role of inflammation-associated smooth muscle cells in tissue remodeling and potential therapeutic targeting

4. **Spatial context**: Analyze the physical proximity of interacting cell types to validate CellPhoneDB predictions with spatial colocalization

5. **Disease stage mapping**: Correlate neighborhoods with DSS treatment timepoints (DSS3, DSS9, DSS21) to establish temporal disease progression

6. **Cross-trajectory comparison**: Compare this trajectory with other neighborhood progressions (e.g., follicle-associated trajectories) to identify common vs. unique inflammatory patterns

7. **Therapeutic validation**: Test predicted therapeutic targets (MMP9 inhibitors, anti-TGF-β, Wnt agonists) in experimental colitis models

---

## Methods Summary

**Data Source**: MERFISH spatial transcriptomics of mouse colon (colitis model)

**Analysis Tool**: CellPhoneDB statistical method with permutation testing

**Neighborhoods Analyzed**: MU3, MU4, MU6, MU7, MU9 (sequential trajectory)

**Cell Type Annotation**: Tier3 annotations including inflammation-associated subtypes (IAE, IAF, IASMC)

**Interaction Filtering**: Significant interactions (p < 0.05) with mean expression > 0

**Focus**: Inflammation-associated cell types, immune-epithelial crosstalk, stromal-epithelial interactions, and ligand-receptor pairs relevant to IBD pathology

---

**Report Generated**: 2026-02-02 23:36:03
**Analysis Directory**: /tmp/trajectory_analysis_MU3_MU4_MU6_MU7_MU9/
