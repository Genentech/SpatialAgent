# Neighborhood Trajectory Inference in DSS-Induced Colitis
## Spatial Tissue Reorganization Pathways

---

## Executive Summary

This analysis infers **7 major neighborhood trajectory paths** during DSS-induced colitis progression in mouse colon, based on temporal abundance dynamics, cell type composition shifts, and spatial continuity.

### Key Findings

**1. Seven Distinct Trajectory Paths:**
- Epithelial Homeostasis → Inflammation → Recovery
- Stem Cell Niche Disruption → Recovery
- Lymphoid Structure Formation
- Smooth Muscle Remodeling
- Mesenchymal Activation
- Acute Immune Infiltration
- Plasma Cell Niche Formation

**2. Temporal Waves:**
- Wave 1 (DSS3): Early activation - 7 neighborhoods peak
- Wave 2 (DSS9): Peak inflammation - 12 neighborhoods peak
- Wave 3 (DSS21): Recovery - 4 specialized repair neighborhoods emerge

**3. Gene Expression Transitions:**
- Inflammatory: ↑ Cxcl10, Il1b, Igfbp5
- ECM loss: ↓ Col1a2, Col6a1, Dpt
- Repair: ↑ Arg1, Clu, Aicda

---

## Detailed Trajectory Paths

### Epithelial Homeostasis → Inflammation → Recovery

**Path**: MU3 → MU4 → MU6 → MU7 → MU9

**Stages**: Healthy → DSS3 → DSS9 → DSS9 → DSS21

**Description**: Differentiated epithelium → Early IAE → Peak IAE → IAE-stromal interface → Regenerative epithelium

**Key Transitions:**
- MU3→MU4: Colonocytes/Goblet → IAE 1 (early stress response)
- MU4→MU6: IAE 1 → IAE 2 (peak inflammation)
- MU6→MU7: IAE 2 → IAE 3 (epithelial-stromal remodeling)
- MU7→MU9: IAE 3 → Clu+ epithelium (regeneration)

**Abundance Dynamics:**
- MU3: H=1.00, D3=0.43, D9=0.12, D21=0.35
- MU4: H=0.02, D3=1.00, D9=0.04, D21=0.01
- MU6: H=0.00, D3=0.00, D9=1.00, D21=0.03
- MU7: H=0.00, D3=0.00, D9=1.00, D21=0.02
- MU9: H=0.00, D3=0.00, D9=0.02, D21=1.00

---

### Stem Cell Niche Disruption → Recovery

**Path**: MU1 → MU2 → MU8 → MU10

**Stages**: Healthy → DSS3 → DSS9 → DSS21

**Description**: Stem cell crypt → TA expansion → Severe inflammation → Repair macrophages

**Key Transitions:**
- MU1→MU2: Stem cells/Goblet → TA cells (proliferative response)
- MU2→MU8: TA depletion → Severe inflammation (Cxcl10+ Mac, Neutrophils, IAF)
- MU8→MU10: Inflammation resolution → Arg1+ repair macrophages

**Abundance Dynamics:**
- MU1: H=1.00, D3=0.29, D9=0.88, D21=0.25
- MU2: H=0.62, D3=1.00, D9=0.10, D21=0.06
- MU8: H=0.00, D3=0.00, D9=1.00, D21=0.01
- MU10: H=0.00, D3=0.06, D9=0.01, D21=1.00

---

### Lymphoid Structure Formation

**Path**: FOL2 → FOL1 → FOL3

**Stages**: DSS3 → DSS9 → DSS21

**Description**: Innate lymphoid induction → Active B/T follicles → Mature germinal centers

**Key Transitions:**
- FOL2→FOL1: ILC3/M cells → B cell 2/T cell follicles (tertiary lymphoid formation)
- FOL1→FOL3: Active follicles → Aicda+ B cells (germinal center maturation)

**Abundance Dynamics:**
- FOL2: H=0.05, D3=1.00, D9=0.00, D21=0.00
- FOL1: H=0.15, D3=0.06, D9=1.00, D21=0.24
- FOL3: H=0.15, D3=0.00, D9=0.07, D21=1.00

---

### Smooth Muscle Remodeling

**Path**: SM1 → SM2 → SM3

**Stages**: DSS3 → DSS9 → DSS21

**Description**: Lymphatic-immune niche → Inflammation-associated SMC → Fibrotic remodeling

**Key Transitions:**
- SM1→SM2: ILC2/Lymphatic EC/Mrc1+ Mac → IASMC 1/Monocytes (inflammatory activation)
- SM2→SM3: IASMC 1 → Fibro 4 dominance (fibrotic remodeling)

**Abundance Dynamics:**
- SM1: H=0.52, D3=1.00, D9=0.07, D21=0.12
- SM2: H=0.00, D3=0.00, D9=1.00, D21=0.01
- SM3: H=0.01, D3=0.02, D9=0.40, D21=1.00

---

### Mesenchymal Activation

**Path**: ME1 → ME3 → ME4

**Stages**: DSS3 → DSS9 → DSS9

**Description**: Neuromuscular control → IASMC emergence → IASMC-lymphatic interface

**Key Transitions:**
- ME1→ME3: SMC 1/ICC 2 → IASMC 2/Fibro 1 (smooth muscle inflammation)
- ME3→ME4: IASMC 2 → IASMC 3/DC/Lymphatic EC (immune-vascular niche)

**Abundance Dynamics:**
- ME1: H=0.94, D3=1.00, D9=0.36, D21=0.53
- ME3: H=0.00, D3=0.00, D9=1.00, D21=0.02
- ME4: H=0.00, D3=0.02, D9=1.00, D21=0.02

---

### Acute Immune Infiltration

**Path**: MU5 → MU8 → LUM

**Stages**: DSS9 → DSS9 → DSS9

**Description**: Immune cell aggregation → Severe inflammation → Luminal damage

**Key Transitions:**
- MU5→MU8: Itgax+ Mac/NK/IAF 5 → Cxcl10+ Mac/Neutrophil/IAF 2 (inflammatory escalation)
- MU8→LUM: Severe inflammation → Neutrophil-dominant barrier breach

**Abundance Dynamics:**
- MU5: H=0.00, D3=0.00, D9=1.00, D21=0.01
- MU8: H=0.00, D3=0.00, D9=1.00, D21=0.01
- LUM: H=0.00, D3=0.00, D9=1.00, D21=0.02

---

### Plasma Cell Niche Formation

**Path**: FOL1 → MU11

**Stages**: DSS9 → DSS9-DSS21

**Description**: B/T cell follicles → Plasma cell niches

**Key Transitions:**
- FOL1→MU11: B cell 2/Tfh → Plasma cells/Treg/cDC1 (antibody production niche)

**Abundance Dynamics:**
- FOL1: H=0.15, D3=0.06, D9=1.00, D21=0.24
- MU11: H=0.00, D3=0.00, D9=1.00, D21=1.00

---

## Gene Expression Dynamics

### Inflammatory Activation

| Gene | Function | Trajectory |
|------|----------|------------|
| Cxcl10 | T cell chemotaxis | MU8, LUM |
| Il1b | Pro-inflammatory | MU6-MU7 |
| Igfbp5 | Fibrosis | SM3 |

### ECM Degradation

| Gene | Function | Trajectory |
|------|----------|------------|
| Col1a2 | Collagen I | IAF transitions |
| Col6a1 | Collagen VI | IAF transitions |
| Dpt | ECM organization | Stromal remodeling |

---

## Conclusions

Neighborhood trajectories reveal:

1. **Coordinated tissue reorganization** across multiple compartments
2. **Stage-specific neighborhoods** that emerge and resolve in waves
3. **Incomplete recovery** - specialized repair states persist
4. **Gene programs** aligned with neighborhood transitions

See visualizations:
- `neighborhood_trajectory_network.png`
- `trajectory_abundance_dynamics.png`
- `trajectory_gene_markers.png`
