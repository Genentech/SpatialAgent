# Spatial Neighborhood Characterization Report
## MERFISH Mouse Colon During DSS-Induced Colitis

---

## Executive Summary

This comprehensive analysis characterizes **26 distinct spatial neighborhoods** in mouse colon tissue during DSS-induced colitis progression. Using MERFISH spatial transcriptomics data from **1,348,582 cells** across **52 tissue slices**, we mapped the cellular composition, anatomical location, and temporal dynamics of each neighborhood through 4 disease stages: Healthy, DSS3 (day 3), DSS9 (day 9 - peak inflammation), and DSS21 (day 21 - chronic/recovery).

### Key Findings

**Tissue Reorganization:**
- **9 neighborhoods** are inflammation-specific, emerging only during disease (DSS3-DSS21)
- **3 neighborhoods** are healthy-specific, depleted during inflammation  
- **4 neighborhoods** are recovery-specific, emerging at DSS21
- **10 neighborhoods** show multi-stage presence with dynamic abundance shifts

**Spatial Compartmentalization:**
- **Mucosal neighborhoods (MU1-MU11)**: 11 neighborhoods representing epithelial crypts, immune infiltration, and regeneration
- **Smooth muscle neighborhoods (SM1-SM3)**: 3 neighborhoods in muscularis layers
- **Mesenchymal neighborhoods (ME1-ME4, MES1-MES3)**: 7 neighborhoods representing stromal, neuronal, and vascular niches
- **Follicular neighborhoods (FOL1-FOL3)**: 3 neighborhoods representing organized lymphoid structures

**Inflammation-Associated Populations:**
- **IAE (Inflammation-Associated Epithelium)**: Dominant in MU4, MU6, MU7
- **IAF (Inflammation-Associated Fibroblasts)**: Dominant in MU5, MU7, MU8
- **IASMC (Inflammation-Associated Smooth Muscle)**: Dominant in ME3, ME4, SM2

---

## Detailed Neighborhood Descriptions


### Homeostatic Epithelial/Mucosal Neighborhoods

#### **MU1**

**Cellular Composition:**
- Goblet 1: 1.00
- Stem cells: 1.00
- EEC: 1.00
- Fibro 6: 0.97
- Fibro 13: 0.81

**Temporal Dynamics:**
- Healthy: 1.00 → DSS3: 0.29 → DSS9: 0.88 → DSS21: 0.25
- Peak stage: **Healthy**

#### **MU2**

**Cellular Composition:**
- TA: 1.00
- Fibro 13: 1.00
- Goblet 1: 0.96
- Macrophage (Mrc1+): 0.54
- Fibro 6: 0.51

**Temporal Dynamics:**
- Healthy: 0.62 → DSS3: 1.00 → DSS9: 0.10 → DSS21: 0.06
- Peak stage: **DSS3**

#### **MU3**

**Cellular Composition:**
- Colonocytes: 1.00
- Capillarial EC: 1.00
- Goblet 2: 1.00
- Fibro 2: 1.00
- Fibro 13: 0.81

**Temporal Dynamics:**
- Healthy: 1.00 → DSS3: 0.43 → DSS9: 0.12 → DSS21: 0.35
- Peak stage: **Healthy**


### Acute Inflammation Mucosal Neighborhoods

#### **MU4**

**Cellular Composition:**
- IAE 1: 1.00
- Goblet 2: 0.68
- Fibro 13: 0.53
- Colonocytes: 0.52
- Capillarial EC: 0.51

**Temporal Dynamics:**
- Healthy: 0.02 → DSS3: 1.00 → DSS9: 0.04 → DSS21: 0.01
- Peak stage: **DSS3**

#### **MU5**

**Cellular Composition:**
- Macrophage (Itgax+): 1.00
- NK: 1.00
- T (Mki67+): 1.00
- IAF 5: 1.00
- cDC1: 0.93

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.01
- Peak stage: **DSS9**

#### **MU6**

**Cellular Composition:**
- IAE 2: 1.00
- Goblet 2: 0.59
- Capillarial EC: 0.38
- DC (Ccl22+): 0.30
- EEC: 0.30

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.03
- Peak stage: **DSS9**

#### **MU7**

**Cellular Composition:**
- IAE 3: 1.00
- IAF 3: 0.94
- Macrophage (Itgax+): 0.63
- cDC1: 0.55
- IAF 2: 0.53

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.02
- Peak stage: **DSS9**

#### **MU8**

**Cellular Composition:**
- Macrophage (Cxcl10+): 1.00
- IAF 2: 1.00
- T (Cd8+): 1.00
- Neutrophil 1: 1.00
- IAF 4: 1.00

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.01
- Peak stage: **DSS9**


### Recovery/Regeneration Mucosal Neighborhoods

#### **MU9**

**Cellular Composition:**
- Epithelial (Clu+): 1.00
- Fibro 2: 0.45
- Capillarial EC: 0.43
- Neuron (Nos1+): 0.41
- Goblet 2: 0.39

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 0.02 → DSS21: 1.00
- Peak stage: **DSS21**

#### **MU10**

**Cellular Composition:**
- Repair associated  (Arg1+): 1.00
- ILC3-LTi-like: 0.42
- cDC1: 0.37
- Macrophage (Itgax+): 0.37
- Lymphatic EC: 0.37

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.06 → DSS9: 0.01 → DSS21: 1.00
- Peak stage: **DSS21**

#### **MU11**

**Cellular Composition:**
- Plasma cell: 1.00
- cDC1: 0.86
- Macrophage (Itgax+): 0.74
- Treg: 0.67
- NK: 0.56

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 1.00
- Peak stage: **DSS9**


### Mesenchymal/Neuronal Neighborhoods

#### **ME1**

**Cellular Composition:**
- SMC 1: 1.00
- ICC 2: 1.00
- Fibro 15: 0.74
- Glia (Gfra3+): 0.58
- ICC 1: 0.37

**Temporal Dynamics:**
- Healthy: 0.94 → DSS3: 1.00 → DSS9: 0.36 → DSS21: 0.53
- Peak stage: **DSS3**

#### **ME2**

**Cellular Composition:**
- Neuron (Chat+): 1.00
- ICC 1: 1.00
- SMC 2: 1.00
- Neuron (Nos1+): 1.00
- Glia (Gfap+): 1.00

**Temporal Dynamics:**
- Healthy: 0.71 → DSS3: 1.00 → DSS9: 0.36 → DSS21: 0.34
- Peak stage: **DSS3**

#### **ME3**

**Cellular Composition:**
- IASMC 2: 1.00
- Fibro 1: 0.80
- Neuron (Nos1+): 0.70
- Glia (Gfap+): 0.57
- Neuron (Chat+): 0.57

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.02
- Peak stage: **DSS9**

#### **ME4**

**Cellular Composition:**
- IASMC 3: 1.00
- DC (Ccl22+): 0.93
- Fibro 1: 0.79
- Lymphatic EC (Ccl21a+): 0.62
- Neuron (Nos1+): 0.59

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.02 → DSS9: 1.00 → DSS21: 0.02
- Peak stage: **DSS9**


### Vascular/Stromal Mesenchymal Neighborhoods

#### **MES1**

**Cellular Composition:**
- Adipose: 1.00
- Arterial EC: 1.00
- Pericyte 2: 1.00
- Pericyte 1: 1.00
- Venous EC: 1.00

**Temporal Dynamics:**
- Healthy: 0.50 → DSS3: 1.00 → DSS9: 0.57 → DSS21: 0.28
- Peak stage: **DSS3**

#### **MES2**

**Cellular Composition:**
- Macrophage (Lyve1+): 1.00
- Monocyte: 1.00
- Fibro 5: 1.00
- cDC1: 0.90
- Fibro 1: 0.89

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.03
- Peak stage: **DSS9**

#### **MES3**

**Cellular Composition:**
- SMC Mesentery: 1.00
- Glia (Mpz+): 1.00
- Glia (Gfra3+): 1.00
- Glia (Apod+): 1.00
- Pericyte 1: 0.58

**Temporal Dynamics:**
- Healthy: 1.00 → DSS3: 0.26 → DSS9: 0.03 → DSS21: 0.22
- Peak stage: **Healthy**


### Smooth Muscle Neighborhoods

#### **SM1**

**Cellular Composition:**
- ILC2: 1.00
- Lymphatic EC: 1.00
- Macrophage (Mrc1+): 1.00
- Fibro 6: 1.00
- DC (Ccl22+): 0.61

**Temporal Dynamics:**
- Healthy: 0.52 → DSS3: 1.00 → DSS9: 0.07 → DSS21: 0.12
- Peak stage: **DSS3**

#### **SM2**

**Cellular Composition:**
- Lymphatic EC (Ccl21a+): 1.00
- IASMC 1: 1.00
- Monocyte: 0.80
- Fibro 4: 0.57
- DC (Fscn1+): 0.53

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.01
- Peak stage: **DSS9**

#### **SM3**

**Cellular Composition:**
- Fibro 4: 1.00
- cDC1: 0.85
- Fibro 1: 0.65
- Lymphatic EC: 0.64
- Macrophage (Itgax+): 0.51

**Temporal Dynamics:**
- Healthy: 0.01 → DSS3: 0.02 → DSS9: 0.40 → DSS21: 1.00
- Peak stage: **DSS21**


### Lymphoid Follicular Neighborhoods

#### **FOL1**

**Cellular Composition:**
- B cell 2: 1.00
- B cell 1: 1.00
- T (Cd4+ Ccr7+): 1.00
- DC (Fscn1+): 1.00
- Tfh: 1.00

**Temporal Dynamics:**
- Healthy: 0.15 → DSS3: 0.06 → DSS9: 1.00 → DSS21: 0.24
- Peak stage: **DSS9**

#### **FOL2**

**Cellular Composition:**
- ILC3-LTi-like: 1.00
- M cells: 1.00
- FRC: 1.00
- cDC1: 0.99
- Macrophage (Itgax+): 0.90

**Temporal Dynamics:**
- Healthy: 0.05 → DSS3: 1.00 → DSS9: 0.00 → DSS21: 0.00
- Peak stage: **DSS3**

#### **FOL3**

**Cellular Composition:**
- B cell (Aicda+): 1.00
- Fibro 1: 1.00
- Tfh: 0.52
- Macrophage (Itgax+): 0.46
- B cell 1: 0.35

**Temporal Dynamics:**
- Healthy: 0.15 → DSS3: 0.00 → DSS9: 0.07 → DSS21: 1.00
- Peak stage: **DSS21**


### Luminal Neighborhood

#### **LUM**

**Cellular Composition:**
- Neutrophil 2: 1.00
- DC (Ccl22+): 0.73
- T (Cd8+): 0.71
- Neutrophil 1: 0.64
- Treg: 0.42

**Temporal Dynamics:**
- Healthy: 0.00 → DSS3: 0.00 → DSS9: 1.00 → DSS21: 0.02
- Peak stage: **DSS9**


### Unclassified

#### **others**

**Cellular Composition:**
- DC (Ccl22+): 1.00
- IAF 1: 1.00
- Treg: 1.00
- ILC2: 0.99
- Monocyte: 0.71

**Temporal Dynamics:**
- Healthy: 0.20 → DSS3: 0.07 → DSS9: 1.00 → DSS21: 0.08
- Peak stage: **DSS9**


---

## Disease Progression Summary

### Healthy → DSS3 (Early Inflammation)

**Depleted Neighborhoods:**
- MU1 (1.00 → 0.29): Stem cell niche disruption
- MU3 (1.00 → 0.43): Differentiated epithelium loss
- MES3 (1.00 → 0.26): Peripheral glia depletion

**Emerging Neighborhoods:**
- MU4 (0.02 → 1.00): IAE 1-dominant early stress
- FOL2 (0.05 → 1.00): Innate lymphoid structures

**Expanding Neighborhoods:**
- MU2 (0.62 → 1.00): Transit-amplifying zone
- ME1, ME2 (peak): Neuromuscular activation
- MES1, SM1 (peak): Vascular/immune remodeling

**Biological Interpretation:**
Early inflammation triggers epithelial stress responses (IAE 1 emergence), disrupts homeostatic epithelial compartments, and activates vascular/immune networks. The transit-amplifying zone expands, possibly attempting regeneration.

---

### DSS3 → DSS9 (Peak Inflammation)

**Newly Emerging Neighborhoods (DSS9-specific):**
- LUM (0.00 → 1.00): Neutrophil-dominant luminal damage
- MU5 (0.00 → 1.00): Immune infiltration niche
- MU6 (0.00 → 1.00): IAE 2-dominant barrier
- MU7 (0.00 → 1.00): IAE 3-stromal remodeling
- MU8 (0.00 → 1.00): Severe inflammation zone
- ME3 (0.00 → 1.00): IASMC 2-dominant
- ME4 (0.00 → 1.00): IASMC 3-lymphatic interface
- MES2 (0.00 → 1.00): Immune-stromal interface
- SM2 (0.00 → 1.00): IASMC 1-vascular niche
- FOL1 (0.06 → 1.00): B/T cell follicles

**Collapsed Neighborhoods:**
- MU2 (1.00 → 0.10): Transit-amplifying zone destruction
- MU3 (0.43 → 0.12): Further epithelial loss
- MES3 (0.26 → 0.03): Peripheral structure loss
- SM1 (1.00 → 0.07): Lymphatic-immune disruption

**Biological Interpretation:**
Peak inflammation shows:
1. **Complete epithelial barrier breakdown** (MU5-MU8 emergence, MU2-MU3 collapse)
2. **Massive immune infiltration** (neutrophils, macrophages, T cells)
3. **Smooth muscle remodeling** (IASMC emergence in ME3, ME4, SM2)
4. **Tertiary lymphoid structure formation** (FOL1 expansion)
5. **Stromal activation** (IAF populations in multiple neighborhoods)

---

### DSS9 → DSS21 (Recovery/Chronic Phase)

**Recovery-Specific Neighborhoods (DSS21 peak):**
- MU9 (0.02 → 1.00): Clu+ epithelial regeneration
- MU10 (0.01 → 1.00): Arg1+ repair macrophages
- SM3 (0.40 → 1.00): Fibrotic remodeling
- FOL3 (0.07 → 1.00): Germinal center formation

**Persistent Neighborhoods:**
- MU11 (1.00 → 1.00): Plasma cell niches remain

**Collapsed Acute Inflammation Neighborhoods:**
- LUM (1.00 → 0.02): Neutrophil clearance
- MU5-MU8 (1.00 → 0.01-0.03): Acute inflammation resolution
- ME3-ME4 (1.00 → 0.02): IASMC normalization
- SM2 (1.00 → 0.01): Inflammation-associated smooth muscle reduction

**Partial Recovery:**
- MU3 (0.12 → 0.35): Some epithelial restoration
- MES3 (0.03 → 0.22): Partial peripheral recovery

**Biological Interpretation:**
Recovery phase shows:
1. **Specialized repair neighborhoods emerge** (MU9, MU10) - not a return to healthy state
2. **Fibrotic remodeling** (SM3 expansion with Fibro 4 dominance)
3. **Persistent adaptive immunity** (FOL3 germinal centers, MU11 plasma cells)
4. **Incomplete epithelial restoration** (MU1, MU2, MU3 remain depleted)
5. **Clearance of acute inflammation** (neutrophil neighborhoods collapse)

---

## Biological Insights

### Inflammation-Associated Cell State Transitions

**Epithelial Transitions:**
- Healthy: Colonocytes, Goblet cells, Stem cells (MU1-MU3)
- Inflammation: IAE 1 → IAE 2 → IAE 3 (MU4 → MU6 → MU7)
- Recovery: Clu+ epithelium (MU9)

**Fibroblast Transitions:**
- Healthy: Fibro 1-7, 13, 15 (distributed across neighborhoods)
- Inflammation: IAF 1-5 (MU5, MU7, MU8, others)
- Recovery: Fibro 4 dominance (SM3 fibrotic remodeling)

**Smooth Muscle Transitions:**
- Healthy: SMC 1, SMC 2 (ME1, ME2)
- Inflammation: IASMC 1-3 (ME3, ME4, SM2)
- Recovery: Mixed (partial normalization)

### Spatial Organization Principles

**Compartmentalization:**
- **Mucosal layer**: Epithelial neighborhoods (MU1-MU11) show greatest dynamics
- **Muscularis layer**: Smooth muscle neighborhoods (SM1-SM3, ME1-ME4) undergo remodeling
- **Submucosa/stroma**: Mesenchymal neighborhoods (MES1-MES3) provide vascular/immune support
- **Lymphoid structures**: Follicular neighborhoods (FOL1-FOL3) represent organized immunity

**Temporal Waves:**
1. **Wave 1 (DSS3)**: Vascular activation, early epithelial stress, innate lymphoid induction
2. **Wave 2 (DSS9)**: Massive immune infiltration, epithelial collapse, IASMC emergence, tertiary lymphoid formation
3. **Wave 3 (DSS21)**: Specialized repair, fibrotic remodeling, mature germinal centers

### Incomplete Recovery Mechanisms

**Why tissue doesn't return to healthy state:**
1. **Persistent plasma cell niches** (MU11) maintain ongoing antibody production
2. **Fibrotic remodeling** (SM3) replaces normal smooth muscle
3. **Specialized repair neighborhoods** (MU9, MU10) differ from homeostatic epithelium
4. **Stem cell niche depletion** (MU1 remains at 0.25 vs 1.00 healthy)
5. **Mature tertiary lymphoid organs** (FOL3) establish permanent immune structures

---

## Clinical and Therapeutic Implications

### Biomarker Potential

**Disease Stage Biomarkers:**
- **DSS3 (early)**: MU4 (IAE 1), FOL2 (ILC3-M cells)
- **DSS9 (acute)**: MU5-MU8 (immune infiltration), LUM (neutrophils), FOL1 (B/T follicles)
- **DSS21 (chronic)**: MU9-MU10 (repair), SM3 (fibrosis), FOL3 (germinal centers)

**Severity Markers:**
- High MU8 (Cxcl10+ macrophages, neutrophils): Severe inflammation
- High LUM (neutrophil dominance): Active barrier breach
- Low MU1-MU3: Epithelial barrier dysfunction

### Therapeutic Targets

**Anti-Inflammatory Targets:**
- **MU8 neighborhood**: Cxcl10+ macrophages, IAF 2/4
- **LUM neighborhood**: Neutrophil recruitment pathways
- **ME3/ME4/SM2**: IASMC populations

**Pro-Regenerative Targets:**
- **MU9 neighborhood**: Support Clu+ epithelial regeneration
- **MU10 neighborhood**: Enhance Arg1+ repair macrophage function
- **MU1 restoration**: Promote stem cell niche recovery

**Anti-Fibrotic Targets:**
- **SM3 neighborhood**: Target Fibro 4-mediated fibrosis
- **IAF populations**: Prevent chronic fibroblast activation

### Precision Medicine Approaches

**Neighborhood-Specific Therapies:**
- **Epithelial protection**: Target MU4-MU8 to preserve barrier
- **Immune modulation**: Target FOL1-FOL3 to regulate adaptive immunity
- **Fibrosis prevention**: Target SM3 to prevent chronic remodeling
- **Regeneration enhancement**: Support MU9-MU10 repair processes

---

## Conclusions

This comprehensive spatial neighborhood analysis reveals that DSS-induced colitis triggers **organized tissue reorganization** rather than random inflammation:

1. **26 distinct neighborhoods** represent functional tissue compartments
2. **Stage-specific neighborhoods** mark disease progression (9 inflammation-specific, 4 recovery-specific)
3. **Inflammation-associated cell states** (IAE, IAF, IASMC) emerge in specific spatial contexts
4. **Incomplete recovery** involves establishment of new tissue architecture, not restoration of healthy state
5. **Tertiary lymphoid structures** form progressively (FOL2 → FOL1 → FOL3)
6. **Fibrotic remodeling** (SM3) persists in chronic disease

**Key Biological Insight:**
Colitis progression involves **sequential waves of tissue reorganization**: vascular activation (DSS3) → immune infiltration and epithelial collapse (DSS9) → specialized repair and fibrosis (DSS21). Each stage has characteristic neighborhood signatures that could serve as biomarkers and therapeutic targets.

**Clinical Relevance:**
Understanding neighborhood dynamics provides:
- **Diagnostic biomarkers** for disease staging
- **Therapeutic targets** for neighborhood-specific interventions
- **Prognostic indicators** for recovery potential
- **Mechanistic insights** into IBD pathogenesis

---

## Methods Summary

**Data Source:**
- MERFISH spatial transcriptomics
- Mouse colon tissue
- DSS-induced colitis model
- 1,348,582 cells, 990 genes, 52 tissue slices

**Neighborhood Analysis:**
- 26 spatial neighborhoods (Leiden clustering)
- 72 cell types (Tier3 annotations)
- 4 disease stages: Healthy, DSS3, DSS9, DSS21

**Computational Approach:**
- Normalized cell type composition per neighborhood
- Normalized neighborhood abundance per disease stage
- Spatial distribution mapping across all slices
- Temporal dynamics analysis

**Visualization:**
- Heatmaps and dotplots for composition/abundance
- Individual spatial maps for each neighborhood
- Disease stage progression tracking

---

## Supplementary Information

**Data Files:**
- `celltype_composition_per_neighborhood.csv`: Normalized fractions (26 neighborhoods × 72 cell types)
- `neighborhood_abundance_per_stage.csv`: Normalized abundances (26 neighborhoods × 4 stages)

**Spatial Maps:**
- `neighborhood_spatial/`: 26 PNG files showing spatial distribution of each neighborhood
- `celltype_spatial/`: 72 PNG files showing spatial distribution of each cell type

**Summary Plots:**
- Cell type composition heatmap and dotplot
- Neighborhood abundance heatmap and dotplot

---

**Report Generated:** Automated spatial transcriptomics analysis pipeline
**Output Directory:** `output/breakdown/`

