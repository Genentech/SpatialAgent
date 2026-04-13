
# Cell-Cell Interaction Analysis: MU1 → MU2 → MU8 → MU10 Trajectory

## Executive Summary

This analysis examines cell-cell interactions along a single neighborhood trajectory 
in a mouse colon MERFISH dataset (colitis/IBD model). The trajectory shows progressive 
inflammatory changes from MU1 through MU10, with dramatic shifts in cellular composition 
and communication patterns.

### Key Findings:

1. **Interaction Complexity Increases 2.9-fold**: From 2,233 interactions (MU1) to 6,410 (MU10)
2. **Inflammation-Associated (IA) Cells Emerge**: 29-fold increase in IA cell interactions (MU2→MU8)
3. **Immune Infiltration Intensifies**: 3.9-fold increase in immune cell interactions
4. **Repair Mechanisms Activate**: Repair-associated cells appear in MU8 and expand in MU10
5. **Dominant Signaling Shifts**: From ECM-epithelial (MU1-MU2) to immune-stromal (MU8-MU10)

---

## Neighborhood-by-Neighborhood Analysis

### MU1: Early/Baseline State
**Cellular Composition**: 29 cell types
- Dominated by epithelial cells (Colonocytes, TA, Stem cells, Goblet cells)
- Fibroblast subtypes (Fibro 2, 5, 6, 7, 13)
- Smooth muscle cells (SMC 1, 2)
- Limited inflammation-associated cells (IAE 1, IAF 2, IAF 3)
- Resident immune cells (Macrophages, cDC1, Plasma cells)

**Interaction Profile**:
- Total interactions: 2,233 (all significant p<0.05)
- IA cell interactions: 54
- Inflammation-related: 729
- Mean expression: 0.207

**Key Cell-Cell Communication Patterns**:

1. **Fibroblast → Epithelial Dominance**:
   - Fibro 2 → TA: 24 interactions
   - Fibro 6 → Stem cells: 24 interactions
   - Fibro 2 → Stem cells: 22 interactions
   - **Biological significance**: Fibroblasts provide structural and trophic support to epithelium

2. **Top Ligand-Receptor Pairs**:
   - **Col1a2-Sdc1** (avg expression: 0.408): ECM-epithelial adhesion
   - **Col1a2-Flt4** (0.384): ECM-lymphatic endothelial signaling
   - **Sparc-Fgfr1** (0.354): Fibroblast growth factor signaling
   - **Csf1-Csf1r** (0.299): Macrophage survival/differentiation

3. **Inflammation-Associated Interactions** (Top 5):
   - Plasma cell → cDC1: Selplg-Itgb2 (0.366) - Immune cell adhesion
   - Macrophage (Itgax+) → Arterial EC: Tgfb1-Acvrl1 (0.363) - Anti-inflammatory signaling
   - cDC1 → Arterial EC: Tgfb1-Acvrl1 (0.362) - Vascular regulation
   - Macrophage (Itgax+) → Fibro 13: Tgfb1-Tgfbr3 (0.353) - Fibroblast activation
   - Pericyte 1 → Macrophage (Mrc1+): Csf1-Csf1r (0.331) - M2 macrophage maintenance

4. **IA Cell Interactions** (54 total):
   - IAF 3 → TA: 5 interactions (early fibroblast-epithelial crosstalk)
   - Fibro 2 → IAE 1: 4 interactions (ECM support to inflamed epithelium)
   - **Top interaction**: Fibro 5 → IAE 1: Col1a2-Sdc1 (0.593)

**Biological Interpretation**:
MU1 represents a relatively homeostatic or early inflammatory state with:
- Intact epithelial-mesenchymal communication
- Balanced ECM deposition (Col1a2, Col6a1, Sparc)
- Resident immune surveillance (TGFβ signaling dominant)
- Limited inflammation-associated cell types
- Glia-fibroblast signaling (Sparc-Fgfr1: 0.569) suggests intact neuronal support

---

### MU2: Early Inflammatory Transition
**Cellular Composition**: 25 cell types (slight reduction)
- Epithelial cells maintained
- Increased IAE presence (IAE 1, IAE 2)
- Reduced diversity (loss of some vascular subtypes)
- Maintained fibroblast populations

**Interaction Profile**:
- Total interactions: 2,827 (+27% from MU1)
- IA cell interactions: 128 (+137% from MU1)
- Inflammation-related: 1,067 (+46% from MU1)
- Mean expression: 0.185 (decreased - more diverse but weaker interactions)

**Key Cell-Cell Communication Patterns**:

1. **Fibroblast → Epithelial Intensification**:
   - Fibro 2 → Fibro 13: 38 interactions (fibroblast-fibroblast communication increases)
   - Fibro 2 → TA: 33 interactions
   - Fibro 2 → Colonocytes: 28 interactions
   - **Biological significance**: Fibroblasts responding to epithelial stress

2. **Emergence of IAE Interactions**:
   - Fibro 2 → IAE 1: 16 interactions
   - Fibro 6 → IAE 1: 14 interactions
   - IAE 1 → Macrophage (Itgax+): 8 interactions (inflamed epithelium recruiting immune cells)
   - **Top interaction**: Fibro 5 → IAE 1: Col1a2-Sdc1 (0.682)

3. **Top Ligand-Receptor Pairs**:
   - **Col1a2-Sdc1** (0.450): Increased ECM-epithelial signaling
   - **Col6a1-Sdc1** (0.356): Alternative ECM component engagement
   - **Sparc-Fgfr1** (0.320): Maintained growth factor signaling
   - **Jag2-Notch3** (0.292): Notch signaling emerges (cell fate decisions)

4. **Inflammation-Associated Interactions** (Top 5):
   - Plasma cell → cDC1: Selplg-Itgb2 (0.385) - Maintained immune adhesion
   - cDC1 → Arterial EC: Tgfb1-Acvrl1 (0.371) - Vascular remodeling
   - Macrophage (Itgax+) → Arterial EC: Tgfb1-Acvrl1 (0.363)
   - EEC → Macrophage (Itgax+): Vegfa-Sirpa (0.335) - Angiogenic signaling
   - Pericyte 1 → Macrophage (Mrc1+): Csf1-Csf1r (0.333)

**Biological Interpretation**:
MU2 shows early inflammatory response with:
- Epithelial stress response (IAE emergence)
- Increased fibroblast activation and ECM remodeling
- Maintained but shifting immune landscape
- Notch signaling activation (tissue repair/regeneration attempts)
- Vegfa signaling suggests early angiogenic response

---

### MU8: Active Inflammation Peak
**Cellular Composition**: 48 cell types (+92% from MU2)
- Massive immune cell infiltration (B cells, T cells, DCs, Neutrophils)
- Full spectrum of IA cells (IAE 1-3, IAF 1-4, IASMC 1-3)
- Lymphoid structures (FRC, Lymphatic EC Ccl21a+)
- **Repair-associated (Arg1+) cells appear**

**Interaction Profile**:
- Total interactions: 4,445 (+57% from MU2)
- IA cell interactions: 1,502 (+1,073% from MU2) **DRAMATIC INCREASE**
- Inflammation-related: 2,439 (+129% from MU2)
- Mean expression: 0.194

**Key Cell-Cell Communication Patterns**:

1. **IASMC Emerges as Central Hub**:
   - IASMC 1 → IAF 3: Thbs1-Sdc1 (0.592) - Smooth muscle-fibroblast crosstalk
   - IASMC 1 → Repair-associated (Arg1+): Thbs1-Sdc1 (0.647) **HIGHEST**
   - IASMC 1 → Plasma cell: Thbs1-Sdc1 (0.568)
   - **Biological significance**: Inflammation-associated smooth muscle cells coordinate tissue remodeling

2. **Immune Cell Dominance**:
   - Neutrophil 1 → Monocyte: Mmp9-Itgam (0.564) - Neutrophil-monocyte recruitment
   - NK → Venous EC: Selplg-Selp (0.587) - Lymphocyte trafficking
   - IAF 3 → Monocyte: 25 interactions (fibroblast recruiting monocytes)
   - IAF 4 → Monocyte: 21 interactions

3. **Top Ligand-Receptor Pairs**:
   - **Selplg-Selp** (0.440): Immune cell trafficking/rolling
   - **Cd34-Selp** (0.416): Endothelial-immune adhesion
   - **Mmp9-Ifnar1** (0.347): Neutrophil matrix degradation + interferon signaling
   - **Thbs1-Sdc1** (0.329): Thrombospondin-mediated ECM remodeling

4. **Repair-Associated Interactions** (56 total - NEW):
   - IASMC 1 → Repair (Arg1+): Thbs1-Sdc1 (0.647)
   - Fibro 5 → Repair (Arg1+): Col1a2-Sdc1 (0.577)
   - Fibro 4 → Repair (Arg1+): Col1a2-Sdc1 (0.548)
   - **Biological significance**: M2-like repair macrophages receiving signals from stromal cells

5. **Inflammation-Associated Interactions** (Top 5):
   - NK → Venous EC: Selplg-Selp (0.587) - Lymphocyte extravasation
   - Neutrophil 1 → Monocyte: Mmp9-Itgam (0.564) - Myeloid cell recruitment
   - cDC1 → Venous EC: Selplg-Selp (0.535) - DC trafficking
   - T (Cd4+ Ccr7+) → Venous EC: Selplg-Selp (0.530) - T cell homing
   - Neutrophil 1 → T (Cd8+): Mmp9-Ifnar1 (0.526) - Neutrophil-T cell crosstalk

**Biological Interpretation**:
MU8 represents peak inflammatory activity with:
- Massive immune infiltration (neutrophils, T cells, B cells, DCs)
- Tertiary lymphoid structure formation (FRC, Ccl21a+ lymphatics)
- Matrix remodeling (Mmp9, Thbs1 dominant)
- Emergence of repair mechanisms (Arg1+ cells)
- Selectin-mediated immune trafficking is dominant signaling
- IASMC cells coordinate between immune and stromal compartments

---

### MU10: Chronic Inflammation with Active Repair
**Cellular Composition**: 44 cell types (slight reduction from MU8)
- Maintained immune diversity
- **Massive expansion of Repair-associated (Arg1+) cells**
- Reduced IAE presence (only IAE 3)
- Maintained IAF and IASMC populations
- Appearance of specialized epithelium (Epithelial Clu+)

**Interaction Profile**:
- Total interactions: 6,410 (+44% from MU8) **HIGHEST**
- IA cell interactions: 1,571 (+5% from MU8 - plateaued)
- Inflammation-related: 3,086 (+27% from MU8)
- Repair-associated: 500 (9-fold increase from MU8)
- Mean expression: 0.167 (lowest - most diverse network)

**Key Cell-Cell Communication Patterns**:

1. **Repair-Associated Cells Central**:
   - Repair (Arg1+) → Repair (Arg1+): 27 interactions (autocrine signaling)
   - Fibro 5 → Repair (Arg1+): Col1a2-Sdc1 (0.653)
   - IASMC 1 → Repair (Arg1+): Thbs1-Sdc1 (0.628)
   - Neutrophil 1 → Repair (Arg1+): Mmp9-Cd44 (0.498)

2. **IAF-Endothelial Dominance**:
   - IAF 3 → Venous EC: 22 interactions
   - IAF 3 → Capillarial EC: 21 interactions
   - IAF 3 → Arterial EC: 20 interactions
   - **Biological significance**: Fibroblasts regulating vascular remodeling

3. **Top Ligand-Receptor Pairs**:
   - **Selplg-Selp** (0.380): Continued immune trafficking
   - **Thbs1-Sdc1** (0.380): Thrombospondin-mediated repair
   - **Mmp9-Ifnar1** (0.358): Sustained neutrophil activity
   - **Sparc-Fgfr1** (0.329): Growth factor signaling restored
   - **Thbs1-Tnfrsf11b** (0.250): Osteoprotegerin signaling (bone/tissue protection)

4. **Repair-Associated Interactions** (500 total):
   - Fibro 5 → Repair (Arg1+): Col1a2-Sdc1 (0.653) **HIGHEST**
   - IASMC 1 → Repair (Arg1+): Thbs1-Sdc1 (0.628)
   - IASMC 1 → Repair (Arg1+): Thbs1-Tnfrsf11b (0.479) - Protective signaling
   - Neutrophil 1 → Repair (Arg1+): Mmp9-Cd44 (0.498) - Matrix remodeling
   - **Biological significance**: Coordinated tissue repair program

5. **Inflammation-Associated Interactions** (Top 5):
   - Neutrophil 1 → Neutrophil 1: Mmp9-Ifnar1 (0.555) - Autocrine activation
   - Fibro 1 → cDC1: C3-Itgax (0.548) - Complement-mediated immune activation
   - ILC3-LTi-like → Venous EC: Selplg-Selp (0.532) - Innate lymphoid trafficking
   - IASMC 1 → Epithelial (Clu+): Thbs1-Tnfrsf11b (0.531) - Smooth muscle-epithelial protection
   - Neutrophil 1 → Monocyte: Mmp9-Itgam (0.526)

**Biological Interpretation**:
MU10 represents chronic inflammation with active repair:
- Repair macrophages (Arg1+) are central coordinators
- Fibroblasts and IASMC provide ECM scaffolding for repair
- Sustained immune activity (neutrophils, T cells)
- Vascular remodeling (IAF-endothelial interactions)
- Protective signaling (Thbs1-Tnfrsf11b, osteoprotegerin pathway)
- Specialized epithelium (Clu+) suggests adaptive responses

---

## Trajectory-Level Synthesis

### Progressive Changes:

1. **Interaction Network Expansion** (2.9-fold total):
   - MU1→MU2: +27% (early response)
   - MU2→MU8: +57% (inflammatory burst)
   - MU8→MU10: +44% (repair expansion)

2. **IA Cell Interaction Dynamics**:
   - MU1→MU2: +137% (IAE emergence)
   - MU2→MU8: +1,073% **EXPLOSIVE GROWTH** (full IA spectrum)
   - MU8→MU10: +5% (plateau, shift to repair)

3. **Immune Infiltration**:
   - MU1: 790 interactions (resident immune cells)
   - MU2: 1,141 (+44%)
   - MU8: 2,836 (+149%) - peak infiltration
   - MU10: 3,046 (+7%) - sustained

4. **Repair Mechanisms**:
   - MU1-MU2: Absent
   - MU8: 56 interactions (emergence)
   - MU10: 500 interactions (+793%) - dominant repair program

### Dominant Signaling Transitions:

**MU1-MU2 (Homeostasis → Early Inflammation)**:
- Col1a2-Sdc1: ECM-epithelial adhesion
- Sparc-Fgfr1: Growth factor signaling
- Tgfb1-Acvrl1: Anti-inflammatory control
- **Shift**: Increased fibroblast-epithelial communication

**MU2-MU8 (Early → Active Inflammation)**:
- Selplg-Selp: Immune cell trafficking **NEW DOMINANT**
- Mmp9-Itgam/Ifnar1: Matrix degradation + interferon
- Thbs1-Sdc1: ECM remodeling
- **Shift**: From epithelial support to immune recruitment

**MU8-MU10 (Active Inflammation → Repair)**:
- Thbs1-Sdc1: Maintained ECM remodeling
- Sparc-Fgfr1: Growth factor signaling restored
- Thbs1-Tnfrsf11b: Protective signaling **NEW**
- **Shift**: From recruitment to tissue protection/repair

---

## Key Biological Insights

### 1. Inflammation-Associated Cell Evolution

**IAE (Inflammation-Associated Epithelium)**:
- Emerges in MU2 (IAE 1, 2)
- Peaks in MU8 (IAE 1, 2, 3)
- Declines in MU10 (only IAE 3)
- **Interpretation**: Epithelial damage peaks mid-trajectory, then either repairs or is lost

**IAF (Inflammation-Associated Fibroblasts)**:
- Present early (MU1: IAF 2, 3)
- Expands in MU8 (IAF 1-4)
- Maintained in MU10
- **Interpretation**: Fibroblasts are early responders and persist throughout inflammation

**IASMC (Inflammation-Associated Smooth Muscle)**:
- Absent in MU1-MU2
- Emerges in MU8 (IASMC 1-3)
- Maintained in MU10
- **Interpretation**: Smooth muscle cells undergo phenotypic switching during active inflammation

### 2. Matrix Remodeling Programs

**Early (MU1-MU2)**: Structural ECM
- Col1a2, Col6a1 (fibrillar collagens)
- Sparc (matricellular protein)
- **Function**: Tissue architecture maintenance

**Active Inflammation (MU8)**: Degradation + Remodeling
- Mmp9 (matrix metalloproteinase) - dominant
- Thbs1 (thrombospondin-1) - emerges
- **Function**: Tissue breakdown and immune cell infiltration

**Repair (MU10)**: Protective Remodeling
- Thbs1-Sdc1: Maintained
- Thbs1-Tnfrsf11b: NEW (osteoprotegerin pathway)
- Sparc-Fgfr1: Restored
- **Function**: Controlled remodeling with tissue protection

### 3. Immune Cell Dynamics

**Resident → Recruited → Sustained**:
- MU1: Macrophages (Mrc1+, Itgax+), cDC1, Plasma cells
- MU8: +Neutrophils, T cells (Cd4+, Cd8+, Treg, Tfh), B cells, NK, Monocytes
- MU10: Maintained diversity + Repair macrophages (Arg1+)

**Selectin-Mediated Trafficking**:
- Absent/low in MU1-MU2
- Dominant in MU8-MU10 (Selplg-Selp, Cd34-Selp)
- **Interpretation**: Vascular activation enables sustained immune infiltration

**Neutrophil Signaling**:
- Mmp9-Itgam: Integrin-mediated adhesion and migration
- Mmp9-Ifnar1: Interferon receptor engagement (antiviral/inflammatory)
- Mmp9-Cd44: Hyaluronan receptor (matrix interaction)
- **Interpretation**: Neutrophils are multi-functional effectors

### 4. Repair Mechanisms

**Arg1+ Repair-Associated Cells**:
- Appear in MU8 (56 interactions)
- Expand in MU10 (500 interactions)
- Receive signals from: Fibroblasts (Col1a2), IASMC (Thbs1), Neutrophils (Mmp9)
- **Interpretation**: M2-like macrophages coordinate tissue repair

**Osteoprotegerin Pathway** (Thbs1-Tnfrsf11b):
- Emerges in MU10
- IASMC → Epithelial, Stem cells
- **Function**: Inhibits RANKL, protects against tissue destruction
- **Interpretation**: Active tissue protection mechanism

### 5. Vascular Remodeling

**Lymphatic Expansion**:
- Lymphatic EC (Ccl21a+) appears in MU8-MU10
- FRC (fibroblastic reticular cells) in MU8-MU10
- **Interpretation**: Tertiary lymphoid structure formation

**Angiogenic Signaling**:
- Col1a2-Flt4: ECM-VEGFR3 signaling (lymphangiogenesis)
- Maintained across trajectory
- **Interpretation**: Sustained vascular remodeling

---

## Therapeutic Implications

### 1. Potential Drug Targets

**Early Intervention (MU1-MU2)**:
- **TGFβ pathway**: Abundant Tgfb1-Acvrl1/Tgfbr3 signaling
  - *Strategy*: Modulate (not block) to prevent fibrosis while maintaining homeostasis
- **Notch pathway**: Jag2-Notch3 emerges in MU2
  - *Strategy*: Target to prevent epithelial damage progression

**Active Inflammation (MU8)**:
- **Selectins** (Selp, Sele): Dominant immune trafficking
  - *Strategy*: Block to reduce immune infiltration (e.g., anti-Selp antibodies)
- **MMP9**: Matrix degradation
  - *Strategy*: Inhibit to reduce tissue damage
- **Integrin pathways**: Itgam, Itgax, Itgb2
  - *Strategy*: Block immune cell adhesion (e.g., vedolizumab targets α4β7)

**Repair Enhancement (MU10)**:
- **Arg1+ cell recruitment**: CSF1-CSF1R pathway
  - *Strategy*: Enhance to promote M2 macrophage differentiation
- **Osteoprotegerin pathway**: Thbs1-Tnfrsf11b
  - *Strategy*: Augment tissue protection
- **SPARC-FGFR1**: Growth factor signaling
  - *Strategy*: Enhance epithelial regeneration

### 2. Biomarkers

**Disease Activity**:
- **Selplg-Selp interaction strength**: Correlates with immune infiltration
- **Mmp9 expression**: Tissue damage marker
- **IASMC presence**: Smooth muscle phenotype switching

**Repair Status**:
- **Arg1+ cell abundance**: Repair macrophage activity
- **Thbs1-Tnfrsf11b signaling**: Tissue protection
- **Sparc-Fgfr1 restoration**: Regenerative capacity

### 3. Timing of Intervention

**Early (MU1-MU2)**:
- Target fibroblast activation
- Prevent epithelial damage (IAE emergence)
- Modulate TGFβ signaling

**Mid (MU8)**:
- Block immune trafficking (selectins)
- Inhibit matrix degradation (MMP9)
- Promote early repair (CSF1)

**Late (MU10)**:
- Enhance repair mechanisms
- Support vascular remodeling
- Prevent chronic fibrosis

---

## Surprising Findings

### 1. IASMC as Central Coordinator
- **Unexpected**: Smooth muscle cells (IASMC) emerge as major signaling hubs in MU8-MU10
- **Significance**: Thbs1-Sdc1 interactions with nearly all cell types
- **Implication**: Smooth muscle phenotypic switching may be therapeutic target

### 2. Osteoprotegerin Pathway Activation
- **Unexpected**: Thbs1-Tnfrsf11b signaling in MU10
- **Significance**: Bone protection pathway active in gut inflammation
- **Implication**: Cross-tissue protective mechanisms may be exploited

### 3. Neutrophil-Interferon Axis
- **Unexpected**: Mmp9-Ifnar1 is dominant neutrophil interaction
- **Significance**: Links matrix remodeling to interferon signaling
- **Implication**: Neutrophils may have antiviral/regulatory roles beyond inflammation

### 4. Repair Cell Autocrine Signaling
- **Unexpected**: Repair (Arg1+) → Repair (Arg1+) is top interaction in MU10
- **Significance**: Self-amplifying repair program
- **Implication**: Positive feedback loop could be therapeutically enhanced

### 5. Early IAF Presence
- **Unexpected**: IAF cells present even in MU1 (relatively early state)
- **Significance**: Fibroblasts are early sensors/responders
- **Implication**: Fibroblast targeting may prevent disease progression

---

## Canonical vs. Novel Patterns

### Canonical Patterns Recovered:

1. **Selectin-mediated immune trafficking** (MU8-MU10)
   - Selplg-Selp, Cd34-Selp
   - Well-established in IBD pathogenesis

2. **MMP9 in neutrophil-mediated tissue damage** (MU8-MU10)
   - Mmp9-Itgam, Mmp9-Cd44
   - Known contributor to colitis

3. **TGFβ in fibrosis** (MU1-MU2)
   - Tgfb1-Acvrl1, Tgfb1-Tgfbr3
   - Classic fibrogenic pathway

4. **CSF1-CSF1R in macrophage biology** (all neighborhoods)
   - Macrophage survival and differentiation
   - Established pathway

5. **Tertiary lymphoid structures** (MU8-MU10)
   - FRC, Ccl21a+ lymphatics
   - Described in chronic IBD

### Novel/Unexpected Patterns:

1. **IASMC-mediated coordination** (MU8-MU10)
   - Thbs1-Sdc1 hub signaling
   - *Rare*: Smooth muscle as central coordinator

2. **Osteoprotegerin pathway in gut** (MU10)
   - Thbs1-Tnfrsf11b
   - *Novel*: Bone pathway in intestinal protection

3. **Neutrophil-interferon coupling** (MU8-MU10)
   - Mmp9-Ifnar1
   - *Unexpected*: Direct link between matrix degradation and interferon

4. **Glia-fibroblast signaling** (MU1)
   - Sparc-Fgfr1
   - *Rare*: Neuronal support cells in early inflammation

5. **Repair cell autocrine amplification** (MU10)
   - Repair → Repair interactions
   - *Novel*: Self-sustaining repair program

---

## Conclusions

This trajectory analysis reveals a **progressive inflammatory cascade** with distinct phases:

1. **MU1 (Homeostasis/Early)**: ECM-epithelial communication, resident immune surveillance
2. **MU2 (Transition)**: IAE emergence, fibroblast activation, early immune recruitment
3. **MU8 (Peak Inflammation)**: Massive immune infiltration, IA cell expansion, matrix degradation
4. **MU10 (Chronic/Repair)**: Sustained inflammation with active repair programs

**Key Transitions**:
- **MU2→MU8**: 29-fold increase in IA cell interactions - critical inflection point
- **MU8→MU10**: Shift from recruitment to repair (9-fold increase in repair interactions)

**Therapeutic Windows**:
- **Early (MU1-MU2)**: Prevent IA cell emergence, modulate TGFβ
- **Mid (MU8)**: Block immune trafficking (selectins), inhibit MMP9
- **Late (MU10)**: Enhance repair (Arg1+ cells), support osteoprotegerin pathway

**Most Promising Targets**:
1. Selectins (Selp/Sele) - reduce immune infiltration
2. MMP9 - limit tissue damage
3. CSF1-CSF1R - promote repair macrophages
4. Thbs1-Tnfrsf11b - enhance tissue protection
5. IASMC signaling - modulate smooth muscle phenotype

**Critical Gaps for Future Study**:
- Spatial organization of IA cells and immune cells
- Temporal dynamics within each neighborhood
- Cross-neighborhood communication
- Role of IASMC in disease progression vs. repair
- Mechanisms of repair cell autocrine amplification

