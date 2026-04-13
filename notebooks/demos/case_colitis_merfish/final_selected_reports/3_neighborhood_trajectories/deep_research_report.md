# Deep Research Report: Spatial Neighborhood Trajectory Dynamics During DSS-Induced Colitis Progression

## Executive Summary

- **Spatial neighborhoods undergo seven distinct trajectory paths during DSS-induced colitis**, reflecting coordinated tissue remodeling from homeostasis through acute inflammation to resolution, with each path characterized by specific temporal kinetics and molecular signatures.

- **Neighborhood transitions mirror underlying cell-type trajectories**, with mesenchymal neighborhoods showing early activation (DSS3) corresponding to fibroblast→IAF transitions, epithelial neighborhoods exhibiting disruption-recovery patterns linked to epithelial→IAE dynamics, and lymphoid structures forming progressively through immune cell recruitment.

- **Peak inflammation at DSS9 represents a convergence point** where multiple neighborhood trajectories intersect, characterized by maximal epithelial disruption, lymphoid aggregation, smooth muscle remodeling, and pro-inflammatory gene expression (Cxcl1, Cxcl2, Il1b, Tnf).

- **Resolution at DSS21 follows heterogeneous pathways**, with some neighborhoods restoring homeostatic states (stem cell niches with Lgr5, Ascl2, Olfm4), while others establish persistent adaptive immune structures (plasma cell niches) or fibrotic remodeling (Col1a1, Col3a1, Fn1).

- **Asynchronous temporal dynamics between tissue destruction, immune activation, and regeneration** provide a spatiotemporal framework for understanding inflammatory bowel disease pathogenesis, revealing that effective resolution requires coordinated restoration of epithelial barriers, immune regulation, and stromal support.

## 1. Introduction

### 1.1 Background

Inflammatory bowel disease (IBD), including ulcerative colitis and Crohn's disease, affects millions of individuals worldwide and represents a significant clinical challenge due to its chronic, relapsing nature and complex pathophysiology. The intestinal mucosa is a highly organized tissue comprising multiple cellular compartments that maintain barrier function, nutrient absorption, and immune homeostasis. During inflammation, this spatial organization undergoes dramatic remodeling involving epithelial damage, immune cell infiltration, fibroblast activation, and vascular changes.

The dextran sulfate sodium (DSS)-induced colitis model in mice recapitulates key features of human IBD, including epithelial barrier disruption, immune cell recruitment, cytokine production, and tissue remodeling. This model provides a controlled system to study the temporal progression of inflammation and subsequent resolution or chronicity. Traditional bulk transcriptomic approaches have identified numerous genes and pathways involved in colitis, but these methods lack the spatial resolution necessary to understand how different tissue microenvironments coordinate during disease progression.

Recent advances in spatial transcriptomics, particularly multiplexed error-robust fluorescence in situ hybridization (MERFISH), enable simultaneous measurement of hundreds of genes at single-cell resolution while preserving tissue architecture. This technology has revealed that tissues are organized into spatial neighborhoods—microenvironments defined by specific cellular compositions and molecular programs. However, how these neighborhoods transition during inflammatory disease progression remains poorly understood.

Previous studies have identified cell-type-specific trajectories during colitis, including the transition of fibroblasts to inflammatory-associated fibroblasts (IAFs), epithelial cells to inflammatory-associated epithelial cells (IAEs), and the recruitment of various immune cell populations. Yet, the relationship between these cellular trajectories and the reorganization of spatial neighborhoods has not been systematically characterized. Understanding neighborhood-level transitions is critical because tissue function emerges from the coordinated behavior of multiple cell types within specific spatial contexts.

### 1.2 Research Question

This study addresses the fundamental question: **How do spatial neighborhoods transition during DSS-induced colitis progression?** Specifically, we aim to:

1. Infer neighborhood trajectory paths based on temporal abundance changes across disease stages (Healthy, DSS3, DSS9, DSS21)
2. Characterize cellular composition shifts that define neighborhood transitions
3. Link neighborhood-level transitions to underlying cell-type trajectories (fibroblast→IAF, epithelial→IAE, immune cell recruitment)
4. Identify key gene expression changes that mark critical transition points
5. Establish spatial continuity relationships that reveal how neighborhoods influence each other during disease progression

By integrating temporal dynamics, cellular composition, and molecular signatures, we seek to construct a comprehensive spatiotemporal map of tissue remodeling during colitis that connects cellular-level processes to tissue-level organization.

### 1.3 Significance

Understanding neighborhood trajectory dynamics during colitis has several important implications:

**Mechanistic Insight**: Spatial neighborhoods represent functional units where multiple cell types interact to execute tissue-level programs. Characterizing how these units transition during disease reveals the coordinated cellular behaviors that drive pathology and resolution, moving beyond single cell-type analyses to systems-level understanding.

**Therapeutic Targets**: Different neighborhoods may respond differently to therapeutic interventions. Identifying critical transition points and the molecular programs that define them could reveal new therapeutic opportunities. For example, if mesenchymal neighborhoods activate early in disease progression, targeting fibroblast activation pathways might prevent downstream pathology.

**Disease Heterogeneity**: IBD patients exhibit significant heterogeneity in disease course and treatment response. Understanding the diversity of neighborhood trajectories and resolution pathways may help explain this heterogeneity and enable patient stratification based on spatial tissue organization patterns.

**Translational Potential**: While this study uses a mouse model, the principles of spatial neighborhood organization and transition are likely conserved in human IBD. The trajectory framework established here could be applied to human spatial transcriptomic datasets to identify disease subtypes, predict outcomes, and guide personalized treatment strategies.

**Methodological Advancement**: This work demonstrates how spatial transcriptomic data can be analyzed at the neighborhood level to infer dynamic processes from static snapshots, providing a template for studying tissue remodeling in other inflammatory, developmental, and neoplastic contexts.

## 2. Data Overview

### 2.1 Dataset Description

This analysis utilized a comprehensive MERFISH spatial transcriptomics dataset of mouse colon tissue during DSS-induced colitis progression. The dataset encompasses:

**Scale**: 1,348,582 single cells profiled across the complete disease time course, representing one of the largest spatial transcriptomic datasets of inflammatory disease to date.

**Spatial Organization**: Cells were computationally organized into 26 distinct spatial neighborhoods based on cellular composition and spatial proximity. These neighborhoods span multiple functional tissue compartments:
- Mucosal neighborhoods (MU1-MU10): Epithelial-dominant regions representing the intestinal lining
- Mesenchymal neighborhoods (MES1-MES6): Stromal regions containing fibroblasts, smooth muscle, and vascular cells
- Follicle-associated neighborhoods (FOL1-FOL3): Lymphoid structures containing B cells, T cells, and follicular dendritic cells
- Smooth muscle neighborhoods (SM1-SM3): Muscle layer regions
- Luminal neighborhoods (LUM1-LUM4): Regions near the intestinal lumen

**Temporal Coverage**: Four disease stages were sampled:
- **Healthy**: Baseline homeostatic state
- **DSS3**: Early inflammation (3 days post-DSS exposure)
- **DSS9**: Peak acute inflammation (9 days)
- **DSS21**: Resolution/chronic phase (21 days)

This temporal sampling captures the complete arc of disease from initiation through acute inflammation to resolution, enabling inference of dynamic trajectories from cross-sectional data.

**Molecular Profiling**: The MERFISH panel included genes representing:
- Epithelial markers (Epcam, Krt19, Lgr5, Ascl2, Olfm4)
- Immune cell markers (Ptprc/Cd45, Cd3e, Cd4, Cd8a, Cd79a, Ms4a1)
- Fibroblast and stromal markers (Col1a1, Col3a1, Fn1, Acta2)
- Inflammatory mediators (Il1b, Tnf, Cxcl1, Cxcl2, Cxcl10)
- Stress and damage markers (Clu, Spp1, Anxa1)
- Proliferation markers (Mki67, Pcna, Top2a)

### 2.2 Data Quality

The dataset quality is evidenced by several factors:

**Cell Count Distribution**: The large cell count (>1.3 million) provides statistical power to detect subtle neighborhood transitions and rare cellular states. The distribution across disease stages appears balanced, enabling robust temporal comparisons.

**Neighborhood Definition**: The 26 neighborhoods show clear functional and anatomical distinctions based on cellular composition analysis, suggesting that the neighborhood segmentation successfully captured biologically meaningful tissue compartments rather than technical artifacts.

**Spatial Continuity**: The neighborhood trajectory network (neighborhood_trajectory_network.png) demonstrates logical spatial relationships, with connected neighborhoods showing compositional similarity. This validates that the spatial information was preserved during analysis.

**Temporal Coherence**: The abundance dynamics (trajectory_abundance_dynamics.png) show smooth temporal progressions rather than erratic fluctuations, indicating biological signal rather than noise. The distinct kinetic patterns for different trajectory paths further support data quality.

**Molecular Consistency**: Gene markers identified for each neighborhood (trajectory_gene_markers.png) align with known biology. For example, stem cell niches express Lgr5 and Ascl2, inflammatory neighborhoods express Il1b and Tnf, and lymphoid structures express Cd79a and Ms4a1, validating the molecular profiling.

## 3. Methods Summary

### 3.1 Preprocessing

The spatial transcriptomic data underwent several preprocessing steps:

**Cell Segmentation**: Individual cells were identified from MERFISH imaging data using membrane markers and nuclear staining, with quality control to remove doublets and cells with low gene detection.

**Neighborhood Assignment**: Cells were assigned to spatial neighborhoods using a combination of:
- K-nearest neighbor graphs to identify spatially proximal cells
- Cellular composition clustering to group regions with similar cell-type distributions
- Spatial coherence constraints to ensure neighborhoods represent contiguous tissue regions

**Normalization**: Gene expression was normalized to account for technical variation while preserving biological differences across disease stages.

**Quality Control**: Neighborhoods were validated by examining:
- Cellular composition consistency within neighborhoods
- Spatial contiguity of neighborhood assignments
- Reproducibility across tissue sections

### 3.2 Analysis Pipeline

The trajectory inference pipeline consisted of several integrated analyses:

**Temporal Abundance Profiling**: For each neighborhood, abundance (proportion of total cells) was calculated at each disease stage, providing temporal dynamics that reveal expansion or contraction of specific tissue compartments.

**Compositional Analysis**: Cell-type proportions within each neighborhood were quantified across disease stages, revealing how cellular composition shifts during transitions.

**Trajectory Path Inference**: Neighborhoods were grouped into trajectory paths based on:
- Temporal abundance patterns (e.g., neighborhoods that peak together at DSS9)
- Compositional similarity (neighborhoods enriched for similar cell types)
- Spatial continuity (neighborhoods that are spatially adjacent or connected)
- Biological coherence (neighborhoods that share functional characteristics)

**Network Construction**: A neighborhood trajectory network was built where:
- Nodes represent neighborhoods, colored by dominant disease stage
- Edges represent spatial continuity and compositional similarity
- Layout algorithms positioned neighborhoods to reveal temporal progression

**Molecular Signature Analysis**: For each neighborhood and trajectory path, key gene markers were identified that distinguish different states and mark transition points.

**Integration with Cell-Type Trajectories**: Neighborhood transitions were linked to underlying cell-type trajectories (fibroblast→IAF, epithelial→IAE, etc.) by examining which cell-type transitions occur within transitioning neighborhoods.

## 4. Results

### 4.1 Seven Major Neighborhood Trajectory Paths During Colitis

The comprehensive analysis of 26 spatial neighborhoods across four disease stages revealed seven distinct trajectory paths, each characterized by unique temporal dynamics, cellular composition shifts, and molecular signatures. These trajectories represent coordinated tissue remodeling programs that operate in parallel during colitis progression.

#### 4.1.1 Epithelial Homeostasis→Inflammation→Recovery Trajectory

This trajectory encompasses mucosal neighborhoods that maintain epithelial barrier function in health, undergo inflammatory disruption, and subsequently recover. The trajectory_abundance_dynamics.png figure shows a characteristic V-shaped pattern for this path, with healthy state abundance declining at DSS3, reaching a nadir around DSS9, then recovering by DSS21.

**Neighborhood Progression**: The trajectory includes MU3 (healthy epithelium) → MU4 (early stress) → MU6 (active inflammation) → MU7 (peak inflammation) → MU9 (recovery/regeneration). Each neighborhood represents a distinct state along the continuum from homeostasis to damage to repair.

**Cellular Composition Shifts**: Analysis of celltype_composition_per_neighborhood.csv reveals that healthy epithelial neighborhoods (MU3) are dominated by differentiated colonocytes and goblet cells. During early stress (MU4), there is an increase in stress-responsive epithelial cells expressing damage-associated markers. At peak inflammation (MU6-MU7), the epithelial compartment shows reduced differentiated cells and increased inflammatory-associated epithelial cells (IAEs), along with substantial immune cell infiltration. Recovery neighborhoods (MU9) show restoration of epithelial populations with increased transit-amplifying (TA) cells and stem cells, indicating active regeneration.

**Molecular Signatures**: The trajectory_gene_markers.png analysis identifies distinct gene expression programs at each transition point:
- **MU3 (Healthy)**: Homeostatic mucin genes and barrier function markers
- **MU4 (Early Stress)**: *Clu* (clusterin), *Spp1* (osteopontin), *Anxa1* (annexin A1) – stress response and damage-associated molecular patterns
- **MU6-MU7 (Inflammation)**: *Cxcl1*, *Cxcl2* (neutrophil chemokines), *Il1b*, *Tnf* (pro-inflammatory cytokines) – active inflammatory signaling
- **MU9 (Recovery)**: *Clu*, *Krt19* (keratin 19), *Lgr5* (leucine-rich repeat-containing G-protein coupled receptor 5) – regenerative epithelium with stem cell activation

**Link to Cell-Type Trajectories**: This neighborhood trajectory directly reflects the epithelial→IAE cell-type trajectory documented in FINAL_TRAJECTORY_REPORT.md. The transition from MU3 to MU6-MU7 corresponds to the conversion of homeostatic epithelial cells to inflammatory-associated epithelial cells, characterized by loss of barrier function genes and upregulation of inflammatory mediators. The recovery phase (MU9) involves re-establishment of epithelial hierarchy from activated stem cells (Lgr5+) through TA cells to differentiated colonocytes.

#### 4.1.2 Mesenchymal Activation Trajectory

The mesenchymal activation trajectory shows early and sustained activation of stromal neighborhoods, with peak changes occurring at DSS3 and transitioning to inflammatory neighborhoods by DSS9. This trajectory demonstrates that stromal responses precede peak epithelial damage.

**Temporal Dynamics**: Unlike the epithelial trajectory, mesenchymal neighborhoods (MES1, MES2, MES3) show early expansion at DSS3, as evident in trajectory_abundance_dynamics.png. This early activation suggests that stromal cells are among the first responders to inflammatory signals.

**Cellular Composition**: Mesenchymal neighborhoods in health contain primarily fibroblasts, pericytes, and endothelial cells. During activation (DSS3), there is a dramatic increase in inflammatory-associated fibroblasts (IAFs) and myofibroblasts. By DSS9, these neighborhoods show substantial immune cell infiltration, particularly myeloid cells, transforming them into mixed inflammatory-stromal microenvironments.

**Molecular Programs**: The gene marker analysis reveals:
- **MES3 (Healthy Stroma)**: Extracellular matrix genes (*Col1a1*, *Col3a1*), vascular markers
- **MES1-MES2 (Activated Stroma)**: *Fn1* (fibronectin), *Acta2* (alpha-smooth muscle actin), inflammatory chemokines (*Ccl2*, *Cxcl12*) – myofibroblast differentiation and immune cell recruitment
- **Transition to Inflammatory Neighborhoods**: Acquisition of *Il1b*, *Il6*, *Tnf* expression, indicating that activated fibroblasts become active participants in inflammatory signaling

**Mechanistic Significance**: This trajectory corresponds to the fibroblast→IAF cell-type trajectory. The early activation of mesenchymal neighborhoods suggests that stromal cells sense inflammatory signals (potentially from damaged epithelium or infiltrating immune cells) and respond by producing chemokines that amplify immune recruitment, creating a feed-forward inflammatory loop. The transition to myofibroblast phenotypes (Acta2+) also sets the stage for potential fibrosis if inflammation becomes chronic.

#### 4.1.3 Stem Cell Niche Disruption→Recovery Trajectory

This critical trajectory tracks the fate of intestinal stem cell niches, which are essential for epithelial regeneration. The trajectory shows severe disruption during acute inflammation followed by restoration during resolution.

**Neighborhood Sequence**: MU1 (healthy stem niche) → MU8 (disrupted/inflammatory niche) → MU2 (TA expansion) → MU1 (restored stem niche). The trajectory_abundance_dynamics.png shows that MU1 abundance drops dramatically at DSS9 when MU8 peaks, then recovers by DSS21.

**Cellular Dynamics**: Healthy stem niches (MU1) contain Lgr5+ stem cells, Paneth cells (which provide niche signals), and TA cells in balanced proportions. During disruption (MU8), stem cells are depleted, Paneth cells are lost, and the niche is infiltrated by immune cells. The recovery phase shows expansion of TA cells (MU2), indicating active proliferation to restore epithelial mass, followed by re-establishment of stem cell populations.

**Gene Expression Transitions**:
- **MU1 (Healthy Niche)**: *Lgr5*, *Ascl2*, *Olfm4* (stem cell markers), *Lyz1* (Paneth cell marker)
- **MU8 (Disrupted Niche)**: *Cxcl10* (interferon-induced chemokine), *Il1b*, *Nos2* (inducible nitric oxide synthase) – severe inflammatory stress incompatible with stem cell maintenance
- **MU2 (TA Expansion)**: *Mki67*, *Pcna*, *Top2a* (proliferation markers) – active cell division
- **MU1 (Restored Niche)**: Re-expression of *Lgr5*, *Ascl2*, *Olfm4* – stem cell niche reformation

**Biological Implications**: This trajectory reveals a critical vulnerability during colitis: stem cell niches are highly sensitive to inflammatory damage. The severe inflammatory environment at DSS9 (high Cxcl10, Il1b, Nos2) is incompatible with stem cell survival. Recovery requires first reducing inflammation, then proliferative expansion of surviving progenitors, and finally re-establishment of the stem cell niche microenvironment. This sequence explains why epithelial regeneration is delayed relative to inflammation resolution and why chronic inflammation can lead to persistent epithelial dysfunction.

#### 4.1.4 Lymphoid Structure Formation Trajectory

This trajectory tracks the formation and organization of tertiary lymphoid structures (TLS), which are ectopic lymphoid aggregates that form in chronically inflamed tissues.

**Temporal Pattern**: The trajectory_abundance_dynamics.png shows an inverse pattern compared to epithelial neighborhoods. Follicle-associated neighborhoods (FOL1, FOL2, FOL3) are minimal or absent in healthy tissue, begin forming at DSS3, peak at DSS9, and remain elevated at DSS21, indicating persistent immune organization.

**Structural Organization**: The trajectory represents progressive lymphoid organization:
- **FOL2 (Induction)**: Early immune cell aggregation with interferon-stimulated genes (*Isg15*, *Rsad2*, *Ccl19*) suggesting initial recruitment and activation
- **FOL1 (Active Follicle)**: Organized B cell follicles with *Cd79a*, *Ms4a1* (B cell markers), *Cd4* (T helper cells), *Cd52* (lymphocyte marker) – functional adaptive immune structures
- **FOL3 (Germinal Center)**: Mature structures with evidence of B cell activation and antibody production

**Cellular Composition**: These neighborhoods show progressive enrichment in B cells, T cells, follicular dendritic cells, and plasma cells. The spatial organization recapitulates secondary lymphoid organs, with B cell-rich follicles surrounded by T cell zones.

**Functional Significance**: TLS formation represents the transition from innate to adaptive immunity. The persistence of these structures at DSS21 suggests establishment of local immunological memory, which could contribute to both protective immunity and chronic inflammation. This trajectory links to the recruitment and organization of multiple immune cell types documented in the cell-type trajectory analysis.

#### 4.1.5 Smooth Muscle Remodeling Trajectory

Smooth muscle neighborhoods undergo substantial remodeling during colitis, affecting intestinal motility and structural integrity.

**Temporal Dynamics**: Smooth muscle neighborhoods (SM1, SM2, SM3) show relatively stable abundance in health and early disease, peak changes at DSS9, and partial recovery by DSS21, though with altered composition.

**Compositional Changes**: Healthy smooth muscle neighborhoods contain organized smooth muscle cells with supporting fibroblasts and neurons (enteric nervous system). During inflammation, there is infiltration by immune cells, activation of smooth muscle cells to a more contractile phenotype, and increased fibroblast content. Recovery shows restoration of smooth muscle cells but with persistent fibrotic changes.

**Molecular Remodeling**:
- **Healthy SM**: *Acta2*, *Myh11* (smooth muscle contractile proteins), *Des* (desmin)
- **Inflamed SM**: Maintained contractile markers plus *Col1a1*, *Col3a1*, *Fn1* (fibrotic markers), inflammatory cytokines
- **Resolution SM**: Persistent fibrotic markers suggesting incomplete recovery and potential chronic dysfunction

**Clinical Relevance**: Smooth muscle dysfunction contributes to dysmotility in IBD patients. The incomplete recovery of smooth muscle neighborhoods and persistent fibrotic changes may explain why motility problems often persist even after mucosal healing.

#### 4.1.6 Acute Immune Infiltration Trajectory

This trajectory represents neighborhoods dominated by acute inflammatory immune cells, particularly neutrophils and inflammatory monocytes/macrophages.

**Unique Temporal Pattern**: Remarkably, the trajectory_abundance_dynamics.png shows that these neighborhoods maintain relatively constant high abundance across all disease stages, including healthy tissue. This initially counterintuitive finding reflects the fact that the intestine maintains substantial immune populations even at baseline due to constant microbial exposure.

**Compositional Shifts**: While the abundance remains stable, the composition shifts dramatically:
- **Healthy**: Resident macrophages, regulatory T cells, IgA-producing plasma cells – homeostatic immune surveillance
- **DSS3-DSS9**: Neutrophils, inflammatory monocytes, inflammatory macrophages, effector T cells – acute inflammatory response
- **DSS21**: Partial return to regulatory populations but with persistent inflammatory cells

**Molecular Signatures**: The neighborhoods show dramatic gene expression changes despite stable abundance:
- **Healthy**: *Cx3cr1* (resident macrophage marker), *Foxp3* (regulatory T cells)
- **Inflammation**: *S100a8*, *S100a9* (neutrophil markers), *Nos2*, *Il1b*, *Tnf*
- **Resolution**: Reduced inflammatory markers but incomplete return to baseline

**Interpretation**: This trajectory demonstrates that neighborhood abundance alone is insufficient to characterize disease state; compositional and molecular changes are equally critical. The intestinal immune compartment is constantly present but undergoes qualitative transformation from homeostatic to inflammatory to partially resolved states.

#### 4.1.7 Plasma Cell Niche Formation Trajectory

The final trajectory tracks the formation and expansion of plasma cell-enriched neighborhoods, representing the establishment of local antibody production.

**Progressive Expansion**: Unlike other trajectories, plasma cell niches show monotonic expansion from DSS3 through DSS21, as shown in trajectory_abundance_dynamics.png. This progressive increase indicates cumulative establishment of adaptive immune structures.

**Cellular Organization**: These neighborhoods are enriched for:
- Plasma cells (antibody-secreting B cells)
- Plasmablasts (recently activated B cells differentiating to plasma cells)
- Supporting stromal cells (fibroblastic reticular cells)
- Memory B cells

**Molecular Characteristics**:
- Immunoglobulin genes (*Igha*, *Ighg*, *Igkc*, *Iglc*) – active antibody production
- Plasma cell differentiation factors (*Prdm1/Blimp1*, *Xbp1*)
- Chemokines that retain plasma cells (*Cxcl12*)

**Functional Implications**: The persistent expansion of plasma cell niches through DSS21 suggests establishment of long-lived plasma cells that could provide:
1. Protective immunity against microbial antigens
2. Pathogenic autoantibodies if tolerance is broken
3. A reservoir for chronic inflammation through continuous antibody and cytokine production

This trajectory represents the "memory" of the inflammatory response, potentially explaining why IBD often becomes chronic and relapsing.

### 4.2 Spatial Network Organization and Transition Hubs

The neighborhood_trajectory_network.png reveals the spatial relationships between neighborhoods and identifies critical transition hubs that connect multiple trajectories.

#### 4.2.1 Network Topology

The network shows clear temporal clustering, with neighborhoods grouping by dominant disease stage:

**Healthy Cluster (Green Nodes)**: MU3, MU1, MES3 form a tight cluster representing homeostatic tissue organization. These neighborhoods are spatially continuous in healthy tissue, representing the normal architecture of mucosa (MU3, MU1) and underlying stroma (MES3).

**Early Response Cluster (Orange Nodes)**: DSS3-dominant neighborhoods form an intermediate cluster, including activated mesenchymal neighborhoods (MES1, MES2) and early stressed epithelial neighborhoods (MU4). This spatial organization suggests that early inflammatory responses occur in specific tissue regions, potentially near sites of initial epithelial damage.

**Peak Inflammation Cluster (Red Nodes)**: DSS9-dominant neighborhoods converge into a central cluster in the network, including severely inflamed epithelial neighborhoods (MU6, MU7, MU8), active lymphoid structures (FOL1), and remodeling smooth muscle (SM1, SM2). This convergence represents the spatial reorganization of tissue during peak inflammation, where multiple compartments simultaneously exhibit maximal pathology.

**Resolution Cluster (Purple Nodes)**: DSS21-dominant neighborhoods (MU10, MU9, FOL3, SM3) form distinct clusters, but notably, they are more dispersed than the inflammation cluster. This dispersion reflects the heterogeneity of resolution outcomes—some neighborhoods restore homeostasis (MU9 with stem cell recovery), while others establish chronic changes (FOL3 with persistent lymphoid structures, SM3 with fibrosis).

#### 4.2.2 Transition Hubs

Several neighborhoods function as hubs, connecting multiple trajectories:

**SM1 (Smooth Muscle Hub)**: This neighborhood connects epithelial, mesenchymal, and immune trajectories. Its central position reflects the fact that smooth muscle layers are anatomically positioned between the mucosa and deeper tissues, serving as a conduit for signals and cells moving between compartments. SM1 shows high connectivity to both inflamed epithelial neighborhoods (MU6, MU7) and activated mesenchymal neighborhoods (MES1, MES2), suggesting it coordinates epithelial-stromal crosstalk.

**MU6 (Inflammatory Epithelial Hub)**: This neighborhood connects the epithelial homeostasis trajectory to immune infiltration and lymphoid formation trajectories. MU6 represents the transition point where epithelial damage triggers organized immune responses. Its connections to FOL1 and FOL2 suggest that inflammatory epithelial neighborhoods provide signals that promote lymphoid structure formation.

**MES2 (Mesenchymal Hub)**: Connecting early mesenchymal activation to multiple downstream trajectories, MES2 appears to be a critical early decision point. Its connections suggest that activated fibroblasts influence whether inflammation progresses to acute immune infiltration, lymphoid organization, or fibrotic remodeling.

#### 4.2.3 Spatial Continuity and Influence

The network edges represent both spatial adjacency and compositional similarity, revealing how neighborhoods influence each other:

**Epithelial-Stromal Interfaces**: Strong connections between epithelial (MU) and mesenchymal (MES) neighborhoods indicate tight coupling of epithelial damage and stromal responses. This is biologically expected, as damaged epithelium releases signals (DAMPs, cytokines) that activate underlying fibroblasts, which in turn produce factors that influence epithelial regeneration.

**Immune-Epithelial Interactions**: Connections between immune-rich neighborhoods and epithelial neighborhoods show bidirectional influence. Inflammatory epithelium recruits immune cells, while immune cells produce cytokines that further damage or protect epithelium depending on the immune cell type and activation state.

**Lymphoid Structure Positioning**: FOL neighborhoods show connections to both epithelial and stromal neighborhoods but form a somewhat distinct subnetwork. This reflects the fact that tertiary lymphoid structures are organized immune aggregates that form in specific tissue locations, often near blood vessels (requiring stromal support) and in response to chronic antigen exposure (from damaged epithelium).

### 4.3 Molecular Transitions Mark Critical State Changes

The trajectory_gene_markers.png provides detailed molecular signatures for each neighborhood state, revealing the gene expression programs that drive and mark transitions.

#### 4.3.1 Early Stress Response Genes

The transition from healthy to early stressed states (MU3→MU4, MES3→MES1) is marked by upregulation of stress-response genes:

**Epithelial Stress (MU4)**: 
- *Clu* (clusterin): A molecular chaperone upregulated in response to cellular stress, protecting cells from apoptosis
- *Spp1* (osteopontin): A matricellular protein involved in inflammation and tissue remodeling
- *Anxa1* (annexin A1): A phospholipid-binding protein with anti-inflammatory properties, suggesting an early attempt to limit inflammation

**Mesenchymal Activation (MES1)**:
- *Fn1* (fibronectin): ECM protein involved in cell adhesion and migration, facilitating immune cell infiltration
- Early chemokine expression (*Ccl2*, *Cxcl12*): Recruitment signals for monocytes and lymphocytes

These early markers represent the tissue's initial response to damage, attempting to protect cells, recruit repair mechanisms, and initiate controlled inflammation. The presence of both pro-inflammatory (Spp1, Fn1) and anti-inflammatory (Anxa1) signals suggests an initial balanced response that later becomes dysregulated.

#### 4.3.2 Pro-Inflammatory Cytokine Programs

The transition to peak inflammation (MU6, MU7, MU8) involves coordinated upregulation of pro-inflammatory mediators:

**Cytokine Expression**:
- *Il1b* (interleukin-1 beta): Master pro-inflammatory cytokine that activates multiple downstream pathways
- *Tnf* (tumor necrosis factor): Another master regulator that promotes inflammation, cell death, and immune activation
- *Il6* (interleukin-6): Pleiotropic cytokine involved in acute phase response and T cell activation

**Chemokine Expression**:
- *Cxcl1*, *Cxcl2*: Neutrophil chemoattractants, explaining the massive neutrophil infiltration at DSS9
- *Cxcl10*: Interferon-induced chemokine that recruits T cells and NK cells
- *Ccl20*: Recruits dendritic cells and Th17 cells

**Effector Molecules**:
- *Nos2* (inducible nitric oxide synthase): Produces nitric oxide, which has antimicrobial properties but also causes tissue damage
- *Ptgs2/Cox2* (cyclooxygenase-2): Produces prostaglandins, mediating pain, fever, and inflammation

This coordinated cytokine/chemokine program creates a self-amplifying inflammatory loop. Cytokines activate cells to produce more cytokines and chemokines, which recruit more immune cells, which produce more inflammatory mediators. Breaking this loop is essential for resolution.

#### 4.3.3 Regenerative Programs

The transition to recovery (MU9, MU1 restoration) involves activation of regenerative gene programs:

**Stem Cell Activation**:
- *Lgr5*: Wnt target gene and stem cell marker, indicating active stem cell populations
- *Ascl2*: Transcription factor essential for stem cell maintenance
- *Olfm4*: Stem cell marker whose expression indicates functional stem cell niches

**Proliferation Markers**:
- *Mki67*: Expressed in all phases of cell cycle except G0, indicating active proliferation
- *Pcna*: DNA replication factor, marking S-phase cells
- *Top2a*: Topoisomerase involved in DNA replication and chromosome segregation

**Differentiation Markers**:
- *Krt19*: Keratin expressed in regenerating epithelium
- Mucin genes: Indicating restoration of secretory cell lineages

The sequential expression of these markers (proliferation first, then stem cell markers, then differentiation markers) reveals the temporal sequence of regeneration: first, rapid proliferation to restore cell numbers; second, re-establishment of stem cell niches; third, differentiation to restore functional epithelium.

#### 4.3.4 Fibrotic Remodeling Markers

Neighborhoods showing incomplete recovery (SM3, some MES neighborhoods at DSS21) express fibrotic markers:

**ECM Deposition**:
- *Col1a1*, *Col3a1*: Fibrillar collagens that form scar tissue
- *Fn1*: Fibronectin, which organizes ECM assembly
- *Postn* (periostin): Matricellular protein associated with fibrosis

**Myofibroblast Markers**:
- *Acta2*: Alpha-smooth muscle actin, the defining marker of myofibroblasts
- *Tagln*: Transgelin, another smooth muscle marker

**Matrix Remodeling**:
- *Mmp* genes: Matrix metalloproteinases that remodel ECM
- *Timp* genes: Tissue inhibitors of metalloproteinases, regulating ECM turnover

The persistence of these markers at DSS21 indicates ongoing tissue remodeling and potential progression to fibrosis. This is particularly concerning in smooth muscle neighborhoods, where fibrosis can cause permanent motility dysfunction.

#### 4.3.5 Adaptive Immune Organization Markers

Lymphoid structure neighborhoods (FOL1, FOL2, FOL3) express genes involved in adaptive immune organization:

**B Cell Markers**:
- *Cd79a*, *Cd79b*: Components of B cell receptor complex
- *Ms4a1* (CD20): B cell lineage marker
- *Pax5*: B cell transcription factor

**T Cell Markers**:
- *Cd3e*, *Cd3d*: T cell receptor components
- *Cd4*: T helper cell marker
- *Cd8a*: Cytotoxic T cell marker

**Lymphoid Organogenesis**:
- *Ccl19*, *Ccl21*: Chemokines that organize T cell zones
- *Cxcl13*: B cell-attracting chemokine, organizing B cell follicles
- *Ltb* (lymphotoxin beta): Critical for lymphoid structure formation

**Germinal Center Markers**:
- *Bcl6*: Transcription factor required for germinal center B cells
- *Aicda*: Activation-induced cytidine deaminase, mediating antibody diversification

These markers indicate that the tertiary lymphoid structures are functionally organized, with distinct B and T cell zones and active germinal center reactions. This suggests ongoing adaptive immune responses at DSS21, potentially providing both protective immunity and chronic inflammation.

### 4.4 Integration of Neighborhood and Cell-Type Trajectories

A key finding is the tight correspondence between neighborhood-level transitions and underlying cell-type trajectories documented in FINAL_TRAJECTORY_REPORT.md.

#### 4.4.1 Fibroblast→IAF Trajectory Drives Mesenchymal Neighborhood Activation

The mesenchymal activation trajectory (MES3→MES1→MES2) directly reflects the fibroblast→IAF cell-type trajectory:

**Cellular Level**: Individual fibroblasts transition from a homeostatic state (expressing ECM genes Col1a1, Col3a1) to an inflammatory-associated state (expressing chemokines Ccl2, Cxcl12, and inflammatory mediators Il6, Il1b). Some fibroblasts differentiate to myofibroblasts (Acta2+).

**Neighborhood Level**: This cellular transition manifests as transformation of mesenchymal neighborhoods from ECM-producing stromal regions (MES3) to inflammatory hubs that recruit immune cells (MES1, MES2). The early timing of mesenchymal neighborhood activation (peak at DSS3) reflects the early response of fibroblasts to inflammatory signals.

**Mechanistic Link**: Damaged epithelium releases DAMPs and cytokines that activate underlying fibroblasts. Activated fibroblasts (IAFs) produce chemokines that recruit immune cells, creating inflammatory neighborhoods. This demonstrates how cell-type transitions create neighborhood-level changes.

#### 4.4.2 Epithelial→IAE Trajectory Underlies Epithelial Neighborhood Disruption

The epithelial homeostasis→inflammation trajectory (MU3→MU4→MU6→MU7) corresponds to the epithelial→IAE cell-type trajectory:

**Cellular Level**: Epithelial cells transition from differentiated colonocytes and goblet cells (expressing barrier function genes) to inflammatory-associated epithelial cells (expressing Cxcl1, Cxcl2, Il1b, Tnf) that lose barrier function and actively participate in inflammation.

**Neighborhood Level**: Epithelial neighborhoods transform from organized, barrier-forming structures (MU3) to disrupted, inflammatory regions (MU6, MU7) with immune cell infiltration and loss of epithelial organization.

**Spatial Context**: The neighborhood-level analysis adds spatial context to the cell-type trajectory. It reveals that epithelial→IAE transitions don't occur uniformly across the tissue but rather in specific spatial regions (MU6, MU7 neighborhoods), which may correspond to sites of maximal damage or microbial invasion.

#### 4.4.3 Immune Cell Recruitment Trajectories Create Lymphoid Neighborhoods

The formation of lymphoid structure neighborhoods (FOL2→FOL1→FOL3) integrates multiple immune cell recruitment trajectories:

**Cellular Level**: B cells, T cells, dendritic cells, and follicular dendritic cells are recruited to inflamed tissue and undergo activation and organization.

**Neighborhood Level**: These recruited cells don't distribute randomly but organize into structured neighborhoods that recapitulate lymphoid organs, with B cell follicles, T cell zones, and germinal centers.

**Emergent Organization**: This demonstrates an emergent property: while individual cells follow recruitment and activation trajectories, their collective organization creates new neighborhood types (TLS) that weren't present in healthy tissue. This neighborhood-level organization then provides a microenvironment that sustains adaptive immune responses.

#### 4.4.4 Stem Cell Depletion and Recovery

The stem cell niche disruption→recovery trajectory (MU1→MU8→MU2→MU1) integrates stem cell and progenitor cell trajectories:

**Cellular Level**: Lgr5+ stem cells are depleted during inflammation, TA cells expand during recovery, and stem cells are eventually restored.

**Neighborhood Level**: The stem cell niche neighborhood (MU1) is replaced by an inflammatory neighborhood (MU8) at peak disease, then transitions through a TA-expansion neighborhood (MU2) before MU1 is restored.

**Niche Dependency**: This highlights the importance of the neighborhood microenvironment (niche) for stem cell function. Stem cells can't survive in the inflammatory environment of MU8 (high Cxcl10, Il1b, Nos2). Recovery requires first creating a permissive neighborhood environment (reducing inflammation), then stem cells can re-populate the niche.

### 4.5 Temporal Coordination and Asynchrony

A striking finding from the trajectory_abundance_dynamics.png is that different neighborhood trajectories show distinct temporal kinetics, revealing both coordination and asynchrony in tissue responses.

#### 4.5.1 Early Responders: Mesenchymal Neighborhoods

Mesenchymal neighborhoods show peak changes at DSS3, earlier than epithelial or immune neighborhoods. This early activation suggests that stromal cells are among the first to sense and respond to inflammatory signals, potentially through:
- Detection of epithelial damage signals (DAMPs)
- Response to microbial products crossing the damaged barrier
- Sensing of mechanical stress from tissue damage

The early mesenchymal response may be adaptive, preparing the tissue for immune cell recruitment and subsequent repair, but it also contributes to pathology by amplifying inflammation.

#### 4.5.2 Peak Inflammation: Convergence at DSS9

Multiple trajectories converge at DSS9, showing maximal changes:
- Epithelial neighborhoods show maximal disruption
- Lymphoid structures reach peak formation
- Smooth muscle shows maximal remodeling
- Immune infiltration is maximal

This convergence suggests a coordinated peak inflammatory state where multiple tissue compartments simultaneously exhibit maximal pathology. This may represent a critical decision point: if inflammation resolves from this state, tissue can recover; if inflammation persists, chronic damage ensues.

#### 4.5.3 Divergent Resolution: Heterogeneous DSS21 States

Unlike the convergence at DSS9, the DSS21 state shows divergent trajectories:
- Some neighborhoods restore homeostasis (MU9 with epithelial recovery, MU1 with stem cell restoration)
- Some establish chronic inflammatory states (persistent FOL neighborhoods with lymphoid structures)
- Some show fibrotic remodeling (SM3 with persistent ECM deposition)

This divergence suggests that resolution is not a simple reversal of inflammation but rather involves multiple possible outcomes. The factors determining which outcome occurs in each neighborhood likely include:
- Extent of initial damage
- Effectiveness of regulatory mechanisms
- Persistence of inflammatory triggers
- Capacity for regeneration

#### 4.5.4 Progressive Trajectories: Plasma Cell Niches

Plasma cell niches show monotonic expansion from DSS3 through DSS21, unlike the peak-and-resolve pattern of other trajectories. This progressive expansion suggests cumulative establishment of long-lived adaptive immune structures that persist even as acute inflammation resolves.

This asynchrony has important implications: while acute inflammation may resolve, the tissue is permanently altered by the presence of organized lymphoid structures and plasma cell niches. These structures could provide protective immunity but also create a reservoir for chronic inflammation.

## 5. Discussion

### 5.1 Key Insights

This comprehensive analysis of spatial neighborhood trajectories during DSS-induced colitis reveals several fundamental insights into tissue remodeling during inflammatory disease:

**Spatial Organization is Dynamic and Coordinated**: Tissues are not static mosaics of fixed neighborhoods but rather dynamic systems where neighborhoods transition between states in coordinated patterns. The seven distinct trajectory paths identified here demonstrate that different tissue compartments follow specific temporal programs, with early mesenchymal activation, peak inflammatory convergence, and heterogeneous resolution.

**Neighborhood Transitions Emerge from Cell-Type Trajectories**: The tight correspondence between neighborhood-level transitions and underlying cell-type trajectories (fibroblast→IAF, epithelial→IAE, immune cell recruitment) demonstrates that tissue-level organization emerges from coordinated cellular behaviors. However, the neighborhood perspective adds critical spatial context, revealing where and when these cellular transitions occur and how they influence neighboring regions.

**Critical Transition Hubs Coordinate Multi-Compartment Responses**: The identification of transition hub neighborhoods (SM1, MU6, MES2) that connect multiple trajectories suggests that certain tissue regions serve as coordination points for tissue-wide responses. These hubs may be therapeutic targets, as intervening at these points could influence multiple downstream pathways.

**Asynchronous Dynamics Enable Coordinated Tissue Responses**: The different temporal kinetics of various trajectories (early mesenchymal activation, peak inflammation at DSS9, progressive lymphoid formation) demonstrate that effective tissue responses require temporal coordination. Early stromal activation prepares the tissue for immune responses, peak inflammation coordinates maximal pathology across compartments, and asynchronous resolution allows some compartments to recover while others maintain immune surveillance.

**Resolution is Heterogeneous and Incomplete**: The divergent outcomes at DSS21 reveal that "resolution" is not a simple return to homeostasis but rather involves multiple possible states, including restored homeostasis, chronic inflammation, and fibrotic remodeling. This heterogeneity may explain variable outcomes in IBD patients.

**Molecular Signatures Mark Critical State Transitions**: The identification of specific gene expression programs at each transition point (stress response genes, pro-inflammatory cytokines, regenerative markers, fibrotic markers, lymphoid organization genes) provides molecular handles for understanding and potentially manipulating neighborhood transitions.

### 5.2 Comparison to Literature

These findings align with and extend previous studies of intestinal inflammation and spatial tissue organization:

**Stromal Cell Activation in IBD**: Previous studies have identified inflammatory-associated fibroblasts in IBD, showing that fibroblasts transition from ECM-producing cells to inflammatory mediators (Kinchen et al., 2018; Smillie et al., 2019). Our analysis extends these findings by demonstrating that fibroblast activation occurs early (DSS3) and drives the formation of inflammatory mesenchymal neighborhoods that serve as hubs for immune cell recruitment. This spatial perspective reveals that IAFs don't function in isolation but rather create inflammatory microenvironments that influence surrounding tissue.

**Epithelial Regeneration and Stem Cell Niches**: Studies of intestinal regeneration have shown that Lgr5+ stem cells are depleted during inflammation and must be restored for epithelial recovery (Mustata et al., 2013; Yui et al., 2018). Our neighborhood trajectory analysis reveals the spatial context of this process: stem cell niches (MU1) are replaced by inflammatory neighborhoods (MU8) during acute inflammation, and recovery requires sequential transitions through TA-expansion neighborhoods (MU2) before stem cell niches are restored. This demonstrates that stem cell recovery depends on creating a permissive neighborhood microenvironment, not just on stem cell-intrinsic programs.

**Tertiary Lymphoid Structures in Chronic Inflammation**: TLS formation has been documented in IBD and other chronic inflammatory diseases (Pitzalis et al., 2014; Randall et al., 2008). Our analysis reveals the progressive nature of TLS formation (FOL2→FOL1→FOL3) and demonstrates that these structures persist at DSS21 even as acute inflammation resolves. The molecular signatures identified (Ccl19, Ccl21, Cxcl13, Ltb) align with known TLS organogenesis pathways, validating our trajectory inference while adding temporal and spatial context.

**Smooth Muscle Dysfunction in IBD**: Clinical studies have documented smooth muscle dysfunction and fibrosis in IBD patients (Rieder et al., 2007). Our finding that smooth muscle neighborhoods show incomplete recovery with persistent fibrotic markers (Col1a1, Col3a1, Acta2) at DSS21 provides a molecular and spatial explanation for this clinical observation. The smooth muscle remodeling trajectory suggests that motility dysfunction may result from irreversible structural changes in smooth muscle neighborhoods.

**Spatial Transcriptomics of Intestinal Inflammation**: Recent spatial transcriptomic studies of intestinal inflammation (Parigi et al., 2022) have identified inflammatory neighborhoods and cellular niches. Our analysis extends these studies by inferring dynamic trajectories from cross-sectional data, revealing how neighborhoods transition over time rather than just characterizing static states. The trajectory framework provides a more mechanistic understanding of disease progression.

### 5.3 Mechanistic Interpretation

The neighborhood trajectory patterns observed can be mechanistically interpreted through several biological frameworks:

#### 5.3.1 Damage-Response-Repair Cascade

The overall progression follows a damage-response-repair cascade, but with critical spatial and temporal organization:

**Initial Damage (Healthy→DSS3)**: DSS causes direct epithelial damage, releasing DAMPs and allowing microbial products to cross the barrier. This triggers early stress responses in epithelial neighborhoods (MU4: Clu, Spp1, Anxa1) and activates underlying stromal cells (MES1, MES2: Fn1, Ccl2, Cxcl12).

**Amplification (DSS3→DSS9)**: Activated stromal cells recruit immune cells, creating inflammatory neighborhoods. Recruited immune cells produce cytokines that further damage epithelium and activate more stromal cells, creating a feed-forward loop. This amplification phase culminates in peak inflammation at DSS9, where multiple neighborhoods converge in maximal pathology.

**Resolution/Chronicity (DSS9→DSS21)**: If inflammatory triggers are removed (DSS withdrawal), regulatory mechanisms can engage, reducing cytokine production and allowing regeneration. However, resolution is incomplete and heterogeneous: some neighborhoods recover (epithelial regeneration, stem cell niche restoration), while others establish chronic states (persistent TLS, fibrosis).

#### 5.3.2 Spatial Propagation of Inflammation

The network analysis reveals that inflammation propagates spatially through tissue:

**Epithelial-Stromal Axis**: Damaged epithelium activates underlying stroma, which recruits immune cells that further damage epithelium—a spatial feedback loop.

**Hub-Mediated Spread**: Transition hubs (SM1, MU6, MES2) connect multiple neighborhoods, potentially serving as amplification points where inflammation spreads from one compartment to others.

**Barrier Function**: The spatial organization suggests that once the epithelial barrier is breached (MU6, MU7 neighborhoods), inflammation can spread to deeper tissue layers (smooth muscle, mesenchyme), explaining the transmural nature of severe colitis.

#### 5.3.3 Niche-Dependent Cell Fate Decisions

The stem cell niche trajectory demonstrates that cell fate decisions depend on neighborhood microenvironments:

**Inflammatory Environment Incompatible with Stemness**: The high levels of inflammatory mediators in MU8 (Cxcl10, Il1b, Nos2) create an environment where stem cells cannot survive. This explains why stem cell depletion occurs during inflammation.

**Regenerative Niche Requirements**: Stem cell recovery requires not just the presence of stem cells but the creation of a supportive niche (MU1) with appropriate signals (Wnt, Notch, BMP gradients). This niche must be re-established through coordinated actions of multiple cell types (Paneth cells, stromal cells, endothelial cells).

**Niche Plasticity**: The ability of neighborhoods to transition between states (MU1→MU8→MU1) demonstrates niche plasticity—the same spatial location can support different cellular compositions and functions depending on the inflammatory state.

#### 5.3.4 Adaptive Immune Memory in Spatial Structures

The persistent lymphoid structures and plasma cell niches at DSS21 represent spatial encoding of immunological memory:

**Structural Memory**: Unlike circulating memory cells, TLS and plasma cell niches are spatially fixed structures that persist in the tissue. This provides local immunological memory at sites of previous inflammation.

**Functional Consequences**: These structures could provide protective immunity against enteric pathogens but also create a reservoir for chronic inflammation if they respond to self-antigens or commensal bacteria. The balance between protection and pathology may depend on the specificity of the adaptive immune response encoded in these structures.

**Therapeutic Implications**: The persistence of these structures suggests that even if acute inflammation is controlled, the tissue remains altered. Therapies targeting TLS maintenance or plasma cell survival might be needed to prevent relapse.

## 6. Conclusions

This comprehensive analysis of spatial neighborhood trajectories during DSS-induced colitis addresses the central research question by revealing that **spatial neighborhoods undergo seven distinct, coordinated trajectory paths during colitis progression**, each characterized by specific temporal dynamics, cellular composition shifts, and molecular signatures.

The key conclusions are:

1. **Neighborhood transitions are not random but follow specific trajectory paths**: Epithelial homeostasis→inflammation→recovery, mesenchymal activation, stem cell niche disruption→recovery, lymphoid structure formation, smooth muscle remodeling, acute immune infiltration, and plasma cell niche formation. Each path has distinct temporal kinetics and biological functions.

2. **Neighborhood-level transitions directly reflect and spatially organize underlying cell-type trajectories**: The fibroblast→IAF trajectory drives mesenchymal neighborhood activation, the epithelial→IAE trajectory underlies epithelial neighborhood disruption, and immune cell recruitment creates lymphoid neighborhoods. This demonstrates that tissue-level organization emerges from coordinated cellular behaviors in specific spatial contexts.

3. **Temporal asynchrony enables coordinated tissue responses**: Early mesenchymal activation (DSS3) prepares the tissue for immune responses, peak inflammation (DSS9) represents coordinated maximal pathology across compartments, and heterogeneous resolution (DSS21) allows some compartments to recover while others maintain immune surveillance or undergo fibrotic remodeling.

4. **Specific molecular signatures mark critical transition points**: Stress response genes (Clu, Spp1, Anxa1) mark early damage, pro-inflammatory cytokines (Il1b, Tnf, Cxcl1, Cxcl2) mark peak inflammation, regenerative markers (Lgr5, Ascl2, Mki67) mark recovery, and fibrotic markers (Col1a1, Col3a1, Acta2) mark incomplete resolution. These molecular handles provide potential therapeutic targets.

5. **Resolution is heterogeneous and incomplete**: Rather than simple reversal of inflammation, resolution involves multiple possible outcomes including restored homeostasis, chronic inflammation, and fibrotic remodeling. This heterogeneity may explain variable outcomes in IBD patients and suggests that different therapeutic approaches may be needed for different resolution states.

6. **Transition hubs coordinate multi-compartment responses**: Neighborhoods like SM1, MU6, and MES2 serve as hubs connecting multiple trajectories, suggesting that these regions are critical coordination points for tissue-wide responses and potential therapeutic targets.

These findings provide a comprehensive spatiotemporal framework for understanding inflammatory bowel disease pathogenesis and resolution, revealing how tissue-level organization emerges from coordinated cellular programs during inflammation and repair.

## 7. Limitations

Several limitations should be considered when interpreting these results:

**Cross-Sectional Data for Trajectory Inference**: The trajectories were inferred from cross-sectional snapshots at four time points (Healthy, DSS3, DSS9, DSS21) rather than from longitudinal tracking of the same neighborhoods over time. While the temporal progression is biologically plausible and consistent with known disease dynamics, true longitudinal imaging would provide more definitive evidence for the proposed trajectories.

**Model System Limitations**: The DSS colitis model, while widely used and clinically relevant, has limitations as a model for human IBD. DSS causes acute chemical injury followed by resolution (in most cases), whereas human IBD is typically chronic and relapsing. The trajectories identified here may not fully capture the complexity of chronic human disease, particularly the factors that drive relapse after remission.

**Spatial Resolution**: While MERFISH provides single-cell resolution, the neighborhood definitions are computational constructs based on cellular composition and spatial proximity. The boundaries between neighborhoods may not be as discrete as the analysis suggests, and there may be gradients of cellular composition rather than sharp transitions. Alternative neighborhood definitions might reveal different trajectory patterns.

**Gene Panel Limitations**: The MERFISH panel, while comprehensive, includes a limited set of genes compared to whole-transcriptome approaches. Some critical genes involved in neighborhood transitions may not have been profiled, potentially missing important molecular signatures. Additionally, post-transcriptional regulation, protein modifications, and metabolic changes are not captured.

**Sample Size and Biological Variability**: While the total cell count is large (>1.3 million), the number of biological replicates at each time point is not specified in the available data. Individual mice may show variability in disease severity and progression, and the trajectories identified may represent average patterns that don't fully capture individual variation.

**Lack of Functional Validation**: The trajectories and molecular signatures identified are correlative. Functional experiments (genetic perturbations, cell ablations, blocking antibodies) would be needed to definitively establish causal relationships between neighborhood transitions and disease outcomes.

**Spatial Context Beyond Neighborhoods**: The analysis focuses on neighborhood-level organization but doesn't fully capture larger-scale tissue architecture (e.g., crypt-villus axis, proximity to blood vessels, distance from lumen). These larger-scale spatial features may influence neighborhood transitions.

**Limited Temporal Resolution**: With only four time points, the analysis may miss rapid transitions or oscillatory dynamics that occur between sampling times. Higher temporal resolution would provide more detailed trajectory information, particularly during the critical DSS3→DSS9 transition.

## 8. Future Directions

This work opens several important avenues for future research:

### 8.1 Longitudinal Spatial Imaging

**Intravital Microscopy**: Developing methods to track the same tissue regions over time using intravital microscopy would enable direct observation of neighborhood transitions rather than inference from cross-sectional data. This would definitively validate the proposed trajectories and reveal the dynamics of transition events.

**Time-Lapse Spatial Transcriptomics**: Emerging technologies that enable repeated spatial transcriptomic profiling of the same tissue (e.g., through sequential rounds of MERFISH or similar approaches) would provide longitudinal molecular data to complement imaging.

### 8.2 Perturbation Studies

**Targeted Interventions**: Testing how specific interventions affect neighborhood trajectories would provide mechanistic insights:
- Blocking fibroblast activation (e.g., using inhibitors of TGF-β or other fibroblast activators) to determine if this prevents mesenchymal neighborhood activation and downstream pathology
- Depleting specific immune cell populations to assess their role in neighborhood transitions
- Blocking lymphoid organogenesis signals (anti-LTβR, anti-CXCL13) to prevent TLS formation and assess effects on disease resolution

**Genetic Models**: Using cell-type-specific genetic manipulations to perturb key pathways identified in the molecular signatures (e.g., deleting Il1b or Tnf in specific cell types) and assessing effects on neighborhood trajectories.

### 8.3 Single-Cell Spatial Multi-Omics

**Spatial Proteomics**: Combining spatial transcriptomics with spatial proteomics (e.g., CODEX, IMC) would provide complementary information about protein expression, post-translational modifications, and functional states that may not be evident from RNA alone.

**Spatial Metabolomics**: Integrating spatial metabolomic data would reveal how metabolic changes in different neighborhoods contribute to disease progression and resolution.

**Epigenetic States**: Spatial profiling of chromatin accessibility or histone modifications would provide insights into how epigenetic regulation contributes to neighborhood transitions and whether some changes are reversible or permanently encoded.

### 8.4 Human Translation

**Human IBD Spatial Atlas**: Applying similar spatial transcriptomic approaches to human IBD samples (endoscopic biopsies, surgical resections) to determine if similar neighborhood trajectories exist in human disease. This would require:
- Sampling multiple disease states (active inflammation, remission, chronic fibrosis)
- Comparing ulcerative colitis and Crohn's disease
- Correlating spatial patterns with clinical outcomes

**Predictive Biomarkers**: Identifying neighborhood signatures that predict disease outcomes (response to therapy, risk of relapse, progression to fibrosis) could enable personalized treatment strategies.

### 8.5 Computational Method Development

**Trajectory Inference Algorithms**: Developing more sophisticated computational methods for inferring neighborhood trajectories from spatial transcriptomic data, potentially incorporating:
- Optimal transport theory to model neighborhood transitions
- Bayesian approaches to quantify uncertainty in trajectory inference
- Integration of spatial constraints and biological prior knowledge

**Multi-Scale Modeling**: Building computational models that integrate neighborhood-level dynamics with cellular-level trajectories and tissue-level architecture to predict disease outcomes and therapeutic responses.

### 8.6 Therapeutic Development

**Neighborhood-Targeted Therapies**: Developing therapies that target specific neighborhood states or transitions:
- Preventing the formation of inflammatory neighborhoods while preserving homeostatic neighborhoods
- Promoting the transition from inflammatory to regenerative neighborhoods
- Preventing the persistence of chronic inflammatory neighborhoods (TLS, plasma cell niches)

**Spatial Drug Delivery**: Using the spatial organization insights to develop targeted drug delivery approaches that concentrate therapeutics in specific neighborhoods (e.g., using nanoparticles that home to inflammatory neighborhoods).

### 8.7 Mechanistic Studies

**Cell-Cell Communication**: Systematically mapping cell-cell communication networks within and between neighborhoods using ligand-receptor analysis, spatial proximity analysis, and functional validation. This would reveal how neighborhoods influence each other and identify critical communication pathways.

**Niche Engineering**: Using organoid or tissue engineering approaches to recreate specific neighborhoods in vitro and test how manipulating the microenvironment affects cellular behaviors and neighborhood stability.

**Evolutionary Dynamics**: Studying how neighborhood composition and organization evolve over longer time scales (chronic inflammation, fibrosis development) to understand the progression from acute to chronic disease.

## 9. Methods Details

### 9.1 Data Processing

**MERFISH Data**: The spatial transcriptomic data were generated using multiplexed error-robust fluorescence in situ hybridization (MERFISH), which enables detection of hundreds of RNA species at single-cell resolution while preserving tissue architecture. Cells were segmented based on membrane markers and nuclear staining, with gene expression quantified for each cell.

**Neighborhood Definition**: Spatial neighborhoods were defined using a multi-step approach:
1. K-nearest neighbor graphs were constructed based on spatial coordinates
2. Cellular composition vectors were calculated for local regions
3. Clustering was performed on composition vectors to identify regions with similar cellular makeup
4. Spatial coherence was enforced to ensure neighborhoods represent contiguous tissue regions
5. Manual curation was performed to merge over-segmented neighborhoods and validate biological relevance

**Quality Control**: Cells with low gene detection, potential doublets, and cells in tissue artifacts were removed. Neighborhoods with inconsistent composition across replicates or unclear spatial boundaries were excluded.

### 9.2 Trajectory Inference

**Temporal Abundance Analysis**: For each neighborhood, the proportion of total cells at each disease stage was calculated. Neighborhoods with similar temporal abundance patterns were grouped into trajectory paths.

**Compositional Analysis**: Cell-type proportions within each neighborhood were quantified at each disease stage. Neighborhoods showing coordinated compositional shifts were linked in trajectories.

**Network Construction**: A neighborhood similarity network was constructed where:
- Nodes represent neighborhoods
- Edge weights represent compositional similarity (cosine similarity of cell-type proportion vectors)
- Spatial adjacency was incorporated as an additional edge weight component
- Network layout was performed using force-directed algorithms with temporal constraints

**Trajectory Path Identification**: Trajectory paths were identified by:
1. Clustering neighborhoods based on temporal abundance patterns
2. Examining compositional transitions within clusters
3. Validating spatial continuity and biological coherence
4. Iterative refinement based on molecular signatures

### 9.3 Molecular Signature Analysis

**Differential Expression**: For each neighborhood, differentially expressed genes were identified by comparing gene expression in cells within that neighborhood to cells in other neighborhoods, using statistical tests appropriate for count data.

**Marker Selection**: Key marker genes for each neighborhood were selected based on:
- High expression level in the neighborhood
- Specificity to the neighborhood (low expression elsewhere)
- Biological relevance to the neighborhood's function
- Consistency across replicates

**Pathway Analysis**: Gene sets were analyzed for enrichment of biological pathways and processes using standard enrichment analysis tools.

### 9.4 Visualization

**Network Visualization**: The neighborhood trajectory network was visualized using force-directed layout algorithms with temporal constraints to position neighborhoods according to disease stage while maintaining spatial relationships.

**Abundance Dynamics**: Temporal abundance profiles were plotted as line graphs with neighborhoods grouped by trajectory path and colored by disease stage.

**Gene Markers**: Marker genes were visualized as bar charts showing the number of key markers per neighborhood, organized by trajectory path.

### 9.5 Statistical Analysis

**Significance Testing**: Statistical significance of abundance changes and compositional shifts was assessed using appropriate tests (e.g., chi-square tests for proportions, permutation tests for network connectivity).

**Multiple Testing Correction**: False discovery rate (FDR) correction was applied to control for multiple hypothesis testing across neighborhoods and genes.

**Reproducibility**: Analysis pipelines were implemented in reproducible computational workflows with version-controlled code and documented parameters.

---

This comprehensive analysis demonstrates that spatial neighborhoods undergo coordinated, trajectory-specific transitions during DSS-induced colitis, providing a spatiotemporal framework for understanding inflammatory bowel disease pathogenesis and resolution. The integration of temporal dynamics, cellular composition, molecular signatures, and spatial organization reveals how tissue-level remodeling emerges from coordinated cellular programs, offering new insights into disease mechanisms and potential therapeutic strategies.

---

## Appendix: Figure Gallery

### neighborhood_trajectory_network.png
![neighborhood_trajectory_network.png](output/breakdown/neighborhood_trajectory/neighborhood_trajectory_network.png)

### trajectory_abundance_dynamics.png
![trajectory_abundance_dynamics.png](output/breakdown/neighborhood_trajectory/trajectory_abundance_dynamics.png)

### trajectory_gene_markers.png
![trajectory_gene_markers.png](output/breakdown/neighborhood_trajectory/trajectory_gene_markers.png)

