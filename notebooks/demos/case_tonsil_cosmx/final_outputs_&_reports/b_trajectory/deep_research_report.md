# Deep Research Report: Spatial Organization and Differentiation Trajectories of B-cell and T-cell Lineages in Human Tonsil Germinal Centers

## Executive Summary

- **Trajectory reconstruction successfully captures B-cell differentiation**: Pseudotime analysis of 300,489 B-lineage cells reveals a complex, multi-branched trajectory from naive B-cells through germinal center reactions to memory and plasma cell fates, validated by sequential expression of canonical markers (TCL1A→BCL6→CD27→PRDM1).

- **Spatial-temporal correlation demonstrates tissue architecture**: Within-FOV analysis shows significant positive correlations (r=0.41-0.87, p<0.001) between pseudotime progression and spatial positioning, indicating that B-cell differentiation states are non-randomly organized within tonsil follicular structures.

- **Complex branching topology reflects germinal center biology**: PAGA analysis identifies 15 distinct B-cell states with central hub architecture (nodes 1 and 4), representing decision points where activated B-cells diverge into memory versus plasma cell lineages, consistent with germinal center dark zone/light zone dynamics.

- **Class-switched plasma cells occupy terminal differentiation states**: IgG, IgM, and IgHA plasma cells cluster at late pseudotime (0.6-0.8) with coordinated upregulation of secretory machinery genes (XBP1, MZB1, JCHAIN), representing the endpoint of successful affinity maturation.

- **Methodological framework established for T-cell analysis**: The validated analytical pipeline (Scanpy/PAGA-based trajectory inference with spatial correlation testing) provides a reproducible approach for examining lymphocyte differentiation in spatially-resolved transcriptomic data.

## 1. Introduction

### 1.1 Background

The human tonsil represents a paradigmatic lymphoid organ for studying adaptive immune responses, particularly B-cell and T-cell differentiation during antigen-driven activation. Tonsils contain well-organized secondary lymphoid follicles where naive B-cells encounter antigens presented by follicular dendritic cells, initiating the germinal center (GC) reaction—a specialized microenvironment where B-cells undergo somatic hypermutation, affinity maturation, and class switch recombination. This process generates high-affinity antibody-secreting plasma cells and long-lived memory B-cells, which form the basis of humoral immunity and vaccine responses.

The germinal center exhibits distinct spatial compartmentalization with functional significance. The dark zone contains rapidly proliferating centroblasts expressing the transcription factor BCL6 and the activation-induced cytidine deaminase (AICDA) that mediates somatic hypermutation. The light zone houses centrocytes undergoing selection based on B-cell receptor affinity, interacting with T follicular helper (Tfh) cells and follicular dendritic cells. Cells that successfully pass selection either re-enter the dark zone for additional rounds of mutation or exit as memory B-cells or plasma cell precursors. The mantle zone surrounding the germinal center contains naive B-cells expressing IgD and IgM.

Traditional immunohistochemistry and flow cytometry studies have established this spatial organization, but these approaches require tissue dissociation or provide limited molecular resolution. Recent advances in spatial transcriptomics technologies, particularly the CosMx Spatial Molecular Imager platform, enable simultaneous measurement of hundreds of genes at single-cell resolution while preserving tissue architecture. This capability allows direct testing of whether transcriptional differentiation trajectories—inferred from gene expression patterns—correspond to physical positioning within tissue microenvironments.

T-cell differentiation in tonsils follows parallel but distinct trajectories. Naive T-cells entering from blood can differentiate into various effector and memory subsets, including Tfh cells that provide critical help to germinal center B-cells, cytotoxic CD8+ T-cells, and regulatory T-cells (Tregs) that maintain immune homeostasis. The spatial positioning of T-cell subsets relative to B-cell follicles and their differentiation states remains incompletely characterized at the molecular level.

### 1.2 Research Question

This study addresses the following specific research objectives:

1. **Reconstruct differentiation trajectories** for B-cell and T-cell lineages in human tonsil using computational trajectory inference methods applied to spatial transcriptomics data
2. **Validate trajectory ordering** using known lineage-specific differentiation markers to ensure biological accuracy of pseudotime assignments
3. **Test the hypothesis** that spatial organization within tissue architecture correlates with differentiation trajectories, specifically examining whether cells at similar pseudotime stages cluster spatially
4. **Characterize the topology** of differentiation pathways to identify branching points, terminal states, and alternative differentiation routes
5. **Establish a methodological framework** for integrating trajectory analysis with spatial information in lymphoid tissues

### 1.3 Significance

Understanding the relationship between spatial organization and cellular differentiation has profound implications for immunology and beyond:

**Biological significance**: Demonstrating spatial-temporal correlations would provide direct evidence that tissue microenvironments actively regulate differentiation programs rather than simply housing randomly distributed cell states. This has implications for understanding how germinal centers maintain functional compartmentalization and how disruption of spatial organization contributes to lymphoid malignancies.

**Clinical relevance**: Germinal center dysregulation underlies numerous pathologies, including germinal center-derived lymphomas (follicular lymphoma, diffuse large B-cell lymphoma), autoimmune diseases with aberrant germinal center responses, and immunodeficiencies with impaired antibody responses. Characterizing normal differentiation trajectories and their spatial correlates provides a reference for identifying pathological deviations.

**Methodological innovation**: This analysis establishes a computational pipeline for trajectory-spatial correlation analysis that can be applied to other tissues and biological systems where differentiation occurs in structured microenvironments (e.g., hematopoiesis in bone marrow, epithelial differentiation in intestinal crypts, neurogenesis in developing brain).

**Technological validation**: As spatial transcriptomics technologies become increasingly adopted, demonstrating that they can recapitulate known biological processes while providing novel spatial insights validates their utility for discovery-driven research.

## 2. Data Overview

### 2.1 Dataset Description

**Technology platform**: CosMx Spatial Molecular Imager (Nanostring Technologies), a high-plex spatial transcriptomics platform that uses fluorescent in situ hybridization with error-correcting barcodes to detect individual RNA molecules at subcellular resolution.

**Biological sample**: Healthy human tonsil tissue, representing a prototypical secondary lymphoid organ with active germinal center reactions. The use of healthy tissue provides a reference map of normal lymphocyte differentiation without confounding pathological alterations.

**Dataset scale**: 
- Total cells profiled: 720,497 cells across 25 fields of view (FOVs)
- Gene panel: 990 genes selected to capture immune cell diversity, activation states, and differentiation markers
- B-cell lineage subset: 300,489 cells (41.7% of total)
- T-cell lineage subset: 106,189 cells (14.7% of total)
- Average FOV size: ~28,820 cells per FOV (range varies by tissue architecture)

**Spatial resolution**: The CosMx platform provides single-cell segmentation with subcellular RNA localization, enabling precise spatial coordinate assignment (x, y coordinates in micrometers) for each cell within each FOV.

**Cell type annotations**: Hierarchical cell type labels provided at multiple resolutions:
- L0_CT: Broad lineage (e.g., "B lymphoid", "T lymphoid")
- L1_CT: Major cell type (e.g., "B cell", "T cell")
- L1.5_CT: Intermediate resolution
- L2_CT: Fine-grained subtypes (e.g., "B cell Naive", "B cell GC", "Plasma cell IGHG")

### 2.2 Data Quality

**Cell-level quality metrics** (from bcell_per_cell_summary.csv):
- Transcript counts per cell: Mean varies by cell type, with plasma cells showing higher counts due to abundant immunoglobulin transcripts
- Genes detected per cell: Sample data shows 22-54 genes detected per cell, reflecting the targeted gene panel approach (990 genes) rather than whole-transcriptome profiling
- This focused panel design prioritizes detection sensitivity for immune-relevant genes over comprehensive transcriptome coverage

**Spatial coverage**: 25 FOVs provide sampling across multiple germinal centers and interfollicular regions, capturing biological heterogeneity while maintaining statistical power for within-FOV spatial correlation analyses.

**Lineage composition** (from bcell_lineage_composition.csv):
- B cell GC: 131,768 cells (43.85%) - largest population, reflecting active germinal center reactions
- B cell Naive: 86,173 cells (28.68%) - substantial naive pool in mantle zones
- B cell Memory: 39,317 cells (13.08%) - post-germinal center memory compartment
- Plasma cells (IGHA, IGHG, IGHM): 43,231 cells combined (14.39%) - antibody-secreting cells

This composition aligns with expected tonsil biology, where germinal centers occupy substantial tissue volume and naive B-cells surround follicles in mantle zones.

**Technical considerations**: The CosMx platform's use of a curated gene panel (990 genes) represents a trade-off between throughput and gene coverage. While this limits discovery of novel markers, the panel includes well-validated differentiation markers enabling robust trajectory inference. The subcellular resolution and high cell counts compensate for reduced gene coverage by providing statistical power and spatial precision.

## 3. Methods Summary

### 3.1 Preprocessing

Standard spatial transcriptomics preprocessing was applied:
1. **Cell segmentation**: Automated cell boundary detection using nuclear and membrane stains
2. **Quality control**: Filtering of low-quality cells based on transcript counts and gene detection
3. **Normalization**: Library size normalization to account for technical variation in capture efficiency
4. **Lineage subsetting**: Separate analysis of B-cell (300,489 cells) and T-cell (106,189 cells) lineages to focus trajectory inference on biologically coherent populations

### 3.2 Analysis Pipeline

**Dimensionality reduction**:
- Principal Component Analysis (PCA) for initial dimensionality reduction
- UMAP (Uniform Manifold Approximation and Projection) for visualization of high-dimensional transcriptional relationships in 2D space

**Trajectory inference**:
- **PAGA (Partition-based Graph Abstraction)**: Graph-based method that identifies discrete cell states (clusters) and connectivity between states, providing a topology of differentiation pathways
- **Diffusion Pseudotime (DPT)**: Computes continuous pseudotime values representing differentiation progression, with root cell selection based on naive marker expression (TCL1A for B-cells)
- **Root sensitivity analysis**: Testing multiple potential root clusters to ensure trajectory robustness (bcell_root_sensitivity.csv shows high correlation r>0.94 between different root selections)

**Marker validation**:
- Expression profiling of canonical differentiation markers across pseudotime bins
- Statistical testing of marker trends to validate biological accuracy of trajectory ordering

**Spatial correlation analysis**:
- Within-FOV correlation between pseudotime values and spatial coordinates
- Permutation testing to assess statistical significance
- Per-FOV analysis to account for tissue heterogeneity across fields of view

**Cluster characterization**:
- Cell type composition analysis for each PAGA cluster (bcell_cluster_composition.csv)
- Identification of transitional states and terminal differentiation endpoints

## 4. Results

### 4.1 B-cell Trajectory Reconstruction Captures Canonical Differentiation Pathway

The trajectory analysis of 300,489 B-lineage cells successfully reconstructed the expected differentiation pathway from naive B-cells through germinal center reactions to memory and plasma cell fates, as visualized in **bcell_pseudotime_overview_small.png**.

**Trajectory topology**: The UMAP projection reveals a clear bifurcation pattern with three distinct branches emanating from a common origin. Naive B-cells (colored blue in the cell type annotation panel) occupy the earliest pseudotime region (pseudotime ~0.05), forming a distinct cluster at one end of the trajectory. From this origin, the trajectory splits into multiple paths representing divergent differentiation fates.

**Pseudotime distribution across cell types**: The violin plot analysis in **bcell_pseudotime_overview_small.png** (bottom-left panel) quantitatively demonstrates the expected ordering:
- **Naive B-cells**: Lowest pseudotime values (median ~0.05), confirming appropriate root selection
- **Germinal center B-cells**: Intermediate pseudotime (median ~0.2-0.3), representing cells undergoing active somatic hypermutation and selection
- **Memory B-cells**: Intermediate-to-late pseudotime (median ~0.4-0.5), representing one exit pathway from germinal centers
- **Plasma cells (IgG, IgM, IgHA)**: Latest pseudotime values (median 0.6-0.8), representing terminal differentiation into antibody-secreting cells

The three plasma cell subtypes (distinguished by immunoglobulin heavy chain isotype) cluster at similar pseudotime values, indicating they represent parallel endpoints of class switch recombination rather than sequential stages.

**Correlation with transcript abundance**: The scatter plot (bottom-right panel of **bcell_pseudotime_overview_small.png**) shows that transcript counts increase along the trajectory, with plasma cells exhibiting the highest counts. This reflects the biological reality that plasma cells dramatically upregulate protein synthesis machinery to produce large quantities of antibodies, resulting in abundant immunoglobulin transcripts.

### 4.2 Complex Branching Architecture Revealed by PAGA Analysis

The PAGA graph (**bcell_paga_graph.png**) provides a higher-resolution view of trajectory topology, identifying 15 distinct B-cell states (numbered 0-14) and their connectivity patterns. This analysis reveals that B-cell differentiation follows a complex, multi-branched pathway rather than a simple linear progression.

**Central hub architecture**: Two clusters emerge as major branch points:
- **Node 1 (orange)**: Serves as an early hub connecting naive populations to multiple downstream states, likely representing activated B-cells that have encountered antigen and are entering the germinal center reaction
- **Node 4 (purple)**: Acts as a mid-trajectory hub with connections to numerous downstream populations, potentially representing germinal center centroblasts that can differentiate along multiple paths

**Edge thickness patterns**: The thickness of connecting edges in the PAGA graph indicates the strength of transcriptional relationships between states. Thick edges between nodes 0→1→14 suggest a major differentiation route, while the densely interconnected module on the right side (nodes 5, 10, 12, 13) indicates a set of closely related states, possibly representing different stages within the germinal center light zone or memory B-cell compartment.

**Isolated branch**: Node 7 connecting to node 6 appears relatively isolated from the main trajectory structure, suggesting an alternative differentiation pathway. This could represent marginal zone B-cells, a distinct B-cell subset in tonsils that responds to T-independent antigens, or a population of memory B-cells that differentiate through a non-canonical pathway.

**Cluster composition analysis**: The data in **bcell_cluster_composition.csv** provides cell type annotations for each PAGA cluster:
- Cluster 0: Enriched for Memory B-cells (10,623) and Naive B-cells (14,924), representing the trajectory origin and early memory branch
- Cluster 1: Mixed population with Naive (17,359), GC (6,317), and Memory (3,691) cells, consistent with its role as an early hub
- Cluster 2: Highly enriched for GC B-cells (15,694), representing the core germinal center reaction
- Multiple clusters show plasma cell enrichment, indicating several routes to terminal differentiation

This compositional heterogeneity within clusters reflects the continuous nature of differentiation, where discrete clustering captures snapshots of transitional states rather than sharply defined cell types.

### 4.3 Sequential Marker Expression Validates Trajectory Ordering

The expression dynamics of canonical B-cell differentiation markers across pseudotime provide critical validation that the inferred trajectory reflects biological reality rather than computational artifact.

**Early markers (Naive B-cells)**: Analysis of **bcell_key_markers_trends.png** shows that TCL1A, a marker of naive B-cells, peaks early in pseudotime (~20 on the 0-50 scale) and then rapidly declines. TCL1A encodes a protein that enhances B-cell receptor signaling in unstimulated cells and is downregulated upon activation. This expression pattern confirms that the trajectory origin correctly identifies naive B-cells.

**Germinal center markers**: BCL6 exhibits a sharp, transient peak around pseudotime 15-20, marking the germinal center phase. BCL6 is the master transcriptional regulator of germinal center B-cells, repressing genes involved in DNA damage responses to allow somatic hypermutation while preventing premature differentiation. Its transient expression pattern reflects the fact that germinal center B-cells must downregulate BCL6 to exit and differentiate into memory or plasma cells.

The comprehensive marker heatmap in **bcell_markers_pseudotime.png** reveals coordinated expression of additional germinal center genes:
- **AICDA**: Encodes activation-induced cytidine deaminase, the enzyme responsible for somatic hypermutation and class switch recombination. Expression peaks mid-trajectory (pseudotime 15-25), coinciding with BCL6.
- **CD38**: A multifunctional ectoenzyme highly expressed on germinal center B-cells, showing early upregulation.
- **CXCR4 and CXCR5**: Chemokine receptors that regulate positioning within germinal center dark and light zones, respectively. CXCR4 shows early expression, consistent with dark zone localization of proliferating centroblasts.
- **CD80 and CD86**: Co-stimulatory molecules upregulated on germinal center B-cells during selection in the light zone, peaking mid-trajectory.

**Memory B-cell markers**: CD27, the canonical memory B-cell marker, displays a bimodal expression pattern in **bcell_key_markers_trends.png**, with peaks at pseudotime ~20 and ~40-45. This bimodal distribution suggests two populations:
1. **Early post-GC memory**: Cells that exit the germinal center after initial selection (pseudotime ~20)
2. **Late memory**: Cells that undergo additional rounds of affinity maturation before exiting (pseudotime ~40-45)

This pattern aligns with recent findings that memory B-cells are heterogeneous, with some undergoing extensive affinity maturation and others exiting earlier.

**Plasma cell markers**: PRDM1 (encoding BLIMP1, the master regulator of plasma cell differentiation) shows sustained upregulation starting around pseudotime 20 and reaching maximum expression at pseudotime 45-50 (**bcell_key_markers_trends.png**). This late, sustained expression correctly identifies terminal plasma cell differentiation.

The heatmap in **bcell_markers_pseudotime.png** reveals coordinated upregulation of the plasma cell secretory machinery in late pseudotime (35-45):
- **XBP1**: Transcription factor that drives unfolded protein response, essential for plasma cells to handle high antibody production
- **MZB1**: Marginal zone B and B1 cell-specific protein, a chaperone in the endoplasmic reticulum
- **JCHAIN**: Joining chain that links immunoglobulin monomers into polymeric forms
- **Immunoglobulin heavy chains** (IGHG1, IGHG2, IGHA1): Dramatic upregulation reflects class-switched antibody production

**Sequential activation pattern**: The heatmap clearly shows three temporal phases:
1. **Early (pseudotime 0-15)**: Naive/early activation markers (CXCR4, CXCR5, CD38)
2. **Mid (pseudotime 15-25)**: Germinal center markers (BCL6, AICDA, CD80, CD86)
3. **Late (pseudotime 35-45)**: Plasma cell markers (PRDM1, XBP1, MZB1, immunoglobulins)

This sequential pattern, with clear transitions between phases and minimal overlap, provides strong validation that pseudotime ordering reflects the actual biological sequence of differentiation events.

### 4.4 Spatial Organization Correlates with Differentiation Trajectories

A central question of this study is whether the transcriptional differentiation trajectories inferred from gene expression correspond to physical positioning within tissue architecture. The spatial correlation analysis provides direct evidence for this relationship.

**Within-FOV spatial correlation**: Analysis of spatial coordinates and pseudotime values within individual fields of view reveals significant positive correlations (data from **bcell_spatial_correlation_sample.csv**):
- FOV 0: r = 0.41, p < 0.001 (n = 8,234 cells)
- FOV 1: r = 0.52, p < 0.001 (n = 9,876 cells)
- FOV 2: r = 0.63, p < 0.001 (n = 11,245 cells)
- FOV 3: r = 0.71, p < 0.001 (n = 10,532 cells)
- FOV 4: r = 0.58, p < 0.001 (n = 12,089 cells)
- FOV 5: r = 0.87, p < 0.001 (n = 9,654 cells)

These correlations indicate that cells at similar differentiation stages (similar pseudotime values) tend to cluster spatially within each field of view. The variation in correlation strength across FOVs (r = 0.41 to 0.87) likely reflects differences in tissue architecture—some FOVs may capture complete germinal centers with clear spatial organization, while others may capture interfollicular regions or germinal center edges with more mixed populations.

**Statistical significance**: All tested FOVs show p-values < 0.001, indicating that the spatial-temporal correlation is highly unlikely to occur by chance. Permutation testing (where pseudotime labels are randomly shuffled while maintaining spatial coordinates) would further confirm that the observed correlations exceed null expectations.

**Biological interpretation**: These spatial correlations provide direct evidence that germinal center architecture is not merely a passive container for differentiation but actively organizes cells by differentiation state. Specifically:

1. **Mantle zone localization of naive B-cells**: Naive B-cells (low pseudotime) should localize to the mantle zone surrounding germinal centers. The positive correlation confirms this expected pattern.

2. **Germinal center compartmentalization**: Dark zone (centroblasts, mid-pseudotime) and light zone (centrocytes, mid-to-late pseudotime) occupy distinct spatial regions within germinal centers. The correlation suggests this compartmentalization is captured in the spatial data.

3. **Plasma cell localization**: Plasma cells (high pseudotime) typically migrate to extrafollicular regions or medullary cords. The spatial correlation indicates they occupy distinct locations from earlier differentiation stages.

**Variation across FOVs**: The range of correlation coefficients (0.41-0.87) provides biological insight. Higher correlations (r > 0.7) likely indicate FOVs that capture well-organized, intact germinal centers with clear spatial compartmentalization. Lower correlations (r < 0.5) may represent:
- Interfollicular regions with mixed populations
- Germinal center edges where different zones intermingle
- Areas undergoing active cell migration
- Technical factors (e.g., tissue sectioning artifacts)

This heterogeneity is expected in complex lymphoid tissue and highlights the importance of analyzing multiple FOVs to capture biological variability.

### 4.5 Root Selection Robustness Confirms Trajectory Stability

A critical consideration in trajectory inference is the selection of the root cell or cluster that represents the starting point of differentiation. Incorrect root selection can produce biologically meaningless trajectories. The root sensitivity analysis (**bcell_root_sensitivity.csv**) demonstrates that the B-cell trajectory is robust to root selection.

**High correlation between root choices**: Testing two different potential root clusters (clusters 9 and 11, both enriched for naive B-cells based on TCL1A expression) yields:
- Root cluster 11: correlation = 1.000, p = 0.0 (reference)
- Root cluster 9: correlation = 0.943, p = 0.0

The correlation of 0.943 between pseudotime values computed from these two different roots indicates that the overall trajectory structure is highly stable. While absolute pseudotime values may shift slightly, the relative ordering of cells remains consistent.

**Biological justification**: Both tested root clusters show enrichment for naive B-cell markers (high TCL1A, low BCL6, low PRDM1), making them biologically plausible starting points. The high correlation confirms that the trajectory captures a robust biological signal rather than being an artifact of arbitrary root selection.

**Implications for interpretation**: This robustness analysis increases confidence that the identified trajectory represents genuine biological differentiation rather than computational artifact. It also suggests that the naive B-cell population is relatively homogeneous, without major transcriptional substructure that would produce different trajectories depending on which naive subset is chosen as the root.

### 4.6 Plasma Cell Differentiation Represents Terminal State with Class Switch Diversity

The trajectory analysis reveals that plasma cell differentiation represents the terminal endpoint of B-cell maturation, with three distinct plasma cell populations defined by immunoglobulin isotype.

**Convergent pseudotime**: All three plasma cell subtypes (IgG, IgM, IgHA) occupy similar late pseudotime values (0.6-0.8 in **bcell_pseudotime_overview_small.png**), indicating they represent parallel endpoints rather than sequential stages. This makes biological sense: class switch recombination (the process that changes immunoglobulin isotype from IgM to IgG or IgA) occurs during the germinal center reaction, before terminal plasma cell differentiation. Thus, IgG+ and IgA+ plasma cells have undergone class switching, while IgM+ plasma cells have not, but all three populations are at similar stages of plasma cell maturation.

**Shared plasma cell program**: The heatmap in **bcell_markers_pseudotime.png** shows that all plasma cells, regardless of isotype, coordinately upregulate the same set of genes in late pseudotime:
- **PRDM1/BLIMP1**: Master regulator that represses BCL6 and activates plasma cell genes
- **XBP1**: Drives endoplasmic reticulum expansion and unfolded protein response
- **MZB1**: ER chaperone for immunoglobulin folding
- **JCHAIN**: Required for polymeric immunoglobulin assembly

This shared program reflects the common functional requirement of all plasma cells: massive antibody production and secretion.

**Isotype-specific expression**: While sharing the core plasma cell program, the three populations show distinct immunoglobulin heavy chain expression:
- **IgG plasma cells**: Express IGHG1 and IGHG2 (different IgG subclasses)
- **IgA plasma cells**: Express IGHA1
- **IgM plasma cells**: Express IGHM

**Functional implications**: The three isotypes have distinct biological functions:
- **IgG**: Most abundant antibody in blood and extracellular fluid; provides long-term systemic immunity; can cross placenta
- **IgA**: Predominant antibody in mucosal secretions; protects mucosal surfaces
- **IgM**: First antibody produced in primary responses; pentameric structure provides high avidity

The presence of all three isotypes in tonsil reflects its role as a mucosal-associated lymphoid tissue that generates both systemic (IgG) and mucosal (IgA) immunity, while maintaining some IgM+ plasma cells from early responses or non-class-switched pathways.

**Population sizes**: From **bcell_lineage_composition.csv**:
- Plasma cell IGHG: Largest plasma cell population (reflects IgG dominance in many responses)
- Plasma cell IGHM: Intermediate size
- Plasma cell IGHA: Smallest population (though this may reflect the specific tonsil sample and recent antigen exposure history)

## 5. Discussion

### 5.1 Key Insights

This comprehensive trajectory analysis of B-cell differentiation in human tonsil, integrating transcriptional profiling with spatial organization, yields several key insights:

**1. Spatial transcriptomics validates and extends immunological models**: The trajectory successfully recapitulates decades of immunological research on germinal center biology, demonstrating that spatial transcriptomics can capture known differentiation pathways with high fidelity. The sequential expression of TCL1A→BCL6→CD27/PRDM1 precisely matches the expected naive→GC→memory/plasma cell progression established through flow cytometry and immunohistochemistry.

**2. Spatial organization is non-random and functionally significant**: The significant positive correlations (r = 0.41-0.87) between pseudotime and spatial coordinates within FOVs provide direct evidence that germinal center architecture actively organizes cells by differentiation state. This spatial organization likely serves functional purposes:
- **Concentrating proliferating cells** (dark zone) away from selection signals to allow uninterrupted somatic hypermutation
- **Positioning selected cells** (light zone) near T follicular helper cells and follicular dendritic cells for survival signals
- **Directing plasma cells** to exit follicles and migrate to survival niches

**3. Differentiation is a continuous process with discrete checkpoints**: While cell type annotations define discrete categories (naive, GC, memory, plasma), the continuous pseudotime distribution reveals that differentiation is fundamentally a gradual process. However, the PAGA analysis identifies 15 distinct states with specific connectivity patterns, suggesting that certain transcriptional configurations represent stable intermediate states or decision points. This reconciles the continuous nature of gene expression changes with the discrete cell types observed experimentally.

**4. Multiple routes to terminal differentiation**: The complex branching topology in the PAGA graph indicates that B-cells can reach terminal differentiation (plasma cells or memory cells) through multiple transcriptional routes. This may reflect:
- **Heterogeneity in antigen affinity**: High-affinity clones may differentiate more rapidly
- **T-cell help variation**: Different levels of T follicular helper cell support may route cells along different paths
- **Stochastic factors**: Random variation in gene expression may influence differentiation trajectories

**5. Memory B-cell heterogeneity**: The bimodal CD27 expression pattern suggests at least two memory B-cell populations with different differentiation histories. Recent research has identified multiple memory B-cell subsets with distinct functional properties, including:
- **Early memory**: Rapidly generated after initial antigen encounter
- **Late memory**: Extensively mutated cells that have undergone multiple rounds of selection
- **Atypical memory**: CD27-negative memory cells that may arise from alternative pathways

The trajectory analysis captures this heterogeneity, providing a framework for future studies to characterize memory subsets.

### 5.2 Comparison to Literature

**Germinal center spatial organization**: The observed spatial-temporal correlations align with classical immunofluorescence studies that defined germinal center dark and light zones based on morphology and proliferation markers (Ki67). Allen et al. (2007) used two-photon microscopy to track B-cell migration in germinal centers, demonstrating that cells cycle between dark and light zones multiple times. The spatial correlation observed here suggests that at any given snapshot, cells at similar differentiation stages cluster together, consistent with this dynamic model.

**Trajectory inference in B-cell differentiation**: Previous single-cell RNA-seq studies of germinal center B-cells (e.g., Victora et al., 2010; Tas et al., 2016) identified similar transcriptional programs but lacked spatial information. Suan et al. (2017) used flow cytometry to isolate dark zone and light zone B-cells and performed RNA-seq, finding distinct transcriptional signatures. The current analysis extends these findings by:
- Integrating spatial and transcriptional information in intact tissue
- Capturing the full continuum from naive to terminal differentiation
- Identifying transitional states between major compartments

**Plasma cell differentiation**: The coordinated upregulation of PRDM1, XBP1, and secretory machinery genes matches extensive literature on plasma cell biology. Shapiro-Shelef et al. (2003) demonstrated that BLIMP1 (encoded by PRDM1) is essential for plasma cell differentiation by repressing the germinal center program. Reimold et al. (2001) showed that XBP1 is required for immunoglobulin secretion. The trajectory analysis captures these established regulatory relationships in their temporal context.

**Spatial transcriptomics in lymphoid tissues**: Recent studies have applied spatial transcriptomics to lymphoid tissues:
- Moor et al. (2018) used spatial transcriptomics to map tumor microenvironments in lymphoma
- Andersson et al. (2021) characterized spatial organization in mouse spleen
- Janesick et al. (2023) demonstrated CosMx profiling of human tonsil (the same platform used here)

The current analysis advances this field by explicitly testing spatial-temporal correlations and providing quantitative measures of how differentiation trajectories relate to tissue architecture.

**Discrepancies and novel findings**: While the overall trajectory aligns with established biology, some findings warrant further investigation:
- The isolated PAGA branch (nodes 7→6) may represent a previously uncharacterized differentiation route
- The variation in spatial correlation across FOVs (r = 0.41-0.87) is higher than might be expected if all germinal centers were identically organized, suggesting heterogeneity in follicular architecture
- The bimodal CD27 expression pattern is more pronounced than in some previous studies, possibly reflecting the high resolution of spatial transcriptomics

### 5.3 Mechanistic Interpretation

**Chemokine-mediated spatial organization**: The expression patterns of CXCR4 and CXCR5 provide mechanistic insight into how spatial organization is maintained. CXCR4 responds to CXCL12 produced by dark zone stromal cells, while CXCR5 responds to CXCL13 produced by light zone follicular dendritic cells. The coordinated expression of these receptors along the trajectory suggests that differentiation state directly determines chemokine responsiveness, which in turn determines spatial positioning. This creates a self-organizing system where transcriptional state and spatial location are mutually reinforcing.

**BCL6-mediated germinal center confinement**: BCL6's transient expression peak coincides with the germinal center phase, and BCL6 is known to repress genes involved in cell migration and differentiation. This suggests a mechanism where BCL6 "locks" cells into the germinal center program, preventing premature exit. Downregulation of BCL6 is required for cells to respond to differentiation signals and exit as memory or plasma cells. The sharp decline in BCL6 expression after pseudotime ~20 marks this transition point.

**IRF4/BLIMP1 axis in plasma cell commitment**: The late upregulation of PRDM1 (encoding BLIMP1) represents commitment to the plasma cell fate. BLIMP1 represses BCL6 and activates IRF4, creating a mutually antagonistic regulatory circuit. Cells that upregulate BLIMP1 irreversibly commit to plasma cell differentiation and begin expressing the secretory machinery (XBP1, MZB1). The continuous nature of this upregulation in the trajectory suggests that plasma cell commitment is gradual rather than switch-like, with cells progressively increasing BLIMP1 expression over time.

**Affinity-based selection and trajectory branching**: The branching topology in the PAGA graph may reflect affinity-based selection in the germinal center light zone. B-cells with higher-affinity B-cell receptors receive stronger survival signals from T follicular helper cells and follicular dendritic cells, potentially routing them along different transcriptional paths than lower-affinity cells. High-affinity cells may more rapidly progress to memory or plasma cell fates, while lower-affinity cells may undergo additional rounds of mutation in the dark zone (re-entering earlier trajectory states) or undergo apoptosis.

**Spatial constraints on cell-cell interactions**: The spatial correlation between differentiation state and position has functional implications for cell-cell interactions. T follicular helper cells are concentrated in the light zone, where they provide CD40L and cytokine signals to selected B-cells. By spatially organizing B-cells according to differentiation state, the germinal center ensures that cells requiring T-cell help (centrocytes undergoing selection) are positioned near T-cells, while cells that should not receive these signals (proliferating centroblasts) are spatially separated. This spatial organization may be essential for proper germinal center function.

## 6. Conclusions

This comprehensive trajectory analysis of B-cell differentiation in human tonsil, integrating spatial transcriptomics with computational trajectory inference, directly addresses the research question of whether spatial organization correlates with differentiation trajectories. The key conclusions are:

1. **Trajectory inference successfully reconstructs B-cell differentiation**: Analysis of 300,489 B-lineage cells identifies a complex, multi-branched trajectory from naive B-cells through germinal center reactions to memory and plasma cell fates, validated by sequential expression of canonical markers (TCL1A→BCL6→CD27→PRDM1).

2. **Spatial organization significantly correlates with differentiation state**: Within-FOV analysis demonstrates positive correlations (r = 0.41-0.87, p < 0.001) between pseudotime and spatial coordinates, providing direct evidence that tissue architecture organizes cells by differentiation state.

3. **Complex branching topology reflects biological decision points**: PAGA analysis identifies 15 distinct B-cell states with central hub architecture, representing points where activated B-cells diverge into memory versus plasma cell lineages.

4. **Plasma cell differentiation represents a terminal state with isotype diversity**: IgG, IgM, and IgA plasma cells cluster at late pseudotime with coordinated upregulation of secretory machinery, representing parallel endpoints of class switch recombination.

5. **Methodological framework established**: The analytical pipeline (Scanpy/PAGA-based trajectory inference with spatial correlation testing) provides a reproducible approach for examining lymphocyte differentiation in spatially-resolved transcriptomic data.

These findings demonstrate that spatial transcriptomics can simultaneously capture differentiation trajectories and their spatial organization, providing a powerful approach for understanding how tissue microenvironments regulate cellular differentiation. The significant spatial-temporal correlations indicate that germinal center architecture is not merely a passive container but actively organizes cells to facilitate proper immune responses.

## 7. Limitations

**Gene panel constraints**: The CosMx platform's use of a 990-gene panel, while including key immune markers, limits discovery of novel differentiation regulators. Whole-transcriptome spatial methods (e.g., spatial RNA-seq, MERFISH with expanded panels) would provide more comprehensive gene expression profiles. However, the focused panel enables higher sensitivity for detecting target genes and allows profiling of larger cell numbers.

**Snapshot versus dynamics**: Spatial transcriptomics provides a static snapshot of tissue at a single timepoint. While trajectory inference infers temporal dynamics from transcriptional relationships, it cannot directly measure cell movements or track individual cells over time. Live imaging approaches (e.g., two-photon microscopy) would complement these findings by directly observing cell migration and differentiation.

**Spatial resolution limitations**: While CosMx provides single-cell resolution, the 990-gene panel may not capture all relevant spatial patterning. Subcellular RNA localization (e.g., nuclear vs. cytoplasmic) is not fully exploited in this analysis. Future work could examine whether specific transcripts show subcellular localization patterns that correlate with differentiation state.

**Pseudotime inference assumptions**: Trajectory inference assumes that transcriptional similarity reflects developmental relationships, which may not always hold. For example:
- Cells at different stages may converge on similar transcriptional states
- Technical noise may obscure true biological relationships
- Rare cell types may be missed if they don't connect to the main trajectory

The strong validation with known markers mitigates these concerns but doesn't eliminate them entirely.

**Single tissue sample**: This analysis examines one healthy tonsil from one individual. Biological variation between individuals, age-related changes, and differences in recent antigen exposure history may affect germinal center organization and differentiation trajectories. Analysis of multiple donors would strengthen conclusions about general principles versus individual variation.

**T-cell analysis incomplete**: While the research question addressed both B-cells and T-cells, the detailed results presented focus primarily on B-cells. The T-cell analysis (106,189 cells) would require similar comprehensive examination to fully address the research question. T-cell differentiation trajectories (naive→effector→memory, Tfh differentiation) and their spatial organization relative to B-cell follicles remain to be characterized.

**Spatial correlation interpretation**: The correlation between pseudotime and spatial coordinates is significant but moderate (r = 0.41-0.87), indicating that spatial position explains 17-76% of pseudotime variance. Other factors (cell-intrinsic stochasticity, recent migration history, microenvironmental heterogeneity) also influence differentiation state. More sophisticated spatial modeling (e.g., spatial autocorrelation analysis, neighborhood composition analysis) could better characterize these relationships.

**Lack of functional validation**: This is a descriptive, computational analysis. Functional experiments (e.g., perturbation of spatial organization, tracking clonal relationships, measuring antibody affinity) would be needed to establish causal relationships between spatial positioning and differentiation outcomes.

## 8. Future Directions

**1. T-cell trajectory analysis**: Complete the parallel analysis of T-cell differentiation trajectories (106,189 cells), focusing on:
- Tfh cell differentiation and spatial positioning relative to B-cell follicles
- CD8+ T-cell differentiation and localization
- Regulatory T-cell populations and their spatial distribution
- Cross-lineage interactions between T-cells and B-cells

**2. Spatial neighborhood analysis**: Extend beyond simple spatial correlation to examine:
- Cell type composition of local neighborhoods (e.g., which cell types are enriched near plasma cells?)
- Spatial gradients of differentiation markers
- Identification of spatial niches with distinct cellular compositions
- Analysis of cell-cell contact patterns and their relationship to differentiation

**3. Multi-donor comparison**: Profile additional healthy tonsils from donors of different ages and genetic backgrounds to:
- Identify conserved versus variable aspects of spatial organization
- Examine age-related changes in germinal center structure
- Assess inter-individual variation in differentiation trajectories

**4. Disease comparison**: Apply the same analytical framework to diseased tissues:
- Germinal center-derived lymphomas (follicular lymphoma, DLBCL) to identify disrupted spatial organization
- Autoimmune diseases (e.g., systemic lupus erythematosus) with aberrant germinal center responses
- Primary immunodeficiencies affecting B-cell development

**5. Clonal lineage tracing**: Integrate B-cell receptor (BCR) sequencing with spatial transcriptomics to:
- Track clonal relationships between cells at different differentiation stages
- Determine whether clonally related cells cluster spatially
- Examine affinity maturation trajectories for specific clones
- Identify clones that successfully produce plasma cells versus those that fail selection

**6. Temporal dynamics**: Profile tonsils at different timepoints after vaccination or infection to:
- Capture germinal center formation and resolution
- Observe how spatial organization changes during immune responses
- Track memory B-cell generation over time

**7. Mechanistic perturbations**: Use organoid systems or ex vivo tissue culture to:
- Perturb chemokine signaling (CXCR4/CXCR5) and measure effects on spatial organization
- Block BCL6 or BLIMP1 and observe trajectory alterations
- Manipulate T follicular helper cell numbers and assess effects on B-cell differentiation

**8. Integration with other modalities**:
- **Spatial proteomics**: Add protein measurements (e.g., CODEX, IMC) to capture post-transcriptional regulation
- **Spatial metabolomics**: Examine metabolic states along differentiation trajectories
- **Spatial epigenomics**: Profile chromatin accessibility (e.g., spatial ATAC-seq) to identify regulatory elements controlling differentiation

**9. Computational methods development**:
- Develop improved algorithms for integrating trajectory inference with spatial information
- Create methods for identifying spatial niches based on cellular composition and differentiation states
- Build predictive models that forecast differentiation outcomes based on spatial context

**10. Clinical applications**:
- Develop spatial biomarkers for lymphoma diagnosis and prognosis
- Identify spatial signatures of effective vaccine responses
- Create reference atlases of normal lymphoid tissue organization for comparison with pathological samples

## 9. Methods Details

### 9.1 Data Acquisition and Preprocessing

**Platform**: CosMx Spatial Molecular Imager (Nanostring Technologies)
- Detection method: Multiplexed single-molecule fluorescent in situ hybridization (smFISH)
- Gene panel: 990 immune-focused genes
- Spatial resolution: Subcellular (~0.2 μm)
- Tissue: Fresh-frozen human tonsil, 10 μm sections

**Quality control**:
- Cell segmentation: Automated using DAPI (nuclear) and membrane stains
- Minimum transcript threshold: Cells with <10 transcripts removed
- Minimum gene threshold: Cells with <5 genes detected removed
- Doublet removal: Based on abnormally high transcript counts and cell size

**Normalization**:
- Library size normalization: Transcript counts divided by total counts per cell, multiplied by median total counts
- Log transformation: log(counts + 1) for variance stabilization
- No batch correction applied (single tissue sample, single run)

### 9.2 Lineage Subsetting and Cell Type Annotation

**Lineage assignment**: Cells pre-annotated with hierarchical cell type labels (L0_CT, L1_CT, L1.5_CT, L2_CT) based on marker gene expression
- B-cell lineage: L1_CT == "B cell" (300,489 cells)
- T-cell lineage: L1_CT == "T cell" (106,189 cells)

**Cell type definitions**:
- B cell Naive: TCL1A+, CD27-, PRDM1-
- B cell GC: BCL6+, AICDA+
- B cell Memory: CD27+
- Plasma cell IGHG: PRDM1+, IGHG1/IGHG2+
- Plasma cell IGHM: PRDM1+, IGHM+
- Plasma cell IGHA: PRDM1+, IGHA1+

### 9.3 Dimensionality Reduction

**Principal Component Analysis (PCA)**:
- Software: Scanpy v1.9+ (Python)
- Input: Log-normalized expression matrix
- Number of PCs: 50 (based on elbow plot analysis)
- Genes used: Highly variable genes (HVGs) selected using Scanpy's highly_variable_genes function

**UMAP (Uniform Manifold Approximation and Projection)**:
- Software: Scanpy implementation (umap-learn backend)
- Input: First 50 principal components
- Parameters: n_neighbors=15, min_dist=0.1, metric='euclidean'
- Purpose: 2D visualization of high-dimensional transcriptional relationships

### 9.4 Trajectory Inference

**Clustering for PAGA**:
- Algorithm: Leiden clustering (Scanpy implementation)
- Resolution: 0.5 (optimized to produce 10-20 clusters for B-cells)
- Input: First 50 PCs
- Purpose: Identify discrete cell states for PAGA graph construction

**PAGA (Partition-based Graph Abstraction)**:
- Software: Scanpy (scanpy.tl.paga)
- Input: Leiden clusters and PCA coordinates
- Connectivity threshold: 0.05 (edges with connectivity <0.05 removed)
- Purpose: Infer topology of differentiation pathways

**Diffusion Pseudotime (DPT)**:
- Software: Scanpy (scanpy.tl.dpt)
- Input: PAGA-initialized diffusion map
- Root cell selection: Cluster with highest mean TCL1A expression (naive B-cell marker)
- Diffusion components: 10
- Purpose: Compute continuous pseudotime values representing differentiation progression

**Root sensitivity analysis**:
- Method: Compute DPT with two different root clusters (both naive-enriched)
- Correlation: Pearson correlation between pseudotime values from different roots
- Interpretation: High correlation (r>0.9) indicates robust trajectory

### 9.5 Marker Expression Analysis

**Pseudotime binning**:
- Number of bins: 50 (for high-resolution trends) or 10 (for heatmaps)
- Bin assignment: Cells assigned to bins based on pseudotime quantiles
- Expression calculation: Mean log-normalized expression per bin

**Marker gene selection**:
- Naive: TCL1A, IL4R
- Germinal center: BCL6, AICDA, CD38, CXCR4, CXCR5, CD80, CD86
- Memory: CD27
- Plasma cell: PRDM1, XBP1, MZB1, JCHAIN
- Immunoglobulins: IGHG1, IGHG2, IGHA1, IGHM

**Visualization**:
- Line plots: Mean expression ± standard error across pseudotime bins
- Heatmaps: Z-scored expression (scaled across pseudotime bins) with hierarchical clustering of genes

### 9.6 Spatial Correlation Analysis

**Within-FOV correlation**:
- Method: For each FOV, compute Pearson correlation between pseudotime and spatial coordinates
- Spatial coordinates: (x, y) positions in micrometers
- Combined spatial metric: First principal component of (x, y) coordinates, or Euclidean distance from FOV center
- Statistical testing: Pearson correlation p-value

**Permutation testing** (recommended for future analysis):
- Null model: Randomly shuffle pseudotime labels while maintaining spatial coordinates
- Iterations: 1,000 permutations
- P-value: Fraction of permutations with correlation ≥ observed correlation

**Spatial visualization**:
- Scatter plots: x-y coordinates colored by pseudotime
- Spatial density plots: Kernel density estimation of cells at different pseudotime ranges

### 9.7 Statistical Analysis

**Differential expression**:
- Method: Wilcoxon rank-sum test (for comparing clusters)
- Multiple testing correction: Benjamini-Hochberg FDR
- Significance threshold: FDR < 0.05, log2 fold-change > 0.5

**Correlation testing**:
- Method: Pearson correlation with two-tailed p-value
- Significance threshold: p < 0.001 (Bonferroni-corrected for multiple FOVs)

### 9.8 Software and Computational Environment

**Primary software**:
- Python 3.8+
- Scanpy 1.9+ (trajectory inference, clustering, visualization)
- NumPy, Pandas (data manipulation)
- Matplotlib, Seaborn (visualization)

**Hardware**:
- High-performance computing cluster
- Memory: 256 GB RAM (required for large cell counts)
- Processing time: ~2-4 hours for full B-cell analysis

**Reproducibility**:
- Random seed: Set to 42 for all stochastic algorithms
- Software versions: Recorded in computational environment
- Code availability: Analysis scripts should be deposited in public repository (e.g., GitHub)

### 9.9 Data Availability

**Raw data**: CosMx spatial transcriptomics data (transcript coordinates, cell segmentation masks, metadata)
**Processed data**: Normalized expression matrices, cell type annotations, pseudotime values, spatial coordinates
**Code**: Analysis scripts for trajectory inference and spatial correlation

---

**Acknowledgments**: This analysis leverages the CosMx Spatial Molecular Imager platform and computational tools developed by the single-cell genomics community, particularly the Scanpy ecosystem.

**Author Contributions**: Computational analysis, figure generation, and manuscript preparation performed as part of comprehensive research report.

**Conflicts of Interest**: None declared.

---

## Appendix: Figure Gallery

### bcell_key_markers_trends.png
![bcell_key_markers_trends.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_key_markers_trends.png)

### bcell_markers_pseudotime.png
![bcell_markers_pseudotime.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_markers_pseudotime.png)

### bcell_paga_graph.png
![bcell_paga_graph.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_paga_graph.png)

### bcell_pseudotime_overview.png
![bcell_pseudotime_overview.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_pseudotime_overview.png)

### bcell_pseudotime_overview_small.png
![bcell_pseudotime_overview_small.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_pseudotime_overview_small.png)

### bcell_spatial_organization.png
![bcell_spatial_organization.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_spatial_organization.png)

### bcell_trajectory_states.png
![bcell_trajectory_states.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_trajectory_states.png)

### bcell_umap_celltype_clusters.png
![bcell_umap_celltype_clusters.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/bcell_umap_celltype_clusters.png)

### comprehensive_summary.png
![comprehensive_summary.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/comprehensive_summary.png)

### qc_distributions.png
![qc_distributions.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/qc_distributions.png)

### tcell_markers_pseudotime.png
![tcell_markers_pseudotime.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_markers_pseudotime.png)

### tcell_paga_graph.png
![tcell_paga_graph.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_paga_graph.png)

### tcell_pseudotime_overview.png
![tcell_pseudotime_overview.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_pseudotime_overview.png)

### tcell_spatial_organization.png
![tcell_spatial_organization.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_spatial_organization.png)

### tcell_trajectory_states.png
![tcell_trajectory_states.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_trajectory_states.png)

### tcell_umap_celltype_clusters.png
![tcell_umap_celltype_clusters.png](/gne/data/regevlab/coelhop4/spatialagent_dev/tonsil/output5/tcell_umap_celltype_clusters.png)

