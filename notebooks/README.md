# Notebooks

Jupyter notebooks for SpatialAgent demos and case studies.

```
notebooks/
└── demos/
    ├── annotation.ipynb                            # Cell type annotation (single-cell platforms)
    ├── annotation_visium.ipynb                     # Platform-aware annotation on Visium (auto-routes to deconvolution)
    ├── panel_design.ipynb                          # Gene panel design workflow
    ├── case_study_dataset_harmonization.ipynb      # Harmonize cell-type annotations across scRNA-seq datasets
    ├── case_study_spatial_grn.ipynb                # Spatial gene regulatory network inference (Visium mouse brain)
    ├── case_tonsil_cosmx/                          # Case study: tonsil CosMx
    │   ├── final_notebooks/
    │   │   ├── a_neighborhood/agent-neighb.ipynb
    │   │   ├── b_trajectory/trajectory2.ipynb
    │   │   └── c_gene_imputation/gene-imputation3.ipynb
    │   ├── final_outputs_&_reports/                # Per-task outputs and markdown reports
    │   └── figure_pngs/                            # Rendered figures
    └── case_colitis_merfish/                       # Case study: colitis MERFISH
        ├── final_notebooks/
        │   ├── part1-3_cell-neighborhood_analysis.ipynb
        │   ├── part4_ccc.ipynb
        │   └── part5_ccc-neighb-traj.ipynb
        ├── final_selected_reports/                 # Curated markdown reports (data summary, trajectories, CCC)
        └── figure_pngs/                            # Rendered figures
```
