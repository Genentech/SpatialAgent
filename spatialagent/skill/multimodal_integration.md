# Multimodal Integration

Integrate multiple data modalities (RNA, protein, ATAC) and perform batch correction.

## Tools

### Multimodal Integration
- `totalvi_integration`: Joint RNA + protein analysis (CITE-seq)
- `multivi_integration`: Joint RNA + ATAC analysis (multiome)
- `mofa_integration`: Multi-omics factor analysis (any combination)

### Batch Correction & Label Transfer
- `scanpy_bbknn`: Batch-balanced KNN integration
- `scanpy_ingest`: Transfer labels from reference to query
- `scanpy_score_genes`: Score cells for gene signatures

## Workflow

### 1. CITE-seq Analysis (RNA + Protein)

```
# totalVI for joint RNA + protein modeling
totalvi_integration(
    adata_path="cite_seq_data.h5ad",
    protein_layer="protein_expression",  # Key in adata.obsm
    n_latent=20,
    max_epochs=400,
    batch_key="batch",  # Optional batch correction
    save_path="./totalvi"
)
```

**Requirements:**
- RNA counts in `adata.X` (raw counts)
- Protein counts in `adata.obsm['protein_expression']`

### 2. Multiome Analysis (RNA + ATAC)

```
# MultiVI for joint RNA + ATAC modeling
multivi_integration(
    adata_path="multiome_data.h5ad",
    n_latent=20,
    max_epochs=500,
    batch_key="batch",
    save_path="./multivi"
)
```

### 3. Multi-Omics Factor Analysis (MOFA+)

```
# MOFA+ with multiple modality files
mofa_integration(
    adata_path="rna.h5ad,protein.h5ad,atac.h5ad",  # Comma-separated
    n_factors=10,
    max_iterations=1000,
    save_path="./mofa"
)

# Or single file with modality annotation
mofa_integration(
    adata_path="multimodal.h5ad",
    modality_key="modality",  # Column in adata.var
    n_factors=10,
    save_path="./mofa"
)
```

### 4. Batch Correction with BBKNN

```
scanpy_bbknn(
    adata_path="multi_batch_data.h5ad",
    batch_key="batch",
    n_pcs=30,
    neighbors_within_batch=3,
    run_umap=True,
    save_path="./bbknn"
)
```

### 5. Label Transfer with Ingest

```
scanpy_ingest(
    adata_query_path="query_data.h5ad",
    adata_ref_path="reference_with_annotations.h5ad",
    obs_keys=["cell_type", "cluster"],  # Annotations to transfer
    embedding="umap",
    save_path="./ingest"
)
```

**Requirements:**
- Reference must have PCA and UMAP computed
- Reference must have the annotation columns to transfer

### 6. Gene Signature Scoring

```
scanpy_score_genes(
    adata_path="spatial_data.h5ad",
    gene_list=["CD3D", "CD3E", "CD4", "CD8A"],  # T cell markers
    score_name="T_cell_score",
    save_path="./scores"
)
```

## Method Comparison

| Method | Modalities | Use Case | GPU |
|--------|------------|----------|-----|
| totalVI | RNA + Protein | CITE-seq | Yes |
| MultiVI | RNA + ATAC | 10x Multiome | Yes |
| MOFA+ | Any combination | Exploratory | No |
| BBKNN | Single modality | Batch correction | No |
| Ingest | Single modality | Label transfer | No |

## Outputs

### totalvi_integration
- `totalvi_result.h5ad`: Data with latent space, normalized values
- `totalvi_model/`: Saved model directory

### multivi_integration
- `multivi_result.h5ad`: Data with latent space
- `multivi_model/`: Saved model directory

### mofa_integration
- `mofa_model.hdf5`: MOFA model file
- `mofa_factors.csv`: Latent factors per sample
- `mofa_variance_explained.csv`: Variance per factor per modality

### scanpy_bbknn
- `bbknn_result.h5ad`: Batch-corrected data with UMAP

### scanpy_ingest
- `ingest_result.h5ad`: Query data with transferred annotations
- `ingest_annotations.csv`: Transferred labels

### scanpy_score_genes
- `scored.h5ad`: Data with gene scores in obs
- `{score_name}.csv`: Per-cell scores

## References

- **totalVI**: Gayoso et al. (2021) Nature Methods
- **MultiVI**: Ashuach et al. (2023) Nature Methods
- **MOFA+**: Argelaguet et al. (2020) Genome Biology
- **BBKNN**: Polanski et al. (2020) Bioinformatics
