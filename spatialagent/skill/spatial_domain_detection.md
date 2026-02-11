# Spatial Domain Detection

Identify spatial domains (tissue regions, niches) in spatial transcriptomics data using deep learning methods.

## When to Use

| Method | Use Case | Speed | Key Feature |
|--------|----------|-------|-------------|
| **SpaGCN** | Visium/ST with H&E images | Fast | Integrates histology |
| **GraphST** | Any spatial data | Medium | Self-supervised learning |

---

## Prerequisites

**Required**:
- Spatial transcriptomics data (h5ad) with spatial coordinates in `obsm['spatial']`
- Estimated number of spatial domains (can refine)

**Optional (SpaGCN)**:
- H&E histology image (tif/png/jpg)
- Pixel coordinates in obs (x_pixel, y_pixel)

---

## Method Selection Guide

### Use SpaGCN when:
- You have **Visium data with H&E images**
- Want to leverage histological features
- Need fast results with hexagonal/square grid data
- Data has clear pixel coordinates

### Use GraphST when:
- You have **any spatial platform** (Visium, Slide-seq, MERFISH, Xenium)
- Don't have histology images
- Want self-supervised representation learning
- Need flexibility in clustering methods

---

## SpaGCN: Graph Convolutional Network

**Tool**: `spagcn_clustering`

SpaGCN combines gene expression with spatial location and histology features using a graph convolutional network.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_clusters` | Target number of spatial domains | 7 |
| `histology_path` | Path to H&E image (optional) | "" |
| `x_pixel_col`, `y_pixel_col` | Obs columns for pixel coordinates | "x_pixel", "y_pixel" |
| `p` | Neighborhood expression percentage (0-1) | 0.5 |
| `alpha` | Histology weight (higher = more weight) | 1 |
| `beta` | Spot area parameter for histology | 49 |
| `refine` | Refine clusters using spatial adjacency | True |
| `shape` | Spot shape: "hexagon" (Visium) or "square" (ST) | "hexagon" |

### How SpaGCN Works

1. **Adjacency Matrix**: Builds spatial graph from coordinates
2. **Histology Integration**: Extracts RGB values at spot locations (if image provided)
3. **Parameter Search**: Finds optimal `l` (spatial smoothing) and resolution
4. **Training**: Graph convolution aggregates expression from neighbors
5. **Refinement**: Post-processing refines cluster boundaries

### Example Usage

```
# Basic usage (no histology)
spagcn_clustering(
    adata_path="spatial_data.h5ad",
    n_clusters=7,
    save_path="./experiments/spagcn"
)

# With histology image
spagcn_clustering(
    adata_path="spatial_data.h5ad",
    n_clusters=7,
    histology_path="tissue_image.tif",
    x_pixel_col="x_pixel",
    y_pixel_col="y_pixel",
    alpha=1,  # Increase for more histology influence
    beta=49,  # Spot area in pixels
    shape="hexagon",  # For Visium
    save_path="./experiments/spagcn_histology"
)
```

### Outputs

| File | Description |
|------|-------------|
| `spagcn_domains.csv` | Domain assignments per spot |
| `spagcn_spatial.h5ad` | AnnData with domain labels in obs |
| `spagcn_domains_plot.png` | Spatial visualization |
| `spagcn_refined_plot.png` | Refined clusters (if refine=True) |

---

## GraphST: Graph Self-Supervised Learning

**Tool**: `graphst_clustering`

GraphST uses contrastive learning to learn spatial embeddings, then clusters them using various methods.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_clusters` | Target number of spatial domains | 7 |
| `cluster_method` | "leiden" | "leiden" |
| `n_pcs` | Number of principal components | 30 |
| `n_neighbors` | Neighbors for graph construction | 10 |
| `random_seed` | Random seed for reproducibility | 42 |
| `device` | "cuda" or "cpu" | "cuda" |

### Clustering Methods

- **leiden**: Community detection (recommended, n_clusters as guidance)

### How GraphST Works

1. **Graph Construction**: Builds spatial neighbors graph
2. **Contrastive Learning**: Learns embeddings that preserve spatial structure
3. **Clustering**: Applies chosen method to learned embeddings
4. **Domain Assignment**: Each spot gets a domain label

### Example Usage

```
# Basic usage with leiden (recommended)
graphst_clustering(
    adata_path="spatial_data.h5ad",
    n_clusters=7,
    cluster_method="leiden",
    device="cuda",
    save_path="./experiments/graphst"
)

```

### Outputs

| File | Description |
|------|-------------|
| `graphst_domains.csv` | Domain assignments per spot |
| `graphst_spatial.h5ad` | AnnData with domain labels and embeddings |
| `graphst_domains_plot.png` | Spatial visualization |
| `graphst_embedding.csv` | Learned embeddings (latent space) |

---

## Complete Workflow Examples

### Example 1: Visium with Histology (SpaGCN)

```
1. Load data with spatial and pixel coordinates
2. spagcn_clustering(
       adata_path="visium.h5ad",
       n_clusters=10,
       histology_path="tissue.tif",
       x_pixel_col="array_col",  # Check your data
       y_pixel_col="array_row",
       alpha=1.5,  # Weight histology more
       refine=True,
       shape="hexagon"
   )
3. Review spatial plot
4. Adjust n_clusters if needed
```

### Example 2: Slide-seq/MERFISH (GraphST)

```
1. Load spatial data
2. graphst_clustering(
       adata_path="slideseq.h5ad",
       n_clusters=8,
       cluster_method="leiden",  # Good for subcellular data
       n_neighbors=20,  # More neighbors for dense data
       device="cuda"
   )
3. Review clusters
4. Try different cluster_method if needed
```

### Example 3: Unknown Number of Domains

```
# Strategy: Start with multiple runs
1. Run with n_clusters=5, 7, 10, 15
2. Compare spatial plots
3. Look for:
   - Biologically meaningful regions
   - Spatial coherence
   - Not too fragmented
4. Select best n_clusters
```

---

## Downstream Analysis

After spatial domain detection:

1. **Characterize domains**:
   - Find marker genes per domain
   - Annotate domain identity

2. **Cell type enrichment**:
   - If cell type annotations exist
   - Check which cell types enrich in each domain

3. **Spatial statistics**:
   - Use Squidpy for co-occurrence
   - Analyze domain boundaries

### Example Post-processing

```python
import scanpy as sc
import pandas as pd

# Load result
adata = sc.read_h5ad("experiments/graphst_spatial.h5ad")

# Find marker genes per domain
sc.tl.rank_genes_groups(adata, groupby='spatial_domain', method='wilcoxon')

# Get top markers
markers = sc.get.rank_genes_groups_df(adata, group=None)
markers.to_csv("domain_markers.csv")

# Visualize top markers
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10,
                                 groupby='spatial_domain',
                                 save='domain_markers.png')
```

---

## Tips and Troubleshooting

### Parameter Tuning

1. **n_clusters**: Start with expected tissue complexity
   - Simple tissue: 5-7
   - Complex tissue: 10-15
   - Unknown: Try range and compare

2. **SpaGCN p parameter**: Controls spatial smoothing
   - Higher p (0.7-1.0): More spatially coherent
   - Lower p (0.3-0.5): More expression-driven

3. **GraphST n_neighbors**: Affects cluster smoothness
   - More neighbors: Smoother clusters
   - Fewer neighbors: More fine-grained

### Common Issues

| Problem | Solution |
|---------|----------|
| Fragmented clusters | Increase p (SpaGCN) or n_neighbors (GraphST) |
| Too few domains | Increase n_clusters |
| Clusters not matching tissue | Check coordinate system, try with histology |
| Out of memory (GPU) | Use device="cpu" for GraphST |
| SpaGCN slow | Reduce max_epochs, use without histology |

### Coordinate System

- **SpaGCN with histology**: Needs pixel coordinates matching image
- **GraphST**: Uses `obsm['spatial']` directly
- Check coordinate orientation matches your data

---

## Method Comparison

| Aspect | SpaGCN | GraphST |
|--------|--------|---------|
| Histology integration | Yes | No |
| Platform flexibility | Visium/ST | Any |
| GPU required | No | Recommended |
| Cluster refinement | Built-in | Manual |
| Learned embeddings | No | Yes |
| Speed | Fast | Medium |
| Best for | Visium + H&E | General spatial |
