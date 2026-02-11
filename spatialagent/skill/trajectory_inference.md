# Trajectory Inference

Analyze cellular differentiation trajectories and developmental dynamics using RNA velocity and graph-based methods.

## Tools

### scVelo (RNA Velocity)
- `scvelo_velocity`: Compute RNA velocity from spliced/unspliced counts
- `scvelo_velocity_embedding`: Visualize velocity on UMAP/tSNE

### CellRank (Fate Mapping)
- `cellrank_terminal_states`: Identify terminal/initial differentiation states
- `cellrank_fate_probabilities`: Compute fate probabilities towards terminal states

### PAGA (Graph Abstraction)
- `paga_trajectory`: PAGA graph abstraction with diffusion pseudotime

## Workflow

### 1. RNA Velocity Analysis (requires spliced/unspliced counts)

```
# Step 1: Compute velocity
scvelo_velocity(
    adata_path="spatial_data.h5ad",
    mode="stochastic",  # Options: deterministic, stochastic, dynamical
    n_top_genes=2000,
    save_path="./velocity"
)

# Step 2: Visualize velocity
scvelo_velocity_embedding(
    adata_path="./velocity/velocity_result.h5ad",
    basis="umap",
    color_by="cell_type",
    save_path="./velocity_viz"
)
```

### 2. CellRank Fate Analysis

```
# Step 1: Identify terminal states (requires velocity)
cellrank_terminal_states(
    adata_path="./velocity/velocity_result.h5ad",
    cluster_key="cell_type",
    use_velocity=True,
    save_path="./cellrank"
)

# Step 2: Compute fate probabilities
cellrank_fate_probabilities(
    adata_path="./cellrank/cellrank_terminal.h5ad",
    save_path="./cellrank_fate"
)
```

### 3. PAGA Trajectory (no velocity needed)

```
paga_trajectory(
    adata_path="spatial_data.h5ad",
    groups_key="leiden",  # or "cell_type"
    threshold=0.1,
    root_group="Progenitor",  # Optional: for pseudotime
    save_path="./paga"
)
```

## Data Requirements

### For RNA Velocity (scVelo, CellRank with velocity)
- Must have `spliced` and `unspliced` layers in adata
- Typically from velocyto, kallisto bustools, or STARsolo

### For PAGA / CellRank without velocity
- Standard preprocessed adata with neighbors graph
- Cell type or cluster annotations

## Mode Selection for scVelo

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| deterministic | Fast | Lower | Quick exploration |
| stochastic | Medium | Good | **Recommended default** |
| dynamical | Slow | Highest | Publication-quality analysis |

## Outputs

### scvelo_velocity
- `velocity_result.h5ad`: Data with velocity in layers

### scvelo_velocity_embedding
- `velocity_stream_umap.png`: Stream plot
- `velocity_grid_umap.png`: Grid plot

### cellrank_terminal_states
- `cellrank_terminal.h5ad`: Data with terminal state annotations

### cellrank_fate_probabilities
- `fate_probabilities.csv`: Per-cell fate probabilities
- `cellrank_fate.h5ad`: Data with fate probabilities

### paga_trajectory
- `paga_graph.png`: PAGA connectivity graph
- `paga_adjacency.csv`: Cluster connectivity matrix
- `pseudotime_umap.png`: Pseudotime visualization (if root specified)
- `paga_result.h5ad`: Data with PAGA and pseudotime

## References

- **scVelo**: Bergen et al. (2020) Nature Biotechnology
- **CellRank**: Lange et al. (2022) Nature Methods
- **PAGA**: Wolf et al. (2019) Genome Biology
