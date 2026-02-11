# LIANA Cell-Cell Communication Analysis

Comprehensive ligand-receptor inference using LIANA's multi-method consensus approach, spatial analysis, and downstream integration.

## When to Use LIANA vs CellPhoneDB

| Use Case | Tool |
|----------|------|
| Quick single-method analysis | CellPhoneDB |
| Multi-method consensus ranking | **LIANA** |
| Spatial LR correlations | **LIANA bivariate** |
| Multi-sample tensor decomposition | **LIANA** |
| Learning spatial relationships | **LIANA MISTy** |

---

## Workflow Options

### Option A: Single-Sample Analysis
Basic ligand-receptor inference for one dataset.

### Option B: Multi-Sample Analysis
Compare interactions across conditions/samples with tensor decomposition.

### Option C: Spatial Analysis
Incorporate spatial context for spatially-resolved data.

---

## Option A: Single-Sample Workflow

### Step 1: Run LIANA Inference

**Tool**: `liana_inference`
- `cell_type_key`: Cell type annotation column
- `organism`: "human", "mouse", or "auto"
- `expr_prop`: Min expression proportion (default 0.1)

This combines 7 methods: CellPhoneDB, CellChat, NATMI, LogFC, Connectome, SingleCellSignalR, and geometric mean.

### Step 2: Visualize Results

**Tool**: `liana_plot`
- `plot_type`: "dotplot", "tileplot", or "source_target"
- `source_cells/target_cells`: Filter specific cell types
- `top_n`: Number of interactions to show

---

## Option B: Multi-Sample Workflow

### Step 1: Run Multi-Sample LIANA

**Tool**: `liana_inference`
- `sample_key`: Column with sample IDs
- Runs rank_aggregate per sample automatically

### Step 2: Tensor Decomposition (Full Pipeline)

**Tool**: `liana_tensor`
- `sample_key`: Sample column
- `condition_key`: Condition column (e.g., treatment vs control)
- `cell_type_key`: Cell type column
- Performs: LIANA → Tensor-Cell2Cell → Factorization

### Step 3: Interpret Factors

Each factor represents a communication pattern:
- Context loadings: Which samples/conditions use this pattern
- LR loadings: Which interactions contribute
- Cell type loadings: Which sender/receiver pairs involved

---

## Option C: Spatial Analysis Workflow

### Step 1: Spatial Bivariate Analysis

**Tool**: `liana_spatial`
- `bandwidth`: Spatial kernel width (adjust based on spot distance)
- `local_metric`: "cosine", "pearson", "spearman"
- `global_metric`: "morans" (spatial autocorrelation)

Outputs:
- Local scores per spot for each LR pair
- Global Moran's R for spatial patterns
- Interaction categories (high-high, high-low, low-low)

### Step 2: MISTy Spatial Relationships

**Tool**: `liana_misty`
- `target_key`: obsm key for features to predict (e.g., cell type compositions)
- `predictor_key`: obsm key for predictors (e.g., pathway scores)
- `bandwidth`: Spatial kernel for "para" view
- `n_neighs`: Direct neighbors for "juxta" view

MISTy models three spatial views:
- **Intra**: Within-spot relationships
- **Juxta**: Immediate neighbors
- **Para**: Surrounding microenvironment

Outputs:
- R² for each target
- Gain R²: How much spatial context improves prediction
- Feature importances per view

---

## Output Interpretation

### LIANA Results Columns

| Column | Description |
|--------|-------------|
| `ligand_complex` | Ligand gene(s) |
| `receptor_complex` | Receptor gene(s) |
| `source` | Sender cell type |
| `target` | Receiver cell type |
| `lr_means` | Mean expression of LR pair |
| `magnitude_rank` | Rank by expression magnitude (lower = stronger) |
| `specificity_rank` | Rank by cell type specificity |

### Bivariate Results

| Column | Description |
|--------|-------------|
| `mean` | Average local score across spots |
| `morans` | Moran's R spatial autocorrelation |
| Layer 'pvals' | Permutation p-values per spot |
| Layer 'cats' | Interaction categories (+1=high-high, -1=mixed, 0=low-low) |

---

## Tips

1. **Expression threshold**: Default `expr_prop=0.1` requires 10% of cells expressing each gene. Lower for rare interactions.

2. **Organism detection**: Auto-detects based on gene naming (ACTB=human, Actb=mouse).

3. **Bandwidth selection**: For spatial analysis, use `li.ut.query_bandwidth()` to find optimal value based on spot distances.

4. **Multi-method consensus**: LIANA's strength is combining multiple methods. The `magnitude_rank` and `specificity_rank` aggregate across all methods.

5. **Downstream analysis**: LIANA results can be connected to:
   - Pathway enrichment (via decoupler)
   - TF activity inference
   - Causal network reconstruction
