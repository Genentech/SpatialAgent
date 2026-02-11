# Cell Deconvolution with Tangram

Deconvolve spatial spots into individual cells using Tangram's constrained mapping mode with cell segmentation.

## When to Use

- Spatial technology has multiple cells per spot (Visium, Slide-seq)
- You have histology images with cell segmentation
- Goal: Assign cell types to individual segmented cells

---

## Requirements

Before running deconvolution, you need:

1. **scRNA-seq reference** with cell type annotations
2. **Spatial data** with:
   - Gene expression
   - Cell segmentation from histology (via Squidpy)

---

## Workflow Overview

1. Segment cells from histology image (Squidpy)
2. Calculate image features
3. Preprocess for Tangram
4. Run constrained mapping
5. Assign cell types to segments
6. Visualize deconvolved cells

---

## Step 1: Cell Segmentation (Squidpy)

If not already done, segment cells from histology:

```python
import squidpy as sq
import scanpy as sc

# Load spatial data with image
adata_sp = sc.read_h5ad("spatial.h5ad")
img = sq.datasets.visium_fluo_image_crop()  # or load your image

# Smooth image
sq.im.process(img=img, layer="image", method="smooth")

# Segment nuclei
sq.im.segment(
    img=img,
    layer="image_smooth",
    method="watershed",
    channel=0,  # DAPI channel
)

# Calculate features (cell counts per spot)
sq.im.calculate_image_features(
    adata_sp,
    img,
    layer="image",
    features="segmentation",
    features_kwargs={
        "segmentation": {
            "label_layer": "segmented_watershed",
            "props": ["label", "centroid"],
        }
    },
    mask_circle=True,
)

# Check cell counts
print(adata_sp.obs["cell_count"].describe())
adata_sp.write_h5ad("spatial_with_segmentation.h5ad")
```

---

## Step 2: Visualize Segmentation

Verify segmentation quality:

```python
import matplotlib.pyplot as plt

sc.pl.spatial(adata_sp, color="cell_count", title="Cells per Spot")
plt.savefig("cell_counts.png")

# Check total cells
total_cells = adata_sp.obsm["image_features"]["segmentation_label"].sum()
print(f"Total segmented cells: {total_cells}")
```

---

## Step 3: Preprocess for Tangram

**Tool**: `tangram_preprocess`

```
tangram_preprocess(
    adata_sc_path="scrna.h5ad",
    adata_sp_path="spatial_with_segmentation.h5ad",
    marker_genes="auto",
    cell_type_key="cell_type"
)
```

---

## Step 4: Constrained Mapping

Run Tangram in constrained mode to fit exactly the number of segmented cells:

```python
import scanpy as sc
import tangram as tg
import numpy as np

# Load preprocessed data
adata_sc = sc.read_h5ad("experiments/tangram_sc_prep.h5ad")
adata_sp = sc.read_h5ad("experiments/tangram_sp_prep.h5ad")

# Get cell counts from segmentation
cell_counts = adata_sp.obsm["image_features"]["segmentation_label"]
target_count = int(cell_counts.sum())
density_prior = np.array(cell_counts) / cell_counts.sum()

print(f"Target cell count: {target_count}")

# Constrained mapping
ad_map = tg.map_cells_to_space(
    adata_sc,
    adata_sp,
    mode="constrained",
    target_count=target_count,
    density_prior=density_prior,
    num_epochs=1000,
    device="cuda:0",  # GPU recommended
)

ad_map.write_h5ad("experiments/tangram_constrained_mapping.h5ad")
```

---

## Step 5: Assign Cell Types to Segments

```python
# Create segment dataframe
tg.create_segment_cell_df(adata_sp)

# Count cell types per spot
tg.count_cell_annotations(
    ad_map,
    adata_sc,
    adata_sp,
    annotation="cell_type",
)

# Deconvolve to individual cells
adata_segment = tg.deconvolve_cell_annotations(adata_sp)
adata_segment.write_h5ad("experiments/deconvolved_cells.h5ad")

print(f"Deconvolved {adata_segment.shape[0]} cells")
print(adata_segment.obs["cluster"].value_counts())
```

---

## Step 6: Visualize Deconvolved Cells

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 12))
sc.pl.spatial(
    adata_segment,
    color="cluster",
    size=0.4,
    show=False,
    frameon=False,
    alpha_img=0.2,
    ax=ax,
)
plt.title("Deconvolved Cell Types")
plt.savefig("deconvolved_spatial.png", dpi=200)
```

---

## Output Files

- `spatial_with_segmentation.h5ad` - Spatial data with cell counts
- `tangram_constrained_mapping.h5ad` - Constrained mapping result
- `deconvolved_cells.h5ad` - Individual cell annotations
- Visualization plots

---

## Interpreting Results

The deconvolved AnnData contains:
- `obs['cluster']`: Assigned cell type for each segmented cell
- `obs['x']`, `obs['y']`: Spatial coordinates of cell centroids
- `obsm['spatial']`: Spatial coordinates array

**Note**: Deconvolution assigns cell types probabilistically based on the mapping. Cells are assigned to the most likely type given the spot's expression profile.

---

## Tips

1. **Segmentation quality matters**: Poor segmentation = poor deconvolution
2. **Use GPU**: Constrained mode is computationally intensive
3. **More epochs**: Use 1000+ epochs for constrained mode
4. **Validate patterns**: Check that deconvolved cell types match expected tissue architecture
5. **Compare to spot-level**: Deconvolved proportions should roughly match spot-level predictions
