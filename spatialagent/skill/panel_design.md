# Panel Design

Design a gene panel for spatial transcriptomics by combining literature research, database queries, and expression validation.

## Goal

Generate a gene panel of **exactly N genes** (as specified) for cell type identification. The panel must include markers covering all major cell types in the target tissue with **balanced representation**.

---

## Required Tools

| Tool | Purpose | Required Parameters |
|------|---------|---------------------|
| `query_pubmed` | Literature search | `query`, `max_papers` |
| `query_celltype_genesets` | Curated gene sets | `tissue`, `top_k` |
| `search_czi_datasets` | Find reference datasets | `query`, `n_datasets` |
| `extract_czi_markers` | Extract markers from CZI | `save_path`, `dataset_id`, `iter_round`, **`organism`** |
| `search_panglao` | PanglaoDB markers | `cell_types`, `organism`, `tissue` |
| `search_cellmarker2` | CellMarker 2.0 markers | `cell_types`, `organism`, `tissue` |
| `validate_genes_expression` | Validate tissue expression | `genes`, `target_tissue` |
| `query_tissue_expression` | Check individual gene expression | `gene`, `top_k` |

---

## ⚠️ CRITICAL: Organism Parameter

The `extract_czi_markers` tool **REQUIRES** the `organism` parameter. Without it, the tool returns **0 cell types** and causes cascading failures.

### CORRECT Usage:
```python
extract_czi_markers({
    "save_path": save_path,
    "dataset_id": "xxx-xxx-xxx",
    "iter_round": 1,
    "organism": "Homo sapiens"  # REQUIRED!
})
```

### WRONG Usage (will fail):
```python
extract_czi_markers({
    "save_path": save_path,
    "dataset_id": "xxx-xxx-xxx",
    "iter_round": 1
    # Missing organism - returns 0 cell types!
})
```

### Organism Values by Tool:

| Tool | Human | Mouse |
|------|-------|-------|
| `extract_czi_markers` | `"Homo sapiens"` | `"Mus musculus"` |
| `search_panglao` | `"Hs"` | `"Mm"` |
| `search_cellmarker2` | `"Human"` | `"Mouse"` |

---

## Key Strategy: Balanced Cell Type Representation

**Top-performing panels** organize genes by cell type category with explicit quotas. The categories depend on the tissue:

### Example: Brain (100 genes)
| Category | Genes | Examples |
|----------|-------|----------|
| Excitatory neurons | 20 | SLC17A7, CAMK2A, SATB2 |
| Inhibitory neurons | 15 | GAD1, GAD2, PVALB, SST, VIP |
| Astrocytes | 10 | GFAP, AQP4, ALDH1L1 |
| Oligodendrocytes | 10 | MBP, PLP1, MOG |
| Microglia | 8 | P2RY12, CX3CR1, AIF1 |
| ... | ... | ... |

### Example: Heart (100 genes)
| Category | Genes | Examples |
|----------|-------|----------|
| Cardiomyocytes | 25 | TNNT2, MYH7, ACTC1 |
| Fibroblasts | 15 | COL1A1, DCN, LUM |
| Endothelial | 15 | PECAM1, VWF, CDH5 |
| Smooth muscle | 10 | ACTA2, MYH11, TAGLN |
| Immune cells | 10 | PTPRC, CD68, CD3E |
| ... | ... | ... |

### Example: Liver (100 genes)
| Category | Genes | Examples |
|----------|-------|----------|
| Hepatocytes | 30 | ALB, CYP3A4, HNF4A |
| Cholangiocytes | 10 | KRT19, EPCAM, SOX9 |
| Endothelial | 15 | PECAM1, LYVE1, CLEC4G |
| Stellate cells | 10 | ACTA2, COL1A1, PDGFRB |
| Kupffer cells | 10 | CD68, CLEC4F, MARCO |
| ... | ... | ... |

**Principle**: Allocate more genes to abundant/important cell types, fewer to rare types.

---

## Comprehensive Workflow (7 Steps)

### Step 1: Literature Search + Identify Canonical Markers

Search PubMed to understand the tissue's cell types and known markers.

```python
result = query_pubmed({
    "query": "{tissue} marker genes single cell RNA-seq",
    "max_papers": 5
})

# Parse genes from abstracts
pubmed_genes = [...]  # Extract gene names

# IMPORTANT: Based on literature and your knowledge, define canonical markers
# for the SPECIFIC tissue being studied
canonical_markers = {
    # Define markers for each major cell type in THIS tissue
    # Example structure:
    # "CELL_TYPE_1_MARKER_1", "CELL_TYPE_1_MARKER_2", ...
    # "CELL_TYPE_2_MARKER_1", "CELL_TYPE_2_MARKER_2", ...
}

all_genes = list(set(pubmed_genes) | canonical_markers)
pd.DataFrame({"Gene": all_genes, "Source": "PubMed+Canonical"}).to_csv(f"{save_path}/pubmed_genes.csv", index=False)
```

**Tip**: Use your knowledge to include well-established markers for the tissue's major cell types. This ensures coverage even if database queries return incomplete results.

---

### Step 2: Cell Type Gene Sets

Get curated gene sets from Enrichr/PanglaoDB.

```python
result = query_celltype_genesets({"tissue": "{tissue}", "top_k": 20})

# Extract genes
geneset_genes = [...]  # Parse from results
pd.DataFrame({"Gene": geneset_genes, "Source": "Enrichr"}).to_csv(f"{save_path}/geneset_genes.csv", index=False)
```

---

### Step 3: CZI Reference Dataset

Search for a reference dataset and extract markers.

```python
# Search with multiple queries for robustness
for query in [
    "{species} {tissue} normal",
    "{species} {broader_tissue} normal",
    "{species} {organ} normal"
]:
    result = search_czi_datasets({"query": query, "n_datasets": 3})
    if result:
        break

# Extract markers - INCLUDE ORGANISM!
extract_czi_markers({
    "save_path": save_path,
    "dataset_id": "{dataset_id}",
    "iter_round": 1,
    "organism": "Homo sapiens"  # ← REQUIRED (or "Mus musculus" for mouse)
})

# Verify extraction succeeded
czi_df = pd.read_csv(f"{save_path}/czi_reference_celltype_1.csv")
print(f"Cell types found: {len(czi_df)}")

if len(czi_df) == 0:
    print("⚠️ CZI extraction returned 0 cell types!")
    print("Check: Is 'organism' parameter included?")
```

---

### Step 4: Database Marker Lookup

Search PanglaoDB and CellMarker2 for additional markers.

```python
# Use CZI cell types if available, else use tissue-appropriate defaults
if os.path.exists(f"{save_path}/czi_reference_celltype_1.csv") and len(pd.read_csv(f"{save_path}/czi_reference_celltype_1.csv")) > 0:
    cell_types_source = f"{save_path}/czi_reference_celltype_1.csv"
else:
    # Fallback: define default cell types based on the tissue
    # Examples:
    # Brain: "Excitatory neuron, Inhibitory neuron, Astrocyte, Oligodendrocyte, Microglia, Endothelial cell"
    # Heart: "Cardiomyocyte, Fibroblast, Endothelial cell, Smooth muscle cell, Macrophage"
    # Liver: "Hepatocyte, Cholangiocyte, Endothelial cell, Stellate cell, Kupffer cell"
    cell_types_source = "{comma-separated list of expected cell types for this tissue}"

# PanglaoDB
search_panglao({
    "cell_types": cell_types_source,
    "organism": "Hs",  # or "Mm" for mouse
    "tissue": "{tissue}",
    "save_path": save_path,
    "iter_round": 1
})

# CellMarker2
search_cellmarker2({
    "cell_types": cell_types_source,
    "organism": "Human",  # or "Mouse"
    "tissue": "{tissue}",
    "save_path": save_path,
    "iter_round": 1
})
```

---

### Step 5: Validate Expression

Validate candidate genes are expressed in the target tissue.

```python
# Collect candidates from all sources
all_genes = [...]  # Combine from all files

# Validate in batches (limit ~100 genes per call)
genes_to_validate = ", ".join(all_genes[:100])
result = validate_genes_expression({
    "genes": genes_to_validate,
    "target_tissue": "{tissue}"
})

# For key canonical markers, also verify individually
key_markers = [...]  # Top markers for each cell type
for gene in key_markers:
    result = query_tissue_expression({"gene": gene, "top_k": 5})
    # Check if target tissue is in top tissues

# Save validated genes
pd.DataFrame({"Gene": validated_genes}).to_csv(f"{save_path}/validated_genes.csv", index=False)
```

---

### Step 6: Aggregate & Score with Cell Type Balance

**Key insight from top runs**: Organize genes by cell type category first, then score within each category.

```python
# Define cell type quotas based on:
# 1. Panel size (N genes)
# 2. Tissue type (which cell types are present)
# 3. Cell type abundance (more genes for abundant types)

# Example quota structure:
quotas = {
    'Major_Type_1': int(N * 0.20),   # 20% for most abundant
    'Major_Type_2': int(N * 0.15),   # 15%
    'Major_Type_3': int(N * 0.15),   # 15%
    'Minor_Type_1': int(N * 0.10),   # 10%
    'Minor_Type_2': int(N * 0.10),   # 10%
    'Rare_Type_1': int(N * 0.05),    # 5%
    # ... etc, should sum to N
}

# Assign genes to categories and score
cell_type_genes = {cat: [] for cat in quotas}

for gene in all_candidate_genes:
    category = assign_to_category(gene)  # Based on database annotations
    if category in cell_type_genes and len(cell_type_genes[category]) < quotas[category]:
        cell_type_genes[category].append(gene)

# Combine all categories
final_genes = []
for cat, genes in cell_type_genes.items():
    final_genes.extend(genes)
```

### Scoring Guidelines:
- **Cross-database validation** (+2 per source): Genes in multiple databases are reliable
- **Literature support** (+1): Mentioned in PubMed abstracts
- **Expression validated** (+1): Confirmed expressed in target tissue
- **Canonical marker** (+2): Well-known markers should be prioritized
- **Cell type specificity** (+1): Unique to specific cell types

---

### Step 7: Generate Final Panel with Validation

```python
# Create final DataFrame with detailed Reason column
#
# ⚠️ IMPORTANT: Each Reason must be CONCRETE and SPECIFIC, including:
#   1. Primary cell type this gene marks
#   2. ALL data sources where found (CZI/PanglaoDB/CellMarker2/PubMed)
#   3. Biological function or why it's a good marker
#   4. Expression validation status if checked
#
# ✅ GOOD Reason examples (1-2 coherent scientific sentences):
#   "Canonical hepatocyte marker encoding serum albumin, validated across CZI, PanglaoDB, and CellMarker2 with high liver-specific expression (ARCHS4: 2847 TPM)."
#   "Kupffer cell marker encoding a lysosomal glycoprotein characteristic of tissue-resident macrophages, found in CZI and CellMarker2."
#   "Cholangiocyte-specific cytokeratin that marks biliary epithelial cells, identified in PanglaoDB and CellMarker2 with validated liver expression."
#   "Liver sinusoidal endothelial cell marker encoding a C-type lectin receptor specific to LSECs, identified from CZI reference data."
#   "Hepatic stellate cell marker for alpha-smooth muscle actin, indicating activated myofibroblast phenotype, from PanglaoDB."
#
# ❌ BAD Reason examples (vague, incomplete):
#   "Liver marker" - missing cell type and sources
#   "Found in database" - no specificity
#   "Hepatocyte" - no biological context

final_panel = pd.DataFrame({
    "Gene": final_genes,
    "Score": [...],
    "Reason": [...],  # 1-2 scientific sentences: cell type, biological function, data sources
})

# ⚠️ CRITICAL: Verify panel size before saving
expected_size = num_genes
actual_size = len(final_panel)

if actual_size < expected_size:
    print(f"⚠️ Panel has {actual_size} genes, need {expected_size}")

    # Fill with canonical markers from under-represented categories
    deficit = expected_size - actual_size
    # Add more genes by relaxing score threshold or adding canonical markers

# Verify cell type coverage
print("\n=== Cell Type Coverage ===")
for cat in quotas:
    count = count_genes_in_category(final_panel, cat)
    print(f"{cat}: {count}/{quotas[cat]}")

# Save final panel
final_panel.to_csv(f"{save_path}/final_gene_panel.csv", index=False)
print(f"\n✓ Saved {len(final_panel)}-gene panel")
```

---

## Error Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| CZI returns 0 cell types | Missing `organism` parameter | Add `organism` and retry |
| PanglaoDB empty | Invalid cell type file | Use string list of cell types instead |
| Not enough genes | Over-filtering | Add canonical markers to fill gaps |
| Panel too small | Cascading failures | Use canonical markers for the tissue |
| Unbalanced panel | Missing cell types | Check quotas, add from under-represented categories |

---

## Output Files

| File | Description |
|------|-------------|
| `final_gene_panel.csv` | Final gene panel (Gene, Score, Reason) |
| `pubmed_genes.csv` | Genes from literature + canonical markers |
| `geneset_genes.csv` | Genes from Enrichr/PanglaoDB |
| `czi_reference_celltype_1.csv` | CZI reference markers |
| `pangdb_celltype_1.csv` | PanglaoDB markers |
| `cellmarker_celltype_1.csv` | CellMarker2 markers |
| `validated_genes.csv` | Expression-validated genes |

---

## Validation Checklist (Before Finishing)

1. ✓ Does `final_gene_panel.csv` exist?
2. ✓ Does it have **exactly N genes** (as requested)?
3. ✓ Are **all major cell types** of the tissue represented?
4. ✓ Are well-known canonical markers for this tissue included?
5. ✓ Were at least 2 database sources used?
