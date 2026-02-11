# Bioinformatics Database Query

Query local and remote biological databases for gene sets, variants, miRNA targets, virus-host interactions, and genomic information.

## When to Use This Skill

**Local Database Queries:**

- Find genes in a specific pathway or gene set (MSigDB)
- Query disease-gene associations (DisGeNET, OMIM)
- **Find genes with TF binding sites in promoter regions (GTRD)**
- Find transcription factor targets
- **Query miRNA target predictions (miRDB v6.0)**
- **Query virus-host protein interactions (P-HIPSter)**
- **Query mouse phenotype gene sets (MouseMine/MGI)**

**Remote API Queries:**

- **Determine if a variant is pathogenic or benign (ClinVar)**
- Compare multiple variants to find most/least pathogenic
- Find chromosomal location of a gene (Ensembl)
- Get cytogenetic band information

**Keywords**: miRDB, miRNA, microRNA, MIR, target gene, GTRD, transcription factor, binding site, promoter, TSS, P-HIPSter, virus, viral protein, host, ClinVar, variant, pathogenic, benign, pathogenicity, Ensembl, chromosome, cytogenetic band, MouseMine, MGI, Mouse Genome Informatics, mouse phenotype, MP:

---

## ⚠️ CRITICAL: Path Configuration

**NEVER hardcode paths like `./data/`**. Always use the `data_path` variable which is pre-defined in your Python environment:

```python
import os
DATA_PATH = os.path.join(data_path, "gene_databases")
# data_path is already set - do NOT use "./data" or any hardcoded path!
```

---

## Section 1: Local Gene Set Databases

### Available Local Databases

| File                                                                      | Description                                        | Use For                   |
| ------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------- |
| `msigdb_human_h_hallmark_geneset.parquet`                               | 50 hallmark gene sets                              | Core biological processes |
| `msigdb_human_c2_curated_geneset.parquet`                               | Curated pathways (KEGG, Reactome)                  | Pathway analysis          |
| `msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet` | **TF targets from GTRD**                     | TF binding site queries   |
| `msigdb_human_c5_ontology_geneset.parquet`                              | GO terms (BP, CC, MF)                              | Functional annotation     |
| `msigdb_human_c6_oncogenic_signature_geneset.parquet`                   | Oncogenic signatures                               | Cancer analysis           |
| `msigdb_human_c7_immunologic_signature_geneset.parquet`                 | **Immunologic/vaccine response signatures**  | Vaccine response queries  |
| `DisGeNET.parquet`                                                      | Disease-gene associations                          | Disease queries           |
| `omim.parquet`                                                          | OMIM genetic disorders                             | Genetic disorder queries  |
| `miRDB_v6.0_results.parquet`                                            | **miRNA target predictions**                 | miRNA target queries      |
| `Virus-Host_PPI_P-HIPSTER_2020.parquet`                                 | **Virus-host interactions**                  | Viral protein queries     |
| `mousemine_m5_ontology_geneset.parquet`                                 | **Mouse phenotype (MP:) gene sets from MGI** | Mouse phenotype queries   |
| `mousemine_m2_curated_geneset.parquet`                                  | Mouse curated pathways                             | Mouse pathway queries     |
| `mousemine_m8_celltype_signature_geneset.parquet`                       | Mouse cell type signatures                         | Mouse cell type queries   |

---

## Section 2: GTRD - Transcription Factor Binding Sites

**Use for questions about "binding sites in promoter region (-1000,+100 bp around TSS)"**

The GTRD database contains ChIP-seq data. If a gene appears in `{TF}_TARGET_GENES`, that TF has a binding site in the gene's promoter.

### Check if Gene Has TF Binding Site in Promoter

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet")

# Question: Which gene has a TBX3 binding site in its promoter?
tf_name = "TBX3"
candidates = ["DGAT2-DT", "CXCL1", "RIMS3", "TMEM79"]

# Find TF target gene set (format: {TF}_TARGET_GENES)
tf_entry = df[df['chromosome_id'] == f"{tf_name}_TARGET_GENES"]

if len(tf_entry) > 0:
    target_genes = ast.literal_eval(tf_entry.iloc[0]['geneSymbols'])
    print(f"{tf_name} has {len(target_genes)} target genes in GTRD")

    # Check each candidate
    for gene in candidates:
        if gene in target_genes:
            print(f"  {gene}: YES - has {tf_name} binding site in promoter")
        else:
            print(f"  {gene}: NO")
else:
    print(f"No GTRD data for {tf_name}")
```

**Gene set name format**: `{TF}_TARGET_GENES` (e.g., `TBX3_TARGET_GENES`, `NFKBIA_TARGET_GENES`, `ZNF282_TARGET_GENES`, `DYRK1A_TARGET_GENES`)

---

## Section 3: miRDB - miRNA Target Predictions

**Use for questions about "computationally predicted human gene target of miRNA according to miRDB v6.0"**

### Database Schema

| Column               | Description               | Example             |
| -------------------- | ------------------------- | ------------------- |
| `miRNA`            | miRNA identifier          | `hsa-miR-4795-5p` |
| `target_accession` | RefSeq accession          | `NM_001234`       |
| `score`            | Prediction score (50-100) | `85.5`            |
| `target_symbol`    | Gene symbol               | `UNC5C`           |

### Name Conversion

Benchmark format `MIR4795_5P` → database format `hsa-miR-4795-5p`

- Replace `MIR` with `hsa-miR-`
- Replace `_` with `-`
- Lowercase the arm (5P → 5p)

### Check if Genes are miRNA Targets

```python
import pandas as pd
import os

MIRDB_PATH = os.path.join(data_path, "gene_databases", "miRDB_v6.0_results.parquet")
df = pd.read_parquet(MIRDB_PATH)

# Convert benchmark format to miRDB format
mirna_query = "MIR4795_5P"
mirna_name = mirna_query.replace("MIR", "hsa-miR-").replace("_", "-").lower()
# Result: hsa-miR-4795-5p

# Candidate genes to check
candidates = ["ATP5MGL", "UNC5C", "NTN3", "MACROD1"]

# Get targets for this miRNA
targets = df[df['miRNA'] == mirna_name]
print(f"Found {len(targets)} predicted targets for {mirna_name}")

# Check each candidate
for gene in candidates:
    match = targets[targets['target_symbol'] == gene]
    if len(match) > 0:
        score = match.iloc[0]['score']
        print(f"{gene}: YES (score: {score:.2f})")
    else:
        print(f"{gene}: NO")
```

---

## Section 4: P-HIPSter - Virus-Host Protein Interactions

**Use for questions about "protein predicted to interact with viral protein according to P-HIPSter"**

### Database Schema

| Column            | Description                     | Example                                     |
| ----------------- | ------------------------------- | ------------------------------------------- |
| `Viral Protien` | Viral protein name              | `Hepatitis C virus genotype 5 E2 protein` |
| `Genes`         | List of interacting human genes | `['TRGV3', 'STAT1', 'CD8A']`              |

### Check if Genes Interact with Viral Protein

```python
import pandas as pd
import ast
import os

PHIPSTER_PATH = os.path.join(data_path, "gene_databases", "Virus-Host_PPI_P-HIPSTER_2020.parquet")
df = pd.read_parquet(PHIPSTER_PATH)

# Search for viral protein (use partial match)
virus_query = "Hepatitis C virus genotype 5 E2"
matches = df[df['Viral Protien'].str.contains(virus_query, case=False)]

# Candidate genes to check
candidates = ["TRGV3", "SLAIN1", "FOXK2", "GYS2"]

if len(matches) > 0:
    # Get all interacting genes
    all_genes = set()
    for _, row in matches.iterrows():
        genes = ast.literal_eval(row['Genes'])
        all_genes.update(genes)

    # Check each candidate
    for gene in candidates:
        if gene in all_genes:
            print(f"{gene}: YES - interacts with {virus_query}")
        else:
            print(f"{gene}: NO")
else:
    print(f"No viral protein matching '{virus_query}' found")
```

---

## Section 5: ClinVar - Variant Pathogenicity

**Use for questions about "pathogenic or benign according to ClinVar"**

### Workflow

1. **Compare sequences** to find which positions differ (variants)
2. **Identify the protein** using NCBI BLAST or web search
3. **Query ClinVar** for each variant to get actual pathogenicity classification
4. **Do NOT guess** based on biochemical properties - always look up in ClinVar

### Step 1: Find Variants by Comparing Sequences

```python
def find_variants(sequences):
    """Compare sequences to identify amino acid differences."""
    labels = list(sequences.keys())
    ref_seq = sequences[labels[0]]

    for label in labels:
        seq = sequences[label]
        diffs = []
        for i, (ref_aa, var_aa) in enumerate(zip(ref_seq, seq)):
            if ref_aa != var_aa:
                diffs.append(f"{ref_aa}{i+1}{var_aa}")

        if not diffs:
            print(f"Option {label}: REFERENCE (wild-type)")
        else:
            print(f"Option {label}: {diffs}")

seqs = {'A': "MLLAVLY...", 'B': "MLLAVLY...", 'C': "MLLAVLY..."}
find_variants(seqs)
```

### Step 2: Identify the Protein

Use NCBI BLAST to identify the protein from the sequence:

```python
import requests

def blast_protein(sequence):
    """Submit protein sequence to NCBI BLAST and get gene name."""
    # Submit BLAST job
    put_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    put_params = {
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": "swissprot",
        "QUERY": sequence[:100],  # Use first 100 aa
        "FORMAT_TYPE": "JSON2"
    }
    response = requests.get(put_url, params=put_params)

    # Extract RID and poll for results
    import re
    rid_match = re.search(r"RID = (\w+)", response.text)
    if rid_match:
        rid = rid_match.group(1)
        print(f"BLAST job submitted: {rid}")
        # Poll for results (may take 30-60 seconds)
        # ...
    return rid
```

### Step 3: Query ClinVar for Each Variant

```python
import requests

def query_clinvar(gene, variant):
    """Query ClinVar for a specific gene + variant."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    search_term = f"{gene}[gene] AND {variant}[variant name]"
    search_params = {
        "db": "clinvar",
        "term": search_term,
        "retmax": 10,
        "retmode": "json"
    }

    response = requests.get(f"{base_url}/esearch.fcgi", params=search_params)
    ids = response.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return None

    summary_params = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
    response = requests.get(f"{base_url}/esummary.fcgi", params=summary_params)
    summary = response.json()

    for uid in ids:
        entry = summary.get("result", {}).get(uid, {})
        sig = entry.get("germline_classification", {}).get("description", "N/A")
        print(f"{gene} {variant}: {sig}")

    return summary

# Example: Query each variant found in Step 1
query_clinvar("SCN5A", "A1326S")
query_clinvar("SCN5A", "T455A")
```

**Important**: Always query ClinVar for the actual classification. Do not rely on biochemical properties alone.

---

## Section 6: Ensembl - Gene Locations

**Use for questions about "chromosomal location" or "cytogenetic band"**

### Get Gene Location

```python
import requests

def get_gene_location(gene_symbol):
    """Get chromosomal location from Ensembl."""
    base_url = "https://rest.ensembl.org"
    endpoint = f"/lookup/symbol/homo_sapiens/{gene_symbol}"

    response = requests.get(base_url + endpoint, headers={"Content-Type": "application/json"})
    data = response.json()

    print(f"Gene: {gene_symbol}")
    print(f"  Location: chr{data['seq_region_name']}:{data['start']}-{data['end']}")
    return data

# Example
get_gene_location("BRCA1")
```

### Get Cytogenetic Band

```python
def get_gene_band(gene_symbol):
    """Get cytogenetic band (e.g., 16q22.1) for a gene."""
    base_url = "https://rest.ensembl.org"

    # Get gene coordinates
    gene_info = requests.get(
        f"{base_url}/lookup/symbol/homo_sapiens/{gene_symbol}",
        headers={"Content-Type": "application/json"}
    ).json()

    chrom = gene_info["seq_region_name"]
    pos = gene_info["start"]

    # Get bands
    assembly = requests.get(
        f"{base_url}/info/assembly/homo_sapiens/{chrom}?bands=1",
        headers={"Content-Type": "application/json"}
    ).json()

    for band in assembly.get("karyotype_band", []):
        if band["start"] <= pos <= band["end"]:
            print(f"{gene_symbol} is at chr{chrom}{band['id']}")
            return f"{chrom}{band['id']}"

    return None

# Example: Find which gene is at chr16q22
genes = ["NUTF2", "NPIPA9", "IL9RP3"]
for gene in genes:
    get_gene_band(gene)
```

---

## Section 7: Disease and Pathway Databases

### Query DisGeNET (Disease-Gene Associations)

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/DisGeNET.parquet")

disease_query = "diabetes"
matches = df[df['Disorder'].str.contains(disease_query, case=False)]

for _, row in matches.head(5).iterrows():
    genes = ast.literal_eval(row['Genes'])
    print(f"{row['Disorder']}: {genes[:5]}...")
```

### Query MSigDB Gene Sets (C2 Curated Pathways)

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/msigdb_human_c2_curated_geneset.parquet")

# Find genes in a pathway
pathway_query = "KEGG_APOPTOSIS"
match = df[df['chromosome_id'] == pathway_query]

if len(match) > 0:
    genes = ast.literal_eval(match.iloc[0]['geneSymbols'])
    print(f"{pathway_query}: {len(genes)} genes")
    print(f"  {genes[:10]}...")
```

### Query MSigDB C6 Oncogenic Signatures

**Use for questions about "oncogenic signature gene sets", "C6 collection", genes up/down-regulated in cancer contexts**

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/msigdb_human_c6_oncogenic_signature_geneset.parquet")

# Question: Which gene is in gene set AKT_UP_MTOR_DN.V1_UP?
gene_set_name = "AKT_UP_MTOR_DN.V1_UP"
candidates = ["FBXO11", "ALDH3A2", "MPZL2", "MYO6"]

# Find the gene set
gene_set = df[df['chromosome_id'] == gene_set_name]

if len(gene_set) > 0:
    genes = ast.literal_eval(gene_set.iloc[0]['geneSymbols'])
    print(f"{gene_set_name}: {len(genes)} genes")

    # Check each candidate
    for gene in candidates:
        if gene in genes:
            print(f"  ✓ {gene}: YES - in gene set")
        else:
            print(f"  ✗ {gene}: NO")
else:
    print(f"Gene set '{gene_set_name}' not found")
```

**Gene set naming patterns**: `{GENE/PATHWAY}_{UP/DN}.V1_{UP/DN}` (e.g., `AKT_UP_MTOR_DN.V1_UP`, `BCAT_BILD_ET_AL_DN`)

### Query MSigDB C7 Immunologic Signatures (Vaccine Response)

**Use for questions about "immunologic signature gene sets", "C7 collection", "vaccine response", "FluMist", "blood response"**

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/msigdb_human_c7_immunologic_signature_geneset.parquet")

# Question: Which gene is in gene set CAO_BLOOD_FLUMIST_AGE_05_14YO_1DY_DN?
gene_set_name = "CAO_BLOOD_FLUMIST_AGE_05_14YO_1DY_DN"
candidates = ["GENE1", "GENE2", "GENE3", "GENE4"]

# Find the gene set
gene_set = df[df['chromosome_id'] == gene_set_name]

if len(gene_set) > 0:
    genes = ast.literal_eval(gene_set.iloc[0]['geneSymbols'])
    print(f"{gene_set_name}: {len(genes)} genes")

    # Check each candidate
    for gene in candidates:
        if gene in genes:
            print(f"  ✓ {gene}: YES - in gene set")
        else:
            print(f"  ✗ {gene}: NO")
else:
    print(f"Gene set '{gene_set_name}' not found")
```

---

## Section 8: MouseMine - Mouse Phenotype Gene Sets

**Use for questions about "gene set from Mouse Genome Informatics" or "MouseMine" or "MP:" phenotype IDs**

The MouseMine database contains mouse gene sets from MGI (Mouse Genome Informatics), including MP (Mammalian Phenotype) annotations.

### Database Schema

| Column            | Description         | Example                                                   |
| ----------------- | ------------------- | --------------------------------------------------------- |
| `chromosome_id` | Gene set name       | `MP_INCREASED_SMALL_INTESTINE_ADENOCARCINOMA_INCIDENCE` |
| `geneSymbols`   | List of mouse genes | `['Apc', 'Mlh1', 'Smurf2']`                             |
| `exactSource`   | Source ID           | `MP:0009309`                                            |

### Check if Gene is in Mouse Phenotype Gene Set

```python
import pandas as pd
import ast
import os

DATA_PATH = os.path.join(data_path, "gene_databases")
df = pd.read_parquet(f"{DATA_PATH}/mousemine_m5_ontology_geneset.parquet")

# Question: Which gene is in MP_INCREASED_SMALL_INTESTINE_ADENOCARCINOMA_INCIDENCE?
gene_set_name = "MP_INCREASED_SMALL_INTESTINE_ADENOCARCINOMA_INCIDENCE"
candidates = ["Tes", "Tnfrsf1a", "Rnf8", "Smurf2"]

# Find the gene set
gene_set = df[df['chromosome_id'] == gene_set_name]

if len(gene_set) > 0:
    genes = ast.literal_eval(gene_set.iloc[0]['geneSymbols'])
    print(f"{gene_set_name}: {len(genes)} genes")
    print(f"Genes: {genes}")

    # Check each candidate
    for gene in candidates:
        if gene in genes:
            print(f"  {gene}: YES - in gene set")
        else:
            print(f"  {gene}: NO")
else:
    print(f"Gene set '{gene_set_name}' not found")
```

### Search for Mouse Phenotype Gene Sets

```python
# Search by keyword
keyword = "adenocarcinoma"
matches = df[df['chromosome_id'].str.contains(keyword, case=False)]
print(f"Found {len(matches)} gene sets matching '{keyword}':")
for name in matches['chromosome_id'].head(10):
    print(f"  {name}")
```

**Gene set name format**: `MP_{PHENOTYPE_DESCRIPTION}` (e.g., `MP_INCREASED_TUMOR_INCIDENCE`, `MP_DECREASED_LIVER_TUMOR_INCIDENCE`)

---

## Tips

1. **miRNA name conversion**: `MIR29B_1_5P` → `hsa-miR-29b-1-5p`
2. **GTRD gene set format**: `{TF}_TARGET_GENES`
3. **P-HIPSter**: Use partial matching for viral protein names
4. **ClinVar**: Reference sequence = benign; identify protein with BLAST first
5. **Ensembl**: Use REST API, no authentication needed
6. **Always use `ast.literal_eval()`** to parse gene lists from parquet files
7. **MouseMine**: Use `mousemine_m5_ontology_geneset.parquet` for MP_ mouse phenotype gene sets from MGI
