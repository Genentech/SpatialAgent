# Sequence Analysis

Analyze and identify DNA, RNA, and protein sequences using Biopython and BLAST.

## When to Use This Skill

**Sequence Identification:**

- Given a protein/DNA sequence, need to identify what gene/protein it is
- ClinVar questions that provide sequences (need to identify protein first)
- Any question asking about a specific sequence's identity

**Molecular Biology Analysis:**

- Restriction enzyme digestion (counting fragments, finding cut sites)
- Primer design and verification
- Sequence alignment and comparison
- PCR product prediction
- Finding open reading frames (ORFs)
- Calculating GC content, melting temperature

**Keywords**: sequence, BLAST, blastp, blastn, identify, protein, DNA, restriction enzyme, digest, fragment, primer, PCR, ORF, GC content, melting temperature, Tm, reverse complement, clone, Gibson, Kozak, translation efficiency, pUC19, plasmid

---

## Section 1: Fetch Gene and Plasmid Sequences

### Fetch Gene Coding Sequence from NCBI

```python
from Bio import Entrez, SeqIO

Entrez.email = "your@email.com"

def get_gene_sequence(gene_name, organism="Escherichia coli"):
    """Fetch gene coding sequence from NCBI."""
    # Search for gene
    search_term = f"{gene_name}[Gene Name] AND {organism}[Organism]"
    handle = Entrez.esearch(db="gene", term=search_term, retmax=1)
    record = Entrez.read(handle)

    if not record["IdList"]:
        print(f"Gene {gene_name} not found")
        return None

    gene_id = record["IdList"][0]

    # Get gene record
    handle = Entrez.efetch(db="gene", id=gene_id, rettype="gene_table", retmode="text")
    gene_info = handle.read()
    print(f"Gene ID: {gene_id}")

    # For coding sequence, search nucleotide database
    search_term = f"{gene_name}[Gene Name] AND {organism}[Organism] AND CDS[Feature Key]"
    handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=5)
    nuc_record = Entrez.read(handle)

    if nuc_record["IdList"]:
        handle = Entrez.efetch(db="nucleotide", id=nuc_record["IdList"][0],
                              rettype="fasta", retmode="text")
        seq_record = SeqIO.read(handle, "fasta")
        return str(seq_record.seq)

    return None

# Example: Get rsxD from E. coli
gene_seq = get_gene_sequence("rsxD", "Escherichia coli")
print(f"Start: {gene_seq[:50]}")
print(f"End: {gene_seq[-50:]}")
```

### Fetch Plasmid Sequence (pUC19, pET, etc.)

```python
from Bio import Entrez, SeqIO

Entrez.email = "your@email.com"

def get_plasmid_sequence(plasmid_name):
    """Fetch plasmid sequence from NCBI."""
    handle = Entrez.esearch(db="nucleotide", term=f"{plasmid_name}[Title] AND vector", retmax=1)
    record = Entrez.read(handle)

    if record["IdList"]:
        handle = Entrez.efetch(db="nucleotide", id=record["IdList"][0],
                              rettype="fasta", retmode="text")
        seq_record = SeqIO.read(handle, "fasta")
        return str(seq_record.seq)
    return None
```

### pUC19 Multiple Cloning Site (MCS) - Reference Sequence

**IMPORTANT**: Use this reference for Gibson/restriction cloning into pUC19:

```python
# pUC19 MCS region (positions ~230-290 in full plasmid)
# Format: enzyme sites are marked with their recognition sequences
PUC19_MCS = """
                 EcoRI    SacI     KpnI   SmaI/XmaI  BamHI    XbaI     SalI    PstI    SphI   HindIII
                 |        |        |        |        |        |        |        |        |        |
5'-GAATTCGAGCTCGGTACCCGGGGATCCTCTAGAGTCGACCTGCAGGCATGCAAGCTT-3'
"""

# For Gibson assembly after HindIII linearization:
# The plasmid is cut at AAGCTT, creating these flanking sequences:
PUC19_GIBSON_HOMOLOGY = {
    "HindIII": {
        # 30bp UPSTREAM of HindIII cut (use for forward primer 5' end)
        "upstream": "GATTACGCCAAGCTTGCATGCCTGCAGGTC",
        # 30bp DOWNSTREAM of HindIII cut (take RC for reverse primer 5' end)
        "downstream": "GACTCTAGAGGATCCCCGGGTACCGAGCTC",
        "downstream_rc": "GAGCTCGGTACCCGGGGATCCTCTAGAGTC"  # Pre-computed RC
    },
    "HindII": {  # Note: HindII cuts at GTY^RAC (blunt), different from HindIII
        "upstream": "GATTACGCCAAGCTTGCATGCCTGCAGGTC",
        "downstream": "GACTCTAGAGGATCCCCGGGTACCGAGCTC",
        "downstream_rc": "GAGCTCGGTACCCGGGGATCCTCTAGAGTC"
    }
}

# CRITICAL for Gibson primer verification:
# Forward primer = [upstream homology ~30bp] + [gene start with ATG]
# Reverse primer = [downstream_rc ~30bp] + [gene end RC with stop codon RC]
```

### E. coli Gene Database (Local)

**IMPORTANT**: For E. coli cloning questions, use the local gene database:

```python
import json
import os

# Load E. coli gene database (contains common cloning genes)
ECOLI_GENES_PATH = os.path.join(data_path, "gene_databases", "ecoli_genes.json")

with open(ECOLI_GENES_PATH) as f:
    ECOLI_GENES = json.load(f)

# Available genes: agp, gpp, gudX, hslO, intE, menA, mlaE, napF, plsB, rsxD, torA, ubiI, waaA

def get_ecoli_gene(gene_name):
    """Get E. coli gene sequence from local database.

    Returns dict with: sequence, start, end, length
    """
    gene_name_lower = gene_name.lower()
    for name, data in ECOLI_GENES.items():
        if name.lower() == gene_name_lower:
            return data
    return None

# Example:
rsxd = get_ecoli_gene("rsxD")
if rsxd:
    print(f"rsxD: {rsxd['length']} bp")
    print(f"Start: {rsxd['start']}")  # ATGGTATTCAGAATAGCTAGCTCCCCTTA
    print(f"End: {rsxd['end']}")      # ...ends with stop codon
```

### E. coli Gene Reference Table

| Gene | Length  | Start Sequence (first 25bp)   |
| ---- | ------- | ----------------------------- |
| rsxD | 1059 bp | `ATGGTATTCAGAATAGCTAGCTCCC` |
| napF | 495 bp  | `GTGAAGATTGATGCATCCCGTCGG`  |
| torA | 2547 bp | `ATGAACAATAACGATCTCTTTCAGG` |
| agp  | 1242 bp | `ATGAACAAAACGCTAATCGCCGCC`  |
| menA | 927 bp  | `ATGACTGAACAACAAATTAGTGCC`  |
| mlaE | 783 bp  | `ATGCTGTTAAATGCGCTGGCGCTG`  |

**For Gibson assembly primer verification**:

1. Extract gene start (first ~25bp after homology arm)
2. Compare to the gene's actual start sequence from database
3. The primer with matching gene start is correct

---

## Section 2: Sequence Identification with BLAST

### Identify Protein Sequence

```python
from Bio.Blast import NCBIWWW, NCBIXML

def identify_protein(sequence, max_results=3):
    """Identify a protein sequence using NCBI BLAST.

    Args:
        sequence: Protein sequence (amino acids)
        max_results: Number of top hits to return

    Returns:
        List of top matching proteins with gene names
    """
    print("Submitting BLAST search (may take 30-60 seconds)...")

    result_handle = NCBIWWW.qblast(
        "blastp",       # Program: blastp for protein
        "nr",           # Database: non-redundant protein
        sequence[:500], # Use first 500 aa for speed
        expect=10,
        hitlist_size=max_results
    )

    blast_records = NCBIXML.parse(result_handle)
    blast_record = next(blast_records)

    results = []
    for alignment in blast_record.alignments[:max_results]:
        hsp = alignment.hsps[0]
        identity_pct = (hsp.identities / hsp.align_length) * 100

        results.append({
            "title": alignment.hit_def,
            "accession": alignment.accession,
            "identity": f"{identity_pct:.1f}%",
            "e_value": hsp.expect
        })

        print(f"Match: {alignment.hit_def[:80]}...")
        print(f"  Identity: {identity_pct:.1f}%, E-value: {hsp.expect}")

    return results

# Example
sequence = "MLLAVLYCLLWSFQTSAGHFPRACVSS..."
results = identify_protein(sequence)

# Extract gene name from top hit
if results:
    import re
    top_hit = results[0]['title']
    gene_match = re.search(r'\b([A-Z][A-Z0-9]{2,10})\b', top_hit)
    if gene_match:
        gene_name = gene_match.group(1)
        print(f"Gene: {gene_name}")
```

### Identify DNA Sequence

```python
def identify_dna(sequence, max_results=3):
    """Identify a DNA sequence using NCBI BLAST."""
    print("Submitting BLAST search...")

    result_handle = NCBIWWW.qblast(
        "blastn",        # Program: blastn for nucleotide
        "nt",            # Database: nucleotide collection
        sequence[:1000], # Use first 1000 bp for speed
        expect=10,
        hitlist_size=max_results
    )

    blast_records = NCBIXML.parse(result_handle)
    blast_record = next(blast_records)

    for alignment in blast_record.alignments[:max_results]:
        hsp = alignment.hsps[0]
        print(f"Match: {alignment.hit_def[:80]}")
        print(f"  Identity: {hsp.identities}/{hsp.align_length}")

    return blast_record.alignments
```

### Quick UniProt Search (Faster Alternative)

```python
import requests

def search_uniprot(sequence, max_results=5):
    """Search UniProt by sequence (faster than BLAST for exact matches)."""
    url = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        "query": f"sequence:{sequence[:50]}",
        "format": "json",
        "size": max_results
    }

    response = requests.get(url, params=params)
    data = response.json()

    for entry in data.get("results", []):
        gene = entry.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A")
        protein = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
        print(f"Gene: {gene}, Protein: {protein}")

    return data.get("results", [])
```

---

## Section 3: Restriction Enzyme Digestion

### Count Fragments After Digestion

```python
from Bio import Restriction
from Bio.Seq import Seq

# Example: Digest with MboI (cuts at GATC)
sequence = "AGCATATGGAAGACCAATACATGAGGGGGCATACGCTAGAACGCCCC..."
seq = Seq(sequence.upper())

# Single enzyme
enzyme = Restriction.MboI
cut_sites = enzyme.search(seq, linear=True)  # linear=True for linear DNA
print(f"Cut positions: {cut_sites}")
print(f"Number of fragments: {len(cut_sites) + 1}")

# Multiple enzymes
rb = Restriction.RestrictionBatch(["EcoRI", "BamHI"])
analysis = rb.search(seq, linear=True)
for enzyme, positions in analysis.items():
    print(f"{enzyme}: cuts at {positions}")
```

**Key parameters**:

- `linear=True` for linear DNA, `linear=False` for circular plasmids
- Fragment count = cuts + 1 (linear) or cuts (circular)

### Common Restriction Enzymes

| Enzyme  | Recognition Site | Notes       |
| ------- | ---------------- | ----------- |
| EcoRI   | GAATTC           | Sticky ends |
| BamHI   | GGATCC           | Sticky ends |
| HindIII | AAGCTT           | Sticky ends |
| XbaI    | TCTAGA           | Sticky ends |
| SmaI    | CCCGGG           | Blunt ends  |
| NheI    | GCTAGC           | Sticky ends |
| MboI    | GATC             | Sticky ends |
| BsaI    | GGTCTC           | Type IIS    |

### Find All Restriction Sites

```python
from Bio import Restriction
from Bio.Seq import Seq

seq = Seq("YOUR_SEQUENCE_HERE".upper())

# Check multiple enzymes
enzymes = ["EcoRI", "BamHI", "HindIII", "XbaI", "SmaI", "NheI"]
rb = Restriction.RestrictionBatch(enzymes)
analysis = rb.search(seq, linear=True)

for enzyme, positions in analysis.items():
    if positions:
        print(f"{enzyme} ({enzyme.site}): cuts at {positions}")
```

---

## Section 4: Sequence Properties

### GC Content and Melting Temperature

```python
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, MeltingTemp

seq = Seq("ATGCGATCGATCGATCG")

# GC content
gc = gc_fraction(seq) * 100
print(f"GC content: {gc:.1f}%")

# Melting temperature (for primers)
tm = MeltingTemp.Tm_Wallace(seq)  # Simple calculation
print(f"Tm (Wallace): {tm:.1f}C")

# Reverse complement
rc = seq.reverse_complement()
print(f"Reverse complement: {rc}")

# Translate DNA to protein
protein = seq.translate()
print(f"Protein: {protein}")
```

---

## Section 5: Open Reading Frame (ORF) Detection

```python
from Bio.Seq import Seq

def find_orfs(sequence, min_length=100):
    """Find all ORFs in a DNA sequence."""
    seq = Seq(sequence.upper())
    orfs = []

    # Check all three reading frames
    for frame in range(3):
        for i in range(frame, len(seq) - 2, 3):
            codon = str(seq[i:i+3])
            if codon == "ATG":  # Start codon
                for j in range(i + 3, len(seq) - 2, 3):
                    stop = str(seq[j:j+3])
                    if stop in ["TAA", "TAG", "TGA"]:
                        orf_seq = str(seq[i:j+3])
                        if len(orf_seq) >= min_length:
                            orfs.append({
                                "start": i,
                                "end": j + 3,
                                "length": len(orf_seq),
                                "sequence": orf_seq
                            })
                        break

    return sorted(orfs, key=lambda x: x["length"], reverse=True)

orfs = find_orfs(sequence, min_length=60)
print(f"Found {len(orfs)} ORFs")
for orf in orfs[:5]:
    print(f"  Position {orf['start']}-{orf['end']}: {orf['length']} bp")
```

---

## Section 6: Primer Design and Analysis

### Check Primer Properties

```python
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, MeltingTemp

def check_primer(primer_seq):
    """Check primer properties."""
    seq = Seq(primer_seq.upper())

    gc = gc_fraction(seq) * 100
    tm = MeltingTemp.Tm_Wallace(seq)

    issues = []
    if gc < 40 or gc > 60:
        issues.append(f"GC content {gc:.0f}% outside ideal range (40-60%)")
    if tm < 55 or tm > 65:
        issues.append(f"Tm {tm:.0f}C outside ideal range (55-65C)")
    if len(seq) < 18 or len(seq) > 25:
        issues.append(f"Length {len(seq)} outside ideal range (18-25 bp)")

    return {
        "sequence": str(seq),
        "length": len(seq),
        "gc_content": gc,
        "tm": tm,
        "issues": issues
    }

result = check_primer("ATGCGATCGATCGATCGATCG")
print(f"Length: {result['length']} bp, GC: {result['gc_content']:.1f}%, Tm: {result['tm']:.1f}C")
```

### Find Primer Binding Sites

```python
from Bio.Seq import Seq

def find_primer_binding(template, primer, max_mismatches=1):
    """Find where primer binds to template (both strands)."""
    template = template.upper()
    primer = primer.upper()
    primer_rc = str(Seq(primer).reverse_complement())

    matches = []

    for seq, strand in [(primer, "+"), (primer_rc, "-")]:
        for i in range(len(template) - len(seq) + 1):
            window = template[i:i + len(seq)]
            mismatches = sum(1 for a, b in zip(seq, window) if a != b)
            if mismatches <= max_mismatches:
                matches.append({
                    "position": i,
                    "strand": strand,
                    "mismatches": mismatches
                })

    return matches
```

---

## Section 7: Variant Detection (for ClinVar Queries)

### Compare Sequences to Find Variants

```python
def find_variants(sequences):
    """Compare sequences to identify amino acid differences.

    Args:
        sequences: Dict mapping option label to sequence string

    Returns:
        Dict mapping label to list of variants
    """
    labels = list(sequences.keys())
    ref_seq = sequences[labels[0]]

    variants = {labels[0]: []}  # Reference has no variants

    for label in labels[1:]:
        seq = sequences[label]
        diffs = []
        for i, (ref_aa, var_aa) in enumerate(zip(ref_seq, seq)):
            if ref_aa != var_aa:
                diffs.append(f"{ref_aa}{i+1}{var_aa}")
        variants[label] = diffs

        if not diffs:
            print(f"Option {label}: REFERENCE (wild-type)")
        else:
            print(f"Option {label}: {diffs}")

    return variants

# Example
seqs = {
    'A': "MLLAVLYCLLWSFQTS...",
    'B': "MLLAVLYCLLWSFQTS...",  # with R317C
    'C': "MLLAVLYCLLWSFQTS...",  # with D175Y
}
variants = find_variants(seqs)
```

### Biochemical Analysis of Variants

```python
def analyze_variant(variant):
    """Analyze variant by amino acid properties."""
    from_aa = variant[0]
    to_aa = variant[-1]

    # Conservative substitutions (likely benign)
    conservative_pairs = [
        ("E", "D"), ("K", "R"), ("V", "L"), ("V", "I"),
        ("L", "I"), ("S", "T"), ("F", "Y")
    ]

    for pair in conservative_pairs:
        if (from_aa, to_aa) in [pair, pair[::-1]]:
            return f"{variant}: CONSERVATIVE (likely benign)"

    # Disruptive changes
    if to_aa == "P":
        return f"{variant}: Proline introduction (likely pathogenic)"
    if to_aa == "C":
        return f"{variant}: Cysteine introduction (check disulfide)"

    return f"{variant}: Non-conservative change"
```

---

## Section 8: Translation Efficiency (Kozak Consensus)

**Use for questions about "translation efficiency" or "most likely to be translated"**

The Kozak sequence determines how efficiently a ribosome initiates translation.

### Optimal Kozak Sequence

```
Position:     -6 -5 -4 -3 -2 -1  A  U  G +4 +5 +6
Optimal:       G  C  C  A/G C  C  A  U  G  G  N  N
                         ↑              ↑
                   Critical          Critical
                   (purine)          (G)
```

**Critical positions**:

- Position -3: Must be A or G (purine) - MOST IMPORTANT
- Position +4: Should be G

### Score Kozak Strength

```python
def score_kozak(sequence, aug_position):
    """Score Kozak consensus strength (0-4 scale).

    Args:
        sequence: RNA or DNA sequence (U or T accepted)
        aug_position: Position of A in AUG (0-indexed)

    Returns:
        Score from 0 (weak) to 4 (optimal)
    """
    seq = sequence.upper().replace('T', 'U')
    score = 0

    if aug_position < 3:
        return 0  # Not enough context

    # Position -3 (most critical): should be A or G
    pos_minus_3 = seq[aug_position - 3]
    if pos_minus_3 in ['A', 'G']:
        score += 2  # Worth 2 points (most important)

    # Position +4: should be G
    if aug_position + 3 < len(seq):
        pos_plus_4 = seq[aug_position + 3]
        if pos_plus_4 == 'G':
            score += 1

    # Position -6 to -4: GCC is optimal
    if aug_position >= 6:
        context = seq[aug_position-6:aug_position-3]
        if context == 'GCC':
            score += 1

    return score

# Example: Compare sequences for translation efficiency
sequences = {
    'A': "CCCUGAUGCCUGCUAGC...",  # C at -3, weak
    'E': "CCACCAUGGCUAAUGAC...",  # A at -3, G at +4, optimal
}

for name, seq in sequences.items():
    aug_pos = seq.find('AUG')
    if aug_pos >= 0:
        score = score_kozak(seq, aug_pos)
        context = seq[max(0,aug_pos-6):aug_pos+7]
        print(f"Sequence {name}: score={score}, context={context}")
```

**Interpretation**:

- Score 4: Optimal Kozak (highest translation efficiency)
- Score 2-3: Good Kozak
- Score 0-1: Weak Kozak (poor translation initiation)

---

## Section 9: Cloning Primer Design

### Gibson Assembly Primers

For Gibson assembly into a linearized vector:

- **Forward primer**: [upstream vector homology ~30bp] + [gene start sequence]
- **Reverse primer**: [downstream vector RC homology ~30bp] + [gene end RC sequence]

```python
from Bio.Seq import Seq

def design_gibson_primers(gene_seq, vector_upstream, vector_downstream, homology_len=30):
    """Design primers for Gibson assembly.

    Args:
        gene_seq: Gene coding sequence (start with ATG, end with stop)
        vector_upstream: Sequence upstream of cut site (same strand)
        vector_downstream: Sequence downstream of cut site (same strand)
        homology_len: Length of homology arms (default 30bp)

    Returns:
        Forward and reverse primers
    """
    # Forward primer: upstream homology + gene start
    fwd_homology = vector_upstream[-homology_len:]
    fwd_gene = gene_seq[:25]  # First 25bp of gene
    forward = fwd_homology + fwd_gene

    # Reverse primer: downstream homology (RC) + gene end (RC)
    downstream_rc = str(Seq(vector_downstream[:homology_len]).reverse_complement())
    gene_end_rc = str(Seq(gene_seq[-25:]).reverse_complement())
    reverse = downstream_rc + gene_end_rc

    return {
        "forward": forward,
        "reverse": reverse,
        "fwd_length": len(forward),
        "rev_length": len(reverse)
    }

# Example: Clone gene into pUC19 linearized with HindIII
gene = "ATGGTATTCAGAATAGCTAGCTCCCC...GGCCATCGCAAAGGGTAA"  # rsxD
upstream = "GATTACGCCAAGCTTGCATGCCTGCAGGTC"  # pUC19 upstream of HindIII
downstream = "GACTCTAGAGGATCCCCGGGTACCGAGCTC"  # pUC19 downstream

primers = design_gibson_primers(gene, upstream, downstream)
print(f"Forward: {primers['forward']}")
print(f"Reverse: {primers['reverse']}")
```

### Verify Gibson Primer Homology

```python
def verify_gibson_primers(primer, vector_seq, gene_seq, is_forward=True):
    """Verify that primer has correct homology for Gibson assembly.

    Args:
        primer: Primer sequence
        vector_seq: Full vector sequence
        gene_seq: Gene sequence
        is_forward: True for forward primer, False for reverse

    Returns:
        Dict with verification results
    """
    primer = primer.upper()
    vector_seq = vector_seq.upper()
    gene_seq = gene_seq.upper()

    results = {"homology_found": False, "gene_match": False}

    # Check first 25-35bp against vector
    for homology_len in range(35, 20, -1):
        homology = primer[:homology_len]
        if homology in vector_seq:
            results["homology_found"] = True
            results["homology_len"] = homology_len
            results["homology_seq"] = homology

            # Check remaining matches gene
            gene_part = primer[homology_len:]
            if is_forward:
                if gene_seq.startswith(gene_part):
                    results["gene_match"] = True
            else:
                # Reverse primer: gene part should be RC of gene end
                gene_part_rc = str(Seq(gene_part).reverse_complement())
                if gene_seq.endswith(gene_part_rc):
                    results["gene_match"] = True
            break

    return results
```

### Restriction-Ligation Cloning Primers

For traditional restriction cloning:

- Add restriction site + 2-4 flanking bases to each primer

```python
RESTRICTION_SITES = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "XbaI": "TCTAGA",
    "SmaI": "CCCGGG",
    "XmaI": "CCCGGG",  # Same as SmaI but sticky
    "SalI": "GTCGAC",
    "PstI": "CTGCAG"
}

def design_restriction_primers(gene_seq, enzyme_5prime, enzyme_3prime):
    """Design primers for restriction-ligation cloning.

    Args:
        gene_seq: Gene sequence (ATG to stop)
        enzyme_5prime: Restriction enzyme for 5' end
        enzyme_3prime: Restriction enzyme for 3' end
    """
    site_5 = RESTRICTION_SITES[enzyme_5prime]
    site_3 = RESTRICTION_SITES[enzyme_3prime]

    # Forward: flank + site + gene start (from ATG)
    forward = "CCCC" + site_5 + gene_seq[:20]

    # Reverse: flank + site + gene end RC (including stop codon)
    gene_end_rc = str(Seq(gene_seq[-20:]).reverse_complement())
    reverse = "CCCC" + site_3 + gene_end_rc

    return {"forward": forward, "reverse": reverse}

# Example: Clone with SmaI and XmaI (isoschizomers)
primers = design_restriction_primers(
    "ATGAACAATAACGATCTCTTTCAGGCA...TCATGATTTCACCTGCGACGC",
    "SmaI", "SmaI"
)
```

---

## Tips

1. **BLAST is slow** (30-60+ seconds) - use UniProt for faster exact matches
2. **Use first 500 aa** of protein for faster BLAST
3. **Always uppercase sequences** before analysis
4. **Check linear vs circular** - affects fragment count
5. **For ClinVar questions**: Identify protein first (BLAST), then find variants, then query ClinVar
6. **Reference sequence = Benign**: If one option has NO variants, it's benign
7. **For cloning questions**: Fetch the actual gene and plasmid sequences from NCBI to verify primers
8. **For translation efficiency**: Kozak position -3 (purine) is most critical
9. **Gibson homology**: Use 25-35bp homology arms matching vector flanks
