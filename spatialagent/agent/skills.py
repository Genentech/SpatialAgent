"""
Skill/Template System for SpatialAgent

Manages workflow templates (skills) for common spatial transcriptomics tasks.
Templates are markdown files that guide the agent through standard workflows.
"""

import os
import logging
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate


class SkillManager:
    """Manages loading, selection, and export of workflow skill templates."""

    def __init__(self, skills_dir: str):
        """
        Initialize SkillManager.

        Args:
            skills_dir: Directory containing skill template files (.md)
        """
        self.skills_dir = skills_dir
        self.skills: Dict[str, str] = {}
        self.llm = None

        # Cache for performance
        self.last_task_query = None
        self.last_selected_skill = None

    def load_skills(self) -> Dict[str, str]:
        """
        Load all skill templates from skills directory.

        Returns:
            Dict mapping skill name to skill content
        """
        if self.skills:
            return self.skills

        if not os.path.exists(self.skills_dir):
            logging.warning(f"Skills directory not found: {self.skills_dir}")
            return {}

        for filename in os.listdir(self.skills_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.skills_dir, filename)
                with open(filepath, 'r') as f:
                    skill_name = filename.replace('.md', '')
                    self.skills[skill_name] = f.read().strip()
                    logging.info(f"Loaded skill: {skill_name}")

        return self.skills

    def set_llm(self, llm):
        """Set LLM for intelligent skill selection."""
        self.llm = llm

    # === Multiple Choise Questions (MCQ) Options Removal for Skill Matching ===
    # Set to False to disable removing options from MCQ queries before skill matching
    REMOVE_MCQ_OPTIONS = True

    def _remove_mcq_options(self, task_query: str) -> str:
        """Remove MCQ options (A., B., etc.) to prevent long sequences from confusing skill selector."""
        if not self.REMOVE_MCQ_OPTIONS:
            return task_query
        if '\nOptions:' in task_query:
            return task_query.split('\nOptions:', 1)[0].strip()
        return task_query
    # === End MCQ Options Removal ===

    def select_skill(self, task_query: str, num_skills: int = 1) -> List[str]:
        """
        Select the most relevant skill template(s) for a task query.

        Args:
            task_query: User's task description
            num_skills: Maximum number of skills to retrieve (default: 1)

        Returns:
            List of skill template contents (empty list if no match)
        """
        # Return cached if same query
        if task_query == self.last_task_query and self.last_selected_skill:
            logging.info(f"Using cached skill selection")
            return self.last_selected_skill

        if not self.skills:
            self.load_skills()

        if not self.skills:
            logging.info("No skills available")
            return []

        if not self.llm:
            logging.warning("No LLM set for skill selection")
            return []

        # Remove MCQ options for skill matching (prevents long sequences from confusing LLM)
        query_for_matching = self._remove_mcq_options(task_query)

        # Use LLM to decide if skill is needed and select best matching ones
        skill_names = list(self.skills.keys())

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a workflow matching system for spatial transcriptomics and molecular biology analysis.

Given a user task, decide if it needs workflow templates, and if so, select up to {num_skills} most relevant ones.

Available templates: {skill_names}

**Template descriptions:**

- panel_design: Design gene panels through iterative database queries (CZI CELLxGENE → PanglaoDB → CellMarker2). Use when: "design a gene panel", "find markers for cell types", "build a panel for spatial experiment", marker gene selection for targeted spatial assays.

- annotation: Cell type and tissue niche annotation in spatial transcriptomics data. Use when: "annotate cell types", "identify tissue regions", "label clusters", "what cell types are in this spatial data", cluster annotation, niche identification.

- cell_cell_communication: Cell-cell interaction analysis comparing ligand-receptor pairs across conditions. Use when: "compare interactions between conditions", "how do cell communications change", "cross-condition CCC analysis", differential interaction analysis.

- database_query: Query biological databases including: (1) MSigDB gene sets (C2 curated, C5 ontology/mouse phenotypes, C6 oncogenic signatures, C7 immunologic/vaccine response signatures) and GTRD for transcription factor binding sites in promoter regions, (2) miRDB v6.0 for miRNA target predictions, (3) P-HIPSter for virus-host protein interactions, (4) ClinVar for variant pathogenicity (questions with protein sequences and variant options like L1256P, N414H), (5) Ensembl for gene locations and cytogenetic bands, (6) DisGeNET/OMIM for disease associations, (7) MouseMine/MGI for mouse phenotype (MP:) gene sets. Use when: "which gene is contained in the gene set", "gene set contains", "C6 oncogenic signature", "genes up-regulated", "genes down-regulated", gene set names ending in "_UP" or "_DN" (e.g., AKT_UP_MTOR_DN.V1_UP, BCAT_BILD_ET_AL_DN), "C5 ontology", "mouse phenotype", "binding site in promoter", "TF targets according to GTRD", "miRNA target according to miRDB", "MIR" followed by numbers (e.g., MIR2115_3P, MIR194_5P), "predicted target of MIR", "virus-host interaction", "P-HIPSter", "pathogenic or benign", "According to ClinVar", "ClinVar", "which variant", "variants to the following sequence", "most likely to be pathogenic", "most likely to be benign", "chromosomal location", "cytogenetic band", "transcription factor targets", disease-gene queries, MSigDB queries, "MouseMine", "MGI", "Mouse Genome Informatics", "MP:", "mouse gene", "retrieved from MouseMine". **IMPORTANT:** For gene set membership questions, ALWAYS query the local parquet database files first - do NOT use web search. **Note:** Multiple choice questions about miRNA targets, gene sets, diseases, or variants that require database lookup should use this skill.

- sequence_analysis: DNA/RNA/protein sequence analysis including: (1) BLAST identification of unknown sequences, (2) restriction enzyme digestion and fragment counting, (3) primer design for Gibson assembly and restriction cloning, (4) ORF finding, (5) variant detection, (6) Kozak consensus and translation efficiency analysis, (7) fetching gene/plasmid sequences from NCBI. Use when: "identify this sequence", "BLAST", "how many fragments", "restriction digest", "cut sites", "design primers", "find ORF", "compare sequences", "find variants", "clone gene into plasmid", "Gibson assembly", "pUC19", "translation efficiency", "Kozak sequence", sequence manipulation, enzyme digestion, PCR, cloning.

- cell_deconvolution: Deconvolve spatial spots into individual cells using Tangram with cell segmentation from histology images. Use when: you have H&E images with cell segmentation and want to assign cell types to individual segmented cells, "assign cells to segments", single-cell resolution from spots.

- cellphonedb_analysis: Run CellPhoneDB for ligand-receptor interaction prediction with statistical permutation testing. Use when: "run CellPhoneDB", "predict interactions", "ligand-receptor pairs", "which cells are communicating", single-sample CCC analysis with p-values.

- gene_imputation: Impute or project gene expression from scRNA-seq reference to spatial data using Tangram. Use when: "impute genes", "project expression", "genes not in spatial panel", "expand gene coverage", "correct dropout", predict unmeasured genes in spatial data.

- liana_analysis: LIANA for multi-method consensus ligand-receptor inference, spatial LR correlations, and tensor decomposition. Use when: "consensus LR ranking", "compare multiple CCC methods", "spatial ligand-receptor", "multi-sample tensor analysis", robust interaction ranking.

- ligand_receptor_discovery: Comprehensive workflow to discover and validate ligand-receptor interactions with literature cross-validation. Use when: "find new interactions", "validate LR pairs", "immune checkpoint interactions", "growth factor signaling", discovery and validation pipeline.

- mapping_validation: Validate Tangram spatial mappings using cross-validation, diagnostic plots, and quality metrics. Use when: "validate mapping", "check mapping quality", "is the mapping good", "troubleshoot Tangram", "cross-validate spatial mapping", QC for cell-to-spot mapping.

- multimodal_integration: Integrate multiple modalities - RNA+protein (TotalVI/CITE-seq), RNA+ATAC (MultiVI), or any combination (MOFA). Also handles batch correction with BBKNN. Use when: "integrate CITE-seq", "multiome analysis", "combine RNA and protein", "batch correction", "MOFA integration".

- spatial_deconvolution: Estimate cell type proportions in spatial spots using deep learning methods (DestVI, Cell2location, Stereoscope, gimVI). Use when: "deconvolve Visium spots", "cell type proportions", "what fraction of each cell type", "abundance estimation", bulk-to-single-cell deconvolution.

- spatial_domain_detection: Identify spatial domains and tissue niches using SpaGCN (integrates histology) or GraphST (self-supervised). Use when: "find tissue regions", "spatial clustering", "identify niches", "domain detection", "tissue architecture", spatially-aware clustering.

- spatial_mapping: Map scRNA-seq reference data onto spatial transcriptomics using Tangram for cell type projection and localization. Use when: "map cells to space", "project cell types", "where are T cells located", "Tangram mapping", transfer annotations from scRNA-seq to spatial.

- squidpy_analysis: Squidpy for spatial statistics including neighborhood enrichment, co-occurrence analysis, spatial autocorrelation (Moran's I), Ripley's statistics, and centrality scores. Use when: "are cell types colocalized", "neighborhood enrichment", "spatial patterns", "co-occurrence", "spatially variable genes".

- trajectory_inference: Analyze differentiation trajectories using RNA velocity (scVelo), fate mapping (CellRank), or graph abstraction (PAGA). Use when: "trajectory analysis", "differentiation path", "RNA velocity", "cell fate", "pseudotime", "lineage tracing", developmental dynamics.

**NO_MATCH:** Use for simple queries, general questions, or short commands under 50 characters.

CRITICAL: You MUST output ONLY a template name from the list above, or NO_MATCH. Nothing else.

CORRECT outputs: "database_query" or "annotation" or "NO_MATCH"
WRONG outputs: "I need to search..." or "This question asks about..." or any explanation

For ANY question about gene sets, MSigDB, C6, oncogenic signatures, miRNA targets, variants, or databases → output: database_query"""),
            ("user", "Task: {task_query}")
        ])

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "skill_names": ", ".join(skill_names),
                # "task_query": task_query,  # Original: full query
                "task_query": query_for_matching,  # MCQ options removed
                "num_skills": num_skills
            })

            response = result.content.strip()

            # If response looks like verbose text (long and not a skill name), retry with stricter prompt
            if len(response) > 30 and response not in skill_names and "NO_MATCH" not in response:
                print(f"[SkillManager] Verbose response detected: '{response[:50]}...', retrying")
                retry_prompt = ChatPromptTemplate.from_messages([
                    ("system", """Output EXACTLY one skill name or NO_MATCH.

database_query = MSigDB gene sets (C2/C5/C6/C7), miRNA targets (miRDB), ClinVar variants, P-HIPSter virus-host, GTRD TF targets, disease genes
sequence_analysis = BLAST, restriction digest, primer design, cloning
annotation = cell type annotation in spatial data
NO_MATCH = simple questions, general queries

For gene set membership, oncogenic signatures, miRNA, variant, or virus-host questions → database_query"""),
                    ("user", "{task_query}\n\nOutput only: database_query, sequence_analysis, annotation, or NO_MATCH")
                ])
                retry_chain = retry_prompt | self.llm
                retry_result = retry_chain.invoke({
                    "task_query": query_for_matching
                })
                response = retry_result.content.strip()

            if response == "NO_MATCH" or "NO_MATCH" in response:
                logging.info("No skill template matches task")
                self.last_task_query = task_query
                self.last_selected_skill = []
                return []

            # Parse comma-separated skill names, deduplicate, and limit to num_skills
            selected_names = [s.strip() for s in response.split(",")]
            selected_contents = []
            seen_skills = set()

            for name in selected_names:
                if len(selected_contents) >= num_skills:
                    break  # Already have enough skills

                # Direct match
                if name in self.skills:
                    if name not in seen_skills:
                        logging.info(f"Selected skill: {name}")
                        selected_contents.append(self.skills[name])
                        seen_skills.add(name)
                else:
                    # Try to find skill name in verbose text
                    for skill_name in self.skills.keys():
                        if skill_name in name.lower() and skill_name not in seen_skills:
                            logging.info(f"Extracted skill from verbose response: {skill_name}")
                            selected_contents.append(self.skills[skill_name])
                            seen_skills.add(skill_name)
                            break

            # If no skills found yet, scan entire response for any skill name
            if not selected_contents:
                response_lower = response.lower()
                for skill_name in self.skills.keys():
                    # Check if skill name (with underscores or spaces) appears anywhere
                    if skill_name in response_lower or skill_name.replace('_', ' ') in response_lower:
                        if skill_name not in seen_skills:
                            logging.info(f"Fallback: found skill '{skill_name}' in full response")
                            selected_contents.append(self.skills[skill_name])
                            seen_skills.add(skill_name)
                            if len(selected_contents) >= num_skills:
                                break

            if not selected_contents:
                logging.warning(f"No valid skills found in '{response[:50]}...'")

            self.last_task_query = task_query
            self.last_selected_skill = selected_contents
            return selected_contents

        except Exception as e:
            logging.error(f"Skill selection failed: {e}")
            return []

    def format_skill_guidance(self, skill_content: Optional[str]) -> str:
        """
        Format skill template as guidance for agent.

        Args:
            skill_content: Skill template content

        Returns:
            Formatted guidance string
        """
        if not skill_content:
            return """No pre-defined workflow template matches this task.
Create your own plan based on available tools and the specific task requirements."""

        header = """🎯 Workflow Template Available

The following standardized workflow template matches your task.
Follow these steps as guidance, adapting as needed for your specific requirements:

---
"""
        footer = """
---

Follow these steps systematically. You may adapt or extend the workflow based on:
- Specific requirements of your data
- Available tools and resources
- Intermediate results and findings
"""

        return header + skill_content + footer

    def extract_tools_from_skill(self, skill_content: str) -> List[str]:
        """
        Extract tool names mentioned in a skill template.

        Looks for tool names in backticks (e.g., `tool_name`) and known patterns.

        Args:
            skill_content: Skill template content

        Returns:
            List of tool names found in the skill
        """
        import re

        # Pattern 1: Tool names in backticks (most common in our skills)
        # e.g., `query_pubmed`, `search_panglao`
        backtick_pattern = r'`([a-z_]+)`'
        backtick_matches = re.findall(backtick_pattern, skill_content)

        # Pattern 2: Tool names after "Tool:" or "**Tool**:" markers
        # e.g., **Tool**: `search_panglao`
        tool_marker_pattern = r'\*?\*?Tool\*?\*?:\s*`?([a-z_]+)`?'
        tool_marker_matches = re.findall(tool_marker_pattern, skill_content, re.IGNORECASE)

        # Combine and deduplicate
        all_matches = set(backtick_matches + tool_marker_matches)

        # Filter: only keep names that look like tool names (contain underscore or known prefixes)
        tool_prefixes = [
            'query_', 'search_', 'extract_', 'validate_', 'aggregate_',
            'tangram_', 'cellphonedb_', 'liana_', 'squidpy_', 'scanpy_',
            'scvelo_', 'cellrank_', 'paga_', 'totalvi_', 'multivi_', 'mofa_',
            'destvi_', 'cell2location_', 'stereoscope_', 'gimvi_',
            'spagcn_', 'graphst_', 'harmony_', 'run_utag_',
            'preprocess_', 'annotate_', 'interpret_', 'summarize_', 'infer_',
            'generate_', 'verify_', 'download_', 'fetch_'
        ]

        tool_names = []
        for name in all_matches:
            # Check if it starts with known prefix or contains underscore
            if any(name.startswith(prefix) for prefix in tool_prefixes) or '_' in name:
                # Exclude common non-tool patterns
                if name not in ['cell_type', 'cell_types', 'gene_list', 'gene_panel',
                               'save_path', 'data_path', 'file_path', 'adata_path']:
                    tool_names.append(name)

        return list(set(tool_names))

    def export_skill(self, skill_name: str, skill_content: str, save_path: str = None) -> str:
        """
        Export a new skill template based on successful workflow.

        Args:
            skill_name: Name for the new skill
            skill_content: Skill template content
            save_path: Optional custom save path

        Returns:
            Path to exported skill file
        """
        if save_path is None:
            save_path = self.skills_dir

        os.makedirs(save_path, exist_ok=True)

        # Clean skill name
        skill_name = skill_name.replace(' ', '_').lower()
        if not skill_name.endswith('.md'):
            skill_name += '.md'

        filepath = os.path.join(save_path, skill_name)

        with open(filepath, 'w') as f:
            f.write(skill_content)

        logging.info(f"Exported new skill: {filepath}")

        # Reload skills to include new one
        self.skills = {}
        self.load_skills()

        return filepath

    def generate_skill_from_memory(self, llm, task_query: str, agent_messages: List, save_dir: str = None) -> str:
        """
        Generate a new skill template from agent's conversation memory.

        Args:
            llm: LLM for generating skill template
            task_query: Original task query
            agent_messages: List of messages from agent execution
            save_dir: Directory to save new skill

        Returns:
            Path to new skill file
        """
        # Extract key actions from messages
        actions = []
        for msg in agent_messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            # Look for tool calls and actions
            if '<act>' in content:
                import re
                act_matches = re.findall(r'<act>(.*?)</act>', content, re.DOTALL)
                actions.extend(act_matches)

        if not actions:
            logging.warning("No actions found in memory")
            return None

        # Use LLM to synthesize workflow template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a workflow documentation expert for spatial transcriptomics analysis.

Given a user task and the sequence of actions taken by an agent, create a concise, reusable workflow template.

Output format:
Task: <brief task description>

Plan:
    1. <step name>
        - <key actions>

    2. <next step>
        - <key actions>

    ...

Be concise but complete. Focus on the workflow structure, not implementation details."""),
            ("user", """Task: {task_query}

Actions taken:
{actions}

Create a reusable workflow template:""")
        ])

        try:
            chain = prompt | llm
            result = chain.invoke({
                "task_query": task_query,
                "actions": "\n".join([f"- {act[:200]}..." for act in actions[:10]])
            })

            skill_content = result.content.strip()

            # Generate skill name from task
            name_prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a short, descriptive name (2-4 words, underscores) for this workflow. Output only the name."),
                ("user", skill_content[:300])
            ])
            name_chain = name_prompt | llm
            skill_name = name_chain.invoke({}).content.strip()

            # Export the skill
            return self.export_skill(skill_name, skill_content, save_dir)

        except Exception as e:
            logging.error(f"Failed to generate skill from memory: {e}")
            return None

    def list_skills(self) -> List[str]:
        """List all available skill names."""
        if not self.skills:
            self.load_skills()
        return list(self.skills.keys())

    def get_skill(self, skill_name: str) -> Optional[str]:
        """Get skill content by name."""
        if not self.skills:
            self.load_skills()
        return self.skills.get(skill_name)
