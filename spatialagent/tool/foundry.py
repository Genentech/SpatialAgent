"""
Code inspection and adaptation tools for hybrid CodeAct.

These tools allow the agent to:
1. Read source code of existing tools
2. Understand their implementation
3. Create modified versions for specific needs

All tools are standalone functions following Biomni pattern.
"""

import ast
import inspect
from typing import Annotated
from langchain_core.tools import tool
from pydantic import Field


def _find_function_calls(source_code: str) -> set:
    """Parse source code and find all function calls using AST."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()

    calls = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Direct function call: func_name(...)
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            # Method call on object: obj.method(...) - skip these
            self.generic_visit(node)

    CallVisitor().visit(tree)
    return calls


def _get_module_functions(module) -> dict:
    """Get all functions defined in a module (including private ones)."""
    functions = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            # Check if function is defined in this module (not imported)
            if obj.__module__ == module.__name__:
                functions[name] = obj
    return functions


def _collect_dependencies(func, module, collected: set = None) -> list:
    """Recursively collect all helper functions that a function depends on.

    Returns list of (name, source_code) tuples in dependency order.
    """
    if collected is None:
        collected = set()

    # Get all functions in the module
    module_functions = _get_module_functions(module)

    # Get source code of the function
    try:
        source_code = inspect.getsource(func)
    except Exception:
        return []

    # Find all function calls in the source
    calls = _find_function_calls(source_code)

    # Find which calls are to functions in the same module
    dependencies = []
    for call_name in calls:
        if call_name in module_functions and call_name not in collected:
            collected.add(call_name)
            dep_func = module_functions[call_name]

            # Recursively get dependencies of this dependency
            sub_deps = _collect_dependencies(dep_func, module, collected)
            dependencies.extend(sub_deps)

            # Add this dependency
            try:
                dep_source = inspect.getsource(dep_func)
                dependencies.append((call_name, dep_source))
            except Exception:
                pass

    return dependencies


@tool
def inspect_tool_code(
    tool_name: Annotated[str, Field(description="Name of the tool to inspect (e.g., 'search_panglao', 'preprocess_spatial_data')")],
) -> str:
    """Retrieve the source code of a predefined tool and its helper functions.

    Use this when you want to:
    - Understand how a tool works internally
    - Create a modified version with different logic
    - Combine logic from multiple tools
    - Debug unexpected tool behavior

    Returns the complete source code including:
    - The main tool function
    - All helper functions it depends on (e.g., _private_helpers)
    - Docstrings and implementation details

    Example usage:
    1. Inspect a tool: call_tool("inspect_tool_code", tool_name="search_panglao")
    2. Understand the implementation
    3. Write adapted version using Python code execution

    This is more flexible than tool calling but requires understanding the code.
    """
    # Import all tool modules
    from . import databases, analytics, interpretation

    # Map of tool names to their standalone functions and modules
    tool_map = {
        # Database Tools (9)
        "search_panglao": (databases.search_panglao, databases),
        "search_czi_datasets": (databases.search_czi_datasets, databases),
        "search_cellmarker2": (databases.search_cellmarker2, databases),
        "extract_czi_markers": (databases.extract_czi_markers, databases),
        "download_czi_reference": (databases.download_czi_reference, databases),
        "query_tissue_expression": (databases.query_tissue_expression, databases),
        "query_celltype_genesets": (databases.query_celltype_genesets, databases),
        "validate_genes_expression": (databases.validate_genes_expression, databases),
        "query_disease_genes": (databases.query_disease_genes, databases),

        # Analytics Tools - Core (8)
        "preprocess_spatial_data": (analytics.preprocess_spatial_data, analytics),
        "harmony_transfer_labels": (analytics.harmony_transfer_labels, analytics),
        "run_utag_clustering": (analytics.run_utag_clustering, analytics),
        "aggregate_gene_voting": (analytics.aggregate_gene_voting, analytics),
        "infer_dynamics": (analytics.infer_dynamics, analytics),
        "summarize_conditions": (analytics.summarize_conditions, analytics),
        "summarize_celltypes": (analytics.summarize_celltypes, analytics),
        "summarize_tissue_regions": (analytics.summarize_tissue_regions, analytics),

        # Analytics Tools - Tangram (5)
        "tangram_preprocess": (analytics.tangram_preprocess, analytics),
        "tangram_map_cells": (analytics.tangram_map_cells, analytics),
        "tangram_project_annotations": (analytics.tangram_project_annotations, analytics),
        "tangram_project_genes": (analytics.tangram_project_genes, analytics),
        "tangram_evaluate": (analytics.tangram_evaluate, analytics),

        # Analytics Tools - CellPhoneDB (5)
        "cellphonedb_prepare": (analytics.cellphonedb_prepare, analytics),
        "cellphonedb_analysis": (analytics.cellphonedb_analysis, analytics),
        "cellphonedb_degs_analysis": (analytics.cellphonedb_degs_analysis, analytics),
        "cellphonedb_filter": (analytics.cellphonedb_filter, analytics),
        "cellphonedb_plot": (analytics.cellphonedb_plot, analytics),

        # Analytics Tools - LIANA (5)
        "liana_tensor": (analytics.liana_tensor, analytics),
        "liana_inference": (analytics.liana_inference, analytics),
        "liana_spatial": (analytics.liana_spatial, analytics),
        "liana_misty": (analytics.liana_misty, analytics),
        "liana_plot": (analytics.liana_plot, analytics),

        # Analytics Tools - Squidpy (8)
        "squidpy_spatial_neighbors": (analytics.squidpy_spatial_neighbors, analytics),
        "squidpy_nhood_enrichment": (analytics.squidpy_nhood_enrichment, analytics),
        "squidpy_co_occurrence": (analytics.squidpy_co_occurrence, analytics),
        "squidpy_spatial_autocorr": (analytics.squidpy_spatial_autocorr, analytics),
        "squidpy_ripley": (analytics.squidpy_ripley, analytics),
        "squidpy_centrality": (analytics.squidpy_centrality, analytics),
        "squidpy_interaction_matrix": (analytics.squidpy_interaction_matrix, analytics),
        "squidpy_ligrec": (analytics.squidpy_ligrec, analytics),

        # Analytics Tools - Deconvolution (4)
        "destvi_deconvolution": (analytics.destvi_deconvolution, analytics),
        "cell2location_mapping": (analytics.cell2location_mapping, analytics),
        "stereoscope_deconvolution": (analytics.stereoscope_deconvolution, analytics),
        "gimvi_imputation": (analytics.gimvi_imputation, analytics),

        # Analytics Tools - Spatial Clustering (3)
        "spagcn_clustering": (analytics.spagcn_clustering, analytics),
        "graphst_clustering": (analytics.graphst_clustering, analytics),
        "scanpy_score_genes": (analytics.scanpy_score_genes, analytics),

        # Analytics Tools - Integration (5)
        "scanpy_ingest": (analytics.scanpy_ingest, analytics),
        "scanpy_bbknn": (analytics.scanpy_bbknn, analytics),
        "totalvi_integration": (analytics.totalvi_integration, analytics),
        "multivi_integration": (analytics.multivi_integration, analytics),
        "mofa_integration": (analytics.mofa_integration, analytics),

        # Analytics Tools - Trajectory (6)
        "scvelo_velocity": (analytics.scvelo_velocity, analytics),
        "scvelo_velocity_embedding": (analytics.scvelo_velocity_embedding, analytics),
        "cellrank_terminal_states": (analytics.cellrank_terminal_states, analytics),
        "cellrank_fate_probabilities": (analytics.cellrank_fate_probabilities, analytics),
        "paga_trajectory": (analytics.paga_trajectory, analytics),

        # Interpretation Tools (3)
        "annotate_cell_types": (interpretation.annotate_cell_types, interpretation),
        "annotate_tissue_niches": (interpretation.annotate_tissue_niches, interpretation),
        "interpret_figure": (interpretation.interpret_figure, interpretation),
    }

    # Normalize tool name (handle different formats)
    tool_name_normalized = tool_name.lower().replace("_", "").replace("-", "")
    normalized_map = {k.lower().replace("_", "").replace("-", ""): (k, v) for k, v in tool_map.items()}

    if tool_name_normalized not in normalized_map:
        available = "\n".join([f"  - {name}" for name in sorted(tool_map.keys())])
        return f"Tool '{tool_name}' not found.\n\nAvailable tools:\n{available}"

    # Get the tool function and its module
    original_name, (tool_obj, module) = normalized_map[tool_name_normalized]

    # For LangChain @tool decorated functions, get the underlying function via .func
    if hasattr(tool_obj, 'func'):
        tool_func = tool_obj.func
    else:
        tool_func = tool_obj

    # Get main function source code
    try:
        main_source = inspect.getsource(tool_func)
    except Exception as e:
        return f"Could not retrieve source code: {e}"

    # Collect all dependencies (helper functions)
    dependencies = _collect_dependencies(tool_func, module)

    # Build output
    output_parts = [f"# {original_name}"]

    if dependencies:
        output_parts.append("\n## Helper Functions\n")
        output_parts.append("The following helper functions are used by this tool:\n")
        for dep_name, dep_source in dependencies:
            output_parts.append(f"```python\n{dep_source}```\n")

    output_parts.append("\n## Main Tool Function\n")
    output_parts.append(f"```python\n{main_source}```")

    return "\n".join(output_parts)
