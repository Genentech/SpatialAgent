"""Utility functions for agent system."""

import importlib
import inspect


def load_all_tools(save_path: str = "./experiments", data_path: str = "./data"):
    """
    Auto-discover and load all tool functions from tool modules.

    Args:
        save_path: Path for experiment outputs (for coding tools)
        data_path: Path to reference data (for coding tools)

    Returns:
        list: List of LangChain tool instances
    """
    # Configure tool paths
    from spatialagent.tool.coding import configure_coding_tools
    from spatialagent.tool.databases import configure_database_tools
    configure_coding_tools(save_path, data_path)
    configure_database_tools(data_path)

    # Map module names to their tool modules
    tool_modules = {
        "database": "spatialagent.tool.databases",
        "literature": "spatialagent.tool.literature",
        "analytics": "spatialagent.tool.analytics",
        "interpretation": "spatialagent.tool.interpretation",
        "support_tools": "spatialagent.tool.foundry",
        "coding": "spatialagent.tool.coding",
        "subagent": "spatialagent.tool.subagent",
    }

    all_tools = []

    for category, module_path in tool_modules.items():
        try:
            module = importlib.import_module(module_path)

            # Find all functions decorated with @tool
            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "name") and hasattr(obj, "description"):
                    # This is a LangChain tool
                    all_tools.append(obj)
                    print(f"  Loaded: {obj.name} ({category})")

        except Exception as e:
            print(f"Warning: Could not load tools from {module_path}: {e}")

    return all_tools
