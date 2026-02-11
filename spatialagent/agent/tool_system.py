"""
Model-agnostic tool system with semantic search and programmatic calling.

Supports Claude, OpenAI, Gemini with:
- Dynamic tool loading via semantic search
- Embedding-based tool retrieval
- Programmatic tool calling (tools called by generated code, not injected)
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """Generic tool representation compatible with multiple LLM providers."""

    name: str
    description: str
    function: Callable
    input_schema: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "examples": self.examples,
        }

    def to_text(self) -> str:
        """Convert to searchable text for embedding."""
        parts = [
            f"Tool: {self.name}",
            f"Description: {self.description}",
        ]

        # Add parameter information
        if "properties" in self.input_schema:
            parts.append("Parameters:")
            for param_name, param_info in self.input_schema["properties"].items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "")
                parts.append(f"  - {param_name} ({param_type}): {param_desc}")

        return "\n".join(parts)

    def to_claude_format(self) -> Dict[str, Any]:
        """Convert to Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }

    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class ToolRegistry:
    """
    Central registry for all tools with embedding support.

    Features:
    - Store tools with metadata
    - Generate embeddings for semantic search (using Qwen3-Embedding-0.6B by default)
    """

    def __init__(self, embedding_model_name: str = "qwen3-0.6b"):
        """
        Initialize tool registry.

        Args:
            embedding_model_name: Name of embedding model (default: qwen3-0.6b)
                                  See make_llm.LOCAL_EMBEDDING_MODELS for options.
        """
        self.tools: Dict[str, Tool] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.tool_names_ordered: List[str] = []

        # Lazy load embedding model
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name

    @property
    def embedding_model(self):
        """Lazy load embedding model on first use."""
        if self._embedding_model is None:
            from .make_llm import make_llm_emb_local
            self._embedding_model = make_llm_emb_local(self._embedding_model_name)
        return self._embedding_model

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        # Mark embeddings as stale
        self.embeddings = None

    def register_langchain_tool(self, langchain_tool: Any) -> None:
        """
        Register a LangChain tool by converting it to our Tool format.

        Args:
            langchain_tool: LangChain tool object with name, description, and func
        """
        # Extract schema from LangChain tool
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Try to get schema from tool
        if hasattr(langchain_tool, 'args_schema') and langchain_tool.args_schema:
            # Convert Pydantic model to JSON schema
            try:
                input_schema = langchain_tool.args_schema.model_json_schema()
            except:
                pass

        # Create Tool instance
        tool = Tool(
            name=getattr(langchain_tool, 'name', langchain_tool.__class__.__name__),
            description=getattr(langchain_tool, 'description', 'No description available'),
            function=langchain_tool.func if hasattr(langchain_tool, 'func') else langchain_tool,
            input_schema=input_schema,
        )

        self.register_tool(tool)

    def build_embeddings(self) -> None:
        """Generate embeddings for all registered tools."""
        if not self.tools:
            return

        # Convert tools to text
        self.tool_names_ordered = list(self.tools.keys())
        tool_texts = [self.tools[name].to_text() for name in self.tool_names_ordered]

        # Generate embeddings using LocalEmbeddings interface
        embeddings_list = self.embedding_model.embed_documents(tool_texts)
        self.embeddings = np.array(embeddings_list)

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())


class EmbedToolRetriever:
    """
    Embedding-based tool retrieval using semantic similarity.

    Uses Qwen3-Embedding-0.6B (via ToolRegistry) to embed tool descriptions
    and find relevant tools via cosine similarity.

    Note: Core tools (ALWAYS_LOADED_TOOLS) are always included and don't count
    towards the min_tools/max_tools quota.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        min_tools: int = 5,
        max_tools: int = 20,
        always_loaded_tools: List[str] = None,
    ):
        """
        Initialize embedding-based tool retriever.

        Args:
            registry: ToolRegistry instance
            min_tools: Minimum number of tools to retrieve (default: 5)
            max_tools: Maximum number of tools to retrieve (default: 20)
            always_loaded_tools: Tools to always include (default: ALWAYS_LOADED_TOOLS)
        """
        self.registry = registry
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.always_loaded_tools = always_loaded_tools if always_loaded_tools is not None else ALWAYS_LOADED_TOOLS

    def select(self, query: str, skill_tools: Optional[List[str]] = None) -> List[str]:
        """
        Select relevant tools using semantic similarity.

        Core tools (ALWAYS_LOADED_TOOLS) are always included and don't count
        towards the min_tools/max_tools quota.

        Args:
            query: Natural language query describing needed tools
            skill_tools: Optional list of tools from matched skill (always included)

        Returns:
            List of tool names
        """
        if not self.registry.tools:
            return []

        # Build embeddings if needed
        if self.registry.embeddings is None:
            self.registry.build_embeddings()

        # Start with core tools (always loaded)
        core_tools = set()
        for tool_name in self.always_loaded_tools:
            if tool_name in self.registry.tools:
                core_tools.add(tool_name)

        # Add skill-specific tools (always included, don't count towards quota)
        if skill_tools:
            for tool_name in skill_tools:
                if tool_name in self.registry.tools:
                    core_tools.add(tool_name)

        # Get query embedding and normalize
        query_embedding = np.array(self.registry.embedding_model.embed_query(query))
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities (cosine similarity via dot product for normalized vectors)
        similarities = np.dot(self.registry.embeddings, query_embedding)

        # Get top indices, excluding core tools
        sorted_indices = np.argsort(similarities)[::-1]

        # Select tools up to max_tools (excluding core tools)
        selected = []
        for idx in sorted_indices:
            if len(selected) >= self.max_tools:
                break
            tool_name = self.registry.tool_names_ordered[idx]
            if tool_name not in core_tools:
                selected.append(tool_name)

        # Ensure at least min_tools are selected
        if len(selected) < self.min_tools:
            # Already got all we can from similarity search
            pass

        # Combine: core tools + similarity-selected tools
        final_tools = list(core_tools) + selected
        return final_tools


# Core tools that are always loaded regardless of LLM selection
# These don't count towards min_tools/max_tools quota
ALWAYS_LOADED_TOOLS = [
    # Code execution tools - essential for programmatic analysis
    "execute_python",
    "execute_bash",
    # Code inspection - for viewing tool source code
    "inspect_tool_code",
    # Research tools - literature and web search
    "query_pubmed",
    "web_search",
]


class LLMToolSelector:
    """
    LLM-based tool selection using the main agent's model.

    More accurate than embedding-based retrieval for domain-specific queries.

    Note: Core tools are always loaded and don't count towards min_tools/max_tools quota.
    See ALWAYS_LOADED_TOOLS for the list (code execution, inspection, interpretation,
    preprocessing, literature search, web research).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        min_tools: int = 5,
        max_tools: int = 20,
        always_loaded_tools: List[str] = None,
    ):
        """
        Initialize LLM tool selector.

        Args:
            registry: ToolRegistry instance
            min_tools: Minimum number of tools to select (default: 5)
            max_tools: Maximum number of tools to select (default: 20)
            always_loaded_tools: Tools to always include (default: ALWAYS_LOADED_TOOLS)
        """
        self.registry = registry
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.always_loaded_tools = always_loaded_tools if always_loaded_tools is not None else ALWAYS_LOADED_TOOLS
        self._llm = None

    @property
    def model(self):
        """Get the model name from main agent's config."""
        try:
            from . import get_agent_model
            return get_agent_model()
        except ImportError:
            return "unknown"

    @property
    def llm(self):
        """Lazy load LLM on first use (uses main agent's model)."""
        if self._llm is None:
            from . import get_agent_llm
            self._llm = get_agent_llm()
        return self._llm

    def _build_tool_catalog(self, exclude_tools: Optional[set] = None) -> str:
        """Build a concise catalog of tools for the LLM.

        Args:
            exclude_tools: Set of tool names to exclude from catalog (e.g., core tools)
        """
        exclude = exclude_tools or set()
        lines = ["Available tools:\n"]
        for name, tool in self.registry.tools.items():
            if name in exclude:
                continue
            # Truncate description to first sentence for brevity
            desc = tool.description.split('.')[0] + '.'
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def select(self, query: str, skill_tools: Optional[List[str]] = None) -> List[str]:
        """
        Select relevant tools for a query using LLM.

        Core tools (see ALWAYS_LOADED_TOOLS) are always included and don't count
        towards the min_tools/max_tools quota.

        Args:
            query: User query describing the task
            skill_tools: Optional list of tools from matched skill (always included)

        Returns:
            List of selected tool names (core tools + skill tools + LLM-selected tools)
        """
        # Always include core tools first (these don't count towards quota)
        core_tools = set()
        for tool_name in self.always_loaded_tools:
            if tool_name in self.registry.tools:
                core_tools.add(tool_name)

        # If skill tools provided, add those (also don't count towards quota)
        if skill_tools:
            for tool_name in skill_tools:
                if tool_name in self.registry.tools:
                    core_tools.add(tool_name)

        # LLM-selected tools (these count towards quota)
        selected = set()

        # Build tool catalog (exclude core tools from LLM selection prompt)
        catalog = self._build_tool_catalog(exclude_tools=core_tools)

        # Create selection prompt
        prompt = f"""Given a user query, select the most relevant tools from the catalog.

USER QUERY: {query}

{catalog}

INSTRUCTIONS:
1. Select {self.min_tools}-{self.max_tools} tools that would be needed to complete this task
2. Include database/search tools if information gathering is needed
3. Include analysis tools if data processing is needed
4. Include visualization tools if plots are requested
5. Include utility tools that may be helpful (file I/O, validation, etc.)
6. Return ONLY a JSON list of tool names, nothing else

Note: Core tools (execute_python, execute_bash, inspect_tool_code, query_pubmed, search_google) are already loaded.

SELECTED TOOLS (JSON list of {self.min_tools}-{self.max_tools} tools):"""

        try:
            # Call LLM for selection
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import re
            # Find JSON array in response
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                tool_names = json.loads(match.group())
                # Validate tool names exist and not already in core tools
                for name in tool_names:
                    if name in self.registry.tools and name not in core_tools:
                        selected.add(name)

        except Exception as e:
            # Fallback: return core tools only
            print(f"LLM tool selection failed: {e}")

        # Ensure LLM-selected tools don't exceed max_tools quota
        selected_list = list(selected)[:self.max_tools]

        # Combine: core tools (always loaded) + LLM-selected tools (quota-limited)
        final_tools = list(core_tools) + selected_list
        return final_tools

    def select_with_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Select tools with explanation (for debugging/transparency).

        Returns dict with 'tools' and 'reasoning' keys.
        """
        catalog = self._build_tool_catalog()

        prompt = f"""Given a user query, select the most relevant tools and explain why.

USER QUERY: {query}

{catalog}

Respond in JSON format:
{{
    "tools": ["tool1", "tool2", ...],
    "reasoning": "Brief explanation of why these tools were selected"
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                # Validate tool names
                result['tools'] = [t for t in result.get('tools', []) if t in self.registry.tools]
                return result

        except Exception as e:
            return {"tools": [], "reasoning": f"Selection failed: {e}"}

        return {"tools": [], "reasoning": "Could not parse response"}


class ToolExecutor:
    """
    Generic tool executor supporting multiple LLM providers.

    Handles programmatic tool calling where tools are invoked by
    generated code rather than injected into namespace.
    """

    def __init__(self, registry: ToolRegistry):
        """
        Initialize tool executor.

        Args:
            registry: ToolRegistry instance
        """
        self.registry = registry
        self.execution_context = {}

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        try:
            result = tool.function(**kwargs)
            return result
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """
        Get the executable function for a tool.

        Used for making tools available in code execution context.
        """
        tool = self.registry.get_tool(tool_name)
        return tool.function if tool else None

    def create_tool_context(self, tool_names: List[str]) -> Dict[str, Callable]:
        """
        Create a dict of tool functions for code execution.

        Args:
            tool_names: List of tool names to make available

        Returns:
            Dict mapping tool names to their functions
        """
        context = {}
        for name in tool_names:
            func = self.get_tool_function(name)
            if func:
                context[name] = func
        return context
