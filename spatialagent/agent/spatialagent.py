"""Spatial Transcriptomics Agent using LangGraph."""

from typing import Annotated, List, Dict, Any, TypedDict, Literal
import os, re, operator, warnings, uuid, logging, signal
warnings.filterwarnings('ignore')

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .make_prompt import AgentPrompts
from .tool_system import ToolRegistry, EmbedToolRetriever, ToolExecutor, LLMToolSelector
from .skills import SkillManager



class AgentState(TypedDict):
    """State of the agent graph."""
    messages: list[BaseMessage]
    next_step: str | None


class SpatialAgent:
    """SpatialAgent with simplified execution-focused architecture."""

    def __init__(
        self,
        llm=None,
        tools: List = None,
        data_path: str = "./data",
        save_path: str = "./experiments",
        tool_retrieval: bool = True,
        tool_retrieval_method: str = "llm",  # "llm", "embedding", or "all"
        min_tools: int = 5,
        max_tools: int = 20,
        skill_retrieval: bool = True,
        num_skills: int = 1,
        auto_interpret_figures: bool = True,
        act_timeout: int = 1800,
        web_search_model: str = "gemini-3-flash-preview",
    ):
        """
        Initialize SpatialAgent with dynamic tool loading.

        Args:
            llm: Language model instance. If None, defaults to Claude Sonnet 4.5.
            tools: List of tool instances (LangChain tools). If None, auto-loads all tools.
            data_path: Path to reference data directory (default: "./data").
            save_path: Directory where all results should be saved (default: "./experiments").
            tool_retrieval: Enable dynamic tool retrieval based on query (default: True).
            tool_retrieval_method: Method for retrieving tools (default: "llm"):
                - "llm": LLM-based selection (recommended, most accurate)
                - "embedding": Embedding similarity search (faster, less accurate)
                - "all": Use all tools (no retrieval)
            min_tools: Minimum number of tools to retrieve (default: 5).
            max_tools: Maximum number of tools to retrieve (default: 20).
            skill_retrieval: Enable skill/workflow template retrieval (default: True).
            num_skills: Maximum number of skills to retrieve per query (default: 1).
            auto_interpret_figures: Automatically interpret generated figures using vision LLM (default: True).
            act_timeout: Timeout in seconds for each <act> code execution (default: 1800 = 30 minutes).
            web_search_model: Model to use for web_search tool (default: "gemini-3-flash-preview").
                If None, uses the agent's model. Set to a specific model like "gemini-3-flash-preview"
                to always use that model for web search regardless of the agent's LLM.
        """
        # Default to Claude Sonnet 4.5 if no LLM provided
        if llm is None:
            from .make_llm import make_llm, DEFAULT_CLAUDE_MODEL
            print(f"No LLM provided, using default: {DEFAULT_CLAUDE_MODEL}", flush=True)
            llm = make_llm(DEFAULT_CLAUDE_MODEL)
        self.llm = llm
        self.tool_retrieval = tool_retrieval
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.auto_interpret_figures = auto_interpret_figures
        self.act_timeout = act_timeout

        # Set the model config for subagents and tool selectors to use
        from . import set_agent_model
        # Extract model name from LLM if possible
        # Check multiple attributes for different LLM types
        # Priority: deployment_name (Azure) > model_id (Bedrock) > model_name > model (ensure it's a string)
        model_name = None
        for attr in ['deployment_name', 'model_id', 'model_name', 'model']:
            val = getattr(llm, attr, None)
            if val and isinstance(val, str):
                model_name = val
                break
        if not model_name:
            model_name = "unknown"
        set_agent_model(model_name, llm)

        # web_search_model: if set (default "gemini-3-flash-preview"), use that model
        # if None, use the agent's model for web search
        self.web_search_model = web_search_model if web_search_model else model_name

        # Observation accumulator for deep research reports
        self.observation_log = []
        self._observation_log_path = os.path.join(save_path, "observation_log.jsonl")

        # Create directories if they don't exist
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        # Initialize tool system
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)

        # Load tools: auto-load if not provided
        if tools is None:
            print("Auto-loading tools from tool modules...", flush=True)
            from .utils import load_all_tools
            tools = load_all_tools(save_path=save_path, data_path=data_path)
            print(f"Loaded {len(tools)} tools", flush=True)

        # Register all tools
        for tool in tools:
            self.tool_registry.register_langchain_tool(tool)

        # Inject all tools into Python REPL namespace so they can be called directly
        from spatialagent.tool.coding import inject_tools_into_repl

        # NOTE: Different LLMs generate tool calls with different syntax:
        #
        # Claude generates dict style (what LangChain's invoke() expects):
        #   preprocess_spatial_data({"adata_path": "data.h5ad", "save_path": "./out"})
        #
        # Gemini/GPT generate keyword argument style:
        #   preprocess_spatial_data(adata_path="data.h5ad", save_path="./out")
        #
        # Without wrapping, Gemini fails with:
        #   "BaseTool.invoke() missing 1 required positional argument: 'input'"
        #
        # The wrapper below converts any calling convention to dict style for invoke().
        def make_tool_wrapper(langchain_tool, ws_model: str = None):
            """Create a wrapper that accepts both dict and keyword argument calling styles.

            LangChain's tool.invoke() expects a single dict argument, but LLMs often generate
            code with keyword arguments. This wrapper handles both:
              - tool({"arg1": val1, "arg2": val2})  # dict style
              - tool(arg1=val1, arg2=val2)          # keyword style

            For web_search, automatically injects the configured web_search_model.
            """
            def wrapper(*args, **kwargs):
                # Build input dict based on calling convention
                if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                    input_dict = args[0].copy()
                elif kwargs and not args:
                    input_dict = kwargs.copy()
                elif args and not kwargs:
                    param_names = list(langchain_tool.args_schema.model_fields.keys()) if hasattr(langchain_tool, 'args_schema') else []
                    if len(args) <= len(param_names):
                        input_dict = {param_names[i]: args[i] for i in range(len(args))}
                    else:
                        return langchain_tool.invoke(args[0])
                else:
                    param_names = list(langchain_tool.args_schema.model_fields.keys()) if hasattr(langchain_tool, 'args_schema') else []
                    input_dict = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            input_dict[param_names[i]] = arg
                    input_dict.update(kwargs)

                # For web_search: inject configured model if not explicitly provided
                if langchain_tool.name == "web_search" and ws_model:
                    if "model" not in input_dict or input_dict.get("model") is None:
                        input_dict["model"] = ws_model

                return langchain_tool.invoke(input_dict)

            # Copy metadata for introspection
            wrapper.__name__ = langchain_tool.name
            wrapper.__doc__ = langchain_tool.description
            return wrapper

        tool_functions = {}
        for tool in tools:
            # Skip coding tools themselves to avoid recursion
            if tool.name not in ["execute_python", "execute_bash"]:
                tool_functions[tool.name] = make_tool_wrapper(tool, ws_model=self.web_search_model)
        inject_tools_into_repl(tool_functions)

        # Initialize tool retrieval
        self.tool_retrieval_method = tool_retrieval_method
        self.tool_selector = None
        self.tool_retriever = None

        if self.tool_retrieval:
            if tool_retrieval_method == "llm":
                print(f"Initializing LLM-based tool retrieval ({model_name})...", flush=True)
                self.tool_selector = LLMToolSelector(
                    self.tool_registry,
                    min_tools=min_tools,
                    max_tools=max_tools
                )
            elif tool_retrieval_method == "embedding":
                print(f"Initializing embedding-based tool retrieval...", flush=True)
                self.tool_retriever = EmbedToolRetriever(
                    self.tool_registry,
                    min_tools=min_tools,
                    max_tools=max_tools
                )
            elif tool_retrieval_method == "all":
                print(f"Using all {len(self.tool_registry.tools)} tools (no retrieval)...", flush=True)
            else:
                raise ValueError(f"Unknown tool_retrieval_method: {tool_retrieval_method}")

        # Current active tools (dynamically updated per query)
        self._active_tools = []
        self._last_human_msg_count = 0  # Track human messages to detect new queries
        self._context_injected = False  # Track if tool/skill context has been injected for current query

        # Initialize skill system
        self.skill_retrieval = skill_retrieval
        self.num_skills = num_skills
        if self.skill_retrieval:
            # Skills directory is fixed relative to this file: spatialagent/skill/
            skills_dir = os.path.join(os.path.dirname(__file__), '..', 'skill')
            self.skill_manager = SkillManager(skills_dir)
            self.skill_manager.set_llm(llm)
            skills = self.skill_manager.load_skills()
            if skills:
                print(f"Loaded {len(skills)} skills: {', '.join(skills.keys())}", flush=True)
        else:
            self.skill_manager = None
        self._selected_skill = None  # Cache selected skill for current query

        # Extract cost callback for summary printing
        self.cost_callback = None
        if hasattr(llm, 'callbacks'):
            from .make_llm import CostCallback
            for cb in llm.callbacks:
                if isinstance(cb, CostCallback):
                    self.cost_callback = cb
                    break

        # Build system prompt (without specific tool details)
        self.system_prompt = self._build_system_prompt()

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())

    def _build_system_prompt(self) -> str:
        """Build system prompt with dynamic tool loading capability."""
        # Get generic tool description
        if self.tool_retrieval:
            tool_info = f"""
# Tool Discovery

You have access to {len(self.tool_registry.tools)} specialized tools via **dynamic retrieval**.

**How it works:**
1. When you need a tool, the system automatically searches for relevant tools based on your query
2. Top {self.max_tools} most relevant tools are retrieved and made available
3. Tools are executed by the agent when you use <act> tags with appropriate code

**Important**: Tools are loaded dynamically - you don't need to know all tool names upfront.
The system will retrieve relevant tools based on your task.
"""
        else:
            # List all tools if not using retrieval
            tool_descriptions = [
                "**Available tools:**\n"
            ]
            for tool_name in self.tool_registry.list_tools():
                tool = self.tool_registry.get_tool(tool_name)
                tool_descriptions.append(f"- {tool_name}: {tool.description}")
            tool_info = "\n".join(tool_descriptions)

        # Use existing SYSTEM_PROMPT template but with dynamic tool info
        return AgentPrompts.SYSTEM_PROMPT(tool_info)


    def _build_graph(self) -> StateGraph:
        """Build the simplified LangGraph state machine."""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("act", self._act_node)

        # Add edges
        workflow.add_edge(START, "plan")
        workflow.add_conditional_edges(
            "plan",
            self._routing_function,
            {
                "act": "act",
                "plan": "plan",
                "end": END,
            }
        )
        workflow.add_edge("act", "plan")

        return workflow

    def _format_tool_info(self, tool) -> str:
        """Format a tool's info including parameters for display to LLM."""
        lines = [f"**{tool.name}**: {tool.description}"]
        if "properties" in tool.input_schema:
            params = []
            required = tool.input_schema.get("required", [])
            for param_name, param_info in tool.input_schema["properties"].items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req_marker = "*" if param_name in required else ""
                params.append(f"  - {param_name}{req_marker} ({param_type}): {param_desc}")
            if params:
                lines.append("  Parameters:")
                lines.extend(params)
        return "\n".join(lines)

    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """Plan node: LLM thinks and decides what to do next."""
        # Count human messages to detect new queries
        human_msg_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))

        # Only retrieve tools/skills on first call or when new human message arrives
        if state["messages"] and human_msg_count > self._last_human_msg_count:
            self._last_human_msg_count = human_msg_count

            # Get the latest user query
            user_query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            if user_query:
                # Step 1: Check for matching skills FIRST (skills define required tools)
                skill_context = ""
                skill_tools = []
                if self.skill_retrieval and self.skill_manager:
                    skill_contents = self.skill_manager.select_skill(user_query, num_skills=self.num_skills)
                    if skill_contents:
                        # Process each selected skill
                        skill_names = []
                        for skill_content in skill_contents:
                            # Find skill name for logging
                            for name, content in self.skill_manager.skills.items():
                                if content == skill_content:
                                    skill_names.append(name)
                                    break
                            # Add skill guidance
                            skill_context += self.skill_manager.format_skill_guidance(skill_content)
                            # Extract tools mentioned in the skill
                            tools = self.skill_manager.extract_tools_from_skill(skill_content)
                            skill_tools.extend([t for t in tools if t in self.tool_registry.tools])

                        # Remove duplicates from skill_tools
                        skill_tools = list(dict.fromkeys(skill_tools))
                        self._selected_skill = skill_contents

                        print(f"\033[1m<skill>\033[0m retrieved {', '.join(skill_names)} \033[1m</skill>\033[0m\n", flush=True)
                        if skill_tools:
                            print(f"\033[1m<skill-tools>\033[0m {'; '.join(skill_tools)} \033[1m</skill-tools>\033[0m\n", flush=True)
                    else:
                        print(f"\033[1m<skill>\033[0m no workflow needed \033[1m</skill>\033[0m\n", flush=True)
                        self._selected_skill = []

                # Step 2: Select additional tools based on method
                tool_context = ""
                if self.tool_retrieval:
                    if self.tool_retrieval_method == "llm" and self.tool_selector:
                        # LLM-based retrieval - pass skill_tools to ensure they're included
                        self._active_tools = self.tool_selector.select(user_query, skill_tools=skill_tools)
                    elif self.tool_retrieval_method == "embedding" and self.tool_retriever:
                        # Embedding-based retrieval - pass skill_tools to ensure they're included
                        self._active_tools = self.tool_retriever.select(user_query, skill_tools=skill_tools)
                    elif self.tool_retrieval_method == "all":
                        # Use all tools
                        self._active_tools = list(self.tool_registry.tools.keys())
                    else:
                        # Fallback to just skill tools
                        self._active_tools = skill_tools

                    # Log selected tools
                    if self._active_tools:
                        tool_names = "; ".join(self._active_tools)
                        print(f"\033[1m<tool>\033[0m selected {tool_names} \033[1m</tool>\033[0m\n", flush=True)

                        # Format tool details
                        tool_details = []
                        for name in self._active_tools:
                            tool = self.tool_registry.get_tool(name)
                            if tool:
                                tool_details.append(self._format_tool_info(tool))
                        tool_context = "# Selected Tools (call with dict argument)\n" + "\n\n".join(tool_details)

                # Add tool/skill context as HumanMessage (only once per query, not as AIMessage)
                # This avoids LLM confusion and saves tokens in subsequent calls
                if not self._context_injected and (tool_context or skill_context):
                    context_parts = []
                    if tool_context:
                        context_parts.append(tool_context)
                    if skill_context:
                        context_parts.append(skill_context)
                    context_msg = "[System Context]\n" + "\n\n".join(context_parts)
                    state["messages"].append(HumanMessage(content=context_msg))
                    self._context_injected = True
                    # Update count to include the context message we just added
                    # This prevents re-retrieval on subsequent _plan_node calls
                    self._last_human_msg_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))

        # Build messages with system prompt (tool/skill context is in HumanMessage, only added once)
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]

        # Invoke LLM with error handling
        try:
            logging.debug(f"Invoking LLM with {len(messages)} messages...")
            response = self.llm.invoke(messages)
            msg = str(response.content)
            logging.debug(f"LLM response received, length: {len(msg)}")
        except Exception as e:
            logging.error(f"LLM invocation failed: {type(e).__name__}: {e}", exc_info=True)
            raise

        # Strip hallucinated <observation> tags from LLM response
        if "<observation>" in msg:
            logging.warning("LLM generated hallucinated <observation> tags - stripping them out")
            msg = re.sub(r"<observation>.*?</observation>", "", msg, flags=re.DOTALL)

        # Auto-close unclosed tags
        if "<think>" in msg and "</think>" not in msg:
            msg += "</think>"
        if "<act>" in msg and "</act>" not in msg:
            msg += "</act>"
        if "<conclude>" in msg and "</conclude>" not in msg:
            msg += "</conclude>"

        # Parse for tags
        think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
        act_match = re.search(r"<act>(.*?)</act>", msg, re.DOTALL)
        conclude_match = re.search(r"<conclude>(.*?)</conclude>", msg, re.DOTALL)

        # Add the message to state before checking for errors
        state["messages"].append(AIMessage(content=msg.strip()))

        # Determine next step with error recovery
        if conclude_match:
            state["next_step"] = "end"
        elif act_match:
            state["next_step"] = "act"
        elif think_match:
            state["next_step"] = "plan"
        else:
            # Parsing error recovery
            # Note: Parsing errors can occur when model safety filters block responses
            # (e.g., poxvirus questions return empty content [], causing no tags to be found)
            print("parsing error...", flush=True)
            # Check if we already added an error message to avoid infinite loops
            error_count = sum(
                1 for m in state["messages"]
                if isinstance(m, AIMessage) and "There are no tags" in m.content
            )

            if error_count >= 2:
                # If we've already tried to correct the model twice, just end
                print("Detected repeated parsing errors, ending conversation", flush=True)
                state["next_step"] = "end"
                state["messages"].append(
                    AIMessage(
                        content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                    )
                )
            else:
                # Try to correct it
                state["messages"].append(
                    HumanMessage(
                        content="Each response must include either <act> or <conclude> tag. But there are no tags in the current response. Please include <act> with code to execute, or <conclude> with your final answer."
                    )
                )
                state["next_step"] = "plan"

        return state

    def _log_observation(self, step_number: int, code: str, result: str, figure_interpretations: str = ""):
        """Log an observation for later report generation.

        Args:
            step_number: The current step number in the analysis
            code: The code that was executed
            result: The execution result/output
            figure_interpretations: Any figure interpretations generated
        """
        import json
        from datetime import datetime

        # Create observation entry
        entry = {
            "step": step_number,
            "timestamp": datetime.now().isoformat(),
            # "code_snippet": code[:500] if len(code) > 500 else code,  # Truncate long code
            # "result_summary": result[:2000] if len(result) > 2000 else result,  # Truncate long results
            # "figure_interpretations": figure_interpretations[:5000] if len(figure_interpretations) > 5000 else figure_interpretations,
            "code_snippet": code,
            "result_summary": result,
            "figure_interpretations": figure_interpretations,
        }

        # Add to in-memory log
        self.observation_log.append(entry)

        # Append to file (JSONL format for streaming writes)
        try:
            with open(self._observation_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to observation log: {e}")

    def _display_figures(self, code_context: str = "", user_query: str = "") -> str:
        """Display any new image files and optionally interpret them using vision LLM.

        Args:
            code_context: The code that generated the figures (used as context for interpretation)
            user_query: The user's original query for biological context

        Returns:
            String containing figure interpretations (empty string if no figures or interpretation disabled)
        """
        interpretations = []

        try:
            from spatialagent.tool.coding import get_new_image_files
            image_files = get_new_image_files()
            if not image_files:
                return ""

            # Check if we're in a Jupyter environment
            try:
                from IPython.display import display, Image, SVG
                import os
                print(f"📊 Displaying {len(image_files)} figure(s)...")
                for img_path in image_files:
                    if not os.path.exists(img_path):
                        print(f"⚠️ File not found: {img_path}")
                        continue
                    ext = os.path.splitext(img_path)[1].lower()
                    if ext == '.svg':
                        display(SVG(filename=img_path))
                    elif ext in ('.png', '.jpg', '.jpeg'):
                        display(Image(filename=img_path))
                    elif ext == '.pdf':
                        # For PDFs, just note the file was created
                        print(f"📄 Created: {os.path.basename(img_path)}")
            except ImportError:
                # Not in Jupyter, just note that figures were generated
                import os
                print(f"[{len(image_files)} figure(s) created: {', '.join(os.path.basename(f) for f in image_files)}]")

            # Auto-interpret figures if enabled
            if self.auto_interpret_figures and image_files:
                print(f"🔍 Interpreting {len(image_files)} figure(s)...")

                from spatialagent.tool.interpretation import interpret_figure

                for img_path in image_files:
                    import os
                    if not os.path.exists(img_path):
                        continue

                    ext = os.path.splitext(img_path)[1].lower()
                    # Skip PDFs for now (vision models handle images better)
                    if ext == '.pdf':
                        continue

                    try:
                        # Extract context from the code that generated the figure
                        # Try to infer what type of plot this is from the code
                        context = self._infer_figure_context(code_context, img_path, user_query)

                        # Call interpret_figure tool
                        interpretation = interpret_figure.invoke({
                            "image_path": img_path,
                            "context": context,
                            "analysis_focus": "general"
                        })

                        fig_name = os.path.basename(img_path)
                        interpretations.append(f"\n### Figure Interpretation: {fig_name}\n{interpretation}")

                    except Exception as e:
                        print(f"⚠️ Could not interpret {os.path.basename(img_path)}: {e}")

        except Exception as e:
            # Log error instead of silently ignoring
            print(f"⚠️ Error displaying/interpreting figures: {e}")

        return "\n".join(interpretations) if interpretations else ""

    def _infer_figure_context(self, code: str, img_path: str, user_query: str = "") -> str:
        """Infer the context/type of a figure from the code that generated it.

        Args:
            code: The Python code that generated the figure
            img_path: Path to the generated image
            user_query: The user's original query for biological context
        """
        import os

        # Start with filename as basic context
        fig_name = os.path.basename(img_path)
        context_parts = [f"Figure: {fig_name}"]

        code_lower = code.lower()

        # === 1. Detect plot type (can have multiple) ===
        plot_types = []
        if "umap" in code_lower:
            plot_types.append("UMAP dimensionality reduction")
        if "tsne" in code_lower or "t-sne" in code_lower:
            plot_types.append("t-SNE dimensionality reduction")
        if "pca" in code_lower and "plot" in code_lower:
            plot_types.append("PCA plot")
        if "sc.pl.spatial" in code_lower or "sq.pl.spatial" in code_lower or "spatial_scatter" in code_lower:
            plot_types.append("Spatial plot showing tissue coordinates")
        if "heatmap" in code_lower or "sns.heatmap" in code_lower or "clustermap" in code_lower:
            plot_types.append("Heatmap visualization")
        if "violin" in code_lower:
            plot_types.append("Violin plot")
        if "dotplot" in code_lower or "dot_plot" in code_lower:
            plot_types.append("Dot plot")
        if "stacked_violin" in code_lower:
            plot_types.append("Stacked violin plot")
        if "matrixplot" in code_lower:
            plot_types.append("Matrix plot")
        if "rank_genes" in code_lower:
            plot_types.append("Ranked genes plot")
        if "barplot" in code_lower or "bar(" in code_lower or "barh(" in code_lower:
            plot_types.append("Bar plot")
        if "boxplot" in code_lower:
            plot_types.append("Box plot")
        if "scatter" in code_lower and "spatial" not in code_lower:
            plot_types.append("Scatter plot")

        if plot_types:
            context_parts.append(" + ".join(plot_types))

        # === 2. Detect what's being colored/grouped by (can have multiple) ===
        color_by = []
        if "cell_type" in code_lower or "celltype" in code_lower or "tier3" in code_lower:
            color_by.append("cell type")
        if "leiden" in code_lower:
            color_by.append("Leiden clusters")
        if "louvain" in code_lower:
            color_by.append("Leiden clusters")
        if "batch" in code_lower or "sample" in code_lower:
            color_by.append("batch/sample")
        if "condition" in code_lower or "sample_type" in code_lower:
            color_by.append("condition/disease stage")
        if "leiden_neigh" in code_lower or "neighborhood" in code_lower or "neigh" in code_lower:
            color_by.append("spatial neighborhood")
        if "niche" in code_lower:
            color_by.append("tissue niche")

        if color_by:
            context_parts.append(f"colored/grouped by: {', '.join(color_by)}")

        # === 3. Extract plot title if present ===
        title_patterns = [
            r'plt\.title\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'\.set_title\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'title\s*=\s*[\'"]([^\'"]+)[\'"]',
        ]
        for pattern in title_patterns:
            match = re.search(pattern, code)
            if match:
                context_parts.append(f"Title: {match.group(1)}")
                break

        # === 4. Extract gene names if present ===
        gene_patterns = [
            r'var_names\s*=\s*\[([^\]]+)\]',
            r'genes\s*=\s*\[([^\]]+)\]',
            r"color\s*=\s*['\"]([A-Z][A-Z0-9]+)['\"]",  # Single gene coloring
        ]
        for pattern in gene_patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                genes = match.group(1).strip()
                if len(genes) < 200:  # Avoid very long gene lists
                    context_parts.append(f"Genes: {genes}")
                break

        # === 5. Extract comments that might describe the plot ===
        comment_pattern = r'#\s*(.+?)$'
        comments = re.findall(comment_pattern, code, re.MULTILINE)
        relevant_comments = [c.strip() for c in comments if len(c.strip()) > 10 and len(c.strip()) < 100]
        if relevant_comments:
            # Take first 2 relevant comments
            context_parts.append(f"Code comments: {'; '.join(relevant_comments[:2])}")

        # === 6. Detect comparison/analysis type ===
        if "comparison" in code_lower or "vs" in code_lower or "versus" in code_lower:
            context_parts.append("Comparative analysis")
        if "composition" in code_lower:
            context_parts.append("Composition analysis")
        if "proportion" in code_lower or "percentage" in code_lower:
            context_parts.append("Proportion/percentage analysis")
        if "dynamics" in code_lower or "trajectory" in code_lower:
            context_parts.append("Dynamics/trajectory analysis")
        if "interaction" in code_lower:
            context_parts.append("Cell-cell interaction analysis")

        # === 7. Add user query as biological context (truncated) ===
        if user_query:
            # Extract key biological terms from user query
            query_truncated = user_query[:500] if len(user_query) > 500 else user_query
            context_parts.append(f"Biological context: {query_truncated}")

        return " | ".join(context_parts)

    def _act_node(self, state: AgentState) -> Dict[str, Any]:
        """Act node: runs code from <act> tags using tool-based execution with timeout."""
        last_message = state["messages"][-1].content
        act_match = re.search(r"<act>(.*?)</act>", last_message, re.DOTALL)

        if not act_match:
            state["messages"].append(AIMessage(content="No action to execute"))
            return state

        code = act_match.group(1).strip()
        figure_interpretations = ""

        # Extract user query for context (find the most recent HumanMessage that's not system context)
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                content = msg.content
                # Skip system context messages
                if not content.startswith("[System Context]"):
                    user_query = content
                    break

        # Timeout handler for code execution
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {self.act_timeout} seconds")

        # Execute code with timeout
        result = ""
        try:
            # Set up timeout (Unix only)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.act_timeout)

            try:
                # Determine code type and execute via tool executor
                if code.startswith("#!BASH") or code.startswith("# Bash"):
                    # Bash code - use bash tool
                    bash_code = re.sub(r"^#!BASH|^# Bash script|^# Bash", "", code, 1).strip()
                    result = self.tool_executor.execute_tool("execute_bash", command=bash_code)
                else:
                    # Python code (default) - use python REPL tool
                    result = self.tool_executor.execute_tool("execute_python", code=code)
                    # Display any figures and optionally interpret them
                    figure_interpretations = self._display_figures(code_context=code, user_query=user_query)
            finally:
                # Cancel the alarm and restore old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError as e:
            result = f"ERROR: {str(e)}\n\nThe code execution was terminated. Consider:\n- Breaking the task into smaller steps\n- Using more efficient algorithms\n- Processing data in chunks"

        # Truncate if too long
        if len(result) > 15000:
            result = result[:15000] + "\n... (output truncated)"

        # Build observation with optional figure interpretations
        if figure_interpretations:
            observation = f"<observation>\n{result}\n\n## Auto-Generated Figure Analysis\n{figure_interpretations}\n</observation>"
        else:
            observation = f"<observation>\n{result}\n</observation>"

        # Log observation for deep research report
        step_number = len([m for m in state["messages"] if isinstance(m, AIMessage) and "<act>" in str(m.content)])
        self._log_observation(
            step_number=step_number,
            code=code,
            result=result,
            figure_interpretations=figure_interpretations
        )

        state["messages"].append(AIMessage(content=observation))
        logging.debug(f"_act_node completed, returning state with {len(state['messages'])} messages")
        return state

    def _routing_function(self, state: AgentState) -> str:
        """Route to next node based on state."""
        return state.get("next_step", "plan")

    def _print_message(self, message: BaseMessage):
        """Print a message with appropriate formatting based on its content.

        Tags are displayed with content on separate lines:
        <tag>
        content
        </tag>
        """
        import sys

        # Skip HumanMessage (already displayed as query)
        if isinstance(message, HumanMessage):
            return

        msg = message.content
        if not msg or not isinstance(msg, str):
            return

        # Skip empty or placeholder messages
        msg_stripped = msg.strip()
        if not msg_stripped or msg_stripped == "[]" or msg_stripped == "{}":
            return

        # Skip system tool context messages (internal use only)
        if msg_stripped.startswith("[System]"):
            return

        def format_tag(text, tag, color_code):
            """Replace <tag>content</tag> with formatted multi-line version."""
            pattern = rf"<{tag}>(.*?)</{tag}>"

            def replacer(match):
                content = match.group(1).strip()
                return f"{color_code}<{tag}>\033[0m\n{content}\n{color_code}</{tag}>\033[0m"

            return re.sub(pattern, replacer, text, flags=re.DOTALL)

        # Check for conclude (needs special rich markdown handling)
        conclude_match = re.search(r"<conclude>(.*?)</conclude>", msg, re.DOTALL)

        if conclude_match:
            try:
                from rich.console import Console
                from rich.markdown import Markdown
                from rich.theme import Theme

                custom_theme = Theme({
                    "markdown.code": "bold cyan",
                    "markdown.code_block": "cyan on grey93",
                })
                console = Console(theme=custom_theme, force_terminal=True)

                # Extract conclude content
                conclude_content = conclude_match.group(1).strip()

                # Display everything before conclude with formatted tags
                if conclude_match.start() > 0:
                    pre_conclude = msg[:conclude_match.start()].strip()
                    pre_conclude = format_tag(pre_conclude, "act", "\033[91m")
                    pre_conclude = format_tag(pre_conclude, "observation", "\033[94m")
                    print(pre_conclude)
                    print()
                    sys.stdout.flush()

                # Display conclude with markdown
                print("\033[1m<conclude>\033[0m")
                md = Markdown(conclude_content, code_theme="github-light", inline_code_theme="cyan")
                console.print(md)
                print("\033[1m</conclude>\033[0m")
                print()
                sys.stdout.flush()

            except ImportError:
                # Fallback to plain display with formatted tags
                display_msg = format_tag(msg_stripped, "act", "\033[91m")
                display_msg = format_tag(display_msg, "observation", "\033[94m")
                display_msg = format_tag(display_msg, "conclude", "\033[1m")
                print(display_msg)
                print()
                sys.stdout.flush()
        else:
            # No conclude, display with formatted tags
            display_msg = format_tag(msg_stripped, "act", "\033[91m")
            display_msg = format_tag(display_msg, "observation", "\033[94m")
            print(display_msg)
            print()
            sys.stdout.flush()

    def run(self, user_query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the agent with a user query.

        Args:
            user_query: The user's task/question
            config: Optional LangGraph configuration dict.
                   If not provided, defaults to {"recursion_limit": 50}

        Returns:
            Final agent state
        """
        # Reset retrieval state for new run
        self._last_human_msg_count = 0
        self._selected_skill = None
        self._context_injected = False

        # Agent termination conditions:
        # 1. The agent outputs a <conclude> tag (normal completion)
        # 2. LangGraph's recursion_limit is reached (hard stop after N graph iterations)
        # Note: Each plan->act cycle counts as multiple iterations in the graph.
        if config is None:
            config = {"recursion_limit": 50}
        elif "recursion_limit" not in config:
            config["recursion_limit"] = 50

        # Create proper LangGraph config with checkpointer requirements
        # Use unique thread_id for each run to ensure isolated state
        langgraph_config = {
            "recursion_limit": config.get("recursion_limit", 50),
            "configurable": {
                "thread_id": config.get("thread_id", str(uuid.uuid4()))
            }
        }

        # Display query header
        import sys
        print(f"\033[1m<user query>\033[0m\n{user_query.strip()}\n\033[1m</user query>\033[0m\n")
        sys.stdout.flush()

        # Check if thread exists and get existing messages for multi-turn conversation
        thread_id = langgraph_config["configurable"]["thread_id"]
        existing_messages = []
        try:
            # Try to get existing state from checkpointer
            existing_state = self.app.get_state(langgraph_config)
            if existing_state and existing_state.values and "messages" in existing_state.values:
                existing_messages = existing_state.values["messages"]
        except Exception:
            # Thread doesn't exist yet, start fresh
            pass

        # Build state: append new message to existing conversation or start fresh
        if existing_messages:
            initial_state = {
                "messages": existing_messages + [HumanMessage(content=user_query)],
                "next_step": None,
            }
            prev_message_count = len(existing_messages) + 1  # Skip printing old messages
        else:
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "next_step": None,
            }
            prev_message_count = 1  # Track how many messages we've printed

        # Run the graph with streaming
        try:
            final_state = None
            conclude_reached = False  # Track if conclude has been printed

            # Use stream_mode="values" to get full state updates
            logging.debug("Starting graph stream...")
            step_count = 0
            for state_update in self.app.stream(initial_state, stream_mode="values", config=langgraph_config):
                step_count += 1
                messages = state_update.get("messages", [])
                next_step = state_update.get("next_step")
                logging.debug(f"Stream step {step_count}: {len(messages)} messages, next_step={next_step}")

                # Print any new messages (but stop after conclude)
                for i in range(prev_message_count, len(messages)):
                    if not conclude_reached:
                        self._print_message(messages[i])
                        # Check if this message contains conclude
                        msg_content = messages[i].content if hasattr(messages[i], 'content') else ""
                        if isinstance(msg_content, str) and "<conclude>" in msg_content:
                            conclude_reached = True

                prev_message_count = len(messages)
                final_state = state_update

                # Break the loop if we've reached the end state
                if next_step == "end" or conclude_reached:
                    logging.debug(f"Breaking stream loop: next_step={next_step}, conclude_reached={conclude_reached}")
                    break

            logging.debug(f"Stream loop ended after {step_count} steps")

            # Print cost summary at the end
            if self.cost_callback:
                self.cost_callback.print_summary()

            return final_state

        except Exception as e:
            print(f"Error: {e}", flush=True)
            raise
