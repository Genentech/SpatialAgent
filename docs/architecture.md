# Architecture

## Tool Selection

SpatialAgent uses a hybrid approach combining skill-driven and LLM-based tool selection.

```
User Query
    │
    ▼
Skill Matching (if enabled)
    │  Select matching skill via LLM
    │  Extract tools mentioned in skill (e.g., `query_pubmed`)
    │  These tools are GUARANTEED to be included
    ▼
Additional Tool Selection
    │  LLM-based (default): LLM selects relevant tools from catalog
    │  Embedding-based: Semantic search via Qwen3-Embedding-0.6B
    │  All: Load all 72 tools
    ▼
Final Tool Set = Core Tools (always loaded) + Skill Tools + Selected Tools
```

Core tools (`execute_python`, `execute_bash`, `inspect_tool_code`, `query_pubmed`, `web_search`) are always loaded and don't count towards the selection quota.

### Selection Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `llm` (default) | LLM selects from catalog | Most accurate for domain-specific queries |
| `embedding` | Semantic search with Qwen3-Embedding-0.6B | Fast, no LLM call needed |
| `all` | Load all tools | Debugging, simple tasks |

### Configuration

```python
agent = SpatialAgent(
    llm=make_llm("claude-sonnet-4-5-20250929"),
    tool_selection_method="llm",      # "llm", "embedding", or "all"
    tool_selection_model=None,        # None = use main agent's model
)
```

The `LLMToolSelector` and `EmbedToolRetriever` classes in `spatialagent/agent/tool_system.py` both accept `min_tools` (default: 5) and `max_tools` (default: 20) parameters.

## Skill-Driven Workflows

Skills are markdown templates in `spatialagent/skill/` that encode best practices for common analysis workflows. When a skill is matched, `SkillManager.extract_tools_from_skill()` parses tool names from backticks in the template and ensures they are loaded.

```
<skill> retrieved panel_design </skill>
<skill-tools> query_pubmed; search_panglao; search_cellmarker2; ... </skill-tools>
<tool> selected query_pubmed; search_panglao; ... (+ additional LLM-selected tools) </tool>
```

## Model Consistency

All components use the same LLM as the main agent by default. When `SpatialAgent` is initialized, it calls `set_agent_model()` to share the model name and LLM instance with subcomponents (tool selectors, subagents).

```python
# In spatialagent/agent/__init__.py
_agent_config = {
    "model": None,  # Set by SpatialAgent
    "llm": None,    # Set by SpatialAgent
}
```

Subagents and tool selectors retrieve the shared config via `get_agent_model()` and `get_agent_llm()`.

### Model Name Resolution

The system extracts the model name from the LLM instance by checking these attributes in order:

1. `model_name` (Claude, Gemini)
2. `model` (some OpenAI configurations)
3. `deployment_name` (Azure OpenAI)
4. Fallback to `DEFAULT_CLAUDE_MODEL`

## Subagents

Subagents are autonomous multi-pass analysis tools in `spatialagent/tool/subagent.py`:

| Subagent | Purpose | Passes |
|----------|---------|--------|
| `report_subagent` | Synthesize publication-quality reports | 6 |
| `verification_subagent` | Verify claims against evidence | 5 |

Subagents use `get_agent_model()` to inherit the main agent's model.

## Design Principles

1. **Model Consistency** - All components use the same LLM by default
2. **Skill-Driven** - Skills guarantee correct tools are loaded for known workflows
3. **LLM-Based Selection** - More accurate than embeddings for domain-specific queries
4. **Fallback Safety** - All components have sensible defaults
5. **Transparency** - Console output shows which model/tools are being used
