"""SpatialAgent - Autonomous AI agent for spatial transcriptomics analysis."""

from .hooks import (
    HooksManager,
    HookEvent,
    HookResult,
    HookDefinition,
    get_hooks_manager,
    set_hooks_manager,
    init_hooks,
)

__all__ = [
    "HooksManager",
    "HookEvent",
    "HookResult",
    "HookDefinition",
    "get_hooks_manager",
    "set_hooks_manager",
    "init_hooks",
]
