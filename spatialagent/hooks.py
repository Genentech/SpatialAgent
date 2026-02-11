"""
Hooks system for SpatialAgent.

Provides extensible hook points throughout the agent execution lifecycle,
supporting both bash command hooks and prompt-based (LLM) hooks.

Configuration is loaded from .spatialagent/settings.json
"""

import os
import json
import subprocess
import logging
import re
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Supported hook events throughout the agent lifecycle."""
    # Agent lifecycle
    START = "Start"
    STOP = "Stop"

    # Planning phase
    PRE_PLAN = "PrePlan"
    POST_PLAN = "PostPlan"

    # Action/execution phase
    PRE_ACT = "PreAct"
    POST_ACT = "PostAct"

    # Tool execution (Python/Bash)
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"

    # Routing decisions
    PRE_ROUTE = "PreRoute"

    # Conclusion
    PRE_CONCLUSION = "PreConclusion"


@dataclass
class HookResult:
    """Result from executing a hook."""
    decision: Literal["approve", "block"] = "approve"
    reason: str = ""
    output: str = ""
    modified_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.decision == "approve"

    @property
    def blocked(self) -> bool:
        return self.decision == "block"


@dataclass
class HookDefinition:
    """Definition of a single hook."""
    type: Literal["bash", "prompt"]
    command: Optional[str] = None  # For bash hooks
    prompt: Optional[str] = None   # For prompt hooks
    timeout: int = 30
    matcher: Optional[Dict[str, Any]] = None  # Conditions for when hook applies


class HooksManager:
    """
    Manages hook loading, matching, and execution.

    Hooks are loaded from .spatialagent/settings.json and can be:
    - bash: Execute a shell command
    - prompt: Query an LLM for a decision

    Example settings.json:
    {
        "hooks": {
            "PreAct": [
                {
                    "matcher": {"code_type": "bash"},
                    "type": "prompt",
                    "prompt": "Review this bash command for safety: $CODE",
                    "timeout": 30
                }
            ],
            "Start": [
                {
                    "type": "bash",
                    "command": "echo 'Starting: $QUERY'"
                }
            ]
        }
    }
    """

    def __init__(self, llm=None):
        """
        Initialize HooksManager.

        Args:
            llm: LLM instance for prompt-based hooks
        """
        self.llm = llm
        self.hooks: Dict[str, List[HookDefinition]] = {}
        self.enabled = True

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load hooks configuration from .spatialagent/settings.json in SpatialAgent root directory."""
        # Find the SpatialAgent package root (one level up from spatialagent package)
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file = os.path.join(package_dir, ".spatialagent", "settings.json")

        if not os.path.exists(config_file):
            logger.debug("No .spatialagent/settings.json found, hooks disabled")
            self.enabled = False
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            hooks_config = config.get("hooks", {})

            for event_name, hook_list in hooks_config.items():
                # Validate event name
                try:
                    event = HookEvent(event_name)
                except ValueError:
                    logger.warning(f"Unknown hook event: {event_name}")
                    continue

                self.hooks[event_name] = []

                for hook_def in hook_list:
                    # Handle nested hooks structure (like Claude's format)
                    if "hooks" in hook_def:
                        for nested in hook_def["hooks"]:
                            self.hooks[event_name].append(self._parse_hook(nested, hook_def.get("matcher")))
                    else:
                        self.hooks[event_name].append(self._parse_hook(hook_def))

            logger.info(f"Loaded hooks from {config_file}: {list(self.hooks.keys())}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_file}: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Error loading hooks config: {e}")
            self.enabled = False

    def _parse_hook(self, hook_def: Dict, parent_matcher: Optional[Dict] = None) -> HookDefinition:
        """Parse a hook definition from config."""
        matcher = hook_def.get("matcher", parent_matcher)
        return HookDefinition(
            type=hook_def.get("type", "bash"),
            command=hook_def.get("command"),
            prompt=hook_def.get("prompt"),
            timeout=hook_def.get("timeout", 30),
            matcher=matcher
        )

    def set_llm(self, llm):
        """Set LLM for prompt-based hooks."""
        self.llm = llm

    def _matches(self, hook: HookDefinition, context: Dict[str, Any]) -> bool:
        """Check if hook matcher conditions are satisfied."""
        if not hook.matcher:
            return True

        for key, expected in hook.matcher.items():
            actual = context.get(key)
            if actual is None:
                return False

            # Support regex matching for string values
            if isinstance(expected, str) and isinstance(actual, str):
                if not re.match(expected, actual):
                    return False
            elif actual != expected:
                return False

        return True

    def _substitute_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Substitute $VARIABLE placeholders in template with context values."""
        result = template

        # Standard variable substitution
        for key, value in context.items():
            placeholder = f"${key.upper()}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        # Also handle $ARGUMENTS as JSON of all context
        if "$ARGUMENTS" in result:
            result = result.replace("$ARGUMENTS", json.dumps(context, default=str))

        return result

    def _execute_bash_hook(self, hook: HookDefinition, context: Dict[str, Any]) -> HookResult:
        """Execute a bash command hook."""
        if not hook.command:
            return HookResult(decision="approve", reason="No command specified")

        command = self._substitute_variables(hook.command, context)

        try:
            # Set up environment with context variables
            env = os.environ.copy()
            for key, value in context.items():
                env[key.upper()] = str(value)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                env=env
            )

            output = result.stdout + result.stderr

            # Check for decision in output (JSON format)
            try:
                decision_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', output)
                if decision_match:
                    decision_json = json.loads(decision_match.group())
                    return HookResult(
                        decision=decision_json.get("decision", "approve"),
                        reason=decision_json.get("reason", ""),
                        output=output
                    )
            except json.JSONDecodeError:
                pass

            # Default: approve if command succeeded
            if result.returncode == 0:
                return HookResult(decision="approve", output=output)
            else:
                return HookResult(
                    decision="block",
                    reason=f"Command exited with code {result.returncode}",
                    output=output
                )

        except subprocess.TimeoutExpired:
            return HookResult(
                decision="block",
                reason=f"Hook timed out after {hook.timeout}s"
            )
        except Exception as e:
            logger.error(f"Bash hook error: {e}")
            return HookResult(decision="approve", reason=f"Hook error: {e}")

    def _execute_prompt_hook(self, hook: HookDefinition, context: Dict[str, Any]) -> HookResult:
        """Execute a prompt-based (LLM) hook."""
        if not self.llm:
            logger.warning("Prompt hook requested but no LLM configured")
            return HookResult(decision="approve", reason="No LLM for prompt hook")

        if not hook.prompt:
            return HookResult(decision="approve", reason="No prompt specified")

        prompt = self._substitute_variables(hook.prompt, context)

        try:
            from langchain_core.messages import HumanMessage

            response = self.llm.invoke([HumanMessage(content=prompt)])
            output = str(response.content)

            # Parse JSON decision from response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', output, re.DOTALL)
                if json_match:
                    decision_json = json.loads(json_match.group())
                    return HookResult(
                        decision=decision_json.get("decision", "approve"),
                        reason=decision_json.get("reason", ""),
                        output=output,
                        modified_args=decision_json.get("modified_args", {})
                    )
            except json.JSONDecodeError:
                pass

            # Fallback: look for keywords
            output_lower = output.lower()
            if "block" in output_lower or "deny" in output_lower or "reject" in output_lower:
                return HookResult(decision="block", reason=output, output=output)

            return HookResult(decision="approve", output=output)

        except Exception as e:
            logger.error(f"Prompt hook error: {e}")
            return HookResult(decision="approve", reason=f"Hook error: {e}")

    def execute(self, event: HookEvent, context: Dict[str, Any]) -> HookResult:
        """
        Execute all hooks for a given event.

        Args:
            event: The hook event type
            context: Dictionary of contextual information available to hooks

        Returns:
            HookResult with aggregated decision. If any hook blocks, result is blocked.
        """
        if not self.enabled:
            return HookResult(decision="approve")

        event_name = event.value if isinstance(event, HookEvent) else event
        hooks = self.hooks.get(event_name, [])

        if not hooks:
            return HookResult(decision="approve")

        logger.debug(f"Executing {len(hooks)} hooks for {event_name}")

        all_outputs = []
        modified_args = {}

        for hook in hooks:
            # Check matcher conditions
            if not self._matches(hook, context):
                continue

            # Execute hook based on type
            if hook.type == "bash":
                result = self._execute_bash_hook(hook, context)
            elif hook.type == "prompt":
                result = self._execute_prompt_hook(hook, context)
            else:
                logger.warning(f"Unknown hook type: {hook.type}")
                continue

            all_outputs.append(result.output)
            modified_args.update(result.modified_args)

            # If any hook blocks, stop and return block result
            if result.blocked:
                logger.info(f"Hook blocked {event_name}: {result.reason}")
                return HookResult(
                    decision="block",
                    reason=result.reason,
                    output="\n".join(all_outputs),
                    modified_args=modified_args
                )

        return HookResult(
            decision="approve",
            output="\n".join(all_outputs),
            modified_args=modified_args
        )

    def has_hooks(self, event: HookEvent) -> bool:
        """Check if any hooks are registered for an event."""
        event_name = event.value if isinstance(event, HookEvent) else event
        return bool(self.hooks.get(event_name))


# Global hooks manager instance (can be configured at startup)
_hooks_manager: Optional[HooksManager] = None


def get_hooks_manager() -> HooksManager:
    """Get the global hooks manager, creating one if needed."""
    global _hooks_manager
    if _hooks_manager is None:
        _hooks_manager = HooksManager()
    return _hooks_manager


def set_hooks_manager(manager: HooksManager):
    """Set the global hooks manager."""
    global _hooks_manager
    _hooks_manager = manager


def init_hooks(llm=None) -> HooksManager:
    """Initialize and set the global hooks manager."""
    manager = HooksManager(llm=llm)
    set_hooks_manager(manager)
    return manager
