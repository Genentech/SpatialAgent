# Hooks

Hooks allow you to customize SpatialAgent's behavior at key points during execution. Configure hooks via `.spatialagent/settings.json` in your project directory.

## Quick Start

```json
{
  "hooks": {
    "Start": [
      {
        "type": "bash",
        "command": "echo 'Starting analysis: $QUERY'"
      }
    ],
    "PreAct": [
      {
        "matcher": { "code_type": "bash" },
        "type": "prompt",
        "prompt": "Review this bash command for safety: $CODE\n\nRespond with JSON: {\"decision\": \"approve\" or \"block\", \"reason\": \"explanation\"}",
        "timeout": 30
      }
    ]
  }
}
```

## Hook Events

### Agent Lifecycle

| Event | When | Context Variables |
|-------|------|-------------------|
| `Start` | Agent starts processing | `$QUERY`, `$THREAD_ID`, `$MAX_STEP`, `$TOOLS` |
| `Stop` | Agent completes | `$QUERY`, `$ITERATION_COUNT`, `$CONCLUSION`, `$FINAL_STATE` |

### Planning Phase

| Event | When | Context Variables |
|-------|------|-------------------|
| `PrePlan` | Before LLM reasoning | `$STEP`, `$MESSAGES`, `$SYSTEM_PROMPT` |
| `PostPlan` | After LLM response | `$STEP`, `$RESPONSE`, `$NEXT_STEP`, `$HAS_ACT`, `$HAS_CONCLUSION` |

### Action Phase

| Event | When | Context Variables |
|-------|------|-------------------|
| `PreAct` | Before executing code | `$CODE`, `$CODE_TYPE`, `$RAW_CODE`, `$ITERATION` |
| `PostAct` | After code execution | `$CODE`, `$CODE_TYPE`, `$RESULT`, `$IS_ERROR`, `$ITERATION` |

### Tool Execution

| Event | When | Context Variables |
|-------|------|-------------------|
| `PreToolUse` | Before tool call | `$TOOL`, `$CODE` or `$COMMAND`, `$CODE_TYPE` |
| `PostToolUse` | After tool returns | `$TOOL`, `$CODE` or `$COMMAND`, `$RESULT`, `$SUCCESS` |

### Other

| Event | When |
|-------|------|
| `PreRoute` | Before routing decision |
| `PreConclusion` | Before conclusion |

## Hook Types

### Bash Hooks

Execute shell commands. Exit code 0 = approve, non-zero = block.

```json
{
  "type": "bash",
  "command": "echo 'Processing: $QUERY' >> /tmp/spatialagent.log",
  "timeout": 30
}
```

Environment variables are set from context. Can output JSON `{"decision": "approve|block", "reason": "..."}` to control flow.

### Prompt Hooks

Query an LLM for context-aware decisions. Uses the same LLM configured for the agent.

```json
{
  "type": "prompt",
  "prompt": "Review this code for safety: $CODE\n\nRespond with JSON: {\"decision\": \"approve\" or \"block\", \"reason\": \"explanation\"}",
  "timeout": 30
}
```

Must return JSON with `decision` field. Fallback: keywords like "block", "deny", "reject" trigger blocking.

## Matchers

Filter which hooks run based on context:

```json
{
  "matcher": { "code_type": "bash" },
  "type": "bash",
  "command": "echo 'Bash command detected'"
}
```

| Field | Description | Example |
|-------|-------------|---------|
| `tool` | Tool name | `"execute_bash"`, `"execute_python"` |
| `code_type` | Code type | `"bash"`, `"python"` |
| `step` | Step number | `1`, `5` |

String values support regex patterns.

## Decision Control

| Decision | Effect |
|----------|--------|
| `approve` | Continue execution (default) |
| `block` | Stop execution, return error to agent |

If any hook in a chain blocks, execution stops. Hooks that error out default to `approve`.

## Examples

### Log All Queries

```json
{
  "hooks": {
    "Start": [
      {
        "type": "bash",
        "command": "echo \"$(date): $QUERY\" >> ~/.spatialagent/queries.log"
      }
    ]
  }
}
```

### Block Dangerous Commands

```json
{
  "hooks": {
    "PreAct": [
      {
        "matcher": { "code_type": "bash" },
        "type": "bash",
        "command": "if echo \"$CODE\" | grep -qE '^(rm -rf|sudo|chmod 777)'; then echo '{\"decision\": \"block\", \"reason\": \"Dangerous command\"}'; exit 1; fi"
      }
    ]
  }
}
```

### LLM Code Review

```json
{
  "hooks": {
    "PreAct": [
      {
        "type": "prompt",
        "prompt": "Analyze this code for security issues:\n\nCode type: $CODE_TYPE\n```\n$CODE\n```\n\nCheck for file system access, network requests, and command injection.\n\nRespond with JSON: {\"decision\": \"approve\" or \"block\", \"reason\": \"your analysis\"}",
        "timeout": 45
      }
    ]
  }
}
```

## Programmatic Usage

```python
from spatialagent.hooks import HooksManager, init_hooks
from spatialagent.agent import SpatialAgent

hooks = init_hooks(llm=my_llm)
agent = SpatialAgent(llm=llm, hooks_manager=hooks)
```

## Schema Reference

```json
{
  "hooks": {
    "<EventName>": [
      {
        "matcher": { "<field>": "<value_or_regex>" },
        "type": "bash | prompt",
        "command": "<bash_command>",
        "prompt": "<llm_prompt>",
        "timeout": 30
      }
    ]
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | `"bash"` or `"prompt"` |
| `command` | string | For bash | Shell command to execute |
| `prompt` | string | For prompt | LLM prompt text |
| `timeout` | number | No | Timeout in seconds (default: 30) |
| `matcher` | object | No | Conditions for when hook applies |
