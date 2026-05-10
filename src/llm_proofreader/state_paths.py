"""Platform-correct state-file resolver for agent-tool-llm-proofreader.

Default layout (Linux/macOS):  ~/.local/state/agent-tool-llm-proofreader/
Default layout (Windows):      %LOCALAPPDATA%/agent-tool-llm-proofreader/

Override:
- env var AGENT_TOOL_PROOFREADER_STATE_DIR
- CLI flag --state-dir (caller passes through to functions here)
"""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def state_root(override: str | os.PathLike | None = None) -> Path:
    """Resolve the state-dir root. Caller-supplied override > env var > platformdirs default."""
    if override is not None:
        return Path(override)
    env = os.environ.get("AGENT_TOOL_PROOFREADER_STATE_DIR")
    if env:
        return Path(env)
    return Path(platformdirs.user_state_dir("agent-tool-llm-proofreader"))


def memory_path(override: str | os.PathLike | None = None) -> Path:
    """Path to memory.json (accumulated false-positive patterns). Created on first write."""
    p = state_root(override) / "memory.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def checkpoint_dir(override: str | os.PathLike | None = None) -> Path:
    """Per-book checkpoint directory (replaces __file__-relative .proofreader_state)."""
    p = state_root(override) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def graph_checkpoint_dir(override: str | os.PathLike | None = None) -> Path:
    """SqliteSaver checkpoint dir for the graph proofreader (replaces .proofreader_graph_state)."""
    p = state_root(override) / "graph_checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p
