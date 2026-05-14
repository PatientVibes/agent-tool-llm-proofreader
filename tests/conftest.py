"""Pytest configuration + shared fixtures for agent-tool-llm-proofreader tests."""
import os
import pytest


@pytest.fixture(autouse=True)
def isolate_state_dir(tmp_path, monkeypatch):
    """Force the proofreader to use a per-test state dir so tests don't share
    platformdirs caches across machines or with the user's real run."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("AGENT_TOOL_PROOFREADER_STATE_DIR", str(state_dir))


@pytest.fixture
def env_file(tmp_path):
    """Yields a Path the test can write env-file contents to."""
    return tmp_path / "env"
