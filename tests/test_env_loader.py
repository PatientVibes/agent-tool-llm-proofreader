"""Tests for the env-file loader on both proofreader.py and proofreader_graph.py."""
import os
from importlib import import_module

import pytest


def _loader(module_name):
    """Import and return the _load_env_file_if_present function from a module."""
    mod = import_module(f"llm_proofreader.{module_name}")
    return mod._load_env_file_if_present


# Parametrize over both module variants so both copies stay in sync.
LOADERS = pytest.mark.parametrize(
    "loader",
    [_loader("proofreader"), _loader("proofreader_graph")],
    ids=["proofreader", "proofreader_graph"],
)


def _set_xdg_to(monkeypatch, tmp_path, env_text):
    """Helper: place an env file at the expected path and return its parent."""
    # The loaders look at ~/.config/agent-tool-llm-proofreader/env.
    # Redirect HOME so the loader resolves there inside tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    # Pathlib's Path.home() on Windows also checks USERPROFILE; set both.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg_dir = tmp_path / ".config" / "agent-tool-llm-proofreader"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    env_path = cfg_dir / "env"
    env_path.write_text(env_text, encoding="utf-8")
    return env_path


@LOADERS
def test_non_ascii_identifier_rejected(loader, monkeypatch, tmp_path):
    """Codex P3 repro: `MYé=value` should NOT export `MYé` because POSIX env
    names are ASCII-only. The previous Unicode `str.isalpha()` accepted this."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MYé", raising=False)
    _set_xdg_to(monkeypatch, tmp_path, "MYé=value\n")

    loader()

    assert "MYé" not in os.environ, "non-ASCII identifier should be rejected"


@LOADERS
def test_existing_env_var_not_clobbered(loader, monkeypatch, tmp_path):
    """If OPENROUTER_API_KEY is already set, the loader must be a no-op."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "preset-value")
    _set_xdg_to(monkeypatch, tmp_path, "OPENROUTER_API_KEY=file-value\n")

    loader()

    assert os.environ["OPENROUTER_API_KEY"] == "preset-value"


@LOADERS
def test_single_pair_quote_strip(loader, monkeypatch, tmp_path):
    """K=\"value\" strips to value; K='\"value\"' strips ONE pair to \"value\"."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("K1", raising=False)
    monkeypatch.delenv("K2", raising=False)
    monkeypatch.delenv("K3", raising=False)
    _set_xdg_to(
        monkeypatch,
        tmp_path,
        'OPENROUTER_API_KEY=sk-or-test\nK1="value"\nK2=\'value\'\nK3=\'"value"\'\n',
    )

    loader()

    assert os.environ["K1"] == "value"
    assert os.environ["K2"] == "value"
    assert os.environ["K3"] == '"value"'


@LOADERS
def test_inline_comment_kept_literal(loader, monkeypatch, tmp_path):
    """K=value # not a comment keeps the trailing text as part of the value
    per the loader's documented minimal-reader contract."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("K", raising=False)
    _set_xdg_to(
        monkeypatch,
        tmp_path,
        "OPENROUTER_API_KEY=sk-or-test\nK=value # not a comment\n",
    )

    loader()

    assert os.environ["K"] == "value # not a comment"


@LOADERS
def test_first_wins_on_duplicate_keys(loader, monkeypatch, tmp_path):
    """First-wins within the file for duplicate keys."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("K", raising=False)
    _set_xdg_to(
        monkeypatch,
        tmp_path,
        "OPENROUTER_API_KEY=sk-or-test\nK=first\nK=second\n",
    )

    loader()

    assert os.environ["K"] == "first"


@LOADERS
def test_export_prefix_stripped(loader, monkeypatch, tmp_path):
    """`export K=value` form is accepted; `export ` prefix is stripped."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("K", raising=False)
    _set_xdg_to(
        monkeypatch,
        tmp_path,
        "export OPENROUTER_API_KEY=sk-or-test\nexport K=value\n",
    )

    loader()

    assert os.environ["K"] == "value"
