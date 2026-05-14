# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

LLM-powered OCR proofreader for markdown files extracted from scanned books. Deterministic pre-pass (LaTeX/HTML artifact cleanup, joined-word detection, split-number detection) followed by a LangChain-via-OpenRouter proofreader + verifier two-agent loop with per-chunk checkpointing. Migrated from `C:/Users/chris/OneDrive/Documents/Reading/tools/llm_proofreader.py` on 2026-05-09.

## Workflow (PR-based as of 2026-05-14)

All changes go through pull requests against `master`. **Do NOT push directly to `master`.**

- **Branch naming:** `<type>/<scope>` where `<type>` is one of `fix`, `feat`, `docs`, `chore`, `refactor`. Examples: `fix/proofreader-posix-ascii-check`, `feat/proofreader-token-tracker-adoption`, `docs/proofreader-claude-md`.
- **Pre-push:** Run `agent-tool-pr-reviewer review --base master` locally on the branch before pushing. Address HIGH/BLOCKER findings inline (≤20 LOC) before opening the PR. Drop Gemini date-FP findings without iterating (training-cutoff artifact).
- **Squash-merge** every PR (`gh pr merge --squash --delete-branch`). The branch is auto-deleted on merge.
- **Catalog impact:** Any release-tagged change is mirrored in `D:/ai-agents/README.md` via a separate companion PR in the `ai-agents` repo. Open the catalog PR after the release tag is pushed.

## Release

Version bump goes IN the release commit. Steps:

1. Bump `version = "x.y.z"` in `pyproject.toml`
2. Prepend `## x.y.z — YYYY-MM-DD` entry to `CHANGELOG.md`
3. Stage both files in the release commit
4. After squash-merge: `git tag vx.y.z && git push origin vx.y.z`
5. Open a companion catalog-bump PR in `D:/ai-agents/`

## Tests

```bash
uv sync --group dev
uv run --group dev pytest tests/ -v
```

Dev group declared via PEP 735 `[dependency-groups]`. Tests are isolated from real state via an autouse `conftest.py` fixture that sets `AGENT_TOOL_PROOFREADER_STATE_DIR` to a per-test `tmp_path`.

## Conventions worth knowing

- **`ChatOllama` is an import alias for `ChatOpenAI`** (langchain-openai routed through OpenRouter). The alias preserves call sites from the pre-2026-05-09 Ollama-backed code. Don't rename it unless removing the alias entirely.
- **`<think>`-tag stripping in `_parse_fixes`** is a no-op for the current default model (Gemini 2.5 Pro). Safe to delete in a future cleanup pass — flagged in the v0.1.0 CHANGELOG.
- **State files** live at `platformdirs.user_state_dir("agent-tool-llm-proofreader")` by default. Override via `AGENT_TOOL_PROOFREADER_STATE_DIR` env var or `--state-dir` CLI flag.
- **`books.json` lookup is dropped from this tool** — the harness (`agent-harness-kindle-pipeline`) owns per-book metadata and passes it via CLI flags.
