# agent-tool-llm-proofreader

LLM-powered OCR proofreading agent. Deterministic pre-pass surfaces likely-issue regions; langchain-openai (via OpenRouter) verifies and proposes fixes; a separate verifier subagent reviews each fix before it's applied. Persistent memory of known false positives accumulates across runs in `~/.local/state/agent-tool-llm-proofreader/memory.json` (Linux/macOS) or `%LOCALAPPDATA%\agent-tool-llm-proofreader\memory.json` (Windows).

## Install

```bash
uv tool install --editable D:/agent-tool-llm-proofreader
```

## Configuration

Set `OPENROUTER_API_KEY` in your environment, or store it at `~/.config/agent-tool-llm-proofreader/env` (mode 600) — the CLI auto-sources that file on startup if the env var isn't set:

```bash
mkdir -p ~/.config/agent-tool-llm-proofreader
chmod 700 ~/.config/agent-tool-llm-proofreader
echo "export OPENROUTER_API_KEY='sk-or-...'" > ~/.config/agent-tool-llm-proofreader/env
chmod 600 ~/.config/agent-tool-llm-proofreader/env
```

## Usage

```bash
llm-proofreader run <markdown-file> [--model openrouter:google/gemini-2.5-pro] [--state-dir <path>] [--resume]
llm-proofreader-graph run <markdown-file>     # alternate LangGraph-based runner
crop-flagged-blocks <book-key> [--threshold 0.5] [--top 30] [--dpi 200]
reinsert-blocks <book-key> [--dry-run]
split-and-process <book-key> [...]
```

## State

| Path | Purpose |
|---|---|
| `~/.local/state/agent-tool-llm-proofreader/memory.json` (Linux/macOS) / `%LOCALAPPDATA%\agent-tool-llm-proofreader\memory.json` (Windows) | Accumulated false-positive patterns. Seed copy ships in the package as `share/agent-tool-llm-proofreader/memory.json.example`. |
| `<state>/checkpoints/<book>/` | Per-chunk checkpoint state (used by `--resume`). |
| `<state>/graph_checkpoints/` | SqliteSaver state for the graph runner. |

Override the state root via `AGENT_TOOL_PROOFREADER_STATE_DIR` env var or `--state-dir` CLI flag.

## 12-component agent harness

| Component | Implementation |
|---|---|
| Orchestration | Chunk-by-chunk loop with deterministic pre-analysis + single LLM call per chunk |
| Tools | spell_check, lookup_known_pattern, record_pattern, check_is_quoted, fix_latex_artifact |
| Memory | Session patterns + persistent `memory.json` accumulating known false positives |
| Context Mgmt | Section-aware chunking at heading boundaries (not fixed token windows) |
| Prompt Construction | Dynamic — injects session stats, known FP patterns, pre-analysis hints |
| Output Parsing | JSON extraction with `<think>`-tag handling (preserved as no-op for non-thinking models) |
| State Mgmt | Per-chunk checkpoint, `--resume` flag |
| Error Handling | Retry with fallback; errors don't crash the pipeline |
| Guardrails | No-op rejection, fix cap per chunk, length validation, quote protection |
| Verification | Separate verifier agent reviews all proposed fixes |
| Subagent Orchestration | Proofreader + verifier dual-agent pattern |
| Token/Cost Tracking | Wall-clock timing + input/output token counts (OpenRouter). |

## Origin

Migrated from `C:/Users/chris/OneDrive/Documents/Reading/tools/` on 2026-05-09. See [the migration spec](https://github.com/PatientVibes/ai-agents/blob/master/docs/superpowers/specs/2026-05-09-kindle-pipeline-tool-migration-design.md) and `CHANGELOG.md` for the migration record.

## Development

Install the dev group and run tests:

```bash
uv sync --group dev
uv run --group dev pytest tests/ -v
```

The dev group is declared via PEP 735 `[dependency-groups]` and brings in `pytest`. Tests are isolated from your real state dir via an autouse `conftest.py` fixture that sets `AGENT_TOOL_PROOFREADER_STATE_DIR` to a per-test `tmp_path`.
