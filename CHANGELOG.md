# Changelog

## 0.1.1 — 2026-05-14

### Fixed

- **POSIX-correct ASCII guard for env-loader.** `_load_env_file_if_present` in both `proofreader.py` and `proofreader_graph.py` was using Unicode-aware `str.isalpha()` / `str.isalnum()` to validate env-var names. Non-ASCII names like `MYé=value` passed the guard and got exported into `os.environ` despite POSIX requiring ASCII-only identifiers. Replaced with a module-level compiled regex `^[A-Za-z_][A-Za-z0-9_]*$`. Backport of the Codex P3 finding from the 2026-05-14 `/ship` dogfood (PR #1).

### Added

- **First pytest suite on this repo.** New `tests/conftest.py` + `tests/test_env_loader.py` with 6 cases parametrized across both env-loader call sites (12 total): non-ASCII identifier rejection (Codex case), pre-existing env-var no-clobber, single-pair quote strip, inline-comment literal, first-wins on duplicate keys, `export K=value` prefix strip. Run with `uv run --group dev pytest tests/ -v`.
- **`[dependency-groups]` table in `pyproject.toml`** (PEP 735) with `dev = ["pytest>=8,<9"]`.

## 0.1.0 — 2026-05-09

Initial release. Migrated from `C:/Users/chris/OneDrive/Documents/Reading/tools/llm_proofreader.py`, `llm_proofreader_graph.py`, `crop_flagged_blocks.py`, `reinsert_blocks.py`, `split_and_process.py`, plus the `proofreader_memory.json` and `fixes/` accumulated state, per `https://github.com/PatientVibes/ai-agents/blob/master/docs/superpowers/specs/2026-05-09-kindle-pipeline-tool-migration-design.md`.

### Migrated changes (vs the live source as of 2026-05-09)

- **LLM backend swap:** langchain-ollama -> langchain-openai routed through OpenRouter. Default model: `openrouter:google/gemini-2.5-pro`. The `ChatOllama` symbol is preserved as an import alias for `ChatOpenAI` so call sites stay diff-minimal; Ollama-specific kwargs like `num_ctx` were dropped.
- **State path layout:** `__file__`-relative `.proofreader_state/` + `proofreader_memory.json` -> `platformdirs.user_state_dir("agent-tool-llm-proofreader")`, with `AGENT_TOOL_PROOFREADER_STATE_DIR` / `--state-dir` overrides. Helper module: `llm_proofreader.state_paths`.
- **Metrics:** `extract_ollama_metrics` (Ollama-only `response_metadata` with eval/prompt durations in nanoseconds) -> `extract_openrouter_metrics` (input/output/total tokens from `usage_metadata`) + Python-side wall-clock timing measured around `llm.invoke()` with `time.perf_counter`. The `tokens/sec` server-timing metric is dropped; an approximate rate is recomputed from output-tokens / wall-clock when needed. The old name remains as a backward-compat alias.
- **`load_secrets` import dropped** — replaced by `~/.config/agent-tool-llm-proofreader/env` pattern (mode 600), auto-sourced via `_load_env_file_if_present()` at the start of `main()` if `OPENROUTER_API_KEY` isn't already set. `configure_langsmith` is now a no-op stub; LangSmith tracing setup is migration-out-of-scope.
- **`books.json` lookup dropped** from the proofreader. The harness owns per-book metadata and passes it via flags. `book_key` is still accepted as a CLI arg but no longer triggers a `books.json` read.
- **`analyze_file` inlined** from `prose_quality.py` into `crop_flagged_blocks.py` — avoids a cross-repo Python dependency on `agent-tool-prose-quality`. The inlined helpers (`split_into_blocks`, `score_block`, `score_*`, `should_skip_block`, `is_bibliography_entry`, `is_index_entry`) match the source verbatim. Re-sync manually if the source changes.
- **`<think>`-tag stripping** retained as a no-op for non-thinking models. Marked as previous-model code path; safe to delete in a future cleanup.
