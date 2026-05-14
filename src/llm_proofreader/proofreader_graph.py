"""
LLM Proofreader (LangGraph version)
====================================
Ports the procedural llm_proofreader.py to a LangGraph StateGraph.

Why LangGraph:
- Built-in checkpointing via SqliteSaver (replaces manual .proofreader_state/)
- Native streaming of intermediate states
- Graph visualization (graph.get_graph().draw_mermaid())
- Interrupt-based human-in-the-loop support
- Automatic LangSmith tracing of node execution

Graph structure:
    start → check_skip → [skip branch] → next_chunk
                      → pre_analyze → llm_detect → guardrails → llm_verify → save_fixes → next_chunk
    next_chunk → [more chunks?] → check_skip
               → [done] → apply_fixes → END

Usage:
    python llm_proofreader_graph.py run "path/to/file.md" --book-key perrow
    python llm_proofreader_graph.py run "path/to/file.md" --book-key borges --max-chunks 5
    python llm_proofreader_graph.py draw      # export graph visualization
"""

import sys, io, os, re, json, argparse, time, hashlib, sqlite3
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dataclasses import asdict
from pathlib import Path

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Migrated 2026-05-09: load_secrets replaced by per-tool env file pattern.
# Set OPENROUTER_API_KEY in ~/.config/agent-tool-llm-proofreader/env (mode 600)
# and source it before invoking, or export directly. main() auto-sources via
# _load_env_file_if_present() if the env var isn't already set.
# LangSmith tracing config is migration-out-of-scope; configure_langsmith is a no-op.
def configure_langsmith(*args, **kwargs):
    pass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
# Migrated 2026-05-09: Ollama -> OpenRouter via langchain-openai.
# OPENROUTER_API_KEY required (set in env or ~/.config/agent-tool-llm-proofreader/env).
from langchain_openai import ChatOpenAI as ChatOllama  # noqa: N814 (alias preserves call sites)
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse existing helpers from the procedural version (now packaged as
# `llm_proofreader.proofreader`; the package shares its name with the module).
from llm_proofreader.proofreader import (
    chunk_by_sections,
    should_skip_chunk,
    build_system_prompt,
    pre_analyze_chunk,
    _parse_fixes,
    apply_guardrails,
    verify_fixes,
    preprocess_deterministic,
    load_known_patterns,
    extract_ollama_metrics,
    extract_openrouter_metrics,
    format_metrics,
    ProofreadState,
    DEFAULT_MODEL,
    MAX_RETRIES,
)
from llm_proofreader import proofreader as _lp


# ─── Graph State ─────────────────────────────────────────────────────────────

class InputState(BaseModel):
    """What the user provides when starting a run.
    Studio uses this to build the input form."""
    input_path: str = Field(
        ...,
        title="Input Path",
        description="Absolute path to the markdown file to proofread. Must exist and be readable.",
        examples=[
            r"C:\Users\chris\OneDrive\Documents\Reading\source\On Exactitude in Science — Jorge Luis Borges\index.md"
        ],
    )
    book_key: str = Field(
        default="",
        title="Book Key",
        description="Book key from books.json. Controls per-book vocabulary (e.g. Perrow's technical abbreviations) and LangSmith tracing privacy (private=true → tracing disabled). Leave empty for generic settings. Default: \"\" (no book).",
        examples=["borges", "perrow", "scott", "bowker", "hochschild", "suchman", "polanyi"],
    )
    model: str = Field(
        default="google/gemini-2.5-pro",
        title="Model",
        description="OpenRouter model id (langchain-openai routed to OpenRouter). Default: google/gemini-2.5-pro",
        examples=[
            "google/gemini-2.5-pro",
            "openrouter:anthropic/claude-sonnet-4.5",
            "openrouter:openai/gpt-4o",
        ],
    )
    timeout_per_chunk: int = Field(
        default=120,
        title="Timeout Per Chunk (seconds)",
        description="Hard cap on each LLM call. If exceeded, the chunk is skipped. Default: 120",
        examples=[60, 90, 120, 180],
    )
    max_chunks: int = Field(
        default=0,
        title="Max Chunks",
        description="Stop after processing N chunks, for quick testing. 0 = unlimited (process entire document). Default: 0",
        examples=[0, 1, 5, 10],
    )


class OutputState(BaseModel):
    """What the graph returns when complete."""
    fixes_applied: list = Field(default_factory=list)
    fixes_rejected: list = Field(default_factory=list)
    total_chunks: int = 0


class GraphState(BaseModel):
    """Full internal state flowing through the proofreader graph.
    Pydantic gives reliable JSON schemas — only input_path is required."""
    # User inputs
    input_path: str = Field(..., description="Path to markdown file")
    book_key: str = ""
    model: str = "google/gemini-2.5-pro"
    timeout_per_chunk: int = 120
    max_chunks: int = 0

    # Runtime state (populated by initialize node)
    content: str = ""
    original_content: str = ""
    chunks: list = Field(default_factory=list)
    total_chunks: int = 0
    current_chunk: int = 0
    fixes_found: list = Field(default_factory=list)
    fixes_applied: list = Field(default_factory=list)
    fixes_rejected: list = Field(default_factory=list)
    preprocess_fixes: list = Field(default_factory=list)
    session_patterns: dict = Field(default_factory=dict)
    known_patterns: dict = Field(default_factory=dict)
    book_config: dict = Field(default_factory=dict)

    # Per-chunk working state
    chunk_text: str = ""
    chunk_fixes: list = Field(default_factory=list)
    skip_reason: str = ""

    # Per-call LLM timing metrics (eval_count, eval_duration → tok/s)
    llm_metrics: list = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ─── Nodes ───────────────────────────────────────────────────────────────────

def initialize(state: GraphState) -> dict:
    """Load content, run deterministic preprocess, build chunks."""
    print(f"\n{'='*60}")
    print(f"LangGraph Proofreader")
    print(f"{'='*60}")

    with open(state.input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    known_patterns = load_known_patterns()

    # Deterministic pre-processing
    content, preprocess_fixes = preprocess_deterministic(content)

    # Migrated 2026-05-09: books.json lookup removed. Per-book metadata is now
    # passed explicitly by the harness via CLI flags (or omitted). LangSmith
    # tracing config is migration-out-of-scope; configure_langsmith is a no-op.
    book_config = {}
    books_json = None
    configure_langsmith(book_config or {'private': True})

    # Section-aware chunking
    chunks = chunk_by_sections(content)

    print(f"File:    {os.path.basename(state.input_path)}")
    print(f"Words:   {len(content.split())}")
    print(f"Chunks:  {len(chunks)}")
    print(f"Model:   {(state.model if state.model else DEFAULT_MODEL)}")
    print(f"Memory:  {len(known_patterns['false_positives'])} known false positives")
    print(f"Pre-processed: {len(preprocess_fixes)} artifacts")
    print(f"{'='*60}\n")

    return {
        'content': content,
        'original_content': content,
        'chunks': chunks,
        'total_chunks': len(chunks),
        'current_chunk': 0,
        'fixes_found': [],
        'fixes_applied': list(preprocess_fixes),
        'fixes_rejected': [],
        'preprocess_fixes': preprocess_fixes,
        'session_patterns': {'error_types_seen': {}},
        'known_patterns': known_patterns,
        'book_config': book_config,
    }


def load_chunk(state: GraphState) -> dict:
    """Pull the current chunk into working state."""
    idx = state.current_chunk
    total = state.total_chunks
    chunks = state.chunks

    if idx >= total:
        return {'chunk_text': '', 'skip_reason': 'out_of_range'}

    chunk = chunks[idx]
    print(f"Chunk {idx+1}/{total} (lines {chunk['start_line']}-{chunk['end_line']}, "
          f"{chunk['word_count']} words)")

    # Check skip conditions
    skip, reason = should_skip_chunk(chunk)
    if skip:
        print(f"  → SKIPPED: {reason}")
        return {'chunk_text': chunk['text'], 'skip_reason': reason, 'chunk_fixes': []}

    return {'chunk_text': chunk['text'], 'skip_reason': '', 'chunk_fixes': []}


def detect_fixes(state: GraphState) -> dict:
    """LLM detects potential OCR fixes in the current chunk."""
    import threading

    idx = state.current_chunk
    total = state.total_chunks
    chunk_text = state.chunk_text

    # Build dynamic system prompt
    pseudo_state = ProofreadState(
        input_file=state.input_path,
        input_hash='',
        model=(state.model if state.model else DEFAULT_MODEL),
        total_chunks=total,
        current_chunk=idx,
        session_patterns=state.session_patterns,
    )
    system_prompt = build_system_prompt(
        pseudo_state,
        state.known_patterns,
        book_key=state.book_key,
    )

    # Deterministic pre-analysis hints
    hints = pre_analyze_chunk(chunk_text, state.known_patterns)
    hints_text = ""
    if hints:
        hints_text = "\n\nPre-analysis detected these potential issues:\n" + "\n".join(
            f"- {h}" for h in hints[:10]
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Proofread chunk {idx+1}/{total}.{hints_text}\n\n"
                     f"Text:\n```\n{chunk_text}\n```\n\n"
                     f"Output ONLY a JSON array of fixes. If clean, output []"),
    ]

    # Migrated 2026-05-09: ChatOllama -> ChatOpenAI (alias) routed through OpenRouter.
    # `num_ctx` was Ollama-specific (server-side context window); not passed here.
    llm = ChatOllama(
        model=(state.model if state.model else DEFAULT_MODEL),
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    timeout = (state.timeout_per_chunk if state.timeout_per_chunk else 120)
    response = None
    detect_elapsed = None

    def invoke_with_result(container):
        # Migrated 2026-05-09: capture wall-clock around invoke() since OpenRouter
        # doesn't expose Ollama-style server timings.
        try:
            _t0 = time.perf_counter()
            container['response'] = llm.invoke(messages)
            container['elapsed'] = time.perf_counter() - _t0
        except Exception as e:
            container['error'] = e

    for attempt in range(MAX_RETRIES + 1):
        container = {}
        t = threading.Thread(target=invoke_with_result, args=(container,))
        t.daemon = True
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            print(f"  → LLM timeout ({timeout}s), skipping chunk")
            return {'chunk_fixes': []}

        if 'error' in container:
            print(f"  → LLM error (attempt {attempt}): {container['error']}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return {'chunk_fixes': []}

        response = container.get('response')
        if response:
            detect_elapsed = container.get('elapsed')
            break

    if response is None:
        return {'chunk_fixes': []}

    content = response.content if hasattr(response, 'content') else ""
    fixes = _parse_fixes(content, idx + 1)
    if fixes:
        print(f"  → {len(fixes)} potential fixes found")

    new_metrics = []
    detect_m = extract_openrouter_metrics(
        response,
        label=f"detect/chunk{idx+1}",
        elapsed_seconds=detect_elapsed,
    )
    if detect_m:
        print(f"  metrics: {format_metrics(detect_m)}")
        new_metrics.append(detect_m)

    return {
        'chunk_fixes': fixes,
        'llm_metrics': state.llm_metrics + new_metrics,
    }


def guardrail_filter(state: GraphState) -> dict:
    """Apply guardrails (no-ops, length checks, caps)."""
    fixes = state.chunk_fixes
    if not fixes:
        return {}
    safe = apply_guardrails(fixes, state.chunk_text)
    return {'chunk_fixes': safe}


def verify(state: GraphState) -> dict:
    """Second LLM reviews proposed fixes."""
    fixes = state.chunk_fixes
    if not fixes:
        return {}

    print(f"  → Verifying {len(fixes)} fixes...")
    _lp._LAST_CALL_METRICS.clear()
    verified = verify_fixes(fixes, state.chunk_text)
    drained = list(_lp._LAST_CALL_METRICS)
    _lp._LAST_CALL_METRICS.clear()

    confirmed = [f for f in verified if f.get('status') == 'confirmed']
    rejected = [f for f in verified if f.get('status') == 'rejected']
    print(f"  → {len(confirmed)} confirmed, {len(rejected)} rejected")

    # Update session patterns
    patterns = dict(state.session_patterns)
    error_types = dict(patterns.get('error_types_seen', {}))
    for fix in confirmed:
        reason = fix.get('reason', 'unknown')
        error_types[reason] = error_types.get(reason, 0) + 1
    patterns['error_types_seen'] = error_types

    return {
        'fixes_found': state.fixes_found + confirmed,
        'fixes_rejected': state.fixes_rejected + rejected,
        'session_patterns': patterns,
        'llm_metrics': state.llm_metrics + drained,
    }


def next_chunk(state: GraphState) -> dict:
    """Advance to next chunk."""
    return {'current_chunk': state.current_chunk + 1}


def apply_all_fixes(state: GraphState) -> dict:
    """Apply all confirmed fixes to content and write output."""
    print(f"\n{'='*60}")
    print(f"Applying {len(state.fixes_found)} confirmed fixes")
    print(f"{'='*60}\n")

    modified = state.content
    applied = []
    for fix in state.fixes_found:
        original = fix.get('original', '')
        corrected = fix.get('corrected', '')
        if original and original in modified:
            modified = modified.replace(original, corrected, 1)
            applied.append(fix)

    # Normalize multiple blank lines
    modified = re.sub(r'\n{3,}', '\n\n', modified)

    # Write output
    with open(state.input_path, 'w', encoding='utf-8') as f:
        f.write(modified)

    total_applied = len(applied) + len(state.preprocess_fixes)
    print(f"Applied:  {total_applied} "
          f"({len(state.preprocess_fixes)} deterministic + {len(applied)} LLM)")
    print(f"Rejected: {len(state.fixes_rejected)}")
    print(f"Output:   {state.input_path}")

    # LLM timing summary
    # Migrated 2026-05-09: switched to OpenRouter token counts (input_tokens /
    # output_tokens / total_tokens) plus caller-supplied wall-clock elapsed_seconds.
    # The old Ollama-only tok/s rate is recomputed from elapsed when available.
    metrics = state.llm_metrics
    if metrics:
        total_output_tokens = sum(m.get('output_tokens', 0) for m in metrics)
        total_input_tokens = sum(m.get('input_tokens', 0) for m in metrics)
        total_wall = sum(m.get('elapsed_seconds') or 0 for m in metrics)
        avg_tok_s = (total_output_tokens / total_wall) if total_wall else 0
        by_label = {}
        for m in metrics:
            bucket = m['label'].split('/')[0]
            by_label.setdefault(bucket, []).append(m)

        print(f"\nLLM timing ({len(metrics)} calls):")
        print(f"  input tokens:  {total_input_tokens}")
        print(f"  output tokens: {total_output_tokens}")
        print(f"  avg gen speed: {avg_tok_s:.1f} tok/s")
        print(f"  wall time:     {total_wall:.1f}s")
        for label, items in by_label.items():
            g = sum(x.get('output_tokens', 0) for x in items)
            s = sum((x.get('elapsed_seconds') or 0) for x in items)
            rate = g / s if s else 0
            print(f"    {label}: {len(items)} calls, {g} gen tok, {rate:.1f} tok/s avg")

        # Persist as JSONL for later analysis
        # Migrated 2026-05-09: state moved out of __file__-relative path.
        from llm_proofreader.state_paths import graph_checkpoint_dir
        metrics_dir = graph_checkpoint_dir()
        metrics_dir.mkdir(exist_ok=True)
        file_hash = hashlib.md5(state.input_path.encode()).hexdigest()[:12]
        metrics_path = metrics_dir / f"metrics_{file_hash}.jsonl"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")
        print(f"  saved: {metrics_path}")

    print(f"{'='*60}\n")

    return {
        'content': modified,
        'fixes_applied': state.fixes_applied + applied,
    }


# ─── Conditional Edges ──────────────────────────────────────────────────────

def route_after_load(state: GraphState) -> Literal['detect_fixes', 'next_chunk']:
    """Skip LLM if chunk was filtered, otherwise proceed to detection."""
    if state.skip_reason:
        return 'next_chunk'
    return 'detect_fixes'


def route_after_next(state: GraphState) -> Literal['load_chunk', 'apply_all_fixes']:
    """Loop back to load_chunk if more to process, else finalize."""
    idx = state.current_chunk
    total = state.total_chunks
    max_chunks = state.max_chunks or total

    if idx >= total or idx >= max_chunks:
        return 'apply_all_fixes'
    return 'load_chunk'


def route_after_detect(state: GraphState) -> Literal['guardrail_filter', 'next_chunk']:
    """Skip verify if no fixes found."""
    if not state.chunk_fixes:
        return 'next_chunk'
    return 'guardrail_filter'


def route_after_guardrails(state: GraphState) -> Literal['verify', 'next_chunk']:
    """Skip verify if guardrails removed all fixes."""
    if not state.chunk_fixes:
        return 'next_chunk'
    return 'verify'


# ─── Build Graph ─────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the proofreader StateGraph.

    input=InputState tells Studio which fields to prompt the user for.
    output=OutputState declares the public result schema.
    state_schema=GraphState is the full internal state.
    """
    graph = StateGraph(GraphState, input=InputState, output=OutputState)

    # Add nodes
    graph.add_node('initialize', initialize)
    graph.add_node('load_chunk', load_chunk)
    graph.add_node('detect_fixes', detect_fixes)
    graph.add_node('guardrail_filter', guardrail_filter)
    graph.add_node('verify', verify)
    graph.add_node('next_chunk', next_chunk)
    graph.add_node('apply_all_fixes', apply_all_fixes)

    # Edges
    graph.set_entry_point('initialize')
    graph.add_edge('initialize', 'load_chunk')

    graph.add_conditional_edges('load_chunk', route_after_load, {
        'detect_fixes': 'detect_fixes',
        'next_chunk': 'next_chunk',
    })

    graph.add_conditional_edges('detect_fixes', route_after_detect, {
        'guardrail_filter': 'guardrail_filter',
        'next_chunk': 'next_chunk',
    })

    graph.add_conditional_edges('guardrail_filter', route_after_guardrails, {
        'verify': 'verify',
        'next_chunk': 'next_chunk',
    })

    graph.add_edge('verify', 'next_chunk')

    graph.add_conditional_edges('next_chunk', route_after_next, {
        'load_chunk': 'load_chunk',
        'apply_all_fixes': 'apply_all_fixes',
    })

    graph.add_edge('apply_all_fixes', END)

    return graph


# ─── Module-level compiled graph for LangGraph Studio ───────────────────────
# Studio imports this module and looks for a compiled graph named `graph`.
# It uses its own checkpointer, so we compile without one here.
# The CLI runner below uses its own SqliteSaver-backed compile instead.
graph = build_graph().compile()


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_graph(input_path, book_key=None, model=DEFAULT_MODEL,
              timeout_per_chunk=120, max_chunks=None, thread_id=None):
    """Execute the proofreader graph with SqliteSaver checkpointing."""
    # Checkpoint database
    # Migrated 2026-05-09: state moved out of __file__-relative path into
    # platformdirs.user_state_dir("agent-tool-llm-proofreader")/graph_checkpoints.
    from llm_proofreader.state_paths import graph_checkpoint_dir
    checkpoint_dir = graph_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / "checkpoints.db"

    # Use thread_id based on input file hash (stable across runs)
    if thread_id is None:
        thread_id = hashlib.md5(input_path.encode()).hexdigest()[:12]

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    memory = SqliteSaver(conn)

    graph = build_graph()
    app = graph.compile(checkpointer=memory)

    initial_state = {
        'input_path': input_path,
        'book_key': book_key or '',
        'model': model,
        'timeout_per_chunk': timeout_per_chunk,
        'max_chunks': max_chunks or 10**9,  # effectively unlimited
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10000}

    print(f"Thread ID: {thread_id}")
    print(f"Checkpoint DB: {db_path}")

    final_state = app.invoke(initial_state, config=config)

    conn.close()
    return final_state


def draw_graph(output_path=None):
    """Export the graph as a mermaid diagram."""
    graph = build_graph()
    app = graph.compile()
    mermaid = app.get_graph().draw_mermaid()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("```mermaid\n")
            f.write(mermaid)
            f.write("\n```\n")
        print(f"Graph diagram written to: {output_path}")
    else:
        print(mermaid)


# ─── CLI ─────────────────────────────────────────────────────────────────────

# POSIX env-var name: [A-Za-z_][A-Za-z0-9_]*. ASCII-only — see PR #1
# follow-up for the Unicode-vs-ASCII fix.
_POSIX_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_env_file_if_present():
    """Load ~/.config/agent-tool-llm-proofreader/env into os.environ if OPENROUTER_API_KEY is unset.

    Minimal env-file reader for `KEY=value` and `export KEY=value` lines. Handles
    a single pair of surrounding quotes, full-line `#` comments, and blank lines.
    This is NOT a `source <file>` replacement — inline comments, interpolation,
    multi-line values, and duplicate-key last-wins are unsupported.

    Invariants:
      - No-op if OPENROUTER_API_KEY is already set OR the file doesn't exist.
      - Pre-existing env vars are NEVER overwritten (shell-exported values
        always win, for ALL keys — not just OPENROUTER_API_KEY).
      - **First-wins within the file** for duplicate keys.
      - Bad lines are silently skipped. Downstream KeyError on the missing key
        is the actionable error.

    Migration note (2026-05-09): replaces the old load_secrets / .env pattern.
    Bug-fix backport 2026-05-14: previously overwrote pre-set non-required env
    vars and stripped all chained quote chars; now no-clobber for all keys and
    strips ONE matching pair. See PatientVibes/agent-skills#1 for the canonical
    loader documented in the config-env-loading skill.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    env_path = Path.home() / ".config" / "agent-tool-llm-proofreader" / "env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        # POSIX env-var name (ASCII only). Skip anything else.
        if not _POSIX_ENV_NAME.match(k):
            continue
        if k in os.environ:
            # Don't clobber pre-set env vars — caller's shell wins.
            continue
        v = v.strip()
        # Strip ONE pair of matching surrounding quotes (not all of them).
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        os.environ[k] = v


def main():
    _load_env_file_if_present()
    parser = argparse.ArgumentParser(description="LangGraph-based LLM Proofreader")
    parser.add_argument("command", choices=["run", "draw"])
    parser.add_argument("input", nargs='?', help="Input markdown file")
    parser.add_argument("--book-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--thread-id", default=None,
                        help="Checkpoint thread ID (defaults to hash of input path)")
    parser.add_argument("--output", help="Output path (for draw command)")
    parser.add_argument("--state-dir", default=None,
                        help="Override the state directory (default: platformdirs.user_state_dir)")
    args = parser.parse_args()

    if args.state_dir:
        os.environ["AGENT_TOOL_PROOFREADER_STATE_DIR"] = args.state_dir

    if args.command == "draw":
        draw_graph(args.output)
    elif args.command == "run":
        if not args.input:
            print("ERROR: input file required for 'run'")
            return
        run_graph(
            args.input,
            book_key=args.book_key,
            model=args.model,
            timeout_per_chunk=args.timeout,
            max_chunks=args.max_chunks,
            thread_id=args.thread_id,
        )


if __name__ == "__main__":
    main()
