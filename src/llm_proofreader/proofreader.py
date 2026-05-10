"""
LLM Proofreader Agent Harness
=============================
A proper agent harness for proofreading OCR'd markdown, built against
the 12-component agent harness methodology.

Components implemented:
  1. Orchestration Loop — TAO cycle per chunk (Think → Act → Observe)
  2. Tools — dictionary lookup, pattern memory, context retrieval, fix verification
  3. Memory — short-term (session patterns), long-term (known false positives file)
  4. Context Management — section-aware chunking, surrounding context on demand
  5. Prompt Construction — dynamic, adapts based on patterns found so far
  6. Output Parsing — structured JSON with retry on failure
  7. State Management — checkpoint per chunk, resume on crash
  8. Error Handling — errors fed back to model as observations
  9. Guardrails — protect quotes/citations, cap changes per chunk, proper noun shield
  10. Verification Loops — second pass reviews proposed fixes before applying
  11. Subagent Orchestration — verifier agent reviews fixer agent's proposals
  12. Lifecycle — progress tracking, stats, cleanup

Usage:
    python llm_proofreader.py run "path/to/file.md" --output cleaned.md
    python llm_proofreader.py run "path/to/file.md" --resume  # resume from checkpoint
    python llm_proofreader.py status "path/to/file.md"        # check progress
"""

import sys, io, os, re, json, argparse, time, hashlib
from typing import Optional
from dataclasses import dataclass, field, asdict

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Migrated 2026-05-09: Ollama -> OpenRouter via langchain-openai.
# OPENROUTER_API_KEY required (set in env or ~/.config/agent-tool-llm-proofreader/env).
from langchain_openai import ChatOpenAI as ChatOllama  # noqa: N814 (alias preserves call sites)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


def extract_openrouter_metrics(response, label="", elapsed_seconds=None):
    """Pull token counts from a langchain-openai response (used via OpenRouter).

    OpenRouter / OpenAI-compatible endpoints expose ``response.usage_metadata`` with
    ``input_tokens``, ``output_tokens``, ``total_tokens`` (langchain-core normalises
    OpenAI's ``prompt_tokens``/``completion_tokens`` into this shape). They do NOT
    expose Ollama-style server-side timing, so wall-clock duration is caller-supplied
    via ``elapsed_seconds`` (measured around ``llm.invoke(...)`` with ``time.perf_counter``).

    Returns a dict with: label, model, input_tokens, output_tokens, total_tokens,
    elapsed_seconds. Returns ``None`` if usage metadata is missing.
    """
    usage = getattr(response, 'usage_metadata', None)
    if not usage:
        return None
    meta = getattr(response, 'response_metadata', None) or {}
    return {
        'label': label,
        'model': meta.get('model_name') or meta.get('model', ''),
        'input_tokens': usage.get('input_tokens', 0),
        'output_tokens': usage.get('output_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
        'elapsed_seconds': elapsed_seconds,
    }


# Backward-compatible alias for any internal call sites (and proofreader_graph.py
# which imports the old name from this module).
extract_ollama_metrics = extract_openrouter_metrics


_LAST_CALL_METRICS = []  # module-level sink: callers drain after each call batch


def format_metrics(m):
    """One-line summary of extract_openrouter_metrics output."""
    if not m:
        return "(no metrics)"
    elapsed = m.get('elapsed_seconds')
    elapsed_part = f", {elapsed:.1f}s" if elapsed is not None else ""
    return (
        f"[{m.get('model', '')}] "
        f"input {m.get('input_tokens', 0)}tok, "
        f"output {m.get('output_tokens', 0)}tok, "
        f"total {m.get('total_tokens', 0)}tok"
        f"{elapsed_part}"
    )

# Migrated 2026-05-09: load_secrets replaced by per-tool env file pattern.
# Set OPENROUTER_API_KEY in ~/.config/agent-tool-llm-proofreader/env (mode 600)
# and source it before invoking, or export directly. main() auto-sources via
# _load_env_file_if_present() if the env var isn't already set.
# LangSmith tracing config is migration-out-of-scope; configure_langsmith is a no-op.
def configure_langsmith(*args, **kwargs):
    pass

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_MODEL = "google/gemini-2.5-pro"
VERIFIER_MODEL = "google/gemini-2.5-pro"  # Can use a different model for verification
MAX_FIXES_PER_CHUNK = 15  # Guardrail: if model finds more, something's wrong
MAX_RETRIES = 2

# Migrated 2026-05-09: state files moved out of __file__-relative paths into
# platformdirs.user_state_dir("agent-tool-llm-proofreader"). Override via the
# AGENT_TOOL_PROOFREADER_STATE_DIR env var or --state-dir CLI flag.
from llm_proofreader.state_paths import checkpoint_dir as _checkpoint_dir
CHECKPOINT_DIR = str(_checkpoint_dir())


# ─── Dictionary / Word List ──────────────────────────────────────────────────

_WORD_SET = None

def _load_word_set():
    global _WORD_SET
    if _WORD_SET is None:
        import nltk
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        from nltk.corpus import words
        _WORD_SET = set(w.lower() for w in words.words())
        # Add common words nltk misses
        _WORD_SET.update(['covid', 'internet', 'email', 'website', 'online',
                          'metadata', 'dataset', 'workflow', 'checkbox'])
    return _WORD_SET


# ─── Long-Term Memory: Known Patterns ────────────────────────────────────────

# Migrated 2026-05-09: memory.json now lives at platformdirs.user_state_dir(...).
from llm_proofreader.state_paths import memory_path as _memory_path
KNOWN_PATTERNS_FILE = str(_memory_path())

def load_known_patterns():
    """Load accumulated knowledge from previous sessions."""
    if os.path.exists(KNOWN_PATTERNS_FILE):
        with open(KNOWN_PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "false_positives": [
            # Mc/Mac/De surnames — universal false positives
            "McCleary", "McCloskey", "MacKenzie", "McKeown", "McGoogan",
            "McFarland", "McGraw", "McDowell", "MacDonald", "DeKlerk",
            "McDougall", "MacArthur", "DeVault", "DeCoster", "McCready",
            "McDermot", "McCarthy", "MacMillan", "McGrath", "McCann",
            "MacFadden", "DeLisser", "DiMatteo", "McKim", "MacKay",
            "McCorduck", "McDermott", "JoAnne",
        ],
        "known_corrections": {},  # "original" -> "corrected" mappings seen before
        "book_specific": {},  # per-book patterns
    }

def save_known_patterns(patterns):
    """Persist accumulated knowledge for future sessions."""
    with open(KNOWN_PATTERNS_FILE, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)


# ─── State Management ────────────────────────────────────────────────────────

@dataclass
class ProofreadState:
    """Checkpoint state for resume capability."""
    input_file: str
    input_hash: str  # MD5 of input file for change detection
    model: str
    total_chunks: int = 0
    current_chunk: int = 0
    fixes_found: list = field(default_factory=list)
    fixes_applied: list = field(default_factory=list)
    fixes_rejected: list = field(default_factory=list)
    session_patterns: dict = field(default_factory=lambda: {
        "error_types_seen": {},  # "joined_words": 5, "split_number": 3, etc.
        "false_positives_this_session": [],
    })
    started_at: str = ""
    last_checkpoint: str = ""
    status: str = "initialized"  # initialized, running, verifying, complete

    def checkpoint_path(self):
        name = hashlib.md5(self.input_file.encode()).hexdigest()[:12]
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        return os.path.join(CHECKPOINT_DIR, f"{name}.json")

    def save(self):
        self.last_checkpoint = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.checkpoint_path(), 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, input_file):
        name = hashlib.md5(input_file.encode()).hexdigest()[:12]
        path = os.path.join(CHECKPOINT_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        return None

    @classmethod
    def fresh(cls, input_file, model):
        with open(input_file, 'r', encoding='utf-8') as f:
            content_hash = hashlib.md5(f.read().encode()).hexdigest()
        return cls(
            input_file=input_file,
            input_hash=content_hash,
            model=model,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


# ─── Tools (Component 2) ────────────────────────────────────────────────────

@tool
def spell_check(word: str) -> str:
    """Check if a word exists in the English dictionary. Returns 'valid', 'invalid', or 'proper_noun'."""
    if not word:
        return "invalid"
    words = _load_word_set()
    # Check as-is
    if word.lower() in words:
        return "valid"
    # Check if it's a known proper noun pattern (Mc/Mac/De/Di + Capital)
    if re.match(r'^(Mc|Mac|De|Di|Le|La|Van|Von)[A-Z]', word):
        return "proper_noun"
    # Check without common suffixes
    for suffix in ['s', 'ed', 'ing', 'ly', 'er', 'est', 'tion', 'ment']:
        stem = word.lower().rstrip(suffix) if word.lower().endswith(suffix) else None
        if stem and stem in words:
            return "valid"
    return "invalid"


@tool
def lookup_known_pattern(text: str) -> str:
    """Check if this text is a known false positive or has a known correction.
    Returns JSON with status and any known correction."""
    patterns = load_known_patterns()
    # Check false positives
    for fp in patterns["false_positives"]:
        if fp.lower() in text.lower():
            return json.dumps({"status": "known_false_positive", "match": fp})
    # Check known corrections
    if text in patterns["known_corrections"]:
        return json.dumps({
            "status": "known_correction",
            "corrected": patterns["known_corrections"][text]
        })
    return json.dumps({"status": "unknown"})


@tool
def record_pattern(pattern: str, pattern_type: str, correction: str = "") -> str:
    """Record a discovered pattern for future reference.
    pattern_type: 'false_positive' or 'correction'"""
    patterns = load_known_patterns()
    if pattern_type == "false_positive":
        if pattern not in patterns["false_positives"]:
            patterns["false_positives"].append(pattern)
            save_known_patterns(patterns)
            return f"Recorded false positive: {pattern}"
    elif pattern_type == "correction" and correction:
        patterns["known_corrections"][pattern] = correction
        save_known_patterns(patterns)
        return f"Recorded correction: {pattern} → {correction}"
    return "No action taken"


@tool
def check_is_quoted(text: str, context: str) -> str:
    """Check if text appears inside a quotation, citation, or blockquote.
    These should NOT be modified."""
    # Check for surrounding quotes
    for quote_char in ['"', "'", '\u201c', '\u201d', '\u2018', '\u2019']:
        idx = context.find(text)
        if idx > 0 and (context[idx-1] == quote_char or
                        any(context[max(0,idx-20):idx].count(q) % 2 == 1
                            for q in ['"', '\u201c'])):
            return "inside_quote"
    # Check for blockquote marker
    lines = context.split('\n')
    for line in lines:
        if text in line and line.strip().startswith('>'):
            return "inside_blockquote"
    return "not_quoted"


# ─── LaTeX/Encoding Artifact Reference ────────────────────────────────────────

# Common LaTeX artifacts from OCR and their plain-text equivalents
LATEX_ARTIFACT_PATTERNS = {
    # \mbox{text} → text (used for speaker labels in dialogue)
    r'\$\\mbox\{([^}]+)\}\$': lambda m: m.group(1).upper().rstrip(':') + ':',
    r'\\mbox\{([^}]+)\}': lambda m: m.group(1),
    # \textbf, \textit, \emph
    r'\\textbf\{([^}]+)\}': lambda m: m.group(1),
    r'\\textit\{([^}]+)\}': lambda m: m.group(1),
    r'\\emph\{([^}]+)\}': lambda m: m.group(1),
    # Common math-mode leaks
    r'\$([^$]{1,30})\$': lambda m: m.group(1),  # short inline $math$
    # HTML artifacts
    r'<br\s*/?>': lambda m: '\n',
    r'</?p>': lambda m: '\n',
    r'&amp;': lambda m: '&',
    r'&lt;': lambda m: '<',
    r'&gt;': lambda m: '>',
    r'&nbsp;': lambda m: ' ',
}


@tool
def fix_latex_artifact(text: str) -> str:
    """Clean a LaTeX or HTML artifact and return the plain-text equivalent.
    Use this when you find LaTeX commands or HTML tags in the OCR text.
    Returns the cleaned text, or 'no_match' if no artifact pattern found."""
    import re as _re
    for pattern, replacer in LATEX_ARTIFACT_PATTERNS.items():
        match = _re.search(pattern, text)
        if match:
            cleaned = _re.sub(pattern, replacer, text)
            return json.dumps({"original": text, "cleaned": cleaned, "pattern": pattern})
    return json.dumps({"status": "no_match", "text": text})


# ─── Context-Aware Chunking (Component 4) ────────────────────────────────────

def chunk_by_sections(text, max_words=2000):
    """Split text at section boundaries (## headers) rather than arbitrary word counts."""
    lines = text.split('\n')
    chunks = []
    current_lines = []
    current_words = 0
    chunk_start = 0

    for i, line in enumerate(lines):
        is_heading = re.match(r'^#{1,4}\s', line.strip())
        words_in_line = len(line.split())

        # Split at heading boundaries if chunk is already substantial
        if is_heading and current_words > 500:
            chunks.append({
                'text': '\n'.join(current_lines),
                'start_line': chunk_start,
                'end_line': i - 1,
                'word_count': current_words,
            })
            current_lines = [line]
            current_words = words_in_line
            chunk_start = i
        # Split at max size
        elif current_words + words_in_line > max_words and current_words > 200:
            chunks.append({
                'text': '\n'.join(current_lines),
                'start_line': chunk_start,
                'end_line': i - 1,
                'word_count': current_words,
            })
            current_lines = [line]
            current_words = words_in_line
            chunk_start = i
        else:
            current_lines.append(line)
            current_words += words_in_line

    if current_lines:
        chunks.append({
            'text': '\n'.join(current_lines),
            'start_line': chunk_start,
            'end_line': len(lines) - 1,
            'word_count': current_words,
        })

    return chunks


def should_skip_chunk(chunk):
    """Return (skip_bool, reason) for whether to skip LLM proofreading of a chunk.

    Skip reasons:
    - High pipe density (markdown table) — the LLM can't repair tables
    - Bibliography/reference list — legitimate structure that looks like non-prose
    - Index-like content
    - Very short chunks
    """
    text = chunk['text']

    if len(text.strip()) < 100:
        return True, "too short"

    # Pipe density (tables)
    pipes = text.count('|')
    if pipes > 20 and pipes / len(text) > 0.03:
        return True, f"table (pipe density {pipes/len(text):.3f})"

    # Table separator rows
    if re.search(r'\|[\s\-:]+\|[\s\-:]+\|', text):
        separator_rows = len(re.findall(r'\n\|[\s\-:]+\|', text))
        if separator_rows >= 1:
            return True, f"markdown table ({separator_rows} separator rows)"

    # Bibliography: many lines starting with author patterns
    lines = text.split('\n')
    bib_lines = sum(1 for line in lines if re.match(r'^[A-Z][a-z]+,\s+[A-Z]', line.strip()))
    if bib_lines > 5 and bib_lines / max(1, len(lines)) > 0.3:
        return True, f"bibliography ({bib_lines} entries)"

    # High number-to-word ratio (index)
    words = re.findall(r'\b[A-Za-z]+\b', text)
    numbers = re.findall(r'\b\d+\b', text)
    if len(words) > 20 and len(numbers) / len(words) > 0.5:
        return True, f"index-like ({len(numbers)} numbers / {len(words)} words)"

    return False, None


# ─── Dynamic Prompt Construction (Component 5) ───────────────────────────────

def build_system_prompt(state: ProofreadState, known_patterns: dict, book_key: str = None):
    """Build a dynamic system prompt based on what we've learned so far.

    If book_key is provided, loads book-specific vocabulary from known_patterns."""
    base = """You are a precise OCR proofreader agent. Your job is to find OCR errors in text that was extracted from scanned books.

## Your Mission
Find OCR artifacts and extraction errors ONLY. You must NOT rewrite, rephrase, or "improve" the text.

## What to Fix (CONFIRM these)
- **Joined words missing a space**: "SeeingLike" → "Seeing Like"
- **Split words/numbers**: "M IT" → "MIT", "1 997" → "1997", "I nstitute" → "Institute"
  - Yes, even when the result is a proper noun — "Institute" with an OCR-inserted space IS an error
- **Character substitutions**:
  - "vVould" → "Would" (OCR misread W as vV)
  - "satisfY" → "satisfy" (stray uppercase)
  - "classifYing" → "classifying" (uppercase mid-word)
  - "rn" misread as "m", "cl" as "d"
- **OCR strikethrough artifacts**: "s~~ddenly" → "suddenly", "elir~~inates" → "eliminates"
- **LaTeX leaks**: "$\\mbox{owen:}$" → "OWEN:" (these are ALWAYS errors)
- **Stray HTML**: isolated `<br>`, `<p>`, `&amp;`, `&lt;`, `&gt;`

## What to NEVER Fix (REJECT these)
- **Quoted text, citations, or blockquotes** — leave entirely alone
- **Text inside markdown tables** (rows with `|` separators)
- **Proper nouns with Mc/Mac/De/Di/Le/La/Van/Von/Du prefixes** — these are legitimate surnames (McCleary, MacArthur, DeKlerk, etc.)
- **CamelCase technical terms, product names, or brand names** (BasicBooks, McGraw-Hill, JavaScript, iPhone)
- **British vs American spelling** — preserve as-is
- **Style choices**:
  - Hyphenation ("razor-sharp" vs "razor sharp")
  - Punctuation choices (periods vs commas unless clearly garbled)
- **The author's word choices** — you are NOT an editor

## Common False Positives (DO NOT flag)
1. Running headers — chapter titles repeat at top of pages (already handled elsewhere, ignore)
2. Figure captions: "Figure 3.2: Foo bar baz" — correct format, leave alone
3. Reference list entries with authors, years, volume numbers
4. Section numbers like "2.1 Introduction"
5. Footnote references: `<super>*</super>` or `<super>12</super>` — correct as-is

## Output Format
Output ONLY a valid JSON array of fixes. No explanation, no markdown, no preamble:

```
[
  {"original": "SeeingLike", "corrected": "Seeing Like", "reason": "joined words", "confidence": "high"},
  {"original": "1 997", "corrected": "1997", "reason": "split number", "confidence": "high"}
]
```

If no issues found, output: `[]`

Do NOT propose any fix where `original == corrected` (no-ops are forbidden)."""

    # Add session context
    if state.session_patterns["error_types_seen"]:
        patterns_summary = ", ".join(
            f"{k}: {v} found" for k, v in state.session_patterns["error_types_seen"].items()
        )
        base += f"\n\n## Patterns Found This Session\n{patterns_summary}"

    # Add known false positives (global)
    fps = known_patterns.get("false_positives", [])
    if fps:
        sample = fps[:50]  # Show up to 50 to cover most surnames
        base += f"\n\n## Known False Positives — DO NOT flag these as errors\n"
        base += ', '.join(sample)

    # Add known corrections for context
    corrections = known_patterns.get("known_corrections", {})
    if corrections:
        sample = list(corrections.items())[:15]
        base += f"\n\n## Known Correction Patterns\n"
        for original, corrected in sample:
            base += f'- "{original}" → "{corrected}"\n'

    # Add book-specific vocabulary
    if book_key and "book_specific" in known_patterns:
        book_data = known_patterns["book_specific"].get(book_key, {})
        if book_data:
            base += f"\n\n## Book-Specific Vocabulary — DO NOT flag these\n"
            for category, terms in book_data.items():
                if terms:
                    base += f"**{category}**: {', '.join(terms)}\n"

    return base


# ─── Orchestration Loop (Component 1) ────────────────────────────────────────

def pre_analyze_chunk(chunk_text, known_patterns):
    """Deterministic pre-analysis: run spell checks and pattern lookups
    BEFORE the LLM sees the text. Feed results as context to the model."""
    hints = []
    words = re.findall(r'\b[A-Za-z]+\b', chunk_text)

    # Check for suspicious joined words (lowercase+Uppercase mid-word)
    for match in re.finditer(r'([a-z])([A-Z][a-z])', chunk_text):
        word_match = re.search(r'\S*' + re.escape(match.group()) + r'\S*', chunk_text)
        word = word_match.group() if word_match else match.group()
        # Check against known false positives
        is_fp = any(fp.lower() in word.lower() for fp in known_patterns.get("false_positives", []))
        if not is_fp:
            hints.append(f"Possible joined word: '{word}' (check if space needed)")

    # Check for split numbers
    for match in re.finditer(r'\b(\d+)\s(\d{3})\b', chunk_text):
        hints.append(f"Possible split number: '{match.group()}' → '{match.group(1)}{match.group(2)}'")

    # Check for split words (single letter + space + lowercase)
    for match in re.finditer(r'\b([A-Z])\s([a-z]{3,})\b', chunk_text):
        combined = match.group(1) + match.group(2)
        if combined.lower() in _load_word_set():
            hints.append(f"Possible split word: '{match.group()}' → '{combined}'")

    return hints


def proofread_chunk_simple(llm, chunk_text, chunk_num, total_chunks, state, known_patterns,
                            book_key=None, timeout_seconds=120):
    """Deterministic pre-analysis + single LLM call with per-chunk timeout.

    book_key: enables per-book vocabulary in system prompt
    timeout_seconds: hard cap on LLM call (default 2 min)
    """
    system_prompt = build_system_prompt(state, known_patterns, book_key=book_key)

    # Pre-analyze deterministically
    hints = pre_analyze_chunk(chunk_text, known_patterns)
    hints_text = ""
    if hints:
        hints_text = "\n\nPre-analysis detected these potential issues (verify before including):\n"
        hints_text += "\n".join(f"- {h}" for h in hints[:10])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Proofread chunk {chunk_num}/{total_chunks}.{hints_text}\n\n"
                     f"Text:\n```\n{chunk_text}\n```\n\n"
                     f"Output ONLY a JSON array of fixes. If clean, output []"),
    ]

    response = None
    import threading

    def invoke_llm(result_container):
        try:
            result_container['response'] = llm.invoke(messages)
        except Exception as e:
            result_container['error'] = e

    for attempt in range(MAX_RETRIES + 1):
        container = {}
        thread = threading.Thread(target=invoke_llm, args=(container,))
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            print(f"    LLM timeout ({timeout_seconds}s) on attempt {attempt}, skipping chunk")
            return []

        if 'error' in container:
            print(f"    LLM error attempt {attempt}: {container['error']}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            else:
                return []

        response = container.get('response')
        if response:
            break

    if response is None:
        print(f"    No response for chunk {chunk_num}")
        return []

    content = response.content if hasattr(response, 'content') else ""
    return _parse_fixes(content, chunk_num)


def _parse_fixes(content, chunk_num):
    """Extract JSON fixes from model response, handling various formats."""
    if not content:
        return []

    # Handle qwen3 <think> tags
    if '<think>' in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Try direct JSON parse
    try:
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            fixes = json.loads(json_match.group())
            if isinstance(fixes, list):
                return fixes
    except json.JSONDecodeError:
        pass

    if content.strip() == '[]':
        return []

    print(f"    Warning: could not parse response for chunk {chunk_num}")
    return []


# ─── Verification Agent (Component 10 + 11) ─────────────────────────────────

def verify_fixes(fixes, chunk_text, model=VERIFIER_MODEL):
    """Second agent reviews proposed fixes for correctness."""
    if not fixes:
        return []

    # Migrated 2026-05-09: ChatOllama -> ChatOpenAI (alias) routed through OpenRouter.
    # `num_ctx` was Ollama-specific (server-side context window); not passed here.
    verifier = ChatOllama(
        model=model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    fixes_json = json.dumps(fixes, indent=2, ensure_ascii=False)

    prompt = f"""You are a verification agent. A proofreader proposed these fixes for OCR'd text.
Review each fix and mark it as "confirmed" or "rejected".

RULES:
- Confirm fixes that correct genuine OCR errors
- Reject fixes that would change correct text
- Reject fixes to ACTUAL proper nouns (real people's names, places)
- BUT: confirm fixes where a proper noun has an obvious OCR space insertion (e.g., "I nstitute" → "Institute" is valid because "Institute" is the correct word with an OCR-inserted space)
- Confirm fixes to LaTeX/HTML artifacts — these are ALWAYS OCR errors, never intentional (e.g., LaTeX mbox commands should be converted to plain text)
- Confirm fixes to split numbers (e.g., "1 997" → "1997", "1 850" → "1850")
- Reject if the "corrected" version introduces a new error
- Reject fixes to quoted text or blockquotes

Proposed fixes:
{fixes_json}

Original text:
```
{chunk_text[:2000]}
```

Output a JSON array with the same fixes plus a "status" field ("confirmed" or "rejected") and "verify_reason":
"""

    try:
        # Migrated 2026-05-09: OpenRouter doesn't expose Ollama-style server timings,
        # so measure wall-clock around invoke() and pass through to the metrics helper.
        _t0 = time.perf_counter()
        response = verifier.invoke([HumanMessage(content=prompt)])
        verify_elapsed = time.perf_counter() - _t0
        m = extract_openrouter_metrics(response, label="verify", elapsed_seconds=verify_elapsed)
        if m:
            print(f"    metrics: {format_metrics(m)}")
            _LAST_CALL_METRICS.append(m)
        content = response.content.strip()

        # Handle think tags
        if '<think>' in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            verified = json.loads(json_match.group())
            return verified
    except Exception as e:
        print(f"    Verification error: {e}")

    # If verification fails, return all as unverified
    for f in fixes:
        f['status'] = 'unverified'
    return fixes


# ─── Guardrails (Component 9) ────────────────────────────────────────────────

def apply_guardrails(fixes, chunk_text):
    """Filter out dangerous fixes before they reach verification."""
    safe_fixes = []
    for fix in fixes:
        original = fix.get('original', '')
        corrected = fix.get('corrected', '')

        # Guardrail 1: Cap total fixes per chunk
        if len(safe_fixes) >= MAX_FIXES_PER_CHUNK:
            print(f"    Guardrail: capped at {MAX_FIXES_PER_CHUNK} fixes per chunk")
            break

        # Guardrail 2: Reject no-ops (original == corrected)
        if original == corrected:
            print(f"    Guardrail: skipped no-op '{original[:30]}'")
            continue

        # Guardrail 3: Don't modify very short strings (likely false positive)
        if len(original) < 2:
            continue

        # Guardrail 3: Don't remove large blocks of text
        if not corrected and len(original) > 200:
            fix['confidence'] = 'low'
            fix['guardrail_note'] = 'Large deletion flagged'

        # Guardrail 4: Corrected text shouldn't be drastically different length
        if corrected and abs(len(corrected) - len(original)) > len(original):
            fix['confidence'] = 'low'
            fix['guardrail_note'] = 'Length change too large'

        safe_fixes.append(fix)

    return safe_fixes


# ─── Deterministic Pre-Processing ─────────────────────────────────────────────

def preprocess_deterministic(content):
    """Apply deterministic fixes BEFORE the LLM sees the text.
    These are patterns reliable enough to fix without AI review."""
    import re as _re
    fixes_applied = []

    for pattern, replacer in LATEX_ARTIFACT_PATTERNS.items():
        matches = list(_re.finditer(pattern, content))
        if matches:
            for m in matches:
                original = m.group()
                cleaned = _re.sub(pattern, replacer, original)
                if original != cleaned:
                    fixes_applied.append({
                        'original': original,
                        'corrected': cleaned,
                        'reason': 'Deterministic LaTeX/HTML cleanup',
                        'confidence': 'high',
                        'status': 'auto-applied',
                    })
            content = _re.sub(pattern, replacer, content)

    if fixes_applied:
        print(f"  Pre-processed {len(fixes_applied)} LaTeX/HTML artifacts deterministically")

    return content, fixes_applied


# ─── Main Run ────────────────────────────────────────────────────────────────

def run_proofreader(input_path, output_path=None, model=DEFAULT_MODEL, resume=False,
                    max_chunks=None, book_key=None, timeout_per_chunk=120):
    """Full proofreading pipeline with all harness components.

    book_key: enables per-book vocabulary (e.g. 'perrow', 'scott')
    timeout_per_chunk: hard cap per LLM call in seconds
    """

    # Migrated 2026-05-09: books.json lookup removed. Per-book metadata is now
    # passed explicitly by the harness via CLI flags (or omitted). LangSmith
    # tracing config is migration-out-of-scope; configure_langsmith is a no-op.
    books_json = None
    configure_langsmith(force=False)

    # State management: resume or fresh start
    state = None
    if resume:
        state = ProofreadState.load(input_path)
        if state:
            print(f"Resuming from chunk {state.current_chunk}/{state.total_chunks}")
            print(f"Fixes found so far: {len(state.fixes_found)}")
        else:
            print("No checkpoint found. Starting fresh.")

    if not state:
        state = ProofreadState.fresh(input_path, model)

    state.status = "running"

    # Load content and known patterns
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    known_patterns = load_known_patterns()

    # Deterministic pre-processing (LaTeX/HTML artifacts)
    content, preprocess_fixes = preprocess_deterministic(content)
    state.fixes_applied.extend(preprocess_fixes)

    word_count = len(content.split())

    # Section-aware chunking
    chunks = chunk_by_sections(content)
    state.total_chunks = len(chunks)

    print(f"\n{'='*60}")
    print(f"LLM Proofreader Agent Harness")
    print(f"{'='*60}")
    print(f"File:    {os.path.basename(input_path)}")
    print(f"Words:   {word_count}")
    print(f"Chunks:  {len(chunks)} (section-aware)")
    print(f"Model:   {model}")
    print(f"Verify:  {VERIFIER_MODEL}")
    print(f"Memory:  {len(known_patterns['false_positives'])} known false positives")
    print(f"{'='*60}\n")

    # Migrated 2026-05-09: ChatOllama -> ChatOpenAI (alias) routed through OpenRouter.
    llm = ChatOllama(
        model=model,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    start_time = time.time()
    start_chunk = state.current_chunk

    for i, chunk in enumerate(chunks):
        # Skip already-processed chunks (resume support)
        if i < start_chunk:
            continue

        # Max chunks limit for incremental review
        if max_chunks and (i - start_chunk) >= max_chunks:
            print(f"\n  Stopping at {max_chunks} chunks for review.")
            print(f"  Use --resume to continue from chunk {i+1}.")
            break

        elapsed = time.time() - start_time
        chunks_done = i - start_chunk
        rate = chunks_done / elapsed if elapsed > 0 else 0
        eta = (len(chunks) - i) / rate if rate > 0 else 0

        print(f"Chunk {i+1}/{len(chunks)} "
              f"(lines {chunk['start_line']}-{chunk['end_line']}, "
              f"{chunk['word_count']} words) "
              f"[{elapsed:.0f}s, ~{eta:.0f}s left]")

        # Skip filter: tables, bibliography, index, too-short content
        skip, skip_reason = should_skip_chunk(chunk)
        if skip:
            print(f"  → SKIPPED: {skip_reason}")
            state.current_chunk = i + 1
            state.save()
            continue

        # Step 1: Deterministic pre-analysis + LLM proofreading
        fixes = proofread_chunk_simple(
            llm, chunk['text'], i+1, len(chunks), state, known_patterns,
            book_key=book_key, timeout_seconds=timeout_per_chunk,
        )

        if fixes:
            print(f"  → {len(fixes)} potential fixes found")

            # Step 2: Guardrails
            fixes = apply_guardrails(fixes, chunk['text'])

            # Step 3: Verification agent
            if fixes:
                print(f"  → Verifying {len(fixes)} fixes...")
                fixes = verify_fixes(fixes, chunk['text'])

                confirmed = [f for f in fixes if f.get('status') == 'confirmed']
                rejected = [f for f in fixes if f.get('status') == 'rejected']

                print(f"  → {len(confirmed)} confirmed, {len(rejected)} rejected")

                state.fixes_found.extend(confirmed)
                state.fixes_rejected.extend(rejected)

                # Update session patterns
                for fix in confirmed:
                    reason = fix.get('reason', 'unknown')
                    state.session_patterns["error_types_seen"][reason] = \
                        state.session_patterns["error_types_seen"].get(reason, 0) + 1
        else:
            print(f"  → Clean")

        # Checkpoint after each chunk
        state.current_chunk = i + 1
        state.save()

    # ─── Apply confirmed fixes ───────────────────────────────────────────
    state.status = "applying"
    total_confirmed = len(state.fixes_found)
    print(f"\n{'='*60}")
    print(f"Applying {total_confirmed} confirmed fixes")
    print(f"{'='*60}\n")

    modified = content
    applied = 0
    for fix in state.fixes_found:
        original = fix.get('original', '')
        corrected = fix.get('corrected', '')
        if original and original in modified:
            modified = modified.replace(original, corrected, 1)
            applied += 1
            state.fixes_applied.append(fix)

    # Clean up multiple blank lines
    modified = re.sub(r'\n{3,}', '\n\n', modified)

    # Write output
    out_path = output_path or input_path
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(modified)

    # Update long-term memory with new patterns
    save_known_patterns(known_patterns)

    # Final stats
    state.status = "complete"
    state.save()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Time:           {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Chunks:         {len(chunks)}")
    print(f"Fixes found:    {len(state.fixes_found)}")
    print(f"Fixes applied:  {applied}")
    print(f"Fixes rejected: {len(state.fixes_rejected)}")
    print(f"Output:         {out_path}")

    # Write fix report
    report_path = out_path.replace('.md', '_proofread_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input': input_path,
            'output': out_path,
            'model': model,
            'elapsed_seconds': elapsed,
            'total_chunks': len(chunks),
            'fixes_applied': state.fixes_applied,
            'fixes_rejected': state.fixes_rejected,
            'session_patterns': state.session_patterns,
        }, f, indent=2, ensure_ascii=False)
    print(f"Report:         {report_path}")
    print(f"{'='*60}")

    return state


def show_status(input_path):
    """Show checkpoint status for a file."""
    state = ProofreadState.load(input_path)
    if not state:
        print("No checkpoint found for this file.")
        return

    print(f"File:           {os.path.basename(state.input_file)}")
    print(f"Status:         {state.status}")
    print(f"Progress:       {state.current_chunk}/{state.total_chunks} chunks")
    print(f"Fixes found:    {len(state.fixes_found)}")
    print(f"Fixes rejected: {len(state.fixes_rejected)}")
    print(f"Started:        {state.started_at}")
    print(f"Last checkpoint: {state.last_checkpoint}")
    if state.session_patterns["error_types_seen"]:
        print(f"Error types:    {state.session_patterns['error_types_seen']}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _load_env_file_if_present():
    """Source ~/.config/agent-tool-llm-proofreader/env if OPENROUTER_API_KEY isn't set.

    Mirrors a minimal `source <file>` for `KEY=value` and `export KEY=value` lines.
    Migration note (2026-05-09): replaces the old load_secrets / .env pattern.
    """
    from pathlib import Path
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
        if "=" in line:
            k, _, v = line.partition("=")
            v = v.strip().strip("'").strip('"')
            os.environ[k.strip()] = v


def main():
    _load_env_file_if_present()
    parser = argparse.ArgumentParser(description="LLM Proofreader Agent Harness")
    parser.add_argument("command", choices=["run", "status"],
                        help="run: proofread a file; status: check progress")
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("--output", "-o", help="Output file (default: overwrite input)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after N chunks for review")
    parser.add_argument("--book-key", default=None,
                        help="Book key for per-book vocabulary (perrow, scott, bowker, ...)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per LLM call in seconds (default: 120)")
    parser.add_argument("--state-dir", default=None,
                        help="Override the state directory (default: platformdirs.user_state_dir)")
    args = parser.parse_args()

    # Apply --state-dir override by setting the env var that state_paths reads.
    # Note: CHECKPOINT_DIR / KNOWN_PATTERNS_FILE were resolved at module import,
    # so re-resolve them here when overridden.
    if args.state_dir:
        os.environ["AGENT_TOOL_PROOFREADER_STATE_DIR"] = args.state_dir
        global CHECKPOINT_DIR, KNOWN_PATTERNS_FILE
        CHECKPOINT_DIR = str(_checkpoint_dir(args.state_dir))
        KNOWN_PATTERNS_FILE = str(_memory_path(args.state_dir))

    if args.command == "status":
        show_status(args.input)
    elif args.command == "run":
        run_proofreader(
            args.input, args.output, args.model, args.resume, args.max_chunks,
            book_key=args.book_key, timeout_per_chunk=args.timeout,
        )


if __name__ == "__main__":
    main()
