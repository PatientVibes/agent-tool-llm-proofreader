"""
Reinsert Blocks
===============
Apply manual table/block corrections from the manifests/tables/ folder back
into the corresponding marker_output markdown source.

Workflow:
1. User edits `manifests/tables/{book}/block_XXX_pNNN-MMM/improved.md`
2. This tool finds each non-empty improved.md
3. Loads the matching flagged_text.md (the original mangled version)
4. Searches marker_output/{book}/...md for the flagged text
5. Replaces it with the improved version
6. Updates metadata.json status to "applied"
7. Optionally triggers PDF rebuild

The improved.md is considered "filled in" when:
- It exists AND has at least 20 characters of content (beyond the template comment)
- OR its first line does NOT start with `<!--`

Usage:
    python reinsert_blocks.py perrow
    python reinsert_blocks.py perrow --dry-run    # preview only
    python reinsert_blocks.py perrow --rebuild    # also rebuild PDF after
    python reinsert_blocks.py --all               # all books
"""

import sys, io, os, re, json, argparse, shutil, subprocess
from dataclasses import dataclass

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


BASE_DIR = r"C:\Users\chris\OneDrive\Documents\Reading"
MARKER_OUTPUT = os.path.join(BASE_DIR, "marker_output")
MANIFESTS = os.path.join(BASE_DIR, "manifests")
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirror the config from crop_flagged_blocks.py
BOOK_MARKDOWNS = {
    "polanyi": "Polanyi_Michael_The_Tacit_Dimension",
    "suchman": "Suchman-PlansAndSituatedActions",
    "hochschild": "the-managed-heart-arlie-russell-hochschild",
    "bowker": "Bowker-1999-Sorting-Things-Out-Classification-and-Its-Consequences",
    "scott": "Seeing Like a State - James C. Scott",
    "perrow": "Normal-Accidents-Perrow-decrypted",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_book_markdown_path(book_key):
    """Return the path to the main markdown file for a book."""
    if book_key not in BOOK_MARKDOWNS:
        return None
    subdir = BOOK_MARKDOWNS[book_key]
    path = os.path.join(MARKER_OUTPUT, book_key, subdir, f"{subdir}.md")
    return path if os.path.exists(path) else None


def is_improved_filled_in(improved_path):
    """Check if improved.md has real content (not just the template comment)."""
    if not os.path.exists(improved_path):
        return False
    with open(improved_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip HTML comments
    content_no_comments = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL).strip()

    # Need at least 20 chars of real content
    return len(content_no_comments) >= 20


def load_block(block_dir):
    """Load a block's flagged_text, improved, and metadata."""
    flagged_path = os.path.join(block_dir, "flagged_text.md")
    improved_path = os.path.join(block_dir, "improved.md")
    metadata_path = os.path.join(block_dir, "metadata.json")

    if not os.path.exists(flagged_path):
        return None
    if not os.path.exists(metadata_path):
        return None

    with open(flagged_path, 'r', encoding='utf-8') as f:
        flagged = f.read()

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    improved = None
    if is_improved_filled_in(improved_path):
        with open(improved_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Strip HTML comments and leading/trailing whitespace
        improved = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL).strip()

    return {
        'block_dir': block_dir,
        'flagged': flagged,
        'improved': improved,
        'metadata': metadata,
        'block_id': metadata.get('block_id', os.path.basename(block_dir)),
    }


def normalize_whitespace(text):
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_for_match(text):
    """Normalize text for fuzzy matching: strip HTML tags that strip_marker_artifacts
    would have removed (like <br>), collapse whitespace."""
    # Replace <br> variants with space (matches strip_marker_artifacts behavior)
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    # Convert <sup>x</sup> to x (no tag) for matching purposes
    text = re.sub(r'</?(sup|super|sub)>', '', text)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


def find_and_replace(content, flagged, improved):
    """Find the flagged text in content and replace with improved.
    Returns (new_content, success_bool, match_type)."""
    # Strategy 1: exact match
    if flagged in content:
        return content.replace(flagged, improved, 1), True, "exact"

    # Strategy 2: match ignoring trailing whitespace
    flagged_rstrip = flagged.rstrip()
    if flagged_rstrip in content:
        return content.replace(flagged_rstrip, improved, 1), True, "trimmed"

    # Strategy 3: whitespace-normalized match (reconstruct original positioning)
    flag_norm = normalize_whitespace(flagged)
    content_norm = normalize_whitespace(content)
    if flag_norm in content_norm:
        target_len = len(flag_norm)
        content_lines = content.split('\n')
        for start in range(len(content_lines)):
            accum = []
            for end in range(start, min(start + 50, len(content_lines))):
                accum.append(content_lines[end])
                candidate = '\n'.join(accum)
                if normalize_whitespace(candidate) == flag_norm:
                    return content.replace(candidate, improved, 1), True, "whitespace-normalized"
                if len(normalize_whitespace(candidate)) > target_len * 1.2:
                    break

    # Strategy 4: strip-aware match (handle post strip_marker_artifacts state)
    # The flagged text may contain <br> and other tags that were later removed.
    flag_stripped = normalize_for_match(flagged)
    content_stripped_norm = normalize_for_match(content)
    if flag_stripped in content_stripped_norm:
        target_len = len(flag_stripped)
        content_lines = content.split('\n')
        for start in range(len(content_lines)):
            accum = []
            for end in range(start, min(start + 80, len(content_lines))):
                accum.append(content_lines[end])
                candidate = '\n'.join(accum)
                if normalize_for_match(candidate) == flag_stripped:
                    return content.replace(candidate, improved, 1), True, "tag-stripped"
                if len(normalize_for_match(candidate)) > target_len * 1.2:
                    break

    return content, False, "not_found"


def reinsert_book(book_key, dry_run=False, rebuild=False):
    """Process all filled-in improved.md files for a book."""
    md_path = get_book_markdown_path(book_key)
    if not md_path:
        print(f"No markdown found for book: {book_key}")
        return 0, 0

    tables_dir = os.path.join(MANIFESTS, "tables", book_key)
    if not os.path.isdir(tables_dir):
        print(f"No tables manifest for book: {book_key}")
        return 0, 0

    print(f"\n{'='*60}")
    print(f"Reinsert: {book_key}")
    print(f"{'='*60}")
    print(f"Markdown: {md_path}")
    print(f"Tables:   {tables_dir}")

    # Find all block directories
    block_dirs = sorted([
        os.path.join(tables_dir, d)
        for d in os.listdir(tables_dir)
        if os.path.isdir(os.path.join(tables_dir, d)) and d.startswith('block_')
    ])

    if not block_dirs:
        print("No block directories found.")
        return 0, 0

    # Filter to only blocks with improved.md filled in
    ready = []
    for bd in block_dirs:
        block = load_block(bd)
        if block and block['improved']:
            ready.append(block)

    print(f"\n{len(ready)}/{len(block_dirs)} blocks have improved.md filled in")

    if not ready:
        print("Nothing to apply. Edit improved.md files in block folders first.")
        return 0, 0

    # Read current markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup before modifying
    if not dry_run:
        backup_path = md_path + ".bak.reinsert"
        if not os.path.exists(backup_path):
            shutil.copy2(md_path, backup_path)
            print(f"Backup: {os.path.basename(backup_path)}")

    # Apply each improved block
    applied = 0
    skipped = 0

    for block in ready:
        block_id = block['block_id']
        flagged = block['flagged']
        improved = block['improved']

        print(f"\n  {block_id}")
        print(f"    Flagged length:  {len(flagged)} chars")
        print(f"    Improved length: {len(improved)} chars")

        new_content, success, match_type = find_and_replace(content, flagged, improved)

        if success:
            print(f"    Applied ({match_type})")
            content = new_content
            applied += 1

            if not dry_run:
                # Update metadata
                block['metadata']['status'] = 'applied'
                block['metadata']['applied_at'] = __import__('time').strftime('%Y-%m-%d %H:%M:%S')
                block['metadata']['match_type'] = match_type
                with open(os.path.join(block['block_dir'], 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(block['metadata'], f, indent=2, ensure_ascii=False)
        else:
            print(f"    SKIPPED — flagged text not found in source")
            print(f"    First 80 chars: {flagged[:80]!r}")
            skipped += 1

    # Write the updated markdown
    if not dry_run and applied > 0:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nWrote {applied} updates to {os.path.basename(md_path)}")

        # Also sync to source/
        source_md = find_source_md(book_key)
        if source_md:
            shutil.copy2(md_path, source_md)
            print(f"Synced to source: {os.path.basename(source_md)}")
    elif dry_run:
        print(f"\n[DRY RUN] Would apply {applied} updates, skip {skipped}")

    # Optional rebuild
    if rebuild and applied > 0 and not dry_run:
        print(f"\nRebuilding PDF...")
        result = subprocess.run(
            ["python", os.path.join(TOOLS_DIR, "process_book.py"), "--book", book_key, "--skip-marker"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            # Grep for "Created:" line
            for line in result.stdout.split('\n'):
                if 'Created:' in line and '.pdf' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"  Rebuild failed:\n{result.stderr[-500:]}")

    return applied, skipped


def find_source_md(book_key):
    """Find the source/ folder index.md for a book."""
    source_base = os.path.join(BASE_DIR, "source")
    if not os.path.isdir(source_base):
        return None

    # Book key → title mapping (best-effort)
    title_hints = {
        "polanyi": "The Tacit Dimension",
        "suchman": "Plans and Situated Actions",
        "hochschild": "The Managed Heart",
        "bowker": "Sorting Things Out",
        "scott": "Seeing Like a State",
        "perrow": "Normal Accidents",
    }
    hint = title_hints.get(book_key, book_key)

    for entry in os.listdir(source_base):
        if hint.lower() in entry.lower():
            candidate = os.path.join(source_base, entry, "index.md")
            if os.path.exists(candidate):
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Reinsert improved blocks into source markdown")
    parser.add_argument("book", nargs='?', help="Book key (e.g., 'perrow')")
    parser.add_argument("--all", action="store_true", help="Process all books")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild PDF after applying")
    args = parser.parse_args()

    if args.all:
        total_applied = 0
        total_skipped = 0
        for book_key in BOOK_MARKDOWNS:
            applied, skipped = reinsert_book(book_key, dry_run=args.dry_run, rebuild=args.rebuild)
            total_applied += applied
            total_skipped += skipped
        print(f"\n{'='*60}")
        print(f"TOTAL: {total_applied} applied, {total_skipped} skipped")
        print(f"{'='*60}")
    elif args.book:
        reinsert_book(args.book, dry_run=args.dry_run, rebuild=args.rebuild)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
