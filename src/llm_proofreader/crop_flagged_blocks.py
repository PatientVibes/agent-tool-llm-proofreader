"""
Crop Flagged Blocks
===================
For each block flagged by prose_quality.py, find its source page in the
original PDF, render that page as an image, and save it alongside the
flagged text for manual review and improvement.

Output structure:
    manifests/tables/{book_key}/
        README.md
        block_001_pNNN/
            image.png          (full page crop from source PDF)
            flagged_text.md    (the mangled text)
            metadata.json      (line numbers, score, components)
            improved.md        (empty — for manual improvements)

Usage:
    python crop_flagged_blocks.py perrow --threshold 0.5
    python crop_flagged_blocks.py perrow --top 10
    python crop_flagged_blocks.py perrow --dpi 200
"""

import sys, io, os, re, json, argparse
from dataclasses import dataclass

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import fitz  # PyMuPDF
from statistics import mean, stdev


# ─── Inlined prose-quality scorers ──────────────────────────────────────────
#
# Migrated 2026-05-09: analyze_file (and the helpers it transitively depends on)
# inlined verbatim from agent-tool-prose-quality (`prose_quality.py`) to avoid a
# cross-repo Python dependency. If prose_quality's analyze_file is updated,
# sync this copy manually. Source-of-truth lives in the agent-tool-prose-quality
# repo. Only the symbols transitively reachable from analyze_file are inlined;
# format_report / summarize / main are not used here.

def split_into_blocks(content):
    """Split markdown into logical blocks.
    A block is a sequence of non-empty lines separated by blank lines.
    Returns list of (start_line, end_line, text) tuples."""
    lines = content.split('\n')
    blocks = []
    current = []
    current_start = None

    for i, line in enumerate(lines):
        if line.strip():
            if current_start is None:
                current_start = i
            current.append(line)
        else:
            if current:
                blocks.append({
                    'start_line': current_start + 1,
                    'end_line': i,
                    'text': '\n'.join(current),
                })
                current = []
                current_start = None

    if current:
        blocks.append({
            'start_line': current_start + 1,
            'end_line': len(lines),
            'text': '\n'.join(current),
        })

    return blocks


def score_pipe_density(text):
    """0-1 score: 1 = no pipes (prose), 0 = lots of pipes (table)."""
    if not text:
        return 1.0
    pipes = text.count('|')
    density = pipes / len(text)
    if density == 0:
        return 1.0
    if density > 0.08:
        return 0.0
    return 1.0 - (density / 0.08)


def score_alphabetic_ratio(text):
    """0-1 score: 1 = all letters (prose), 0 = lots of special chars."""
    if not text:
        return 1.0
    letters = sum(1 for c in text if c.isalpha())
    whitespace = sum(1 for c in text if c.isspace())
    content_chars = len(text) - whitespace
    if content_chars == 0:
        return 1.0
    ratio = letters / content_chars
    if ratio >= 0.82:
        return 1.0
    if ratio < 0.5:
        return 0.0
    return (ratio - 0.5) / 0.32


def score_sentence_structure(text):
    """0-1 score based on whether the block looks like sentences."""
    if len(text) < 40:
        return 0.5

    endings = text.count('.') + text.count('?') + text.count('!')
    word_count = len(text.split())

    if word_count < 5:
        return 0.5

    if endings == 0:
        return 0.3 if word_count > 30 else 0.6

    words_per_sentence = word_count / endings

    if 8 <= words_per_sentence <= 40:
        return 1.0
    if words_per_sentence < 3:
        return 0.2
    if words_per_sentence > 80:
        return 0.3
    if words_per_sentence < 8:
        return 0.2 + 0.8 * (words_per_sentence - 3) / 5
    return 0.2 + 0.8 * (80 - words_per_sentence) / 40


def score_word_length_distribution(text):
    """0-1 score: prose has consistent avg word length ~4-6 with moderate variance."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if len(words) < 10:
        return 0.5

    lengths = [len(w) for w in words]
    avg = mean(lengths)
    sd = stdev(lengths) if len(lengths) > 1 else 0

    score = 1.0
    if avg < 3:
        score *= 0.4
    elif avg < 3.5:
        score *= 0.7
    elif avg > 8:
        score *= 0.5

    if sd < 1:
        score *= 0.5

    return max(0.0, min(1.0, score))


def score_capital_density(text):
    """0-1 score: prose has capitals only at sentence starts and proper nouns."""
    words = re.findall(r'\b[A-Za-z]+\b', text)
    if len(words) < 10:
        return 0.5

    cap_words = sum(1 for w in words if w[0].isupper())
    cap_ratio = cap_words / len(words)

    if cap_ratio <= 0.20:
        return 1.0
    if cap_ratio >= 0.50:
        return 0.0
    return 1.0 - (cap_ratio - 0.20) / 0.30


def score_repeated_tokens(text):
    """0-1 score: detect repeated short tokens like 'Yes No Unsure Yes No Unsure'."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 10:
        return 1.0

    three_grams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not three_grams:
        return 1.0

    counts = {}
    for g in three_grams:
        counts[g] = counts.get(g, 0) + 1

    max_count = max(counts.values())
    repetition_ratio = max_count / len(three_grams)

    if repetition_ratio < 0.05:
        return 1.0
    if repetition_ratio > 0.15:
        return 0.0
    return 1.0 - (repetition_ratio - 0.05) / 0.10


def score_br_tag_density(text):
    """0-1 score: HTML <br> tags are usually table cell line breaks."""
    br_count = len(re.findall(r'<br\s*/?>', text, re.IGNORECASE))
    if br_count == 0:
        return 1.0
    words = len(text.split())
    if words == 0:
        return 0.0
    density = br_count / words * 100
    if density < 1:
        return 0.7
    if density > 5:
        return 0.0
    return 1.0 - density / 5


SCORERS = [
    ('pipe_density', score_pipe_density, 1.5),
    ('alphabetic_ratio', score_alphabetic_ratio, 1.2),
    ('sentence_structure', score_sentence_structure, 1.0),
    ('word_length', score_word_length_distribution, 0.8),
    ('capital_density', score_capital_density, 1.0),
    ('repeated_tokens', score_repeated_tokens, 1.0),
    ('br_tags', score_br_tag_density, 1.5),
]


def score_block(text):
    """Return aggregate prose-quality score 0-1 plus component breakdown."""
    components = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for name, scorer, weight in SCORERS:
        s = scorer(text)
        components[name] = round(s, 3)
        weighted_sum += s * weight
        total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else 0
    return round(overall, 3), components


def is_bibliography_entry(text):
    """Detect bibliography/reference list entries."""
    if len(text) > 500:
        return False

    has_year = bool(re.search(r'\b(19|20)\d{2}\b', text))
    has_pp = bool(re.search(r'\bpp?\.\s*\d+', text))
    has_vol = bool(re.search(r'\bvol\.\s*\d+', text, re.IGNORECASE))
    has_journal_italic = '*' in text and re.search(r'\*[A-Z][^*]+\*', text)
    starts_with_name = bool(re.match(r'^[A-Z][a-z]+,\s+[A-Z]', text))

    signals = sum([has_year, has_pp, has_vol, bool(has_journal_italic), starts_with_name])
    return signals >= 2


def is_index_entry(text):
    """Detect index entries like 'Term, 123, 234-45, 345n12'."""
    if len(text) > 800:
        return False

    numbers = re.findall(r'\d+', text)
    words = re.findall(r'\b[A-Za-z]+\b', text)

    if not words:
        return False

    number_ratio = len(numbers) / len(words) if words else 0
    if number_ratio > 0.5:
        return True

    index_patterns = len(re.findall(r'\d+[–-]\d+|\d+n\d+', text))
    if index_patterns >= 3:
        return True

    return False


def should_skip_block(text):
    """Return True if a block shouldn't be scored (e.g., headings, images, frontmatter)."""
    stripped = text.strip()

    if re.match(r'^#{1,6}\s', stripped):
        return True
    if re.match(r'^!\[.*\]\(.*\)$', stripped):
        return True
    if stripped == '---':
        return True
    if re.match(r'^[\-\*_]{3,}$', stripped):
        return True
    if stripped.startswith('<!--') and stripped.endswith('-->'):
        return True
    if len(stripped) < 20:
        return True
    if is_bibliography_entry(stripped):
        return True
    if is_index_entry(stripped):
        return True

    return False


def analyze_file(md_path, threshold=0.5, top_n=None):
    """Analyze a markdown file and return suspicious blocks."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = split_into_blocks(content)
    results = []

    for block in blocks:
        if should_skip_block(block['text']):
            continue

        score, components = score_block(block['text'])
        results.append({
            'start_line': block['start_line'],
            'end_line': block['end_line'],
            'score': score,
            'components': components,
            'length': len(block['text']),
            'preview': block['text'][:150].replace('\n', ' '),
            'text': block['text'],
        })

    # Sort by score ascending (worst first)
    results.sort(key=lambda x: x['score'])

    if top_n:
        return results[:top_n]
    return [r for r in results if r['score'] < threshold]


# ─── End inlined prose-quality scorers ─────────────────────────────────────


BASE_DIR = r"C:\Users\chris\OneDrive\Documents\Reading"
MARKER_OUTPUT = os.path.join(BASE_DIR, "marker_output")
MANIFESTS = os.path.join(BASE_DIR, "manifests")


# ─── Book Config ─────────────────────────────────────────────────────────────

BOOK_CONFIG = {
    "perrow": {
        "markdown": os.path.join(MARKER_OUTPUT, "perrow",
                                 "Normal-Accidents-Perrow-decrypted",
                                 "Normal-Accidents-Perrow-decrypted.md"),
        "source_pdf": r"C:\Users\chris\Downloads\Normal-Accidents-Perrow-decrypted.pdf",
        "chunked": True,
    },
    "polanyi": {
        "markdown": os.path.join(MARKER_OUTPUT, "polanyi",
                                 "Polanyi_Michael_The_Tacit_Dimension",
                                 "Polanyi_Michael_The_Tacit_Dimension.md"),
        "source_pdf": r"C:\Users\chris\Downloads\Polanyi_Michael_The_Tacit_Dimension.pdf",
        "chunked": False,
    },
    "suchman": {
        "markdown": os.path.join(MARKER_OUTPUT, "suchman",
                                 "Suchman-PlansAndSituatedActions",
                                 "Suchman-PlansAndSituatedActions.md"),
        "source_pdf": r"C:\Users\chris\Downloads\Suchman-PlansAndSituatedActions.pdf",
        "chunked": False,
    },
    "hochschild": {
        "markdown": os.path.join(MARKER_OUTPUT, "hochschild",
                                 "the-managed-heart-arlie-russell-hochschild",
                                 "the-managed-heart-arlie-russell-hochschild.md"),
        "source_pdf": r"C:\Users\chris\Downloads\the-managed-heart-arlie-russell-hochschild.pdf",
        "chunked": False,
    },
    "bowker": {
        "markdown": os.path.join(MARKER_OUTPUT, "bowker",
                                 "Bowker-1999-Sorting-Things-Out-Classification-and-Its-Consequences",
                                 "Bowker-1999-Sorting-Things-Out-Classification-and-Its-Consequences.md"),
        "source_pdf": r"C:\Users\chris\Downloads\Bowker-1999-Sorting-Things-Out-Classification-and-Its-Consequences.pdf",
        "chunked": False,
    },
    "scott": {
        "markdown": os.path.join(MARKER_OUTPUT, "scott",
                                 "Seeing Like a State - James C. Scott",
                                 "Seeing Like a State - James C. Scott.md"),
        "source_pdf": r"C:\Users\chris\Downloads\Seeing Like a State - James C. Scott.pdf",
        "chunked": False,
    },
}


# ─── Chunk Mapping (for chunked books like Perrow) ───────────────────────────

def build_chunk_map(md_path):
    """Parse chunk markers from reassembled markdown.
    Returns list of (start_line, end_line, chunk_name, page_start, page_end)."""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    chunks = []
    chunk_starts = []

    for i, line in enumerate(lines):
        match = re.match(r'<!--\s*chunk:\s*chunk_(\d+)_p(\d+)-(\d+)\s*-->', line.strip())
        if match:
            chunk_num = int(match.group(1))
            page_start = int(match.group(2))
            page_end = int(match.group(3))
            chunk_starts.append({
                'line': i + 1,
                'chunk_num': chunk_num,
                'page_start': page_start,
                'page_end': page_end,
            })

    # Build ranges: each chunk covers lines from its marker to the next marker
    for i, cs in enumerate(chunk_starts):
        end_line = chunk_starts[i + 1]['line'] - 1 if i + 1 < len(chunk_starts) else len(lines)
        chunks.append({
            'start_line': cs['line'],
            'end_line': end_line,
            'chunk_num': cs['chunk_num'],
            'page_start': cs['page_start'],
            'page_end': cs['page_end'],
        })

    return chunks


def find_chunk_for_line(chunks, line_num):
    """Find which chunk a line number belongs to."""
    for c in chunks:
        if c['start_line'] <= line_num <= c['end_line']:
            return c
    return None


# ─── Source Page Detection ──────────────────────────────────────────────────

def find_source_page_chunked(flagged_block, chunks, source_pdf, book_key=None):
    """For chunked books: use chunk marker to get page range.
    Since chunked books are usually scanned (no text layer), we search the
    Marker chunk OCR output to find which page within the chunk the text is on."""
    chunk = find_chunk_for_line(chunks, flagged_block['start_line'])
    if not chunk:
        return None, "No chunk found for this line"

    # Extract a distinctive search phrase
    text = flagged_block['text']
    clean_text = re.sub(r'[|<>]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    words = clean_text.split()

    best_phrase = None
    for start in range(max(0, len(words) - 5)):
        phrase = ' '.join(words[start:start + 8])
        if len(phrase) > 30 and phrase[0].isalpha():
            best_phrase = phrase
            break

    # First try: search the source PDF text layer (works for PDFs with OCR layer)
    if best_phrase:
        doc = fitz.open(source_pdf)
        for page_num in range(chunk['page_start'] - 1, chunk['page_end']):
            if page_num >= len(doc):
                break
            page = doc[page_num]
            page_text = page.get_text()
            if best_phrase[:30] in page_text:
                doc.close()
                return page_num + 1, None
        doc.close()

    # Second try: search Marker's individual chunk markdown files.
    # These have the OCR'd text for a specific page range, and Marker often
    # embeds _page_N_ references in image filenames.
    if book_key:
        chunk_name = f"chunk_{chunk['chunk_num']:03d}_p{chunk['page_start']:03d}-{chunk['page_end']:03d}"
        chunk_dir = os.path.join(MARKER_OUTPUT, book_key, "chunks", chunk_name)
        if os.path.isdir(chunk_dir):
            # Find the chunk's markdown
            for root, dirs, files in os.walk(chunk_dir):
                for f in files:
                    if f.endswith('.md'):
                        chunk_md = os.path.join(root, f)
                        with open(chunk_md, 'r', encoding='utf-8') as fh:
                            chunk_content = fh.read()
                        # Look for _page_N_ image references near our text
                        # Marker uses ABSOLUTE page numbers (relative to source PDF)
                        # but when processing a chunk, N is relative to the chunk start:
                        # e.g., chunk 5 pages 41-50 → _page_0_ = page 41
                        # So: absolute_page = chunk['page_start'] + N
                        # (N is 0-indexed, page_start is 1-indexed)
                        if best_phrase and best_phrase[:30] in chunk_content:
                            idx = chunk_content.find(best_phrase[:30])
                            snippet_before = chunk_content[:idx]
                            page_refs = re.findall(r'_page_(\d+)_', snippet_before)
                            if page_refs:
                                n = int(page_refs[-1])
                                # Sanity check: N should be within the chunk's page range
                                chunk_size = chunk['page_end'] - chunk['page_start'] + 1
                                if n < chunk_size:
                                    actual_page = chunk['page_start'] + n
                                    return actual_page, None
                                # Otherwise N might be absolute already
                                if n < 500:  # reasonable page number
                                    return n + 1, "absolute page from marker"

                        # Fallback: use the median page from chunk's image references
                        all_refs = [int(r) for r in re.findall(r'_page_(\d+)_', chunk_content)]
                        if all_refs:
                            median_n = sorted(all_refs)[len(all_refs) // 2]
                            chunk_size = chunk['page_end'] - chunk['page_start'] + 1
                            if median_n < chunk_size:
                                return chunk['page_start'] + median_n, "estimated from chunk image positions"

    # Final fallback: middle of chunk's page range
    fallback_page = (chunk['page_start'] + chunk['page_end']) // 2
    return fallback_page, f"Text search failed, using middle of chunk pages {chunk['page_start']}-{chunk['page_end']}"


def find_source_page_whole(flagged_block, source_pdf):
    """For non-chunked books: search the whole PDF for the flagged text."""
    text = flagged_block['text']
    clean_text = re.sub(r'[|<>]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    words = clean_text.split()

    # Try several phrases from the text
    phrases = []
    for start in range(0, min(len(words), 20), 3):
        phrase = ' '.join(words[start:start + 6])
        if len(phrase) > 25 and phrase[0].isalpha():
            phrases.append(phrase[:40])

    doc = fitz.open(source_pdf)
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        for phrase in phrases:
            if phrase in page_text:
                doc.close()
                return page_num + 1, None

    doc.close()
    return None, "No matching text found in source PDF"


# ─── Page Rendering ─────────────────────────────────────────────────────────

def render_page(source_pdf, page_num_1indexed, output_path, dpi=200):
    """Render a single PDF page as a PNG image."""
    doc = fitz.open(source_pdf)
    page = doc[page_num_1indexed - 1]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()


def render_page_range(source_pdf, page_start_1indexed, page_end_1indexed, output_dir, dpi=200):
    """Render a range of PDF pages as individual PNG images.
    Returns list of (page_num, filepath) tuples."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(source_pdf)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    results = []

    for page_num in range(page_start_1indexed, page_end_1indexed + 1):
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat)
        filename = f"page_{page_num:03d}.png"
        filepath = os.path.join(output_dir, filename)
        pix.save(filepath)
        results.append((page_num, filepath))

    doc.close()
    return results


# ─── Main Processing ────────────────────────────────────────────────────────

def crop_book(book_key, threshold=0.5, top_n=None, dpi=200):
    """Process one book: detect flagged blocks, crop pages, save metadata."""
    if book_key not in BOOK_CONFIG:
        print(f"Unknown book: {book_key}")
        print(f"Available: {list(BOOK_CONFIG.keys())}")
        return

    config = BOOK_CONFIG[book_key]
    md_path = config['markdown']
    source_pdf = config['source_pdf']

    if not os.path.exists(md_path):
        print(f"Markdown not found: {md_path}")
        return
    if not os.path.exists(source_pdf):
        print(f"Source PDF not found: {source_pdf}")
        return

    print(f"\n{'='*60}")
    print(f"Cropping flagged blocks: {book_key}")
    print(f"{'='*60}")
    print(f"Markdown:   {md_path}")
    print(f"Source PDF: {source_pdf}")
    print(f"Chunked:    {config['chunked']}")
    print(f"Threshold:  {threshold}")
    if top_n:
        print(f"Top N:      {top_n}")

    # Step 1: Find flagged blocks
    print(f"\nStep 1: Analyzing prose quality...")
    flagged = analyze_file(md_path, threshold=threshold, top_n=top_n)
    print(f"  Found {len(flagged)} blocks to process")

    if not flagged:
        print("No flagged blocks, nothing to do.")
        return

    # Step 2: Build chunk map if chunked
    chunk_map = None
    if config['chunked']:
        chunk_map = build_chunk_map(md_path)
        print(f"  Built chunk map: {len(chunk_map)} chunks")

    # Step 3: Create output directory
    out_dir = os.path.join(MANIFESTS, "tables", book_key)
    os.makedirs(out_dir, exist_ok=True)

    # Step 4: Process each block
    print(f"\nStep 2: Processing blocks...")
    manifest = []
    for i, block in enumerate(flagged, 1):
        print(f"\n  Block {i}/{len(flagged)} — Line {block['start_line']} "
              f"[score: {block['score']}]")

        # For chunked books: render the ENTIRE chunk's page range so user can
        # find the actual table regardless of exact page mapping errors.
        # For non-chunked: render just the matched page.
        page_range = None
        primary_page = None

        if config['chunked']:
            chunk = find_chunk_for_line(chunk_map, block['start_line'])
            if chunk:
                page_range = (chunk['page_start'], chunk['page_end'])
                primary_page = (chunk['page_start'] + chunk['page_end']) // 2
                print(f"    Chunk pages: {page_range[0]}-{page_range[1]}")
        else:
            primary_page, note = find_source_page_whole(block, source_pdf)
            if primary_page:
                page_range = (primary_page, primary_page)
                print(f"    Source page: {primary_page}" + (f" ({note})" if note else ""))

        if not page_range:
            print(f"    SKIPPED: could not determine page range")
            continue

        # Create block directory
        block_dir = os.path.join(out_dir, f"block_{i:03d}_p{page_range[0]:03d}-{page_range[1]:03d}")
        os.makedirs(block_dir, exist_ok=True)

        # Render all pages in the range
        pages_dir = os.path.join(block_dir, "pages")
        try:
            rendered = render_page_range(source_pdf, page_range[0], page_range[1], pages_dir, dpi=dpi)
            total_kb = sum(os.path.getsize(p) for _, p in rendered) / 1024
            print(f"    Rendered {len(rendered)} pages ({total_kb:.0f} KB total)")
        except Exception as e:
            print(f"    Render failed: {e}")
            continue

        # Save flagged text
        with open(os.path.join(block_dir, "flagged_text.md"), 'w', encoding='utf-8') as f:
            f.write(block['text'])

        # Save metadata
        metadata = {
            'book_key': book_key,
            'block_id': f"block_{i:03d}",
            'page_range': list(page_range),
            'primary_page': primary_page,
            'source_pdf': source_pdf,
            'markdown_line_start': block['start_line'],
            'markdown_line_end': block['end_line'],
            'score': block['score'],
            'components': block['components'],
            'status': 'pending',
            'pages_rendered': [p for p, _ in rendered],
        }
        if config['chunked'] and chunk_map:
            chunk = find_chunk_for_line(chunk_map, block['start_line'])
            if chunk:
                metadata['chunk'] = f"chunk_{chunk['chunk_num']:03d}_p{chunk['page_start']:03d}-{chunk['page_end']:03d}"

        with open(os.path.join(block_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create empty improved.md for manual editing
        improved_path = os.path.join(block_dir, "improved.md")
        if not os.path.exists(improved_path):
            with open(improved_path, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Paste corrected version here. Leave empty to use image in PDF rebuild. -->\n")
                f.write(f"<!-- Actual page in book may differ from PDF page due to front matter. -->\n")
                f.write(f"<!-- Flip through pages/*.png to find the real table. -->\n")

        manifest.append({
            'block_id': f"block_{i:03d}",
            'page_range': f"{page_range[0]}-{page_range[1]}",
            'line': block['start_line'],
            'score': block['score'],
            'preview': block['preview'],
        })

    # Step 5: Write README
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"# Flagged Table/Block Index: {book_key}\n\n")
        f.write(f"Generated from prose quality analysis.\n")
        f.write(f"Threshold: {threshold}, Total blocks: {len(manifest)}\n\n")
        f.write("| # | Block | Pages | Line | Score | Preview |\n")
        f.write("|---|-------|-------|------|-------|--------|\n")
        for m in manifest:
            preview = m['preview'][:60].replace('|', '\\|')
            f.write(f"| {m['block_id']} | {m['block_id']} "
                    f"| {m['page_range']} | {m['line']} | {m['score']} | {preview}... |\n")
        f.write("\n## Review Workflow\n\n")
        f.write("1. Open the `pages/` folder in each block — flip through the page images to find the real table/figure\n")
        f.write("2. Compare with `flagged_text.md` to understand what the OCR got wrong\n")
        f.write("3. If you want to manually improve the text, write the corrected version to `improved.md`\n")
        f.write("4. Leave `improved.md` empty if the image should be used as the replacement instead\n")
        f.write("5. Run the rebuild script to apply changes back to the source markdown\n")
        f.write("\n**Note:** PDF page numbers may differ from printed book page numbers due to front matter.\n")

    print(f"\n{'='*60}")
    print(f"Complete: {len(manifest)} blocks cropped")
    print(f"Output:   {out_dir}")
    print(f"README:   {readme_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Crop flagged blocks from source PDF")
    parser.add_argument("book", help="Book key (e.g., 'perrow')")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Flag blocks below this score (default: 0.5)")
    parser.add_argument("--top", type=int, default=None,
                        help="Process only the top N worst blocks")
    parser.add_argument("--dpi", type=int, default=200, help="Image DPI (default: 200)")
    args = parser.parse_args()

    crop_book(args.book, args.threshold, args.top, args.dpi)


if __name__ == "__main__":
    main()
