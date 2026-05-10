"""
Split and Process — Chunked Marker processing for difficult/large PDFs.

Used when Marker fails on large scanned PDFs (memory, encryption, or other).
Splits the PDF into page-range chunks, runs Marker on each, and reassembles
the markdown with boundary cleanup.

Usage:
    python split_and_process.py <pdf_path> --chunk-size 10
    python split_and_process.py <pdf_path> --chunk-size 10 --resume
    python split_and_process.py <pdf_path> --decrypt  # strip encryption first
"""

import sys, io, os, re, json, argparse, subprocess, time

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from PyPDF2 import PdfReader, PdfWriter


BASE_DIR = r"C:\Users\chris\OneDrive\Documents\Reading"
MARKER_OUTPUT_BASE = os.path.join(BASE_DIR, "marker_output")
MIN_CHUNK_OUTPUT_BYTES = 500  # Below this is considered a failure


def decrypt_pdf(input_path, output_path=None):
    """Strip encryption from a PDF (empty password). Returns output path."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}-decrypted{ext}"

    if os.path.exists(output_path):
        print(f"  Decrypted PDF already exists: {output_path}")
        return output_path

    print(f"  Decrypting: {os.path.basename(input_path)}")
    r = PdfReader(input_path)
    if r.is_encrypted:
        result = r.decrypt("")
        print(f"  Decrypt result: {result}")

    w = PdfWriter()
    for p in r.pages:
        w.add_page(p)

    with open(output_path, 'wb') as f:
        w.write(f)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Decrypted: {output_path} ({size_mb:.1f} MB)")
    return output_path


def get_page_count(pdf_path):
    """Return the number of pages in a PDF."""
    r = PdfReader(pdf_path)
    return len(r.pages)


def build_chunks(total_pages, chunk_size=10):
    """Return list of (chunk_num, start, end) tuples, 0-indexed inclusive ranges."""
    chunks = []
    chunk_num = 1
    start = 0
    while start < total_pages:
        end = min(start + chunk_size - 1, total_pages - 1)
        chunks.append((chunk_num, start, end))
        start = end + 1
        chunk_num += 1
    return chunks


def run_marker_on_range(input_path, output_dir, page_range, min_threshold=True):
    """Run marker_single with --page_range. Returns path to output .md or None."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "marker_single",
        input_path,
        "--output_dir", output_dir,
        "--page_range", page_range,
    ]
    if min_threshold:
        cmd.extend(["--min_document_ocr_threshold", "0"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per chunk
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT: chunk {page_range}")
        return None
    except Exception as e:
        print(f"    ERROR running marker: {e}")
        return None

    if result.returncode != 0:
        print(f"    Marker failed (code {result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[-300:]}")
        return None

    # Find the output markdown file
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.md'):
                return os.path.join(root, f)

    return None


def check_chunk_output(md_path, min_bytes=MIN_CHUNK_OUTPUT_BYTES):
    """Check if chunk output is non-trivially sized."""
    if not md_path or not os.path.exists(md_path):
        return False
    size = os.path.getsize(md_path)
    return size >= min_bytes


def reassemble_chunks(chunk_dir, output_md_path, output_images_dir=None):
    """Concatenate chunk markdowns with boundary cleanup. Returns output path."""
    print(f"\nReassembling chunks → {os.path.basename(output_md_path)}")

    # Find all chunk markdown files in order
    chunk_mds = []
    if os.path.isdir(chunk_dir):
        for entry in sorted(os.listdir(chunk_dir)):
            chunk_path = os.path.join(chunk_dir, entry)
            if os.path.isdir(chunk_path) and entry.startswith("chunk_"):
                for root, dirs, files in os.walk(chunk_path):
                    for f in files:
                        if f.endswith('.md'):
                            chunk_mds.append((entry, os.path.join(root, f)))
                            break
                    if any(f.endswith('.md') for f in files):
                        break

    if not chunk_mds:
        print("  No chunk markdown files found!")
        return None

    print(f"  Found {len(chunk_mds)} chunks to reassemble")

    # Copy all images to the output images dir
    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)
        for chunk_name, md_path in chunk_mds:
            chunk_root = os.path.dirname(md_path)
            for fname in os.listdir(chunk_root):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    src = os.path.join(chunk_root, fname)
                    # Prefix with chunk name to avoid collisions
                    dst_name = f"{chunk_name}_{fname}"
                    dst = os.path.join(output_images_dir, dst_name)
                    if not os.path.exists(dst):
                        try:
                            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                                fdst.write(fsrc.read())
                        except Exception as e:
                            print(f"    Could not copy {src}: {e}")

    # Read and concatenate
    combined_parts = []
    for chunk_name, md_path in chunk_mds:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Strip YAML frontmatter if present
        lines = content.split('\n')
        if lines and lines[0].strip() == '---':
            for j in range(1, len(lines)):
                if lines[j].strip() == '---':
                    lines = lines[j + 1:]
                    break
            content = '\n'.join(lines)

        # Rewrite image references to use chunk-prefixed names
        def rewrite_image(m):
            alt = m.group(1)
            path = m.group(2)
            basename = os.path.basename(path)
            return f"![{alt}]({chunk_name}_{basename})"
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', rewrite_image, content)

        combined_parts.append({
            'name': chunk_name,
            'content': content.strip(),
        })

    # Join with boundary heuristic: if prev ends mid-sentence and next starts lowercase, merge
    combined = []
    for i, part in enumerate(combined_parts):
        content = part['content']
        if i > 0 and combined:
            prev = combined[-1]
            prev_lines = prev.rstrip().split('\n')
            if prev_lines:
                prev_last = prev_lines[-1].rstrip()
                cur_lines = content.lstrip().split('\n')
                if cur_lines:
                    cur_first = cur_lines[0].lstrip()
                    # Merge if prev ends mid-sentence and current starts lowercase
                    if (prev_last and cur_first and
                        not prev_last[-1] in '.?!:;"\'\u201d\u2019' and
                        not prev_last.startswith('#') and
                        not cur_first.startswith('#') and
                        not cur_first.startswith('!') and
                        not cur_first.startswith('|') and
                        cur_first[0].islower()):
                        # Join the last line of prev with the first line of cur
                        merged = prev_last + ' ' + cur_first
                        prev_lines[-1] = merged
                        cur_lines = cur_lines[1:]
                        combined[-1] = '\n'.join(prev_lines)
                        content = '\n'.join(cur_lines)

        combined.append(f"<!-- chunk: {part['name']} -->\n")
        combined.append(content)

    final = '\n\n'.join(combined)
    # Clean up triple+ blank lines
    final = re.sub(r'\n{4,}', '\n\n\n', final)

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(final)

    size_kb = os.path.getsize(output_md_path) / 1024
    print(f"  Reassembled: {size_kb:.1f} KB")
    return output_md_path


def process_difficult_pdf(input_path, output_dir, chunk_size=10, decrypt=False, resume=True):
    """Main orchestrator. Returns path to final markdown or None on failure."""
    print(f"\n{'='*60}")
    print(f"Split and Process: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    # Step 1: decrypt if requested
    if decrypt:
        input_path = decrypt_pdf(input_path)

    # Step 2: get page count and build chunk plan
    total_pages = get_page_count(input_path)
    chunks = build_chunks(total_pages, chunk_size)
    print(f"\nPages: {total_pages}")
    print(f"Chunks: {len(chunks)} ({chunk_size} pages each)")
    print(f"Output: {output_dir}\n")

    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Step 3: process each chunk
    start_time = time.time()
    failed = []
    succeeded = []

    for chunk_num, start, end in chunks:
        chunk_name = f"chunk_{chunk_num:03d}_p{start+1:03d}-{end+1:03d}"
        chunk_dir = os.path.join(chunks_dir, chunk_name)

        # Resume: skip if output already exists and passes check
        if resume and os.path.isdir(chunk_dir):
            existing_md = None
            for root, dirs, files in os.walk(chunk_dir):
                for f in files:
                    if f.endswith('.md'):
                        existing_md = os.path.join(root, f)
                        break
                if existing_md:
                    break
            if existing_md and check_chunk_output(existing_md):
                size_kb = os.path.getsize(existing_md) / 1024
                print(f"[{chunk_num}/{len(chunks)}] {chunk_name} - skipped (exists, {size_kb:.1f} KB)")
                succeeded.append(chunk_name)
                continue

        elapsed = time.time() - start_time
        done = len(succeeded) + len(failed)
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (len(chunks) - done) / rate / 60 if rate > 0 else 0

        print(f"[{chunk_num}/{len(chunks)}] {chunk_name} [{elapsed/60:.1f}m elapsed, ~{eta_min:.0f}m left]")

        page_range = f"{start}-{end}"
        md_path = run_marker_on_range(input_path, chunk_dir, page_range)

        if check_chunk_output(md_path):
            size_kb = os.path.getsize(md_path) / 1024
            print(f"    OK ({size_kb:.1f} KB)")
            succeeded.append(chunk_name)
        else:
            print(f"    FAILED")
            failed.append(chunk_name)

    # Step 4: reassemble
    print(f"\n{'='*60}")
    print(f"Chunking complete: {len(succeeded)} succeeded, {len(failed)} failed")
    print(f"{'='*60}")

    if failed:
        print(f"Failed chunks: {failed}")

    # Output markdown path mimics Marker's structure for downstream compatibility
    book_name = os.path.splitext(os.path.basename(input_path))[0]
    output_md_dir = os.path.join(output_dir, book_name)
    output_md = os.path.join(output_md_dir, f"{book_name}.md")

    reassemble_chunks(chunks_dir, output_md, output_md_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    return output_md


def main():
    parser = argparse.ArgumentParser(description="Split and process difficult PDFs")
    parser.add_argument("input", help="Input PDF path")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=10, help="Pages per chunk (default: 10)")
    parser.add_argument("--decrypt", action="store_true", help="Decrypt PDF first")
    parser.add_argument("--no-resume", action="store_true", help="Re-run even if chunks exist")
    args = parser.parse_args()

    if not args.output_dir:
        # Default to marker_output/{basename}
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output_dir = os.path.join(MARKER_OUTPUT_BASE, base.replace('-decrypted', ''))

    process_difficult_pdf(
        args.input,
        args.output_dir,
        chunk_size=args.chunk_size,
        decrypt=args.decrypt,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
