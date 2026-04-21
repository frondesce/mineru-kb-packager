---
name: mineru-kb-packager
description: Use when converting MinerU PDF parsing output into a retrieval-ready knowledge base package, processing structured JSON content into chunked, searchable documents
---

# MinerU KB Packager

## Overview

Transform MinerU PDF parsing output into a retrieval-ready knowledge base package. This skill defines the complete pipeline: input file discovery, content chunking, metadata enrichment, and structured JSONL output.

**Critical success factors:**
- Image paths must be relative to project root (not input directory)
- Figure `nearby_text` must come from actual adjacent paragraphs, not section titles
- Long tables must be split while repeating headers
- Empty/malformed blocks must be skipped (not output with empty `chunk_text`)
- `section_title` must be cleaned (strip whitespace, collapse multiple spaces)
- Output is minimal JSONL with only 6 fields (no `doc_id`, `doc_title`, `source_pdf`, `metadata`)

## Quick Start

### Usage

```bash
python3 converter.py <input_dir> [--output-dir <output_dir>] [--shared-output <shared_dir>]
```

### Example

```bash
# Basic usage (output to <input_dir>/output/)
python3 converter.py ./mineru_output/my_document.pdf-uuid

# With custom output directory
python3 converter.py ./mineru_output/my_document.pdf-uuid --output-dir ./kb_chunks/

# Batch processing with shared output (collects all kb_chunks into one directory, named by document)
python3 converter.py ./dir1 -s ./output/
python3 converter.py ./dir2 -s ./output/
# Produces: output/my_document_1.jsonl, output/my_document_2.jsonl
```

### Output Files

| File | Description |
|------|-------------|
| `kb_chunks.jsonl` | **Main output** - chunked data for RAG ingestion |
| `kb_manifest.json` | Processing metadata and statistics |
| `error_report.json` | Errors, warnings, and skipped blocks |
| `README_kb.md` | Documentation for the generated files |

These filenames use a neutral `kb_*` prefix. The generated data is generic and can be used with any RAG / vector ingestion pipeline.

### Input Directory Structure

Expected MinerU output directory structure:
```
mineru_output/
├── content_list_v2.json          # Primary input (preferred)
├── content_list.json             # Fallback input
├── layout.json                   # Layout information
├── model.json                    # Model output
├── origin.pdf                    # Original PDF
├── full.md                       # Full markdown (reference only)
└── images/                       # Extracted images
    ├── xxx.jpg
    └── ...
```

## When to Use

**Use this skill when:**
- Processing MinerU-extracted PDF content for RAG/vector search
- Converting `content_list_v2.json` or `content_list.json` to retrieval-ready KB chunks
- Need to preserve document structure (sections, figures, tables) during chunking
- Building searchable knowledge bases from academic papers, reports, or manuals

**Do NOT use for:**
- Direct PDF parsing (use MinerU first)
- Generic text chunking without document structure preservation
- Non-MinerU extraction workflows

## Input File Priority

Source files discovered in this order:

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `content_list_v2.json` or `*_content_list_v2.json` | Primary structured content (preferred) |
| 2 | `content_list.json` or `*_content_list.json` | Fallback structured content |
| 3 | `images/` | Figure extraction and reference |
| 4 | `origin.pdf` / `*_origin.pdf` | Source reference |
| 5 | `full.md` | Supplementary reference only |

**Manifest requirement:** Record BOTH files that exist (`input_files_all`) AND files actually used (`input_files_used`). Include prefixed variants like `63805328-*_content_list.json`.

**Rule:** Never use `full.md` as primary data source. Prefer JSON content lists for structured extraction.

## Block Type Mapping

Map MinerU block types to the target retrieval content types:

| MinerU Type | Target Type | Handling |
|-------------|-----------------|----------|
| `paragraph` | `text` | Merge adjacent short blocks in same section |
| `list` | `text` | Preserve as formatted text |
| `table` | `table` | One table = one chunk; split long tables with repeated headers |
| `image` | `figure` | Generate descriptive text + image_path |
| `equation_interline` | `formula` | Preserve LaTeX/math meaning |
| `title` | - | Use for section hierarchy only; don't output as chunk |
| `page_header` | - | Noise - skip and count |
| `page_footer` | - | Noise - skip and count |
| `page_number` | - | Noise - skip and count |
| `page_aside_text` | - | Noise - skip and count (e.g., page margin text like "Release: 12/15/2021") |

## Critical Implementation Details

### 1. Image Path Handling (Common Failure Point)

**WRONG:** `images/abc123.jpg` (relative to input directory)
**CORRECT:** `mineru_output_dir/images/abc123.jpg` (relative to project root)

**Validation required:**
```python
def _is_valid_image_path(self, path: str) -> bool:
    """Reject directory paths and non-image files"""
    if not path or path.endswith('/'):
        return False
    ext = Path(path).suffix.lower()
    return ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
```

**Path construction:**
```python
full_image_path = self.input_dir / image_path_raw
relative_image_path = self._get_relative_path(full_image_path)  # From project root
```

### 2. Figure Nearby Text (High Error Rate)

**Problem:** 96% of figure chunks initially had `nearby_text` equal to `section_title` instead of actual adjacent text.

**Root cause:** Block index mismatch between processing loop and cache lookup.

**Solution - Two-phase processing:**
```python
def build_page_blocks_cache(self, content_data: list):
    """Cache ALL block types (not just paragraphs) to enable accurate positioning"""
    for page_no, page_blocks in enumerate(content_data, 1):
        self.page_blocks_cache[page_no] = []
        for idx, block in enumerate(page_blocks):  # Use enumerate index!
            block_type = block.get("type", "")
            text = extract_text(block) if block_type in ["paragraph", "list"] else ""
            self.page_blocks_cache[page_no].append((idx, block_type, text))

def find_nearby_text(self, page_no: int, block_idx: int) -> str:
    """Use block_idx from enumerate, NOT global block counter"""
    page_blocks = self.page_blocks_cache.get(page_no, [])
    
    # Find position using the SAME index from cache building
    current_pos = None
    for i, (idx, btype, text) in enumerate(page_blocks):
        if idx == block_idx:
            current_pos = i
            break
    
    # Search forward first (text after image is usually more relevant)
    for offset in range(1, 10):
        forward_pos = current_pos + offset
        if forward_pos < len(page_blocks):
            idx, btype, text = page_blocks[forward_pos]
            if btype in ["paragraph", "list"] and text and len(text) > 20:
                return text[:500]
    
    # Fall back to section title only if no adjacent text found
    return self.current_section_title
```

**Key insight:** Pass `block_idx` from `enumerate()` in processing loop, not a global counter.

### 3. Low-Value Content Filtering (RAG Quality)

**Problem:** Contents pages, index lists, and revision histories pollute RAG retrieval results.

**Sections to skip by default:**
```python
SKIP_SECTION_PATTERNS = [
    r'^\s*contents\s*$',
    r'^\s*table of contents\s*$',
    r'^\s*list of figures\s*$',
    r'^\s*list of tables\s*$',
    r'^\s*rev\.\s*\w+',  # Rev. A, Rev. B, etc.
    r'^\s*revision\s*history\s*$',
    r'^\s*document\s*history\s*$',
    r'^\s*change\s*history\s*$',
]
```

**Implementation:**
```python
def _should_skip_section(self, title: str) -> bool:
    """Check if section should be skipped (TOC, index, revision history)"""
    clean = self.clean_section_title(title).lower()
    return any(re.match(p, clean, re.IGNORECASE) for p in self.SKIP_SECTION_PATTERNS)
```

**Behavior:** Once a skip section is detected, all content blocks (paragraph, list, table) within that section are filtered out until a new section begins.

### 4. Figure Chunk Classification (Three-Tier System)

MinerU often splits a main figure into multiple sub-images. Implement a three-tier classification system:

#### Tier 1: Formal Figures (Keep)
A figure is **formal** if it has:
- A clear figure number (e.g., "Figure 29", "Fig. 3", "图1-2")
- OR a complete caption + sufficient context (≥50 chars total)

#### Tier 2: Subfigure Fragments (Merge)
A figure is a **subfigure fragment** if it has:
- NO figure number
- Short label only (e.g., "MLC Page", "TLC Page", "SLC Page", "Top View", "Bottom View")
- No independent explanatory text

**Action:** Merge into the most recent formal figure on the same page or adjacent blocks.

**Common subfigure labels:**
```python
SUBFIGURE_LABEL_PATTERNS = [
    r'^\s*(MLC|TLC|SLC)\s+Page\s*$',
    r'^\s*Top\s+View\s*$',
    r'^\s*Bottom\s+View\s*$',
    r'^\s*Side\s+View\s*$',
    r'^\s*Detail\s*[A-Z]\s*$',
    r'^\s*Zoom\s*(In|Out)\s*$',
    r'^\s*Close[\s-]*up\s*$',
    r'^\s*Enlarged\s+View\s*$',
]
```

#### Tier 3: Weak Figures (Skip)
A figure is **weak** if it has:
- NO figure number
- NO valid caption
- Only `Context: section_title` (no real adjacent text)
- OR total content < 30 characters

**Action:** Skip entirely - do not add to knowledge base.

#### Implementation

```python
def _classify_figure_chunk(self, caption_text: str, nearby_text: str, chunk_text: str) -> str:
    has_number = self._has_figure_number(caption_text) or self._has_figure_number(nearby_text)
    has_valid_caption = caption_text and len(caption_text.strip()) > 5
    nearby_is_section_title = nearby_text == self.current_section_title
    total_content_len = len((caption_text or "") + (nearby_text or "").strip())
    
    # Formal figure
    if has_number or (has_valid_caption and not nearby_is_section_title and total_content_len >= 50):
        return 'formal'
    
    # Subfigure fragment
    if self._is_subfigure_label(caption_text):
        return 'subfigure'
    
    # Weak figure
    if not has_valid_caption and nearby_is_section_title:
        return 'weak'
    if total_content_len < 30:
        return 'weak'
    
    return 'formal'  # Default

def _process_image_block(self, block, page_no, block_idx, block_index):
    # ... extract info ...
    figure_type = self._classify_figure_chunk(caption_text, nearby_text, chunk_text)
    
    if figure_type == 'weak':
        self.stats["skipped_weak_figures"] += 1
        return  # Skip
    
    if figure_type == 'subfigure':
        if self.last_formal_figure:
            # Merge into parent figure
            last_idx = self.last_formal_figure['index']
            self.chunks[last_idx]['chunk_text'] += f"\n[Subfigure: {caption_text}]"
            self.chunks[last_idx]['metadata']['cleanup_flags'].append('merged_subfigure')
            self.stats["subfigure_fragments_merged"] += 1
            return
        else:
            # No parent found, treat as weak
            self.stats["skipped_weak_figures"] += 1
            return
    
    # Formal figure - create chunk and cache index
    # ... create chunk ...
    self.last_formal_figure = {'index': len(self.chunks), 'page_no': page_no}
    self.chunks.append(chunk)
```

**Key Principle:** Preserve semantic completeness. A figure chunk should be independently searchable and meaningful.

### 5. Table Splitting Logic

**Trigger:** Table exceeds 900 tokens (estimated)

**Algorithm:**
```python
def split_table_into_chunks(self, headers, rows, caption, footnote):
    # Build full text to check length
    full_text = build_table_text(headers, rows, caption, footnote)
    total_tokens = self.estimate_tokens(full_text)
    
    if total_tokens <= self.HARD_CAP_TOKENS:
        return [single_chunk]  # No split needed
    
    # Calculate rows per chunk
    header_tokens = estimate_header_tokens(caption, headers)
    available = self.HARD_CAP_TOKENS - header_tokens - 100  # Buffer for footnote
    rows_per_chunk = max(5, available // (total_tokens // len(rows)))
    
    # Split and repeat headers in each chunk
    total_splits = (len(rows) + rows_per_chunk - 1) // rows_per_chunk
    for split_idx in range(total_splits):
        chunk_rows = rows[split_idx*rows_per_chunk : (split_idx+1)*rows_per_chunk]
        text = f"Table: {caption} (Part {split_idx+1}/{total_splits})\n"
        text += "Columns: " + " | ".join(headers) + "\n"
        for i, row in enumerate(chunk_rows, split_idx*rows_per_chunk + 1):
            text += f"Row {i}: " + " | ".join(row) + "\n"
        if split_idx == total_splits - 1 and footnote:
            text += f"Note: {footnote}"
```

**Handling oversized rows:** If a single row exceeds the token limit (e.g., cells with very long text), do NOT truncate cell content. Preserve semantic completeness by keeping the full row, but log it for manual review.

```python
# If single row exceeds limit, keep it intact for semantic completeness
if row_tokens > SINGLE_ROW_LIMIT:
    if chunk_rows:
        break  # Finish current chunk first
    chunk_rows = [rows[current_row]]  # Row gets its own chunk
    current_row += 1
    break
```

**Metadata for split tables:**
```json
{
  "metadata": {
    "split_info": {
      "is_split": true,
      "split_index": 1,
      "total_splits": 3
    }
  }
}
```

**Principle:** Prefer semantic completeness over strict token limits. A slightly oversized chunk with complete information is better than a truncated cell.

### 4. Empty Block Filtering

**Skip blocks with NO extractable content:**
```python
def _process_table_block(self, block, ...):
    headers, rows, caption, footnote = self.extract_table_data(block)
    
    # Check if there's any actual content
    has_content = (headers or rows) and (caption or footnote or 
                  any(len(str(cell)) > 0 for row in rows for cell in row))
    has_image = self._is_valid_image_path(image_path) and image_exists
    
    if not has_content and not has_image:
        self.stats["skipped_empty"] += 1
        self.error_report["skipped_blocks"].append({
            "reason": "empty table content and no valid image"
        })
        return  # DO NOT create chunk
```

### 5. Section Title Cleaning

**section_title must be cleaned before output:**
```python
def clean_section_title(self, title: str) -> str:
    """Remove leading/trailing whitespace, collapse multiple spaces"""
    if not title:
        return ""
    title = title.strip()  # Remove leading/trailing spaces
    title = re.sub(r'\s+', ' ', title)  # Collapse multiple spaces to single
    return title
```

**Apply at write time:**
```python
minimal_chunk = {
    "section_title": self.clean_section_title(chunk["section_title"]),
    # ...
}
```

**Why this matters:** Prevents "DESCRIPTION " vs "DESCRIPTION" aggregation issues.

### 6. File Discovery with Prefixes

MinerU outputs often have UUID-prefixed filenames:
```python
def discover_files(self):
    # Look for prefixed variants first
    v2_candidates = list(self.input_dir.glob("*_content_list_v2.json"))
    if v2_candidates:
        files["content_list_v2"] = v2_candidates[0]
    
    # Also record old versions that exist but aren't used
    cl_candidates = list(self.input_dir.glob("*_content_list.json"))
    for c in cl_candidates:
        if "_content_list_v2" not in c.name:
            files_exist["content_list"] = c  # Exists
            break
    
    # Separate: what exists vs what we use
    self.input_files_exist = files_exist  # All files found
    self.input_files_used = files_used     # Actually processed
```

## Chunking Strategy

### Text Chunks
- **Target size:** 300-700 tokens
- **Hard cap:** 900 tokens
- **Boundaries:** Prefer paragraph, section, or semantic boundaries
- **Merging:** Combine adjacent short blocks (< 150 tokens) in same section
- **Splitting:** Split long content at sentence boundaries; allow 1-sentence overlap if needed

### Table Chunks
- Default: One table = one chunk
- Long tables: Split into multiple chunks at row boundaries
- **Required:** Each chunk repeats table title and column headers
- Include table caption/footnote in metadata
- Mark split tables with `metadata.split_info`

### Figure Chunks
- Default: One figure = one chunk
- Generate searchable description from:
  - Figure title/caption
  - **Actual adjacent paragraph/list text** (not section title)
  - Image filename reference
- Always set `image_path` with project-root-relative path
- `nearby_text` must differ from `section_title` in >90% of cases

## Output Schema

### JSONL Structure (minimal - 6 fields)

Output is intentionally minimal for RAG ingestion. No `doc_id`, `metadata`, or debug fields.

```json
{
  "chunk_id": "docid_p1_text_1",
  "page_no": 1,
  "content_type": "text|table|figure|formula",
  "section_title": "Clean section heading",
  "chunk_text": "Processed searchable content - NEVER EMPTY",
  "image_path": "mineru_dir/images/abc.jpg or empty"
}
```

### Field Rules

| Field | Rule |
|-------|------|
| `chunk_id` | Format: `{doc_id}:p{page}:{type}:{seq}` - stable unique ID |
| `page_no` | 1-based page number |
| `content_type` | One of: `text`, `table`, `figure`, `formula` |
| `section_title` | **Cleaned** - no leading/trailing spaces, collapsed whitespace |
| `chunk_text` | **Never empty** - skip block if no extractable text |
| `image_path` | **Project root relative path** (not `images/abc.jpg`), empty if no image |

### Why Minimal?

- `doc_id`, `doc_title`, `source_pdf` are constant per file - add at ingest time if needed
- `metadata` fields (bbox, split_info, flags) are for debugging - not needed for retrieval
- Smaller files = faster upload, less storage

## Content Cleaning Rules

### Remove
- Repeated page headers/footers (frequency > threshold)
- Isolated page numbers
- Empty lines (> 2 consecutive)
- HTML fragments (`<div>`, `<span>`, etc.)
- Watermarks (low opacity, repeated text)

### Preserve
- Technical terminology and model numbers
- Units and measurements
- Mathematical symbols
- Code snippets and formulas
- Citation markers

### Table Text Conversion
- Convert HTML tables to natural language description
- Format: "Table {title} shows {description}. Columns: {cols}. Row 1: {data}. Row 2: {data}..."
- Preserve units in column headers

### Figure Description
- Format: "Figure {num}: {title}. {description}. Located on page {page}."
- **Must include actual nearby paragraph text, not just section heading**
- Reference actual image file with project-root-relative path

## Error Handling

### Error Report Schema (`error_report.json`)

```json
{
  "doc_id": "...",
  "summary": {
    "total_missing_images": 3,
    "total_skipped_blocks": 15,
    "total_unsupported_blocks": 2,
    "total_parse_errors": 1
  },
  "missing_images": [
    {"block_id": "p5_b12", "expected_path": "images/missing.jpg", "type": "figure"}
  ],
  "skipped_blocks": [
    {"block_id": "p3_b8", "type": "page_header", "reason": "noise block skipped", "page_no": 3},
    {"block_id": "p15_b23", "type": "table", "reason": "empty table content and no valid image", "page_no": 15}
  ],
  "unsupported_blocks": [
    {"block_id": "p7_b5", "type": "unknown_type", "page_no": 7}
  ],
  "parse_errors": [
    {"block_id": "p2_b3", "error": "HTML parse failed", "raw_content": "..."}
  ]
}
```

### Handling Guidelines

| Situation | Action |
|-----------|--------|
| Extractable but messy | Keep chunk, add `cleanup_flags` |
| Completely unparseable | Skip JSONL, log to error_report |
| Missing image | Keep text chunk, `image_path=""`, log to error_report |
| Unsupported block type | Skip, log to error_report |
| Content conflict (JSON vs MD) | Trust `content_list_v2.json` |
| **Empty table (no html, no image)** | **Skip entirely, log to error_report** |

## Manifest Schema (`kb_manifest.json`)

```json
{
  "doc_id": "...",
  "doc_title": "...",
  "source_dir": "input/mineru_output/",
  "source_pdf": "input/origin.pdf",
  "created_at": "2024-01-15T10:30:00Z",
  "input_files_all": {
    "origin_pdf": "path/to/63805328-*_origin.pdf",
    "content_list_v2": "path/to/content_list_v2.json",
    "content_list": "path/to/63805328-*_content_list.json",
    "layout": "path/to/layout.json",
    "model": "path/to/63805328-*_model.json",
    "full_md": "path/to/full.md",
    "images_dir": "path/to/images"
  },
  "input_files_used": {
    "origin_pdf": "path/to/63805328-*_origin.pdf",
    "content_list_v2": "path/to/content_list_v2.json",
    "layout": "path/to/layout.json",
    "model": "path/to/63805328-*_model.json",
    "full_md": "path/to/full.md",
    "images_dir": "path/to/images"
  },
  "output_files": {
    "chunks_jsonl": "output/kb_chunks.jsonl",
    "manifest": "output/kb_manifest.json",
    "readme": "output/README_kb.md",
    "error_report": "output/error_report.json"
  },
  "chunking_policy": {
    "target_tokens": "300-700",
    "hard_cap": 900,
    "merge_short_blocks": true,
    "table_split": true,
    "type_mapping": {...}
  },
  "stats": {
    "total_chunks": 160,
    "text_chunks": 53,
    "table_chunks": 30,
    "figure_chunks": 64,
    "formula_chunks": 13,
    "missing_images": 0,
    "skipped_empty": 3
  }
}
```

**Note:** `input_files_all` records all files that exist; `input_files_used` records what was actually processed. This preserves audit trail even when preferring v2 over v1 files.

## Common Mistakes & Fixes

| Mistake | Why It Happens | Fix |
|---------|----------------|-----|
| `image_path` like `images/abc.jpg` | Using path from JSON directly | Prepend input_dir, then get relative to project root |
| `nearby_text` equals `section_title` 96% of time | Block index mismatch in cache lookup | Use `enumerate()` index consistently in both cache building and lookup |
| Empty `chunk_text` in output | Not validating table/image content | Check `has_content or has_image` before creating chunk |
| Super long table chunks (>900 tokens) | Not implementing split logic | Calculate tokens, split at row boundaries, repeat headers |
| `section_title` has trailing spaces | MinerU source has whitespace | Clean with `.strip()` and `re.sub(r'\s+', ' ', ...)` before output |
| Missing prefixed files in manifest | Only checking exact filenames | Use `glob("*_content_list.json")` pattern |
| Split tables lose context | Not repeating headers | Always include caption and headers in each split chunk |
| Contents/Rev. history in RAG results | No section filtering | Implement `SKIP_SECTION_PATTERNS` to filter TOC, index, revision history |
| Weak figure chunks pollute results | No quality check on caption/context | Skip chunks with only "Context:" or no caption + no real adjacent text |
| `page_aside_text` in unsupported blocks | Not recognizing margin text as noise | Add `page_aside_text` to noise types, count separately from unsupported |
| Subfigure fragments as separate chunks | No three-tier classification | Implement formal/subfigure/weak classification; merge subfigures to parent |

## Quality Metrics (Self-Check)

Output these metrics after conversion for quality assessment:

| Metric | Description | Target |
|--------|-------------|--------|
| `total_chunks` | Total chunks generated | Varies by document |
| `skipped_low_value_sections` | Contents/List/Figures/Tables/Rev.* filtered | >0 for docs with these sections |
| `skipped_weak_figures` | Figure chunks skipped (insufficient caption/context) | Should decrease with tuning |
| `subfigure_fragments_merged` | Subfigure fragments merged into parent figures | Indicates effective fragment consolidation |
| `oversized_table_chunks` | Tables exceeding token limit (long rows) | Acceptable if <1% of tables |
| `aside_noise` | `page_aside_text` blocks filtered | Should match occurrence count |

**Example output:**
```
自检指标:
  - 被过滤的 Contents/List/Figures/Tables/Rev.*: 19
  - 被过滤的弱 figure chunks: 32
  - 合并的子图碎片: 2
  - 超长 table chunks (供人工审核): 5
  - page_aside_text 噪声块: 434
```

**Quality thresholds:**
- Weak figure chunks should be <10% of total figures after tuning nearby text detection
- Oversized table chunks should be logged but not necessarily "fixed" if they preserve semantic completeness
- Low-value sections should always be filtered to prevent RAG pollution

## Validation Checklist

Before considering conversion complete, verify:

- [ ] **Image paths:** All non-empty `image_path` values are project-root-relative and point to existing files
- [ ] **Nearby text:** >90% of figure chunks have `nearby_text` different from `section_title`
- [ ] **No empty chunks:** Zero chunks with empty `chunk_text`
- [ ] **Table length:** No table chunk exceeds ~3600 chars (900 tokens × 4)
- [ ] **Section title clean:** Zero `section_title` values with leading/trailing spaces or multiple consecutive spaces
- [ ] **Minimal output:** JSONL contains only 6 fields (`chunk_id`, `page_no`, `content_type`, `section_title`, `chunk_text`, `image_path`)
- [ ] **Manifest completeness:** All prefixed files (`*_content_list.json`, `*_model.json`) recorded in `input_files_all`
- [ ] **Split metadata:** Split tables have `metadata.split_info` with `is_split: true`

## Example Implementation

See `converter.py` in this directory for a complete, battle-tested implementation addressing all common mistakes above.

Key architectural decisions:
1. **Two-phase processing:** Build cache first, then process (enables accurate nearby text lookup)
2. **Strict validation:** Reject invalid image paths (directories, missing extensions)
3. **Token-based splitting:** Estimate tokens from character count for table splitting
4. **Comprehensive logging:** Track skipped blocks, missing images, and parse errors separately
