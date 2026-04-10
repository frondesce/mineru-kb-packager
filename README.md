# MinerU KB Packager

Post-process a MinerU extraction directory into a retrieval-ready knowledge base dataset for RAG pipelines.

This project is not a PDF parser. It assumes MinerU has already parsed the source document and produced structured outputs such as `content_list_v2.json`, `content_list.json`, `layout.json`, and `images/`.

The packager takes those intermediate files and turns them into cleaner, more usable ingestion artifacts:

- semantic text/table/figure/formula chunks
- normalized image paths
- low-value section filtering
- figure enrichment using nearby text
- subfigure fragment merging
- manifest and error reporting

## Why This Exists

MinerU is good at extracting structure from PDFs, but its raw output is still an intermediate representation.

In practice, directly indexing `full.md` or raw block JSON often leads to:

- noisy headers, footers, and page numbers in retrieval results
- oversized or fragmented chunks
- tables preserved as HTML instead of retrieval-friendly text
- figures stored as bare image references without usable semantics
- duplicated low-value sections such as contents, lists of figures, and revision history

This project fills that gap by turning MinerU output into a cleaner dataset that is easier to ingest into vector databases and downstream QA systems.

## Features

- Prefer structured MinerU sources over `full.md`
- Semantic chunking for text, tables, figures, and formulas
- Table normalization into retrieval-friendly text
- Figure descriptions enriched with nearby paragraph context
- Weak figure filtering and subfigure fragment merging
- Low-value section filtering:
  - contents
  - list of figures
  - list of tables
  - revision history
- Relative image path normalization
- Minimal JSONL output for ingestion
- Manifest and error report generation

## Input

Expected MinerU output directory:

```text
mineru_output/
├── content_list_v2.json
├── content_list.json
├── layout.json
├── model.json
├── origin.pdf
├── full.md
└── images/
```

Source priority:

1. `content_list_v2.json`
2. `content_list.json`
3. `images/`
4. `origin.pdf`
5. `full.md` as supplementary reference only

## Output

By default, the packager writes these files into `<input_dir>/output/`:

- `kb_chunks.jsonl`: main ingestion file
- `kb_manifest.json`: processing metadata and statistics
- `README_kb.md`: generated dataset-level description
- `error_report.json`: skipped blocks, unsupported blocks, and parse issues

## Quick Start

```bash
python3 converter.py <input_dir>
```

Example:

```bash
python3 converter.py ./mineru_output/my_document.pdf-uuid
```

With custom output directory:

```bash
python3 converter.py ./mineru_output/my_document.pdf-uuid --output-dir ./kb_output
```

## Main Output Schema

Each line in `kb_chunks.jsonl` is a single JSON object with minimal ingestion fields:

```json
{
  "chunk_id": "my_doc_1234abcd:p12:text:42",
  "page_no": 12,
  "content_type": "text",
  "section_title": "Electrical Specifications",
  "chunk_text": "The device supports ...",
  "image_path": ""
}
```

Fields:

- `chunk_id`: stable unique chunk identifier
- `page_no`: 1-based page number
- `content_type`: `text`, `table`, `figure`, or `formula`
- `section_title`: cleaned section title
- `chunk_text`: retrieval text
- `image_path`: relative image path from project root, empty if not applicable

## Processing Rules

### Text

- Merge adjacent short blocks within the same section
- Remove obvious page noise such as repeated headers and footers
- Keep technical terms, units, part numbers, and symbols intact

### Tables

- Convert table structure into retrieval-friendly text
- Split long tables by row groups when possible
- Repeat title and headers for split table chunks
- Preserve very long single-row content when truncation would destroy meaning

### Figures

- Use figure captions when available
- Enrich figure chunks with nearby paragraph/list text
- Merge short subfigure labels such as `MLC Page`, `TLC Page`, `SLC Page`, or `Top View` into the nearest formal figure
- Skip weak figure chunks that have no useful caption and no useful context

### Formulas

- Preserve formula meaning
- Avoid aggressive rewriting of math expressions

## Design Principles

- Structured-source first
- Retrieval quality over markdown fidelity
- Minimal ingestion schema
- Conservative cleanup
- Prefer semantic completeness over aggressive truncation

## When To Use

Use this project when:

- you already have MinerU output
- you want better RAG ingestion quality than raw `full.md`
- you need chunked, searchable knowledge base files
- your documents contain figures, formulas, and dense technical tables

Do not use it when:

- you need PDF parsing from scratch
- your source is not MinerU output
- you only want plain markdown export

## Limitations

- It depends on the quality of MinerU extraction
- Extremely long technical tables may still produce oversized chunks if preserving full row semantics is more important than strict token caps
- Figure enrichment depends on nearby text being recoverable from MinerU block order

## Suggested Use In RAG Pipelines

Use `kb_chunks.jsonl` as the main ingestion file.

Recommended mapping:

- content field: `chunk_text`
- metadata fields:
  - `chunk_id`
  - `page_no`
  - `content_type`
  - `section_title`
  - `image_path`

In general, do not re-chunk this file with a fixed sliding window unless your system strictly requires it.

## Repository Contents

- `converter.py`: main converter
- `SKILL.md`: internal skill-oriented workflow notes
- `README.md`: public project overview

## Community

Special thanks to the community for their support and contributions: https://linux.do/ 

## License

MIT
