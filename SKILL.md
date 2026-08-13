---
name: mineru-kb-packager
description: Convert one or more local PDF files into retrieval-ready JSONL knowledge-base packages by acquiring complete structured MinerU results and applying the established chunking rules. Use for PDF ingestion into the MinerU-based RAG knowledge base. Requires MINERU_TOKEN.
---

# MinerU KB Packager

Run the bundled pipeline:

`PDF → MinerU structured result → KB JSONL`

## Run

Use `acquire_mineru.py` from this Skill directory.

```bash
# One or more PDFs
python3 <skill-dir>/acquire_mineru.py /path/to/document.pdf
python3 <skill-dir>/acquire_mineru.py /path/to/part-1.pdf /path/to/part-2.pdf

# All PDFs directly inside a directory
python3 <skill-dir>/acquire_mineru.py /path/to/pdf-directory
```

Store complete results under a specific root when needed:

```bash
python3 <skill-dir>/acquire_mineru.py /path/to/pdfs \
  --output-root /path/to/knowledge-base
```

Also collect one JSONL file per document in a shared directory when needed:

```bash
python3 <skill-dir>/acquire_mineru.py /path/to/pdfs \
  --output-root /path/to/knowledge-base \
  --shared-output /path/to/knowledge-base/output
```

Read the Token only from `MINERU_TOKEN`. Require a fresh target directory; if the computed target already exists, report the conflict and stop.

## Parsing options

Use `vlm` with table and formula recognition enabled by default.

Use `en` as the default language. Pass `--language ch` when the request, filename, or available context indicates a Chinese document. Add `--ocr` for scanned PDFs or PDFs without usable text.

## Pipeline behavior

`acquire_mineru.py` performs the complete operation:

1. Validate and batch the PDF inputs.
2. Upload them and wait for MinerU processing.
3. Download and safely extract each complete result.
4. Require a valid `*_content_list_v2.json`.
5. Run `converter.py` and validate the generated knowledge-base package.

Keep the command running until it exits. It reports page progress when MinerU provides it and otherwise emits a heartbeat every 60 seconds; continue waiting on that process during temporary stalls. Result Zip validation retries are automatic.

Each PDF produces `<pdf-name>.pdf-mineru/` containing the MinerU result and:

```text
output/
├── kb_chunks.jsonl
├── kb_manifest.json
├── error_report.json
└── README_kb.md
```

## Completion criteria

Report completion only after every successful document has:

- a valid `*_content_list_v2.json`;
- a non-empty `output/kb_chunks.jsonl` with no empty `chunk_text`;
- `output/kb_manifest.json` and `output/error_report.json`;
- valid files for every non-empty `image_path`.
- no unsupported MinerU block types or converter parse errors.

Report successful and failed document counts and the final JSONL paths.

## Output rules

- Prefer `content_list_v2.json`; never use `full.md` as structured input.
- Emit only `chunk_id`, `page_no`, `content_type`, `section_title`, `chunk_text`, and `image_path` in JSONL.
- Filter empty blocks, page furniture, contents pages, and revision-history sections.
- Split long tables by row and repeat their title and headers; do not truncate cells.
- Resolve image paths relative to the knowledge-base root and use adjacent body text as figure context.
- Preserve MinerU `chart` blocks as figures and `code` blocks as searchable text.
- Strip and normalize whitespace in `section_title`.
