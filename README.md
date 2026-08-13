# MinerU KB Packager

Turn local PDFs into retrieval-ready JSONL knowledge-base packages with the MinerU API.

The package runs the complete workflow:

```text
PDF → MinerU structured result → cleaned KB JSONL
```

It uploads one or more PDFs, waits for MinerU parsing, downloads the complete structured result, and converts it into semantic text, table, figure, and formula chunks. The implementation uses only the Python standard library.

## Features

- End-to-end PDF acquisition through the MinerU Precision API
- Batch upload and processing for multiple PDFs
- Live page progress when MinerU provides it, with a heartbeat during long waits
- Automatic retry when a completed result Zip is temporarily unavailable
- Structured-source processing based on `content_list_v2.json`
- Semantic chunking for paragraphs, lists, tables, figures, charts, code, and formulas
- Long-table splitting with repeated captions and headers
- Figure and chart enrichment using nearby document text
- Filtering for page furniture, empty blocks, contents pages, and revision-history sections
- Minimal six-field JSONL output for RAG ingestion
- Manifest, validation, and error reporting

## Requirements

- Python 3.8 or newer
- Network access to MinerU
- A MinerU API token from [MinerU API Management](https://mineru.net/apiManage/docs)

Set the token in the environment before running the package:

```bash
export MINERU_TOKEN="your-token"
```

For unattended use, configure this variable in your shell environment or secret manager. Do not commit tokens to the repository.

## Quick Start

Process one PDF:

```bash
python3 acquire_mineru.py /path/to/document.pdf
```

Process several PDFs in one invocation:

```bash
python3 acquire_mineru.py \
  /path/to/part-1.pdf \
  /path/to/part-2.pdf
```

Process every PDF directly inside a directory:

```bash
python3 acquire_mineru.py /path/to/pdf-directory
```

Store MinerU result directories under a chosen root and collect one final JSONL per document:

```bash
python3 acquire_mineru.py /path/to/pdf-directory \
  --output-root /path/to/knowledge-base \
  --shared-output /path/to/knowledge-base/output
```

## Parsing Options

The defaults are optimized for English technical documents:

- model: `vlm`
- language: `en`
- table recognition: enabled
- formula recognition: enabled

Use Chinese when the document is Chinese:

```bash
python3 acquire_mineru.py document.pdf --language ch
```

Force OCR for scanned PDFs or PDFs without usable embedded text:

```bash
python3 acquire_mineru.py scanned-document.pdf --ocr
```

See every option with:

```bash
python3 acquire_mineru.py --help
```

## Input Limits and Safety

- The package accepts PDF files only.
- Each PDF is checked locally against MinerU's 200 MB file-size limit.
- MinerU enforces its page-count limit and returns an explicit API error when a file exceeds it.
- PDF splitting is intentionally not automated; split large documents at meaningful semantic boundaries before running the package.
- Existing result directories are never overwritten or reused. Move them or choose another `--output-root` before rerunning the same PDF.
- Downloaded Zip files are checked for path traversal, symbolic links, abnormal expansion, and required structured content.

## Output

For `document.pdf`, the default result directory is:

```text
document.pdf-mineru/
├── content_list_v2.json or *_content_list_v2.json
├── images/
├── mineru_acquisition.json
└── output/
    ├── kb_chunks.jsonl
    ├── kb_manifest.json
    ├── error_report.json
    └── README_kb.md
```

When `--shared-output` is provided, the package also writes:

```text
<shared-output>/document.jsonl
```

## JSONL Schema

Each line in `kb_chunks.jsonl` contains exactly six fields:

```json
{
  "chunk_id": "document_abcd1234:p12:text:42",
  "page_no": 12,
  "content_type": "text",
  "section_title": "Electrical Specifications",
  "chunk_text": "The device supports ...",
  "image_path": "document.pdf-mineru/images/example.jpg"
}
```

`content_type` is one of `text`, `table`, `figure`, or `formula`. MinerU `chart` blocks become figures, while `code` blocks are preserved as searchable text.

For ingestion, use `chunk_text` as the content field and the remaining fields as metadata. Avoid applying another fixed sliding-window split unless the downstream system requires it.

## Processing and Validation

`acquire_mineru.py` is the public entry point. It invokes `converter.py` after the complete MinerU result is installed.

The run reports success only when:

- a unique, readable `content_list_v2.json` is present;
- `kb_chunks.jsonl` is non-empty and every chunk has the expected schema;
- every `chunk_text` is non-empty;
- every non-empty `image_path` resolves to an existing file;
- the converter reports no unsupported MinerU block types or parse errors.

The generated `error_report.json` still records intentionally skipped noise and empty placeholder blocks for inspection.

## Repository Contents

- `acquire_mineru.py`: upload, polling, result installation, conversion, and final validation
- `converter.py`: structured MinerU output to knowledge-base JSONL conversion
- `SKILL.md`: concise agent workflow for running the package
- `README.md`: public usage and behavior documentation

## Community

Special thanks to the community for their support and contributions: <https://linux.do/>

## License

MIT
