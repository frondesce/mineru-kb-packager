#!/usr/bin/env python3
"""
MinerU 输出目录转换为通用 RAG / 知识库打包格式

用法:
    python converter.py <input_dir> [--output-dir <output_dir>]

示例:
    python converter.py ./mineru_output/my_document.pdf-uuid
"""

import json
import os
import re
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from collections import defaultdict


class MinerUConverter:
    """将 MinerU 输出转换为通用知识库 / RAG 数据集"""

    # 块类型映射
    TYPE_MAPPING = {
        "paragraph": "text",
        "list": "text",
        "table": "table",
        "image": "figure",
        "equation_interline": "formula",
        "title": "title",
        "page_header": "noise",
        "page_footer": "noise",
        "page_number": "noise",
        "header": "noise",
        "footer": "noise",
        "page_aside_text": "noise",
    }

    # 目标文本长度
    TARGET_TOKENS_MIN = 300
    TARGET_TOKENS_MAX = 700
    HARD_CAP_TOKENS = 900

    # 需要过滤的章节标题模式（目录、索引、修订历史等）
    SKIP_SECTION_PATTERNS = [
        r'^\s*contents\s*$',
        r'^\s*table of contents\s*$',
        r'^\s*list of figures\s*$',
        r'^\s*list of tables\s*$',
        r'^\s*rev\.\s*\w+',  # Rev. A, Rev. B 等
        r'^\s*revision\s*history\s*$',
        r'^\s*document\s*history\s*$',
        r'^\s*change\s*history\s*$',
    ]

    # 子图碎片标签模式（短标签、无独立语义）
    SUBFIGURE_LABEL_PATTERNS = [
        r'^\s*(MLC|TLC|SLC)\s+Page\s*$',
        r'^\s*Top\s+View\s*$',
        r'^\s*Bottom\s+View\s*$',
        r'^\s*Side\s+View\s*$',
        r'^\s*Front\s+View\s*$',
        r'^\s*Back\s+View\s*$',
        r'^\s*Left\s*$',
        r'^\s*Right\s*$',
        r'^\s*Detail\s*[A-Z]\s*$',
        r'^\s*Zoom\s*(In|Out)\s*$',
        r'^\s*Close[\s-]*up\s*$',
        r'^\s*Enlarged\s+View\s*$',
        r'^\s*Partial\s+View\s*$',
        r'^\s*Note[\s:]*$',  # 单独的 "Note:" 标签
        r'^\s*Legend\s*$',
    ]

    # 正式图编号模式
    FIGURE_NUMBER_PATTERN = r'(Figure|Fig\.?|图)\s*\d+[\w\-\.]*'

    def __init__(self, input_dir: str, output_dir: Optional[str] = None,
                 shared_output_dir: Optional[str] = None):
        self.input_dir = Path(input_dir).resolve()
        self.project_root = Path.cwd().resolve()

        # 使用输入目录名生成 doc_id
        self.doc_id = self._generate_doc_id(self.input_dir.name)
        self.doc_title = self._extract_doc_title(self.input_dir.name)

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.input_dir / "output"

        # 共享输出目录（汇集所有文档的 kb_chunks）
        self.shared_output_dir = Path(shared_output_dir).resolve() if shared_output_dir else None

        # 统计数据
        self.stats = {
            "total_chunks": 0,
            "text_chunks": 0,
            "table_chunks": 0,
            "figure_chunks": 0,
            "formula_chunks": 0,
            "missing_images": 0,
            "unparseable_blocks": 0,
            "skipped_noise": 0,
            "skipped_empty": 0,
            "skipped_low_value_sections": 0,
            "skipped_weak_figures": 0,
            "aside_noise": 0,
            "oversized_table_chunks": 0,
            "subfigure_fragments_merged": 0,
        }

        # 错误报告
        self.error_report = {
            "doc_id": self.doc_id,
            "summary": {},
            "missing_images": [],
            "skipped_blocks": [],
            "unsupported_blocks": [],
            "parse_errors": [],
        }

        # 内容列表
        self.chunks = []

        # 输入文件记录 - 区分"存在的文件"和"使用的文件"
        self.input_files_exist = {}  # 实际存在的文件
        self.input_files_used = {}   # 实际使用的文件

        # 当前章节上下文
        self.current_section_title = ""
        self.section_path = []
        self.skip_current_section = False

        # 子图碎片合并：缓存最近的主图 chunk
        self.last_formal_figure = None  # 存储最近正式图的索引和相关信息

        # 页块缓存: page_no -> [(idx, type, text, section_title), ...]
        self.page_blocks_cache = {}

    def _generate_doc_id(self, dirname: str) -> str:
        """生成稳定的 doc_id"""
        base = re.sub(r'-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', '', dirname)
        base = re.sub(r'\.pdf$', '', base)
        hash_suffix = hashlib.md5(dirname.encode()).hexdigest()[:8]
        return f"{base}_{hash_suffix}"

    def _extract_doc_title(self, dirname: str) -> str:
        """提取文档标题"""
        base = re.sub(r'-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', '', dirname)
        base = re.sub(r'\.pdf$', '', base)
        return base

    def _get_relative_path(self, path: Path) -> str:
        """获取相对于项目根目录的路径"""
        if path is None:
            return ""
        try:
            rel = path.relative_to(self.project_root)
            return str(rel).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def _is_valid_image_path(self, path: str) -> bool:
        """检查图片路径是否有效"""
        if not path or path.endswith('/'):
            return False
        ext = Path(path).suffix.lower()
        return ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

    def discover_files(self) -> dict:
        """发现并识别输入目录中的可用文件"""
        files_exist = {
            "origin_pdf": None,
            "content_list_v2": None,
            "content_list": None,
            "layout": None,
            "model": None,
            "full_md": None,
            "images_dir": None,
        }

        files_used = {}

        # 查找 origin.pdf 或 *_origin.pdf
        for f in self.input_dir.iterdir():
            if f.is_file() and (f.name.endswith("_origin.pdf") or f.name == "origin.pdf"):
                files_exist["origin_pdf"] = f
                files_used["origin_pdf"] = f
                break

        # 查找 content_list_v2.json (优先) 或带前缀的 *_content_list_v2.json
        v2_candidates = list(self.input_dir.glob("*_content_list_v2.json"))
        if v2_candidates:
            files_exist["content_list_v2"] = v2_candidates[0]
            files_used["content_list_v2"] = v2_candidates[0]
        elif (self.input_dir / "content_list_v2.json").exists():
            files_exist["content_list_v2"] = self.input_dir / "content_list_v2.json"
            files_used["content_list_v2"] = files_exist["content_list_v2"]

        # 查找 content_list.json 或带前缀的 *_content_list.json (记录存在，但不使用)
        cl_candidates = list(self.input_dir.glob("*_content_list.json"))
        for c in cl_candidates:
            if "_content_list_v2" not in c.name:
                files_exist["content_list"] = c
                break
        if not files_exist["content_list"] and (self.input_dir / "content_list.json").exists():
            files_exist["content_list"] = self.input_dir / "content_list.json"

        # 如果没找到 v2，使用 content_list
        if not files_used.get("content_list_v2") and files_exist.get("content_list"):
            files_used["content_list"] = files_exist["content_list"]

        # 查找 layout.json 或带前缀的 *_layout.json
        layout_candidates = list(self.input_dir.glob("*_layout.json"))
        if layout_candidates:
            files_exist["layout"] = layout_candidates[0]
            files_used["layout"] = layout_candidates[0]
        elif (self.input_dir / "layout.json").exists():
            files_exist["layout"] = self.input_dir / "layout.json"
            files_used["layout"] = files_exist["layout"]

        # 查找 model.json 或带前缀的 *_model.json
        model_candidates = list(self.input_dir.glob("*_model.json"))
        if model_candidates:
            files_exist["model"] = model_candidates[0]
            files_used["model"] = model_candidates[0]
        elif (self.input_dir / "model.json").exists():
            files_exist["model"] = self.input_dir / "model.json"
            files_used["model"] = files_exist["model"]

        # 查找 full.md
        md_file = self.input_dir / "full.md"
        if md_file.exists():
            files_exist["full_md"] = md_file
            files_used["full_md"] = md_file

        # 查找 images 目录
        images_dir = self.input_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            files_exist["images_dir"] = images_dir
            files_used["images_dir"] = images_dir

        self.input_files_exist = files_exist
        self.input_files_used = files_used
        return files_exist, files_used

    def load_content(self) -> list:
        """加载内容列表"""
        content_data = []

        content_file = self.input_files_used.get("content_list_v2") or self.input_files_used.get("content_list")

        if content_file:
            try:
                with open(content_file, "r", encoding="utf-8") as f:
                    content_data = json.load(f)
                print(f"已加载: {content_file.name}")
            except Exception as e:
                self.error_report["parse_errors"].append({
                    "file": content_file.name,
                    "error": str(e),
                })

        return content_data

    def extract_text_from_content(self, content_items: list) -> str:
        """从内容项列表中提取纯文本"""
        texts = []
        for item in content_items:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("content", ""))
                elif item.get("type") == "equation_inline":
                    texts.append(item.get("content", ""))
        return "".join(texts)

    def extract_text_from_paragraph(self, block: dict) -> str:
        """从段落块中提取文本"""
        content = block.get("content", {})
        para_content = content.get("paragraph_content", [])
        return self.extract_text_from_content(para_content)

    def extract_text_from_list(self, block: dict) -> str:
        """从列表块中提取文本"""
        content = block.get("content", {})
        items = content.get("list_items", [])
        texts = []
        for item in items:
            if item.get("item_type") == "text":
                item_content = item.get("item_content", [])
                text = self.extract_text_from_content(item_content)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    def extract_title_text(self, block: dict) -> str:
        """提取标题文本"""
        content = block.get("content", {})
        title_content = content.get("title_content", [])
        return self.extract_text_from_content(title_content)

    def extract_table_data(self, block: dict) -> tuple:
        """提取表格数据，返回 (headers, rows, caption, footnote)"""
        content = block.get("content", {})

        # 表标题
        caption = content.get("table_caption", [])
        caption_text = self.extract_text_from_content(caption)

        # 表注
        footnote = content.get("table_footnote", [])
        footnote_text = self.extract_text_from_content(footnote)

        # HTML 表格内容
        html = content.get("html", "")
        headers = []
        rows = []

        if html:
            try:
                html_rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)
                for i, row in enumerate(html_rows):
                    cells = re.findall(r'<t[dh]>(.*?)</t[dh]>', row, re.DOTALL)
                    cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    if i == 0:
                        headers = cells
                    else:
                        rows.append(cells)
            except Exception as e:
                self.error_report["parse_errors"].append({
                    "block_type": "table",
                    "error": f"HTML parse error: {str(e)}",
                })

        return headers, rows, caption_text, footnote_text

    def split_table_into_chunks(self, headers: list, rows: list, caption: str, footnote: str,
                                 page_no: int, block_index: int, image_path: str) -> list:
        """将表格拆分成多个 chunks，确保每个 chunk 不超过 HARD_CAP_TOKENS"""
        chunks = []

        # 构建完整文本以计算总长度
        texts = []
        if caption:
            texts.append(f"Table: {caption}")
        if headers:
            texts.append("Columns: " + " | ".join(headers))
        for i, row in enumerate(rows, 1):
            if row:
                texts.append(f"Row {i}: " + " | ".join(row))
        if footnote:
            texts.append(f"Note: {footnote}")

        full_text = "\n".join(texts)
        total_tokens = self.estimate_tokens(full_text)

        # 如果总长度未超限，不拆分
        if total_tokens <= self.HARD_CAP_TOKENS:
            is_valid = len(full_text.strip()) > 20
            if is_valid:
                chunks.append({
                    "text": full_text,
                    "is_split": False,
                    "split_index": 0,
                    "total_splits": 1
                })
            return chunks

        # 需要拆分 - 计算每行平均 token 数
        avg_row_tokens = total_tokens // max(len(rows), 1)

        # 计算每个 chunk 能容纳的行数（保守估计，留出余量）
        header_text = ""
        if caption:
            header_text += f"Table: {caption} (Part X/Y)\n"
        if headers:
            header_text += "Columns: " + " | ".join(headers)
        header_tokens = self.estimate_tokens(header_text)

        # 每 chunk 可用 token 数（为脚注留余量）
        available_tokens = self.HARD_CAP_TOKENS - header_tokens - 150  # 增加余量
        rows_per_chunk = max(3, available_tokens // max(avg_row_tokens, 1))

        # 使用动态调整确保每个 chunk 都不超限
        # 注意：如果单行本身超长，为保语义完整，不强行截断，仅记录
        current_row = 0
        chunk_index = 0

        while current_row < len(rows):
            chunk_index += 1
            chunk_rows = []
            chunk_tokens = header_tokens

            # 尽可能多地添加行，但不超过限制
            for i in range(current_row, len(rows)):
                row_text = f"Row {i + 1}: " + " | ".join(rows[i]) if rows[i] else ""
                row_tokens = self.estimate_tokens(row_text)

                # 留出余量给脚注
                margin = 100 if (footnote and i == len(rows) - 1) else 50

                if chunk_tokens + row_tokens > self.HARD_CAP_TOKENS - margin and chunk_rows:
                    break

                chunk_rows.append(rows[i])
                chunk_tokens += row_tokens

            if not chunk_rows:
                # 单行就超限的情况，为保语义完整，单独成行
                chunk_rows = [rows[current_row]]
                current_row += 1
            else:
                current_row += len(chunk_rows)

            # 构建 chunk 文本
            texts = []
            if caption:
                texts.append(f"Table: {caption} (Part {chunk_index}/{max(2, (len(rows) + len(chunk_rows) - 1) // len(chunk_rows))})")
            if headers:
                texts.append("Columns: " + " | ".join(headers))

            for i, row in enumerate(chunk_rows, current_row - len(chunk_rows) + 1):
                if row:
                    texts.append(f"Row {i}: " + " | ".join(row))

            # 只在最后一个 chunk 添加脚注
            if current_row >= len(rows) and footnote:
                texts.append(f"Note: {footnote}")

            chunk_text = "\n".join(texts)

            # 验证 chunk 长度
            actual_tokens = self.estimate_tokens(chunk_text)
            if actual_tokens > self.HARD_CAP_TOKENS:
                self.stats["oversized_table_chunks"] += 1
                self.error_report.setdefault("oversized_table_chunks", []).append({
                    "block_id": f"p{page_no}_b{block_index}",
                    "tokens": actual_tokens,
                    "limit": self.HARD_CAP_TOKENS,
                    "page_no": page_no,
                })

            chunks.append({
                "text": chunk_text,
                "is_split": True,
                "split_index": chunk_index,
                "total_splits": None  # 将在后面更新
            })

        # 更新 total_splits
        total_splits = len(chunks)
        for c in chunks:
            c["total_splits"] = total_splits
            # 修正 Part X/Y 中的 Y
            if caption and "(Part " in c["text"]:
                c["text"] = re.sub(r"\(Part \d+/\d+\)", f"(Part {c['split_index']}/{total_splits})", c["text"])

        return chunks

    def extract_image_info(self, block: dict) -> tuple:
        """提取图片信息"""
        content = block.get("content", {})
        image_source = content.get("image_source", {})
        image_path = image_source.get("path", "")

        caption = content.get("image_caption", [])
        caption_text = self.extract_text_from_content(caption)

        footnote = content.get("image_footnote", [])
        footnote_text = self.extract_text_from_content(footnote)

        texts = []
        if caption_text:
            texts.append(f"Figure: {caption_text}")
        if footnote_text:
            texts.append(f"Note: {footnote_text}")

        return image_path, "\n".join(texts), caption_text

    def extract_formula_text(self, block: dict) -> str:
        """提取公式文本"""
        content = block.get("content", {})
        return content.get("math_content", "")

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量"""
        return len(text) // 4

    def split_long_text(self, text: str, max_tokens: int = HARD_CAP_TOKENS) -> list:
        """将长文本按自然边界切分"""
        if self.estimate_tokens(text) <= max_tokens:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)

            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                sentences = re.split(r'(?<=[.!?。！？])\s+', para)
                temp_chunk = []
                temp_tokens = 0

                for sent in sentences:
                    sent_tokens = self.estimate_tokens(sent)
                    if temp_tokens + sent_tokens <= max_tokens:
                        temp_chunk.append(sent)
                        temp_tokens += sent_tokens
                    else:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [sent]
                        temp_tokens = sent_tokens

                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                if current_tokens + para_tokens <= max_tokens:
                    current_chunk.append(para)
                    current_tokens += para_tokens
                else:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_tokens = para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def merge_short_text_blocks(self, blocks: list) -> list:
        """合并相邻的短正文块"""
        if not blocks:
            return blocks

        merged = []
        current_group = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self.estimate_tokens(block.get("text", ""))

            if block_tokens < self.TARGET_TOKENS_MIN:
                if current_tokens + block_tokens <= self.TARGET_TOKENS_MAX:
                    current_group.append(block)
                    current_tokens += block_tokens
                else:
                    if current_group:
                        merged.append(self._merge_block_group(current_group))
                    current_group = [block]
                    current_tokens = block_tokens
            else:
                if current_group:
                    merged.append(self._merge_block_group(current_group))
                    current_group = []
                    current_tokens = 0
                merged.append(block)

        if current_group:
            merged.append(self._merge_block_group(current_group))

        return merged

    def _merge_block_group(self, group: list) -> dict:
        """合并块组"""
        if len(group) == 1:
            return group[0]

        texts = [b.get("text", "") for b in group]
        merged_text = "\n\n".join(texts)
        block_ids = [b.get("source_block_id", "") for b in group if b.get("source_block_id")]

        return {
            "type": "text",
            "text": merged_text,
            "page_no": group[0].get("page_no", 1),
            "section_title": group[0].get("section_title", ""),
            "source_block_id": f"merged_{group[0].get('source_block_id', '')}",
            "merge_from_block_ids": block_ids,
            "bbox": group[0].get("bbox", []),
        }

    def _find_text_in_page_blocks(
        self,
        page_blocks: list,
        positions: list,
        section_title: str = "",
    ) -> str:
        """从缓存块中查找符合条件的正文"""
        for pos in positions:
            if pos < 0 or pos >= len(page_blocks):
                continue
            _, btype, text, block_section = page_blocks[pos]
            if btype not in ["paragraph", "list"] or not text or len(text) <= 20:
                continue
            if section_title and block_section and block_section != section_title:
                continue
            return text[:500]
        return ""

    def _find_recent_text_chunk(self, page_no: int, section_title: str) -> str:
        """从已生成的 text chunk 中回退查找最近的同章节正文"""
        for chunk in reversed(self.chunks):
            if chunk["content_type"] != "text":
                continue
            if section_title and chunk["section_title"] != section_title:
                continue
            if abs(chunk["page_no"] - page_no) > 2:
                continue
            if chunk["chunk_text"]:
                return chunk["chunk_text"][:500]
        return ""

    def find_nearby_text(self, page_no: int, block_idx: int, section_title: str = "") -> str:
        """查找真正的邻近正文文本，优先同页，其次同章节相邻页/最近文本块"""
        # 从缓存中获取当前页的所有块
        page_blocks = self.page_blocks_cache.get(page_no, [])

        if not page_blocks:
            return self._find_recent_text_chunk(page_no, section_title) or section_title

        # 查找当前块在缓存中的位置
        current_pos = None
        for i, (idx, _, _, _) in enumerate(page_blocks):
            if idx == block_idx:
                current_pos = i
                break

        if current_pos is None:
            return self._find_recent_text_chunk(page_no, section_title) or section_title

        window_size = 10
        forward_positions = [current_pos + offset for offset in range(1, window_size + 1)]
        backward_positions = [current_pos - offset for offset in range(1, window_size + 1)]

        nearby_text = self._find_text_in_page_blocks(page_blocks, forward_positions, section_title)
        if nearby_text:
            return nearby_text

        nearby_text = self._find_text_in_page_blocks(page_blocks, backward_positions, section_title)
        if nearby_text:
            return nearby_text

        # 同章节最近 text chunk 往往比章节标题更有语义
        nearby_text = self._find_recent_text_chunk(page_no, section_title)
        if nearby_text:
            return nearby_text

        # 跨页回退：优先上一页末尾，再下一页开头
        for page_delta in (1, 2):
            prev_blocks = self.page_blocks_cache.get(page_no - page_delta, [])
            if prev_blocks:
                nearby_text = self._find_text_in_page_blocks(
                    prev_blocks,
                    list(range(len(prev_blocks) - 1, -1, -1)),
                    section_title,
                )
                if nearby_text:
                    return nearby_text

            next_blocks = self.page_blocks_cache.get(page_no + page_delta, [])
            if next_blocks:
                nearby_text = self._find_text_in_page_blocks(
                    next_blocks,
                    list(range(len(next_blocks))),
                    section_title,
                )
                if nearby_text:
                    return nearby_text

        return section_title or self.current_section_title

    def build_page_blocks_cache(self, content_data: list):
        """构建页块缓存，用于查找邻近文本和章节上下文"""
        section_path = []
        current_section_title = ""

        for page_no, page_blocks in enumerate(content_data, 1):
            self.page_blocks_cache[page_no] = []
            for idx, block in enumerate(page_blocks):
                block_type = block.get("type", "")
                text = ""

                if block_type == "title":
                    title_text = self.extract_title_text(block)
                    level = block.get("content", {}).get("level", 1)
                    while len(section_path) >= level:
                        section_path.pop()
                    section_path.append(title_text)
                    current_section_title = " > ".join(section_path)

                if block_type == "paragraph":
                    text = self.extract_text_from_paragraph(block)
                elif block_type == "list":
                    text = self.extract_text_from_list(block)
                elif block_type == "title":
                    text = self.clean_section_title(self.extract_title_text(block))

                text = self.clean_text(text)

                self.page_blocks_cache[page_no].append((idx, block_type, text, current_section_title))

    def process_blocks(self, content_data: list):
        """处理所有块并生成 chunks"""
        # 首先构建块位置缓存
        self.build_page_blocks_cache(content_data)

        page_no = 0
        block_index = 0
        text_buffer = []

        for page_blocks in content_data:
            page_no += 1

            for block_idx, block in enumerate(page_blocks):
                block_index += 1
                block_type = block.get("type", "")
                bbox = block.get("bbox", [])

                # 更新章节标题
                if block_type == "title":
                    title_text = self.extract_title_text(block)
                    level = block.get("content", {}).get("level", 1)

                    while len(self.section_path) >= level:
                        self.section_path.pop()
                    self.section_path.append(title_text)
                    self.current_section_title = " > ".join(self.section_path)
                    self.last_formal_figure = None

                    # 检查是否是低价值章节（目录、索引、修订历史）
                    self.skip_current_section = self._should_skip_section(title_text)
                    if self.skip_current_section:
                        self.stats["skipped_low_value_sections"] += 1
                        self.error_report["skipped_blocks"].append({
                            "block_id": f"p{page_no}_b{block_index}",
                            "type": block_type,
                            "reason": "low-value section skipped (contents/index/revision history)",
                            "page_no": page_no,
                            "section_title": title_text,
                        })
                    else:
                        self.stats["skipped_noise"] += 1
                        self.error_report["skipped_blocks"].append({
                            "block_id": f"p{page_no}_b{block_index}",
                            "type": block_type,
                            "reason": "title used for section tracking only",
                            "page_no": page_no,
                        })
                    continue

                # 跳过噪音类型（包括页边文本）
                if block_type in ["page_header", "page_footer", "page_number", "header", "footer", "page_aside_text"]:
                    if block_type == "page_aside_text":
                        self.stats["aside_noise"] += 1
                    else:
                        self.stats["skipped_noise"] += 1
                    self.error_report["skipped_blocks"].append({
                        "block_id": f"p{page_no}_b{block_index}",
                        "type": block_type,
                        "reason": "noise block skipped",
                        "page_no": page_no,
                    })
                    continue

                # 跳过低价值章节中的内容
                if self.skip_current_section and block_type in ["paragraph", "list", "table"]:
                    self.error_report["skipped_blocks"].append({
                        "block_id": f"p{page_no}_b{block_index}",
                        "type": block_type,
                        "reason": "content in low-value section skipped",
                        "page_no": page_no,
                        "section_title": self.current_section_title,
                    })
                    continue

                # 处理段落和列表
                if block_type in ["paragraph", "list"]:
                    if block_type == "paragraph":
                        text = self.extract_text_from_paragraph(block)
                    else:
                        text = self.extract_text_from_list(block)

                    text = self.clean_text(text)

                    if not text or len(text.strip()) < 10:
                        continue

                    text_buffer.append({
                        "type": "text",
                        "text": text,
                        "page_no": page_no,
                        "section_title": self.current_section_title,
                        "source_block_id": f"p{page_no}_b{block_index}",
                        "bbox": bbox,
                    })
                    continue

                # 先处理缓冲区中的文本
                if text_buffer:
                    merged = self.merge_short_text_blocks(text_buffer)
                    for item in merged:
                        self._add_text_chunk(item)
                    text_buffer = []

                # 处理表格
                if block_type == "table":
                    self._process_table_block(block, page_no, block_idx, block_index)
                    continue

                # 处理图片
                if block_type == "image":
                    self._process_image_block(block, page_no, block_idx, block_index)
                    continue

                # 处理行间公式
                if block_type == "equation_interline":
                    self._process_formula_block(block, page_no, block_index)
                    continue

                # 未支持的块类型
                self.error_report["unsupported_blocks"].append({
                    "block_id": f"p{page_no}_b{block_index}",
                    "type": block_type,
                    "page_no": page_no,
                    "content_preview": str(block.get("content", ""))[:200],
                })

        # 处理剩余的文本缓冲
        if text_buffer:
            merged = self.merge_short_text_blocks(text_buffer)
            for item in merged:
                self._add_text_chunk(item)

    def _add_text_chunk(self, item: dict):
        """添加文本 chunk"""
        text = item.get("text", "")
        chunks_texts = self.split_long_text(text)

        for i, chunk_text in enumerate(chunks_texts):
            chunk_id = f"{self.doc_id}:p{item['page_no']}:text:{len(self.chunks) + 1}"
            section_title = item.get("section_title", "")
            section_path = section_title.split(" > ") if section_title else []

            metadata = {
                "section_path": section_path,
                "block_bbox": item.get("bbox", []),
                "merge_from_block_ids": item.get("merge_from_block_ids", []),
                "cleanup_flags": [],
                "parser": "MinerU-postprocessed",
            }

            if i > 0:
                metadata["cleanup_flags"].append("split_continuation")

            chunk = {
                "doc_id": self.doc_id,
                "doc_title": self.doc_title,
                "source_pdf": self._get_relative_path(self.input_files_used.get("origin_pdf")),
                "page_no": item["page_no"],
                "chunk_id": chunk_id,
                "content_type": "text",
                "section_title": section_title,
                "chunk_text": chunk_text,
                "image_path": "",
                "source_block_id": item.get("source_block_id", ""),
                "metadata": metadata,
            }

            self.chunks.append(chunk)
            self.stats["text_chunks"] += 1
            self.stats["total_chunks"] += 1

    def _process_table_block(self, block: dict, page_no: int, block_idx: int, block_index: int):
        """处理表格块，支持拆分"""
        content = block.get("content", {})

        # 提取表格数据
        headers, rows, caption, footnote = self.extract_table_data(block)

        # 检查图片路径
        image_source = content.get("image_source", {})
        image_path_raw = image_source.get("path", "")

        full_image_path = None
        relative_image_path = ""

        if image_path_raw and self._is_valid_image_path(image_path_raw):
            full_image_path = self.input_dir / image_path_raw
            if full_image_path.exists():
                relative_image_path = self._get_relative_path(full_image_path)
            else:
                self.error_report["missing_images"].append({
                    "block_id": f"p{page_no}_b{block_index}",
                    "expected_path": str(image_path_raw),
                    "type": "table",
                })
                self.stats["missing_images"] += 1

        # 如果没有内容，跳过
        if not headers and not rows and not relative_image_path:
            self.stats["skipped_empty"] += 1
            self.error_report["skipped_blocks"].append({
                "block_id": f"p{page_no}_b{block_index}",
                "type": "table",
                "reason": "empty table content and no valid image",
                "page_no": page_no,
            })
            return

        # 拆分表格
        table_chunks = self.split_table_into_chunks(
            headers, rows, caption, footnote, page_no, block_index, relative_image_path
        )

        # 创建 chunks
        section_title = self.current_section_title
        section_path = section_title.split(" > ") if section_title else []

        for tc in table_chunks:
            chunk_id = f"{self.doc_id}:p{page_no}:table:{len(self.chunks) + 1}"

            metadata = {
                "section_path": section_path,
                "table_title": caption,
                "block_bbox": block.get("bbox", []),
                "cleanup_flags": ["table_split"] if tc["is_split"] else [],
                "parser": "MinerU-postprocessed",
                "split_info": {
                    "is_split": tc["is_split"],
                    "split_index": tc["split_index"],
                    "total_splits": tc["total_splits"]
                } if tc["is_split"] else {}
            }

            chunk = {
                "doc_id": self.doc_id,
                "doc_title": self.doc_title,
                "source_pdf": self._get_relative_path(self.input_files_used.get("origin_pdf")),
                "page_no": page_no,
                "chunk_id": chunk_id,
                "content_type": "table",
                "section_title": section_title,
                "chunk_text": tc["text"],
                "image_path": relative_image_path,
                "source_block_id": f"p{page_no}_b{block_index}",
                "metadata": metadata,
            }

            self.chunks.append(chunk)
            self.stats["table_chunks"] += 1
            self.stats["total_chunks"] += 1

    def _process_image_block(self, block: dict, page_no: int, block_idx: int, block_index: int):
        """处理图片块 - 支持正式图、子图碎片、弱图块的分级处理"""
        image_path_raw, image_text, caption_text = self.extract_image_info(block)

        full_image_path = None
        relative_image_path = ""

        if image_path_raw and self._is_valid_image_path(image_path_raw):
            full_image_path = self.input_dir / image_path_raw
            if full_image_path.exists():
                relative_image_path = self._get_relative_path(full_image_path)
            else:
                self.error_report["missing_images"].append({
                    "block_id": f"p{page_no}_b{block_index}",
                    "expected_path": str(image_path_raw),
                    "type": "figure",
                })
                self.stats["missing_images"] += 1

        section_title = self.current_section_title

        # 使用 block_idx（在 page_blocks 中的位置）查找邻近文本
        nearby_text = self.find_nearby_text(page_no, block_idx, section_title)

        texts = []
        if caption_text:
            texts.append(f"Figure: {caption_text}")
        if nearby_text:
            texts.append(f"Context: {nearby_text}")

        chunk_text = "\n".join(texts) if texts else "Figure (see image)"

        # 对 figure chunk 进行分类
        figure_type = self._classify_figure_chunk(caption_text, nearby_text, chunk_text)

        # 1. 弱图块：直接跳过
        if figure_type == 'weak':
            self.stats["skipped_weak_figures"] += 1
            self.error_report["skipped_blocks"].append({
                "block_id": f"p{page_no}_b{block_index}",
                "type": "image",
                "reason": "weak figure chunk skipped (insufficient caption or context)",
                "page_no": page_no,
                "caption": caption_text[:100] if caption_text else "",
                "nearby_text_preview": nearby_text[:100] if nearby_text else "",
            })
            return

        # 2. 子图碎片：合并到最近的主图
        if figure_type == 'subfigure':
            if self._can_merge_into_last_figure(page_no, section_title):
                # 合并到最近的主图
                last_idx = self.last_formal_figure['index']
                # 更新主图的 chunk_text，添加子图信息
                subfigure_info = f"[Subfigure: {caption_text}]"
                if relative_image_path:
                    subfigure_info += f" (Image: {relative_image_path})"

                self.chunks[last_idx]['chunk_text'] += f"\n{subfigure_info}"
                self.chunks[last_idx]['metadata']['cleanup_flags'].append('merged_subfigure')
                self.chunks[last_idx]['metadata'].setdefault('subfigures', []).append({
                    'caption': caption_text,
                    'image_path': relative_image_path,
                    'page_no': page_no,
                })
                self.stats["subfigure_fragments_merged"] += 1
                return
            else:
                # 没有找到主图，当作弱图块跳过
                self.stats["skipped_weak_figures"] += 1
                self.error_report["skipped_blocks"].append({
                    "block_id": f"p{page_no}_b{block_index}",
                    "type": "image",
                    "reason": "subfigure fragment skipped (no parent figure found)",
                    "page_no": page_no,
                    "caption": caption_text[:100] if caption_text else "",
                })
                return

        # 3. 正式图：创建 chunk 并缓存
        chunk_id = f"{self.doc_id}:p{page_no}:figure:{len(self.chunks) + 1}"

        section_path = section_title.split(" > ") if section_title else []

        metadata = {
            "section_path": section_path,
            "figure_title": caption_text,
            "nearby_text": nearby_text,
            "block_bbox": block.get("bbox", []),
            "cleanup_flags": [],
            "parser": "MinerU-postprocessed",
        }

        chunk = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_pdf": self._get_relative_path(self.input_files_used.get("origin_pdf")),
            "page_no": page_no,
            "chunk_id": chunk_id,
            "content_type": "figure",
            "section_title": section_title,
            "chunk_text": chunk_text,
            "image_path": relative_image_path,
            "source_block_id": f"p{page_no}_b{block_index}",
            "metadata": metadata,
        }

        # 缓存这个正式图的索引
        self.last_formal_figure = {
            'index': len(self.chunks),
            'chunk_id': chunk_id,
            'page_no': page_no,
            'section_title': section_title,
        }

        self.chunks.append(chunk)
        self.stats["figure_chunks"] += 1
        self.stats["total_chunks"] += 1

    def _process_formula_block(self, block: dict, page_no: int, block_index: int):
        """处理公式块"""
        formula_text = self.extract_formula_text(block)

        if not formula_text:
            self.error_report["parse_errors"].append({
                "block_id": f"p{page_no}_b{block_index}",
                "type": "equation_interline",
                "error": "Empty formula content",
            })
            return

        chunk_id = f"{self.doc_id}:p{page_no}:formula:{len(self.chunks) + 1}"

        section_title = self.current_section_title
        section_path = section_title.split(" > ") if section_title else []

        metadata = {
            "section_path": section_path,
            "block_bbox": block.get("bbox", []),
            "cleanup_flags": [],
            "parser": "MinerU-postprocessed",
        }

        chunk = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_pdf": self._get_relative_path(self.input_files_used.get("origin_pdf")),
            "page_no": page_no,
            "chunk_id": chunk_id,
            "content_type": "formula",
            "section_title": section_title,
            "chunk_text": formula_text,
            "image_path": "",
            "source_block_id": f"p{page_no}_b{block_index}",
            "metadata": metadata,
        }

        self.chunks.append(chunk)
        self.stats["formula_chunks"] += 1
        self.stats["total_chunks"] += 1

    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'<[^>]+>', '', text)

        watermarks = [
            r'www\.[a-zA-Z0-9.-]+\.(com|cn|net|org)',
            r'Copyright\s*©?\s*\d{4}.*',
            r'All\s+rights\s+reserved',
        ]
        for pattern in watermarks:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def deduplicate_headers_footers(self):
        """删除高频重复的页眉页脚"""
        prefix_counter = defaultdict(int)
        suffix_counter = defaultdict(int)

        for chunk in self.chunks:
            if chunk["content_type"] == "text":
                text = chunk["chunk_text"]
                prefix = text[:50].strip()
                suffix = text[-50:].strip()
                if len(prefix) > 20:
                    prefix_counter[prefix] += 1
                if len(suffix) > 20:
                    suffix_counter[suffix] += 1

        common_prefixes = {p for p, c in prefix_counter.items() if c > 3}
        common_suffixes = {s for s, c in suffix_counter.items() if c > 3}

        for chunk in self.chunks:
            if chunk["content_type"] == "text":
                text = chunk["chunk_text"]

                for prefix in common_prefixes:
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                        chunk["metadata"]["cleanup_flags"].append("removed_header")
                        break

                for suffix in common_suffixes:
                    if text.endswith(suffix):
                        text = text[:-len(suffix)].strip()
                        chunk["metadata"]["cleanup_flags"].append("removed_footer")
                        break

                chunk["chunk_text"] = text

    def clean_section_title(self, title: str) -> str:
        """清洗章节标题：去除首尾空格，压缩连续空白"""
        if not title:
            return ""
        # 去除首尾空白
        title = title.strip()
        # 压缩连续空白为单个空格
        title = re.sub(r'\s+', ' ', title)
        return title

    def _should_skip_section(self, title: str) -> bool:
        """判断章节是否应该被跳过（目录、索引、修订历史等）"""
        if not title:
            return False
        clean = self.clean_section_title(title).lower()
        for pattern in self.SKIP_SECTION_PATTERNS:
            if re.match(pattern, clean, re.IGNORECASE):
                return True
        return False

    def _has_figure_number(self, text: str) -> bool:
        """检查文本是否包含正式图编号（如 Figure 29, Fig. 3, 图1-2 等）"""
        if not text:
            return False
        return bool(re.search(self.FIGURE_NUMBER_PATTERN, text, re.IGNORECASE))

    def _can_merge_into_last_figure(self, page_no: int, section_title: str) -> bool:
        """子图只合并到相邻页、同章节的最近主图，避免跨文意串联"""
        if self.last_formal_figure is None:
            return False
        if abs(self.last_formal_figure["page_no"] - page_no) > 1:
            return False
        last_section = self.last_formal_figure.get("section_title", "")
        if last_section and section_title and last_section != section_title:
            return False
        return True

    def _is_subfigure_label(self, caption_text: str) -> bool:
        """判断是否是子图碎片标签（如 MLC Page, Top View 等）"""
        if not caption_text:
            return False
        clean = caption_text.strip()
        # 如果只是一个短标签，没有图编号，且匹配子图模式
        if len(clean) < 50 and not self._has_figure_number(clean):
            for pattern in self.SUBFIGURE_LABEL_PATTERNS:
                if re.match(pattern, clean, re.IGNORECASE):
                    return True
        return False

    def _classify_figure_chunk(self, caption_text: str, nearby_text: str, chunk_text: str) -> str:
        """
        对 figure chunk 进行分类：
        - 'formal': 正式图（有编号、有标题、有足够上下文）
        - 'subfigure': 子图碎片（无编号、短标签）
        - 'weak': 弱图块（无编号、无标题、无上下文）
        """
        caption_text = (caption_text or "").strip()
        nearby_text = (nearby_text or "").strip()
        has_number = self._has_figure_number(caption_text)
        has_valid_caption = len(caption_text) > 5
        nearby_is_section_title = nearby_text == self.current_section_title
        only_context = chunk_text.startswith("Context:") and not has_valid_caption
        total_content_len = len(caption_text + nearby_text)

        # 子图碎片优先判定，避免被长上下文误判为正式图
        if self._is_subfigure_label(caption_text):
            return 'subfigure'

        # 只有 Context 的图片块，即使上下文里提到了别的图号，也不应当作正式图
        if only_context:
            return 'weak'

        # 正式图判定：有编号 或 （有完整标题 + 有足够上下文）
        if has_number:
            return 'formal'
        if has_valid_caption and not nearby_is_section_title and total_content_len >= 50:
            return 'formal'

        # 弱图块判定：无编号 + 无有效标题 + （只有Context或无真实邻近文本）
        if not has_valid_caption and nearby_is_section_title:
            return 'weak'
        if total_content_len < 30:
            return 'weak'

        # 默认当作正式图处理
        return 'formal'

    def _is_weak_figure_chunk(self, caption_text: str, nearby_text: str, chunk_text: str) -> bool:
        """判断 figure chunk 是否太弱（信息不足）- 兼容旧代码"""
        classification = self._classify_figure_chunk(caption_text, nearby_text, chunk_text)
        return classification == 'weak'

    def write_jsonl(self):
        """写入 JSONL 文件 - 精简版，适合知识库 / RAG 入库"""
        jsonl_path = self.output_dir / "kb_chunks.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                # 清洗 section_title
                section_title = self.clean_section_title(chunk["section_title"])

                # 只保留入库必需的字段
                minimal_chunk = {
                    "chunk_id": chunk["chunk_id"],
                    "page_no": chunk["page_no"],
                    "content_type": chunk["content_type"],
                    "section_title": section_title,
                    "chunk_text": chunk["chunk_text"],
                    "image_path": chunk["image_path"],
                }
                f.write(json.dumps(minimal_chunk, ensure_ascii=False) + "\n")

        print(f"已生成: {jsonl_path}")

        # 写入共享输出目录（以文档名命名）
        if self.shared_output_dir:
            self.shared_output_dir.mkdir(parents=True, exist_ok=True)
            shared_path = self.shared_output_dir / f"{self.doc_title}.jsonl"
            with open(shared_path, "w", encoding="utf-8") as f:
                for chunk in self.chunks:
                    section_title = self.clean_section_title(chunk["section_title"])
                    minimal_chunk = {
                        "chunk_id": chunk["chunk_id"],
                        "page_no": chunk["page_no"],
                        "content_type": chunk["content_type"],
                        "section_title": section_title,
                        "chunk_text": chunk["chunk_text"],
                        "image_path": chunk["image_path"],
                    }
                    f.write(json.dumps(minimal_chunk, ensure_ascii=False) + "\n")
            print(f"已生成 (共享): {shared_path}")

        return jsonl_path

    def write_manifest(self):
        """写入 manifest 文件 - 记录存在的文件和使用的文件"""
        manifest = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_dir": self._get_relative_path(self.input_dir),
            "source_pdf": self._get_relative_path(self.input_files_used.get("origin_pdf")),
            "created_at": datetime.now().isoformat(),
            "input_files_all": {k: self._get_relative_path(v) if v else None
                                for k, v in self.input_files_exist.items()},
            "input_files_used": {k: self._get_relative_path(v) if v else None
                                 for k, v in self.input_files_used.items()},
            "output_files": {
                "chunks_jsonl": self._get_relative_path(self.output_dir / "kb_chunks.jsonl"),
                "manifest": self._get_relative_path(self.output_dir / "kb_manifest.json"),
                "readme": self._get_relative_path(self.output_dir / "README_kb.md"),
                "error_report": self._get_relative_path(self.output_dir / "error_report.json"),
            },
            "chunking_policy": {
                "target_tokens": f"{self.TARGET_TOKENS_MIN}-{self.TARGET_TOKENS_MAX}",
                "hard_cap": self.HARD_CAP_TOKENS,
                "merge_short_blocks": True,
                "deduplicate_headers": True,
                "table_split": True,
                "type_mapping": self.TYPE_MAPPING,
            },
            "stats": self.stats,
        }

        manifest_path = self.output_dir / "kb_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"已生成: {manifest_path}")
        return manifest_path

    def write_error_report(self):
        """写入错误报告"""
        self.error_report["summary"] = {
            "total_missing_images": len(self.error_report["missing_images"]),
            "total_skipped_blocks": len(self.error_report["skipped_blocks"]),
            "total_unsupported_blocks": len(self.error_report["unsupported_blocks"]),
            "total_parse_errors": len(self.error_report["parse_errors"]),
        }

        error_path = self.output_dir / "error_report.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(self.error_report, f, ensure_ascii=False, indent=2)

        print(f"已生成: {error_path}")
        return error_path

    def write_readme(self):
        """写入 README 文件"""
        output_rel_path = self._get_relative_path(self.output_dir)

        # 获取推荐入库文件的统一路径
        chunks_jsonl_rel = self._get_relative_path(self.output_dir / "kb_chunks.jsonl")

        readme_content = f"""# Knowledge Base Dataset Files

## 文档信息

- **文档 ID**: `{self.doc_id}`
- **文档标题**: {self.doc_title}
- **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **源文件**: `{self.input_files_used.get("origin_pdf").name if self.input_files_used.get("origin_pdf") else "N/A"}`

## 生成文件列表

| 文件 | 说明 | 路径 |
|------|------|------|
| `kb_chunks.jsonl` | **知识库 Chunk 数据（推荐入库文件）** | `{chunks_jsonl_rel}` |
| `kb_manifest.json` | 处理元数据和统计信息 | `{output_rel_path}/kb_manifest.json` |
| `README_kb.md` | 本说明文件 | `{output_rel_path}/README_kb.md` |
| `error_report.json` | 错误和警告报告 | `{output_rel_path}/error_report.json` |

说明：输出文件名使用中性的 `kb_*` 前缀，内容本身是通用的 JSONL / manifest / report 结构，可用于任意 RAG / 向量检索系统。

## 推荐入库文件

**使用此文件**: `{chunks_jsonl_rel}`

## kb_chunks.jsonl 字段说明

每行一个 JSON 对象，包含 RAG 检索必需的字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | string | Chunk 唯一标识，格式 `{{doc_id}}:p{{page_no}}:{{type}}:{{seq}}` |
| `page_no` | int | 页码（从 1 开始） |
| `content_type` | string | 内容类型：`text`/`table`/`figure`/`formula` |
| `section_title` | string | 所属章节标题 |
| `chunk_text` | string | Chunk 文本内容（可检索） |
| `image_path` | string | 图片相对项目根目录路径（无图为空字符串） |

## Chunk 切分与清洗规则

### 块类型映射

| MinerU 类型 | 映射类型 | 处理方式 |
|-------------|----------|----------|
| `paragraph` | `text` | 正文，可合并 |
| `list` | `text` | 列表，可合并 |
| `table` | `table` | 表格，支持按行拆分 |
| `image` | `figure` | 图片，结合邻近正文生成描述 |
| `equation_interline` | `formula` | 行间公式 |
| `title` | - | 用于章节追踪，不输出 |
| `page_header/footer/number` | - | 当作噪音跳过 |

### 文本 Chunk 规则

- **目标长度**: 300-700 tokens
- **上限**: 900 tokens（硬切分）
- **合并**: 相邻同章节短正文块自动合并
- **切分**: 超长文本按自然段或句子切分

### 表格 Chunk 规则

- **默认**: 一张表一个 chunk
- **拆分**: 超长表格按行拆分，每个子 chunk 重复表标题和表头
- **格式**: 表格 HTML 转换为自然语言/键值文本

### 图片 Chunk 规则

- 默认一张图一个 chunk
- 优先查找同一页相邻的 paragraph/list 作为 `nearby_text`
- 找不到同页邻近正文时，回退到同章节最近正文、相邻页正文，最后才回退到章节标题
- `image_path` 使用相对项目根目录的路径

### 清洗规则

- 删除高频重复页眉页脚
- 删除孤立页码、无意义空行
- 删除 HTML 残片
- 保留技术术语、型号、单位、符号原样

## RAG / Knowledge Base Ingestion Guide

### 推荐入库文件

**使用此文件**: `{chunks_jsonl_rel}`

### 入库建议

1. 将 `{chunks_jsonl_rel}` 作为主入库文件
2. 将 `chunk_text` 作为主内容字段
3. 将 `chunk_id`, `page_no`, `content_type`, `section_title`, `image_path` 作为 metadata 字段
4. 不要再次做机械滑窗切分；当前 chunk 已按文档语义边界预处理
5. 嵌入 / 重排模型可按你的系统配置选择；当前输出适合现代 embedding + reranker 流水线

### 字段映射建议

```
content_field: chunk_text
metadata_fields: [chunk_id, page_no, content_type, section_title, image_path]
```

## 统计数据

| 指标 | 数值 |
|------|------|
| 总 Chunk 数 | {self.stats["total_chunks"]} |
| 文本 Chunks | {self.stats["text_chunks"]} |
| 表格 Chunks | {self.stats["table_chunks"]} |
| 图片 Chunks | {self.stats["figure_chunks"]} |
| 公式 Chunks | {self.stats["formula_chunks"]} |
| 缺失图片 | {self.stats["missing_images"]} |
| 跳过噪音块 | {self.stats["skipped_noise"]} |
| 跳过空块 | {self.stats["skipped_empty"]} |

## 注意事项

1. 所有路径字段使用相对于项目根目录的格式，统一用 `/` 分隔
2. `image_path` 为空字符串表示无图或图片缺失
3. 公式以 LaTeX 格式保留原始数学含义
4. 表格已转换为可读文本格式，支持超长表格拆分
5. `section_title` 与 `metadata.section_path` 保持一致
6. `figure` 的 `nearby_text` 优先来自真实相邻正文，非章节标题
"""

        readme_path = self.output_dir / "README_kb.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        print(f"已生成: {readme_path}")
        return readme_path

    def convert(self):
        """执行完整转换流程"""
        print(f"开始转换: {self.input_dir.name}")
        print(f"文档 ID: {self.doc_id}")
        print("-" * 50)

        # 1. 发现文件
        files_exist, files_used = self.discover_files()
        print(f"发现文件 (存在):")
        for k, v in files_exist.items():
            status = "✓" if v else "✗"
            print(f"  [{status}] {k}: {v.name if v else 'N/A'}")
        print(f"\n实际使用:")
        for k, v in files_used.items():
            if v:
                print(f"  [→] {k}: {v.name}")
        print()

        # 检查必要文件
        if not files_used.get("content_list_v2") and not files_used.get("content_list"):
            raise ValueError("未找到 content_list_v2.json 或 content_list.json")

        # 2. 加载内容
        content_data = self.load_content()
        print(f"共 {len(content_data)} 页")

        # 3. 处理块
        self.process_blocks(content_data)

        # 4. 后处理
        self.deduplicate_headers_footers()

        # 5. 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 6. 写入文件
        self.write_jsonl()
        self.write_manifest()
        self.write_error_report()
        self.write_readme()

        print("-" * 50)
        print(f"转换完成!")
        print(f"总 Chunks: {self.stats['total_chunks']}")
        print(f"  - 文本: {self.stats['text_chunks']}")
        print(f"  - 表格: {self.stats['table_chunks']}")
        print(f"  - 图片: {self.stats['figure_chunks']}")
        print(f"  - 公式: {self.stats['formula_chunks']}")
        print(f"  - 跳过空块: {self.stats['skipped_empty']}")
        print(f"输出目录: {self.output_dir}")

        # 自检指标
        print("-" * 50)
        print("自检指标:")
        print(f"  - 被过滤的 Contents/List/Figures/Tables/Rev.*: {self.stats['skipped_low_value_sections']}")
        print(f"  - 被过滤的弱 figure chunks: {self.stats['skipped_weak_figures']}")
        print(f"  - 合并的子图碎片: {self.stats['subfigure_fragments_merged']}")
        print(f"  - 超长 table chunks (供人工审核): {self.stats['oversized_table_chunks']}")
        print(f"  - page_aside_text 噪声块: {self.stats['aside_noise']}")


def main():
    parser = argparse.ArgumentParser(
        description="将 MinerU 输出目录转换为通用知识库 / RAG 数据集"
    )
    parser.add_argument(
        "input_dir",
        help="MinerU 输出目录路径"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="输出目录路径（默认: <input_dir>/output）"
    )
    parser.add_argument(
        "--shared-output",
        "-s",
        help="共享输出目录路径，将 kb_chunks 以文档名命名汇集到此目录"
    )

    args = parser.parse_args()

    converter = MinerUConverter(args.input_dir, args.output_dir,
                                shared_output_dir=args.shared_output)
    converter.convert()


if __name__ == "__main__":
    main()
