#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to Chunks (JSONL) - Functionized Module

- 提供可复用的函数 `process_markdown_to_chunks()`：从 Markdown 生成 chunks.jsonl
- 同时保留命令行入口：python src/llm_retrieve.py --input_md xxx.md [--out_dir OUT]

核心规则：
- 标题分段（# / ## / ###）
- 公式块完整性保护（$$, \[\], \begin{..}\end{..}）并追加上下文（2~6行）
- 长度控制：target=320, soft_max=480, hard_max=650, min=180，overlap=80
- 忽略空内容 chunk；输出 JSONL ensure_ascii=False

仅依赖标准库。
"""

from __future__ import annotations
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------------------------- 配置参数 ----------------------------
TARGET_TOKENS = 320
SOFT_MAX = 480
HARD_MAX = 650
MIN_TOKENS = 180
OVERLAP_TOKENS = 80
DEFAULT_OUT_DIR = r"C:\\Project\\LLM_4_QF\\data\\chunk"

# ---------------------------- Token 估计 ----------------------------
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.strip()) // 4)

# ---------------------------- IO ----------------------------
def read_md_lines(md_path: str) -> List[str]:
    with open(md_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n\r') for line in f.readlines()]

# ---------------------------- 标题解析 ----------------------------
def get_heading_level(line: str) -> Optional[int]:
    m = re.match(r'^(#{1,6})\s+', line)
    return len(m.group(1)) if m else None

def extract_heading_title(line: str) -> str:
    return re.sub(r'^#+\s+', '', line)

def extract_heading_number(title: str) -> Optional[str]:
    m = re.match(r'^(\d+(?:\.\d+)*)\s*', title)
    return m.group(1) if m else None

# ---------------------------- 数学块检测 ----------------------------
class MathBlockDetector:
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.math_blocks: List[Tuple[int, int]] = []
        self._detect_all()

    def _detect_all(self):
        i = 0
        while i < len(self.lines):
            if self._match_dollar_block(i):
                start = i
                j = self._find_closing_dollar(i + 1)
                if j is not None:
                    self.math_blocks.append((start, j))
                    i = j + 1
                    continue
            if self._match_bracket_block(i):
                start = i
                j = self._find_closing_bracket(i + 1)
                if j is not None:
                    self.math_blocks.append((start, j))
                    i = j + 1
                    continue
            if self._match_begin_block(i):
                env = self._extract_environment(i)
                start = i
                j = self._find_closing_end(i + 1, env)
                if j is not None:
                    self.math_blocks.append((start, j))
                    i = j + 1
                    continue
            i += 1

    def _match_dollar_block(self, idx: int) -> bool:
        return '$$' in self.lines[idx]

    def _find_closing_dollar(self, idx: int) -> Optional[int]:
        for i in range(idx, len(self.lines)):
            if '$$' in self.lines[i]:
                return i
        return None

    def _match_bracket_block(self, idx: int) -> bool:
        return r'\[' in self.lines[idx]

    def _find_closing_bracket(self, idx: int) -> Optional[int]:
        for i in range(idx, len(self.lines)):
            if r'\]' in self.lines[i]:
                return i
        return None

    def _match_begin_block(self, idx: int) -> bool:
        return re.search(r'\\begin\{', self.lines[idx]) is not None

    def _extract_environment(self, idx: int) -> str:
        m = re.search(r'\\begin\{(\w+)\}', self.lines[idx])
        return m.group(1) if m else ''

    def _find_closing_end(self, idx: int, env: str) -> Optional[int]:
        pattern = r'\\end\{' + re.escape(env) + r'\}'
        for i in range(idx, len(self.lines)):
            if re.search(pattern, self.lines[i]):
                return i
        return None

    def is_in_math_block(self, line_idx: int) -> bool:
        for s, e in self.math_blocks:
            if s <= line_idx <= e:
                return True
        return False

# ---------------------------- 结构段构建 ----------------------------
def parse_headings_and_sections(lines: List[str]) -> List[Dict]:
    sections: List[Dict] = []
    section_path: List[str] = []
    current_start = 0

    for i, line in enumerate(lines):
        level = get_heading_level(line)
        if level is not None:
            if i > current_start:
                sections.append({
                    'section_path': section_path.copy(),
                    'start_line': current_start,
                    'end_line': i - 1,
                })
            section_path = section_path[:level - 1]
            title = extract_heading_title(line)
            section_path.append(title)
            current_start = i

    if current_start < len(lines):
        sections.append({
            'section_path': section_path.copy(),
            'start_line': current_start,
            'end_line': len(lines) - 1,
        })
    return sections

# ---------------------------- 公式上下文扩展（边界辅助） ----------------------------
def expand_math_block_context(lines: List[str], math_block: Tuple[int, int]) -> Tuple[int, int]:
    start, end = math_block
    keywords = ['where', '式中', '其中', '符号', '定义', 'denote', 'here', '参数', '变量']
    has_kw = False
    for i in range(max(0, start - 6), min(len(lines), end + 7)):
        if any(kw in lines[i].lower() for kw in keywords):
            has_kw = True
            break
    ctx = 6 if has_kw else 2
    return max(0, start - ctx), min(len(lines) - 1, end + ctx)

# ---------------------------- 自然边界与数学块边界调整 ----------------------------
def _find_natural_boundary(lines: List[str], start: int, end: int) -> int:
    for i in range(end, start - 1, -1):
        line = lines[i].strip()
        if not line:
            return i - 1 if i > start else start
        if i < len(lines) - 1:
            nxt = lines[i + 1].strip()
            if (nxt.startswith('-') or nxt.startswith('*') or
                (len(nxt) > 1 and nxt[0].isdigit() and nxt[1] == '.')):
                return i
    return end

def _adjust_for_math_block(lines: List[str], start: int, end: int, math_detector: MathBlockDetector) -> int:
    for m_start, m_end in math_detector.math_blocks:
        if m_start >= start and m_end <= end:
            continue
        if m_start <= end <= m_end:
            return min(m_end, len(lines) - 1)
    return end

# ---------------------------- 段内切分 ----------------------------
def split_section_into_chunks(lines: List[str], section: Dict, math_detector: MathBlockDetector) -> List[Dict]:
    start = section['start_line']
    end = section['end_line']
    section_path = section['section_path']

    chunks: List[Dict] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = chunk_start
        acc = 0
        while chunk_end <= end:
            t = estimate_tokens(lines[chunk_end])
            if acc + t > SOFT_MAX:
                break
            acc += t
            chunk_end += 1
        if chunk_end == chunk_start:
            chunk_end = min(chunk_start + 1, end + 1)
        else:
            chunk_end -= 1
        chunk_end = _find_natural_boundary(lines, chunk_start, chunk_end)
        chunk_end = _adjust_for_math_block(lines, chunk_start, chunk_end, math_detector)
        chunks.append({'lines_range': (chunk_start, chunk_end), 'section_path': section_path})
        chunk_start = chunk_end + 1
    return chunks

# ---------------------------- 后处理（最末块、重叠） ----------------------------
def _chunk_tokens(lines: List[str], rng: Tuple[int, int]) -> int:
    s, e = rng
    return sum(estimate_tokens(lines[i]) for i in range(s, e + 1))

def post_process_chunks(chunks: List[Dict], lines: List[str]) -> List[Dict]:
    if not chunks:
        return chunks
    # 处理最后一个 chunk 太短
    if len(chunks) > 1:
        last = chunks[-1]
        lt = _chunk_tokens(lines, last['lines_range'])
        if lt < MIN_TOKENS:
            prev = chunks[-2]
            pt = _chunk_tokens(lines, prev['lines_range'])
            if pt + lt <= HARD_MAX:
                chunks[-2] = {'lines_range': (prev['lines_range'][0], last['lines_range'][1]), 'section_path': prev['section_path']}
                chunks.pop()
            else:
                ps, pe = prev['lines_range']
                take = 0
                i = pe
                while i >= ps and take < OVERLAP_TOKENS:
                    take += estimate_tokens(lines[i])
                    i -= 1
                copy_start = max(ps, i + 1)
                last['lines_range'] = (copy_start, last['lines_range'][1])
    return chunks

# ---------------------------- 页码提取 ----------------------------
def extract_page_hint(text: str) -> Optional[int]:
    patterns = [
        r'[Pp]age\s*[:\s]*(\d+)',
        r'[页码]+\s*[:：]\s*(\d+)',
        r'第\s*(\d+)\s*页',
        r'\[p\.(\d+)\]',
        r'\(p\.(\d+)\)',
        r'（p\.(\d+)）',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))
    return None

# ---------------------------- chunk_id 生成 ----------------------------
def _generate_chunk_id(section_path: List[str], counters: Dict[Tuple[str, ...], int], section_order: Dict[Tuple[str, ...], int]) -> str:
    key = tuple(section_path)
    if key not in counters:
        counters[key] = 0
    counters[key] += 1
    seq = counters[key]
    if section_path:
        num = extract_heading_number(section_path[-1])
        if num:
            return f"{num}::{seq:03d}"
    # fallback: secXX
    if key not in section_order:
        section_order[key] = len(section_order) + 1
    return f"sec{section_order[key]:02d}::{seq:03d}"

# ---------------------------- 记录构建与写入 ----------------------------
def build_chunk_records(chunks: List[Dict], lines: List[str]) -> List[Dict]:
    records: List[Dict] = []
    counters: Dict[Tuple[str, ...], int] = {}
    section_order: Dict[Tuple[str, ...], int] = {}
    for ch in chunks:
        s, e = ch['lines_range']
        section_path = ch['section_path']
        content = '\n'.join(lines[s:e+1])
        if not content.strip():
            continue
        cid = _generate_chunk_id(section_path, counters, section_order)
        page = extract_page_hint(content)
        records.append({
            'chunk_id': cid,
            'content': content,
            'page_hint': page,
            'section_path': section_path,
        })
    return records

def write_jsonl(records: List[Dict], output_path: Path) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------------- 主流程（可复用） ----------------------------
def process_markdown_to_chunks(input_md: str | Path, out_dir: str | Path = DEFAULT_OUT_DIR, verbose: bool = True) -> Path:
    input_path = Path(input_md)
    if verbose:
        print("="*70)
        print(f"[INPUT] 文件: {input_path.resolve()}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_md}")
    if input_path.suffix.lower() != '.md':
        raise ValueError(f"Input file must be Markdown (.md): {input_md}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + '.jsonl')
    if verbose:
        print(f"[OUTPUT] 目录: {output_dir.resolve()}")
        print(f"[OUTPUT] 文件: {output_path.resolve()}")

    lines = read_md_lines(str(input_path))
    if verbose:
        print(f"[INFO] 读取行数: {len(lines)}")
    math_detector = MathBlockDetector(lines)
    if verbose:
        print(f"[INFO] 公式块: {len(math_detector.math_blocks)}")
    sections = parse_headings_and_sections(lines)
    if verbose:
        print(f"[INFO] 结构段: {len(sections)}")

    all_chunks: List[Dict] = []
    for sec in sections:
        all_chunks.extend(split_section_into_chunks(lines, sec, math_detector))
    if verbose:
        print(f"[INFO] 初步 chunks: {len(all_chunks)}")
    all_chunks = post_process_chunks(all_chunks, lines)
    if verbose:
        print(f"[INFO] 后处理 chunks: {len(all_chunks)}")

    records = build_chunk_records(all_chunks, lines)
    if verbose:
        print(f"[INFO] 有效记录: {len(records)}")
    write_jsonl(records, output_path)
    if verbose:
        print(f"[SUCCESS] 已写入: {output_path}")
        print("="*70)
    return output_path

# ---------------------------- CLI 入口 ----------------------------
def cli_main() -> None:
    parser = argparse.ArgumentParser(description='Markdown -> chunks.jsonl (functionized)')
    parser.add_argument('--input_md', required=True, help='Path to input Markdown file')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='Output directory for .jsonl')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose logs')
    args = parser.parse_args()
    process_markdown_to_chunks(args.input_md, args.out_dir, verbose=not args.quiet)

if __name__ == '__main__':
    cli_main()
