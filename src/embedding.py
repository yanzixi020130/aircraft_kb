#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk Embedding (functionized)

- 可作为库导入：process_jsonl_to_embeddings(input_jsonl, out_dir=..., batch_size=32, verbose=True)
- 可命令行运行：python src/embedding.py --input_jsonl data/chunk/tilt_rotor.jsonl
- 参考 latex_to_embed.py 的远端 embedding 服务调用方式

要求：
- 读取 chunks.jsonl，每行含 content 字段
- 通过 SSH 调用远端 http://127.0.0.1:40291/embed 获取 embeddings # $env:SSH_PASS="ximukeji2023"
- 将 embedding 写回记录，输出到 out_dir/<stem>_embedded.jsonl
- 忽略空行；保持 UTF-8；ensure_ascii=False

环境变量：
- SSH_PASS：SSH 登录密码
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from embedding_service_client import SSHConfig, fetch_embeddings_over_ssh

# 默认路径
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_INPUT = str((_DATA_DIR / "chunk" / "tilt_rotor.jsonl").resolve())
DEFAULT_OUTPUT_DIR = str((_DATA_DIR / "embedding").resolve())
DEFAULT_BATCH_SIZE = 32


# ---------------------------- 基础工具 ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk embedding via remote service over SSH")
    parser.add_argument("--input_jsonl", default=DEFAULT_INPUT, help="输入 chunks.jsonl 路径")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小 (默认 32)")
    parser.add_argument("--quiet", action="store_true", help="关闭日志输出")
    return parser.parse_args()


def read_chunks_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] 第 {line_num} 行解析失败: {e}")
    return chunks


def escape_backslashes(text: str) -> str:
    return text.replace("\\", "\\\\")


def write_chunks_with_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]], output_path: Path) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks 与 embeddings 数量不一致：{len(chunks)} vs {len(embeddings)}")
    with open(output_path, "w", encoding="utf-8") as f:
        for ch, emb in zip(chunks, embeddings):
            ch["embedding"] = emb
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


# ---------------------------- 核心流程 ----------------------------
def process_jsonl_to_embeddings(
    input_jsonl: Path | str,
    out_dir: Path | str = DEFAULT_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cfg: SSHConfig | None = None,
    verbose: bool = True,
) -> Path:
    input_path = Path(input_jsonl)
    output_dir = Path(out_dir)
    if cfg is None:
        cfg = SSHConfig()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_embedded.jsonl"

    if verbose:
        print("=" * 70)
        print(f"[INPUT] {input_path.resolve()}")
        print(f"[OUTPUT] {output_path.resolve()}")
        print(f"[CONFIG] batch_size={batch_size}")
        print("=" * 70)

    chunks = read_chunks_jsonl(input_path)
    if verbose:
        print(f"[INFO] 读取 {len(chunks)} 条记录")
    if not chunks:
        raise RuntimeError("输入文件为空或无有效记录")

    texts = []
    for idx, ch in enumerate(chunks):
        content = ch.get("content", "")
        if not content.strip() and verbose:
            print(f"[WARNING] 第 {idx} 条 content 为空")
        texts.append(escape_backslashes(content))

    embeddings = fetch_embeddings_over_ssh(texts, batch_size, cfg, verbose=verbose)
    write_chunks_with_embeddings(chunks, embeddings, output_path)

    if verbose:
        print(f"[SUCCESS] 完成，写入 {len(chunks)} 条 -> {output_path}")
    return output_path


# ---------------------------- CLI ----------------------------
def cli_main() -> None:
    args = parse_args()
    process_jsonl_to_embeddings(
        input_jsonl=args.input_jsonl,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cli_main()
