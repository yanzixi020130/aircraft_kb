#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM helper: generate missing quantity specs.

Goal:
- Given missing quantity names (Chinese), generate quantity specs:
    { "<missing_name_zh>": {"quantity": {id, symbol, symbol_latex, name_zh, unit}} }

Reference context:
- Use ALL PDFs under data/raw as reference. Prefer extracted Markdown under
  data/md/<stem>/<stem>.md (if present). If MD is missing, still include the
  PDF filename.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from llm_client import LLMClient, LLMConfig


# ==================== JSON helpers ====================
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _extract_json_obj(text: str) -> Dict[str, Any]:
    s = _strip_code_fences(text)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", s, 0)
    return json.loads(s[start : end + 1])


# ==================== Reference corpus ====================
def _iter_pdfs(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []
    return sorted([p for p in raw_dir.rglob("*.pdf") if p.is_file()])


def _md_for_pdf(pdf_path: Path, md_root: Path) -> Path:
    stem = pdf_path.stem
    return md_root / stem / f"{stem}.md"


def _read_text_best_effort(path: Path, *, max_chars: int) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=enc)[:max_chars]
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def build_reference_corpus(
    *,
    raw_dir: str,
    md_dir: str,
    max_total_chars: int = 20000,
    per_file_chars: int = 1500,
) -> str:
    """Build a bounded reference corpus string."""
    raw_root = Path(raw_dir)
    md_root = Path(md_dir)
    pdfs = _iter_pdfs(raw_root)

    chunks: List[str] = []
    budget = max_total_chars

    for pdf in pdfs:
        rel_pdf = str(pdf.relative_to(raw_root)) if raw_root in pdf.parents else str(pdf)
        md_path = _md_for_pdf(pdf, md_root)
        excerpt = _read_text_best_effort(md_path, max_chars=per_file_chars)
        if excerpt:
            block = f"[PDF] {rel_pdf}\n[MD_EXCERPT] {excerpt}\n"
        else:
            block = f"[PDF] {rel_pdf}\n[MD_EXCERPT] (missing: {md_path})\n"

        if len(block) > budget:
            break
        chunks.append(block)
        budget -= len(block)

    header = (
        "以下是 data/raw 中 PDF 的参考信息（优先使用已抽取的 Markdown 片段；若缺失则只提供文件名）。\n"
        f"总 PDF 数: {len(pdfs)}，已包含片段数: {len(chunks)}，已做长度截断。\n\n"
    )
    return header + "\n".join(chunks)


# ==================== Prompt ====================
_PROMPT_TEMPLATE = """
你是航空航天/飞行器设计领域的物理量知识建库专家。
现在物理量库中缺少以下物理量，请根据资料补全每个物理量的定义信息。

[任务]
- 必须覆盖所有 missing_quantity_name_zh，不得遗漏。
- 对每个缺失量，给出 id / symbol / symbol_latex / name_zh / unit。
- id / symbol / symbol_latex 必须是简短的拉丁字母/数字/下划线组合，禁止直接使用中文。
- symbol_latex 允许下标，例如 V_{{tip,h}}、S_{{ref}}。
- name_zh 必须等于输入的中文名称。
- 若资料不足，允许保守推断，但字段不得留空。
- unit 使用 SI 单位；无量纲写 "1"。
- 输出必须为合法 JSON 对象。

[示例]
输入:
["展弦比", "旋翼桨尖速度（直升机模式）"]
输出:
{{
    "展弦比": {{
        "quantity": {{
            "id": "AR",
            "symbol": "AR",
            "symbol_latex": "AR",
            "name_zh": "展弦比",
            "unit": "1"
        }}
    }},
    "旋翼桨尖速度（直升机模式）": {{
        "quantity": {{
            "id": "V_tip_h",
            "symbol": "V_tip_h",
            "symbol_latex": "V_{{tip,h}}",
            "name_zh": "旋翼桨尖速度（直升机模式）",
            "unit": "m/s"
        }}
    }}
}}

[输出 JSON 结构]
```
{{
    "<missing_quantity_name_zh>": {{
        "quantity": {{
            "id": "...",
            "symbol": "...",
            "symbol_latex": "...",
            "name_zh": "...",
            "unit": "..."
        }}
    }}
}}
```

[上下文]
- category: {category}
- extractid: {extractid}
- missing_quantity_name_zh: {missing_quantity_name_zh}

[PDF 参考资料]
{reference_corpus}

只输出 JSON，不要输出解释文字。
""".strip()


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _call_llm_json(
    llm: LLMClient,
    prompt: str,
    *,
    retries: int,
    debug: bool = False,
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for i in range(retries + 1):
        # print(f"[llm_fill_missing_quantities] call LLM, try {i+1}/{retries+1}")
        text = llm.completion_text(
            user_prompt=prompt,
            system_prompt="你是一个严格的物理量知识建库专家。",
        )
        # print("[llm_fill_missing_quantities] raw response:", repr(text))
        if debug:
            pass
            # print("[llm_fill_missing_quantities] raw response (debug):", repr(text))
        if not text:
            # print("[llm_fill_missing_quantities] LLM returned empty response!")
            last_err = ValueError("empty response")
            continue
        try:
            return _extract_json_obj(text)
        except Exception as exc:
            # print("[llm_fill_missing_quantities] json parse error:", repr(exc))
            last_err = exc
            if debug:
                pass
                # print("[llm_fill_missing_quantities] json parse error (debug):", repr(exc))
    if last_err:
        # print("[llm_fill_missing_quantities] raise last_err:", repr(last_err))
        raise last_err
    # print("[llm_fill_missing_quantities] LLM JSON parse failed!")
    raise ValueError("LLM JSON parse failed")


def _normalize_quantity_spec(name_zh: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = spec or {}
    _id = str(spec.get("id") or spec.get("symbol") or name_zh).strip()
    symbol = str(spec.get("symbol") or _id).strip()
    symbol_latex = str(spec.get("symbol_latex") or symbol).strip()
    zh = str(spec.get("name_zh") or "").strip()
    if re.search(r"[\u4e00-\u9fff]", str(name_zh)):
        zh = str(name_zh)
    elif not zh:
        zh = _id
    unit = str(spec.get("unit") or "1").strip() or "1"
    return {
        "id": _id,
        "symbol": symbol,
        "symbol_latex": symbol_latex,
        "name_zh": zh,
        "unit": unit,
    }


def generate_missing_quantities(
    *,
    category: str,
    extractid: str,
    missing_quantity_name_zh: List[str],
    raw_dir: str,
    md_dir: str,
    llm_cfg: LLMConfig | None = None,
    key_mode: str = "name_zh",
) -> Dict[str, Dict[str, Any]]:
    """Generate quantity specs for missing quantities.

    Returns a dict mapping missing name_zh -> quantity spec dict.
    """

    # print("[llm_fill_missing_quantities] generate_missing_quantities called")
    # print("  category:", category)
    # print("  extractid:", extractid)
    # print("  missing_quantity_name_zh:", missing_quantity_name_zh)
    # print("  raw_dir:", raw_dir)
    # print("  md_dir:", md_dir)
    # print("  llm_cfg:", llm_cfg)
    # print("  key_mode:", key_mode)

    if not missing_quantity_name_zh:
        print("[llm_fill_missing_quantities] missing_quantity_name_zh is empty, return {}")
        return {}

    # print("[llm_fill_missing_quantities] before LLMClient init")
    llm = LLMClient(llm_cfg or LLMConfig())
    # print("[llm_fill_missing_quantities] LLMClient init success:", llm)

    batch_size = int(os.getenv("LLM_MISSING_QUANTITY_BATCH_SIZE", "6"))
    retries = int(os.getenv("LLM_MISSING_RETRIES", "2"))
    max_total_chars = int(os.getenv("LLM_MISSING_MAX_TOTAL_CHARS", "20000"))
    per_file_chars = int(os.getenv("LLM_MISSING_PER_FILE_CHARS", "1500"))
    debug = os.getenv("LLM_MISSING_DEBUG", "0") == "1"

    reference_corpus = build_reference_corpus(
        raw_dir=raw_dir,
        md_dir=md_dir,
        max_total_chars=max_total_chars,
        per_file_chars=per_file_chars,
    )
    # print("[llm_fill_missing_quantities] reference_corpus preview:", reference_corpus[:300], "...")

    def _fetch_batch(names: List[str]) -> Dict[str, Dict[str, Any]]:
        # print(f"[llm_fill_missing_quantities] _fetch_batch called, names={names}")
        prompt = _PROMPT_TEMPLATE.format(
            category=category,
            extractid=extractid,
            missing_quantity_name_zh=json.dumps(names, ensure_ascii=False),
            reference_corpus=reference_corpus,
        )
        # print(f"[llm_fill_missing_quantities] _fetch_batch prompt preview: {prompt[:200]} ...")
        obj = _call_llm_json(llm, prompt, retries=retries, debug=debug)
        out: Dict[str, Dict[str, Any]] = {}
        for name in names:
            block = obj.get(name, {}) if isinstance(obj, dict) else {}
            if isinstance(block, dict) and "quantity" in block:
                qspec = block.get("quantity") if isinstance(block.get("quantity"), dict) else {}
            else:
                qspec = block if isinstance(block, dict) else {}
            out[name] = _normalize_quantity_spec(name, qspec)
        # print(f"[llm_fill_missing_quantities] _fetch_batch return: {out}")
        return out

    def _fill_batch(names: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            # print(f"[llm_fill_missing_quantities] _fill_batch try, names={names}")
            return _fetch_batch(names)
        except Exception as e:
            # print(f"[llm_fill_missing_quantities] _fill_batch except, fallback to empty spec, names={names}, exc={repr(e)}")
            if len(names) > 1:
                mid = len(names) // 2
                left = _fill_batch(names[:mid])
                right = _fill_batch(names[mid:])
                return {**left, **right}
            return {names[0]: _normalize_quantity_spec(names[0], {})}

    out_by_name: Dict[str, Dict[str, Any]] = {}
    for batch in _chunk_list(missing_quantity_name_zh, batch_size):
        out_by_name.update(_fill_batch(batch))

    if key_mode not in ("name_zh", "symbol_latex"):
        return out_by_name

    if key_mode == "name_zh":
        return out_by_name

    out_by_key: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()
    for name, spec in out_by_name.items():
        key = str(spec.get("symbol_latex") or spec.get("symbol") or spec.get("id") or name).strip()
        if not key:
            key = name
        base = key
        idx = 2
        while key in used:
            key = f"{base}_{idx}"
            idx += 1
        used.add(key)
        out_by_key[key] = spec

    return out_by_key


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate missing quantities via LLM")
    parser.add_argument("--category", required=True)
    parser.add_argument("--extractid", required=True)
    parser.add_argument("--missing", required=True, help="comma-separated missing quantity name_zh")
    parser.add_argument("--key-mode", default="name_zh", choices=["name_zh", "symbol_latex"])
    parser.add_argument(
        "--raw-dir",
        default=str((Path(__file__).resolve().parents[1] / "data" / "raw").resolve()),
    )
    parser.add_argument(
        "--md-dir",
        default=str((Path(__file__).resolve().parents[1] / "data" / "md").resolve()),
    )
    args = parser.parse_args(argv)

    missing = [x.strip() for x in args.missing.split(",") if x.strip()]
    out = generate_missing_quantities(
        category=args.category,
        extractid=args.extractid,
        missing_quantity_name_zh=missing,
        raw_dir=args.raw_dir,
        md_dir=args.md_dir,
        key_mode=args.key_mode,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
