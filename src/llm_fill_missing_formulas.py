#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM helper: generate formulas for quantities that are missing in formulas library.

Goal:
- Given a list of quantity specs, generate formulas for each quantity id:
    { "<quantity_id>": [ {formula_id, formula_name_zh, expr, latex, source:"llm"}, ... ], ... }

Reference context:
- Use ALL PDFs under data/raw as reference. Prefer extracted Markdown under
  data/md/<stem>/<stem>.md (if present). If MD is missing, still include the
  PDF filename.

This module is intentionally lightweight and can be used both as:
- library import from engine.py
- standalone CLI for debugging
"""

from __future__ import annotations

import json
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


# ==================== Normalization helpers ====================
_LATEX_SUBSCRIPT_BREAKS = set(" +-*=(),;:^")


def _wrap_latex_subscripts(s: str) -> str:
    """Ensure subscripts are wrapped with braces, e.g. x_1/4 -> x_{1/4}."""
    out: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "_" and i + 1 < len(s):
            if s[i + 1] == "{":
                out.append(ch)
                i += 1
                continue
            j = i + 1
            token: List[str] = []
            while j < len(s) and s[j] not in _LATEX_SUBSCRIPT_BREAKS:
                token.append(s[j])
                j += 1
            if token:
                out.append("_{")
                out.append("".join(token))
                out.append("}")
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_expr(expr: str) -> str:
    return str(expr or "").replace("_", "").strip()


def _normalize_formula_latex(latex: str) -> str:
    s = str(latex or "").strip()
    if not s:
        return ""
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1].strip()
    else:
        inner = s
    inner = _wrap_latex_subscripts(inner)
    return f"${inner}$"


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
    max_total_chars: int = 30000,
    per_file_chars: int = 2000,
) -> str:
    """Build a bounded reference corpus string.

    Note: requirement says "use all PDFs"; we include all filenames, and include
    extracted MD excerpts when available, bounded by max_total_chars.
    """
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
_PROMPT_TEMPLATE = (
    "你是航空航天/飞行器设计领域的公式抽取与建模专家。\n"
    "现在公式库中缺少以下物理量对应的公式，请根据 data/raw 的资料（已优先提供 Markdown 片段）补全。\n"
    "[任务]\n"
    "- 必须覆盖所有 quantities 列表中的物理量，不得遗漏。\n"
    "- 对每个 quantity_id 给出 3~5 条公式（少于 3 条无效）。\n"
    "- 尽量引用资料中出现的符号/变量；若资料不足，保守推断，并在 formula_name_zh 中标注“候选”。\n"
    "- 输出必须为合法 JSON；最外层 key 使用 quantity_id（避免 LaTeX 反斜杠导致 JSON 解析失败）。\n"
    "- 格式规则：\n"
    "  1) expr 变量名不得包含下划线 '_'。\n"
    "  2) 公式 latex 的下标必须用大括号完整包裹，如 Lambda_{{1/4}}、x_{{root}}。\n"
    "  3) 若输入/资料中出现 Lambda_1/4 或 x_root，规范化为 expr: Lambda1/4 或 xroot；latex: Lambda_{{1/4}} 或 x_{{root}}。\n"
    "[输出 JSON 结构]\n"
    "```\n"
    "{{\n"
    "  \"<quantity_id>\": [\n"
    "    {{\n"
    "      \"formula_id\": \"必须从允许列表中选择\",\n"
    "      \"formula_name_zh\": \"公式中文名（若推断需标注‘候选’）\",\n"
    "      \"expr\": \"等式文本（变量名不含下划线）\",\n"
    "      \"latex\": \"$...$\",\n"
    "      \"source\": \"llm\"\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
    "```\n"
    "[字段要求]\n"
    "- formulas 长度必须为 3~5。\n"
    "- formula_id 必须严格复用允许列表，禁止新增。\n"
    "[允许的 formula_id 列表]\n"
    "{allowed_formula_ids}\n"
    "[上下文]\n"
    "- category: {category}\n"
    "- extractid: {extractid}\n"
    "- quantities: {quantities}\n"
    "[已有公式示例]\n"
    "{existing_formula_examples}\n"
    "[PDF 参考资料]\n"
    "{reference_corpus}\n"
    "只输出 JSON，不要输出解释文字。"
).strip()


def _format_existing_examples(formulas: List[Dict[str, Any]], *, limit: int = 20) -> str:
    rows = []
    for f in formulas[:limit]:
        rows.append({
            "formula_id": f.get("formula_id"),
            "formula_name_zh": f.get("formula_name_zh"),
            "expr": f.get("expr"),
            "source": f.get("source"),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)


def generate_missing_formulas(
    *,
    category: str,
    extractid: str,
    quantities: List[Dict[str, Any]],
    existing_formula_examples: List[Dict[str, Any]],
    raw_dir: str,
    md_dir: str,
    llm_cfg: LLMConfig | None = None,
    allowed_formula_ids: List[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate formulas for quantities.

    Returns a dict mapping quantity_id -> list of formula dicts.
    """
    if not quantities:
        return {}

    llm = LLMClient(llm_cfg or LLMConfig())

    allowed_ids = [x for x in (allowed_formula_ids or []) if isinstance(x, str) and x.strip()]
    if not allowed_ids:
        allowed_ids = [x.get("formula_id") for x in (existing_formula_examples or []) if isinstance(x, dict)]
        allowed_ids = [x for x in allowed_ids if isinstance(x, str) and x.strip()]
    seen = set()
    allowed_ids = [x for x in allowed_ids if not (x in seen or seen.add(x))]

    reference_corpus = build_reference_corpus(raw_dir=raw_dir, md_dir=md_dir)
    prompt = _PROMPT_TEMPLATE.format(
        category=category,
        extractid=extractid,
        quantities=json.dumps(quantities, ensure_ascii=False),
        existing_formula_examples=_format_existing_examples(existing_formula_examples),
        reference_corpus=reference_corpus,
        allowed_formula_ids=json.dumps(allowed_ids, ensure_ascii=False),
    )

    text = llm.completion_text(
        user_prompt=prompt,
        system_prompt="你是一个严格的公式抽取与建模专家。",
    )
    obj = _extract_json_obj(text)

    obj_blocks: Dict[str, Any] = obj if isinstance(obj, dict) else {}

    def _find_block_for_qid(qid: str) -> Any:
        if not obj_blocks:
            return []
        if qid in obj_blocks:
            return obj_blocks.get(qid, [])
        for k, blk in obj_blocks.items():
            if str(k).strip() == str(qid).strip():
                return blk
        return []

    out: Dict[str, List[Dict[str, Any]]] = {}

    for q in quantities:
        qid = str(q.get("id") or "").strip()
        if not qid:
            continue
        block = _find_block_for_qid(qid)
        items: List[Any] = []

        if isinstance(block, dict):
            items = block.get("formulas") if isinstance(block.get("formulas"), list) else []
        elif isinstance(block, list):
            items = block

        normalized: List[Dict[str, Any]] = []
        for idx, it in enumerate(items, start=1):
            if not isinstance(it, dict):
                continue
            formula_id = str(it.get("formula_id") or "").strip()
            if allowed_ids:
                if formula_id not in allowed_ids:
                    formula_id = allowed_ids[(idx - 1) % len(allowed_ids)]
            else:
                if not formula_id:
                    formula_id = f"F_{qid}_llm_{idx}"
            normalized.append(
                {
                    "formula_id": formula_id,
                    "formula_name_zh": str(it.get("formula_name_zh") or f"{qid} candidate formula"),
                    "expr": _normalize_expr(it.get("expr") or ""),
                    "latex": _normalize_formula_latex(it.get("latex") or ""),
                    "source": "llm",
                }
            )
        out[qid] = normalized

    return out


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate missing formulas via LLM")
    parser.add_argument("--category", required=True)
    parser.add_argument("--extractid", required=True)
    parser.add_argument("--quantities", required=True, help="JSON list of quantity specs")
    parser.add_argument(
        "--raw-dir",
        default=str((Path(__file__).resolve().parents[1] / "data" / "raw").resolve()),
    )
    parser.add_argument(
        "--md-dir",
        default=str((Path(__file__).resolve().parents[1] / "data" / "md").resolve()),
    )
    args = parser.parse_args(argv)

    quantities = json.loads(args.quantities)
    out = generate_missing_formulas(
        category=args.category,
        extractid=args.extractid,
        quantities=quantities,
        existing_formula_examples=[],
        raw_dir=args.raw_dir,
        md_dir=args.md_dir,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
