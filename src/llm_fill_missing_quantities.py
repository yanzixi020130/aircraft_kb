#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM helper: generate formulas for missing quantity_ids.

Goal:
- Given missing quantity ids, generate a mapping:
    { "K_max": [ {formula_id, formula_name_zh, expr, latex, source:"llm"}, ... ], ... }

Reference context:
- Use ALL PDFs under data/raw as reference. Practically we prefer to read already-extracted
  Markdown under data/md/<stem>/<stem>.md (if present). If MD is missing, we still include
  the PDF filename.

This module is intentionally lightweight and can be used both as:
- library import from engine.py
- standalone CLI for debugging
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

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
        "以下是 data/raw 下 PDF 的参考信息（优先使用已抽取的 Markdown 片段；若缺失则只提供文件名）。\n"
        f"总PDF数: {len(pdfs)}，已包含片段数: {len(chunks)}，已做长度截断。\n\n"
    )
    return header + "\n".join(chunks)


# ==================== Prompt ====================
_PROMPT_TEMPLATE = """
你是航空航天/飞行器设计领域的公式抽取与建模专家。
Note: missing_quantity_ids may be quantity ids or Chinese names (name_zh). If a missing id looks like name_zh, keep name_zh and choose a reasonable id/symbol.

现在知识库中缺少以下物理量，需要你根据 data/raw 的资料（已优先提供 Markdown 片段）补齐：
1) 该物理量自身的信息（等价于 quantities.yaml 的一条记录）
2) 该物理量可能对应的公式列表

【任务】
- 必须覆盖所有 missing_quantity_id，不得遗漏；即使资料不足也要给出该物理量信息并补齐 3~5 条候选公式。
- 对每个 missing_quantity_id，先给出该物理量的定义（id/symbol/symbol_latex/name_zh/unit），再给出 3~5 条可能的公式（少于 3 条视为无效输出，务必补足）。
- 公式必须尽量引用资料中出现过的符号/变量，不要凭空臆造；如资料不足，请给出最保守的候选，并在 formula_name_zh 里标注“推断/候选”。
- 若缺少可靠资料，仍需输出：unit 用保守的 SI/1，公式列表可以复用【允许的 formula_id 列表】按序循环；所有字段不可留空。
- 输出必须是“合法 JSON 对象”，其中 key 是 missing_quantity_id，value 是一个对象，包含字段 quantity 与 formulas。

【输出 JSON 结构（必须严格遵守）】
```
{{
    "<missing_quantity_id>": {{
        "quantity": {{
            "id": "与 missing_quantity_id 相同或同义映射",
            "symbol": "变量符号（如 g）",
            "symbol_latex": "LaTeX 符号（如 g）",
            "name_zh": "物理量中文名",
            "unit": "SI 单位，无量纲用 '1'"
        }},
        "formulas": [
            {{
                "formula_id": "必须从【允许的 formula_id 列表】选择，不得新增",
                "formula_name_zh": "公式中文名（若推断需标注候选）",
                "expr": "等式文本，如 L = q * Swing * CL",
                "latex": "$...$ 形式",
                "source": "llm"
            }}
        ]
    }}
}}
```

【字段要求】
- quantity.id 应尽量与 missing_quantity_id 一致；若需映射同义写法，可做轻微规范化。
- unit 使用 SI；无量纲用 '1'。
- formulas[*] 数组长度必须 3~5；不足 3 条视为无效输出。
- formulas[*].formula_id 必须严格复用【允许的 formula_id 列表】，不得新增、改写或加后缀。

【允许的 formula_id 列表（只能从这里选）】
{allowed_formula_ids}

【上下文】
- category: {category}
- extractid: {extractid}
- missing_quantity_ids: {missing_quantity_ids}

【已有公式示例（仅供风格参考）】
{existing_formula_examples}

【PDF 参考资料】
{reference_corpus}

只输出 JSON，不要输出解释文字。
""".strip()


_QUANTITY_ONLY_PROMPT = """
You are filling missing quantity definitions. Only output quantity fields, no formulas.
missing_quantity_ids may be quantity ids or Chinese names (name_zh). If a missing id looks like name_zh, keep name_zh and choose a reasonable id/symbol.

Output a pure JSON object with this shape:
{
  "<missing_quantity_id>": {
    "quantity": {
      "id": "...",
      "symbol": "...",
      "symbol_latex": "...",
      "name_zh": "...",
      "unit": "SI unit or '1'"
    }
  }
}

Context:
- category: {category}
- extractid: {extractid}
- missing_quantity_ids: {missing_quantity_ids}
""".strip()


_FORMULA_ONLY_PROMPT = """
Generate formulas for the missing quantities below. Use the provided quantity specs.
Return pure JSON. Each key must be an input id, and include "formulas" only.

Quantity specs:
{quantity_specs}

Allowed formula ids (must reuse, do not invent):
{allowed_formula_ids}

Existing formula examples (style reference only):
{existing_formula_examples}

Reference corpus:
{reference_corpus}

Context:
- category: {category}
- extractid: {extractid}
- missing_quantity_ids: {missing_quantity_ids}

Output JSON:
{
  "<missing_quantity_id>": {
    "formulas": [
      {
        "formula_id": "...",
        "formula_name_zh": "... (mark 推断/候选 when needed)",
        "expr": "...",
        "latex": "...",
        "source": "llm"
      }
    ]
  }
}
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
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for _ in range(retries + 1):
        text = llm.completion_text(user_prompt=prompt, system_prompt="你是一位严谨的公式抽取与建模专家。")
        if not text:
            last_err = ValueError("empty response")
            continue
        try:
            return _extract_json_obj(text)
        except Exception as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise ValueError("LLM JSON parse failed")


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


def generate_missing_quantity_formulas(
    *,
    category: str,
    extractid: str,
    missing_quantity_ids: List[str],
    existing_formula_examples: List[Dict[str, Any]],
    raw_dir: str,
    md_dir: str,
    llm_cfg: LLMConfig | None = None,
    allowed_formula_ids: List[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate formula blocks for missing quantities.

    Returns a dict mapping missing qid -> list of formula dicts.
    """
    if not missing_quantity_ids:
        return {}

    llm = LLMClient(llm_cfg or LLMConfig())

    allowed_ids = [x for x in (allowed_formula_ids or []) if isinstance(x, str) and x.strip()]
    # If caller doesn't pass allowed ids, fall back to using ids from examples.
    if not allowed_ids:
        allowed_ids = [x.get("formula_id") for x in (existing_formula_examples or []) if isinstance(x, dict)]
        allowed_ids = [x for x in allowed_ids if isinstance(x, str) and x.strip()]
    # De-duplicate but keep order
    seen = set()
    allowed_ids = [x for x in allowed_ids if not (x in seen or seen.add(x))]

    batch_size = int(os.getenv("LLM_MISSING_BATCH_SIZE", "6"))
    formula_batch_size = int(os.getenv("LLM_MISSING_FORMULA_BATCH_SIZE", str(batch_size)))
    retries = int(os.getenv("LLM_MISSING_RETRIES", "2"))
    max_total_chars = int(os.getenv("LLM_MISSING_MAX_TOTAL_CHARS", "20000"))
    per_file_chars = int(os.getenv("LLM_MISSING_PER_FILE_CHARS", "1500"))
    allowed_id_limit = int(os.getenv("LLM_MISSING_ALLOWED_ID_LIMIT", "120"))
    if allowed_id_limit > 0 and len(allowed_ids) > allowed_id_limit:
        allowed_ids = allowed_ids[:allowed_id_limit]

    reference_corpus = build_reference_corpus(
        raw_dir=raw_dir,
        md_dir=md_dir,
        max_total_chars=max_total_chars,
        per_file_chars=per_file_chars,
    )

    def _normalize_quantity_spec(
        qid: str,
        spec: Dict[str, Any] | None,
        *,
        fallback: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        spec = spec or {}
        fallback = fallback or {}
        _id = str(spec.get("id") or fallback.get("id") or qid)
        symbol = str(spec.get("symbol") or fallback.get("symbol") or _id)
        symbol_latex = str(spec.get("symbol_latex") or fallback.get("symbol_latex") or symbol)
        name_zh = str(spec.get("name_zh") or fallback.get("name_zh") or "")
        if not name_zh and re.search(r"[\u4e00-\u9fff]", str(qid)):
            name_zh = str(qid)
        unit = str(spec.get("unit") or fallback.get("unit") or "1")
        return {
            "id": _id,
            "symbol": symbol,
            "symbol_latex": symbol_latex,
            "name_zh": name_zh,
            "unit": unit,
        }

    def _fetch_quantity_specs(ids: List[str]) -> Dict[str, Dict[str, Any]]:
        prompt = _QUANTITY_ONLY_PROMPT.format(
            category=category,
            extractid=extractid,
            missing_quantity_ids=json.dumps(ids, ensure_ascii=False),
        )
        obj = _call_llm_json(llm, prompt, retries=retries)
        out: Dict[str, Dict[str, Any]] = {}
        for qid in ids:
            block = obj.get(qid, {})
            qspec = block.get("quantity") if isinstance(block, dict) else {}
            if not isinstance(qspec, dict):
                qspec = {}
            out[qid] = qspec
        return out

    def _fill_quantity_specs(ids: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            return _fetch_quantity_specs(ids)
        except Exception:
            if len(ids) > 1:
                mid = len(ids) // 2
                left = _fill_quantity_specs(ids[:mid])
                right = _fill_quantity_specs(ids[mid:])
                return {**left, **right}
            return {ids[0]: {}}

    quantity_specs: Dict[str, Dict[str, Any]] = {}
    for batch in _chunk_list(missing_quantity_ids, batch_size):
        quantity_specs.update(_fill_quantity_specs(batch))

    def _fetch_formulas(ids: List[str]) -> Dict[str, Any]:
        specs_payload = {
            qid: {"quantity": quantity_specs.get(qid, {})}
            for qid in ids
        }
        prompt = _FORMULA_ONLY_PROMPT.format(
            category=category,
            extractid=extractid,
            missing_quantity_ids=json.dumps(ids, ensure_ascii=False),
            quantity_specs=json.dumps(specs_payload, ensure_ascii=False, indent=2),
            existing_formula_examples=_format_existing_examples(existing_formula_examples, limit=12),
            reference_corpus=reference_corpus,
            allowed_formula_ids=json.dumps(allowed_ids, ensure_ascii=False),
        )
        return _call_llm_json(llm, prompt, retries=retries)

    def _fill_formulas(ids: List[str]) -> Dict[str, Any]:
        try:
            return _fetch_formulas(ids)
        except Exception:
            if len(ids) > 1:
                mid = len(ids) // 2
                left = _fill_formulas(ids[:mid])
                right = _fill_formulas(ids[mid:])
                return {**left, **right}
            return {ids[0]: {}}

    def _fetch_combined(ids: List[str]) -> Dict[str, Any]:
        prompt = _PROMPT_TEMPLATE.format(
            category=category,
            extractid=extractid,
            missing_quantity_ids=json.dumps(ids, ensure_ascii=False),
            existing_formula_examples=_format_existing_examples(existing_formula_examples, limit=12),
            reference_corpus=reference_corpus,
            allowed_formula_ids=json.dumps(allowed_ids, ensure_ascii=False),
        )
        return _call_llm_json(llm, prompt, retries=retries)

    formulas_obj: Dict[str, Any] = {}
    for batch in _chunk_list(missing_quantity_ids, formula_batch_size):
        formulas_obj.update(_fill_formulas(batch))

    # normalize + enforce source + enforce formula_id reuse
    out: Dict[str, List[Dict[str, Any]]] = {}
    for qid in missing_quantity_ids:
        block = formulas_obj.get(qid, {})
        quantity_spec_raw = None
        items = []

        # ?????????
        # 1) ????{ qid: { quantity: {...}, formulas: [...] } }
        # 2) ????{ qid: [ {...}, {...} ] }
        if isinstance(block, dict):
            quantity_spec_raw = block.get("quantity") if isinstance(block.get("quantity"), dict) else None
            items = block.get("formulas") if isinstance(block.get("formulas"), list) else []
        elif isinstance(block, list):
            items = block

        if not items:
            try:
                combined = _fetch_combined([qid])
                block2 = combined.get(qid, {})
                if isinstance(block2, dict):
                    quantity_spec_raw = block2.get("quantity") if isinstance(block2.get("quantity"), dict) else quantity_spec_raw
                    items = block2.get("formulas") if isinstance(block2.get("formulas"), list) else items
                elif isinstance(block2, list):
                    items = block2
            except Exception:
                pass

        quantity_spec = _normalize_quantity_spec(
            qid,
            quantity_spec_raw,
            fallback=quantity_specs.get(qid),
        )

        normalized: List[Dict[str, Any]] = []
        for idx, it in enumerate(items, start=1):
            if not isinstance(it, dict):
                continue
            formula_id = str(it.get("formula_id") or "").strip()
            # Strict reuse: only allow ids from the provided whitelist.
            if allowed_ids:
                if formula_id not in allowed_ids:
                    # Deterministic fallback: assign ids in round-robin order.
                    formula_id = allowed_ids[(idx - 1) % len(allowed_ids)]
            else:
                # No whitelist available: keep original if it looks like an id, otherwise synthesize.
                if not formula_id:
                    formula_id = f"F_{qid}_llm_{idx}"
            normalized.append(
                {
                    "quantity_name_zh": quantity_spec.get("name_zh", ""),
                    "quantity_symbol": quantity_spec.get("symbol", ""),
                    "quantity_symbol_latex": quantity_spec.get("symbol_latex", ""),
                    "quantity_unit": quantity_spec.get("unit", "1"),
                    "quantity_value": "",
                    "formula_id": formula_id,
                    "formula_name_zh": str(it.get("formula_name_zh") or f"{qid} ????"),
                    "expr": str(it.get("expr") or ""),
                    "latex": str(it.get("latex") or ""),
                    "source": "llm",
                }
            )
        out[qid] = normalized

    return out


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate formulas for missing quantities via LLM")
    parser.add_argument("--category", required=True)
    parser.add_argument("--extractid", required=True)
    parser.add_argument("--missing", required=True, help="comma-separated missing quantity ids")
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
    out = generate_missing_quantity_formulas(
        category=args.category,
        extractid=args.extractid,
        missing_quantity_ids=missing,
        existing_formula_examples=[],
        raw_dir=args.raw_dir,
        md_dir=args.md_dir,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
