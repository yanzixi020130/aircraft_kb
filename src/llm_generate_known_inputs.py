#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate plausible known_inputs / variable specs via LLM ONLY.

变更说明（按你的要求）：
- 不再从物理量库(quantities.yaml)检查/补全任何信息；
- 所有信息均由大模型生成：name / value / unit；
- 保持现有输出格式不变：generate_known_inputs_response() 仍返回
  {"variables": [...], "status": "ok", "category": <str>}。

关键约束：
- selectedFormulas 中每个公式块的 target（以及 expr 左侧变量）属于“未知量/目标量”，
  这些变量绝对不能出现在最终输出 variables 中。
- LLM 可能会“额外输出” target 变量或其它无关变量，所以代码必须在返回前做强制裁剪。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from llm_client import LLMClient, LLMConfig

DEFAULT_CATEGORY = "llm"  # 仅用于输出占位

def _format_source_label(source: str, src_file: str | None = None) -> str:
    if source == "expert":
        return "专家库"
    if source == "thesis":
        return f"论文库（{src_file}）" if src_file else "论文库"
    if source == "llm":
        return "大模型生成"
    return str(source)

# -------------------------
# Helpers
# -------------------------

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _norm_varname(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = s.strip("$")
    return s


def _extract_identifiers(expr: str) -> Set[str]:
    """Extract identifier-like tokens from expr (ASCII identifiers only)."""
    if not expr or not isinstance(expr, str):
        return set()
    return set(_IDENTIFIER_RE.findall(expr))


def _lhs_identifier(expr: str) -> str | None:
    """Return left-hand identifier of 'A = ...' if present."""
    if not expr or not isinstance(expr, str) or "=" not in expr:
        return None
    left = expr.split("=", 1)[0].strip()
    # sometimes there are spaces or parentheses; keep first identifier
    m = _IDENTIFIER_RE.search(left)
    return m.group(0) if m else None


def _collect_required_inputs(selected_formulas: Dict[str, Dict[str, Any]]) -> Set[str]:
    """
    Collect required input variables from formula expressions.

    Rules:
    - Exclude targets: keys of selected_formulas (targets), plus each expr's LHS variable.
    - Collect identifiers from RHS of expr.
    """
    targets_norm = set(_norm_varname(k) for k in selected_formulas.keys())

    # Also treat each formula's LHS as forbidden (unknown/target)
    forbidden: Set[str] = set(targets_norm)
    for _, f in selected_formulas.items():
        if isinstance(f, dict):
            lhs = _lhs_identifier(str(f.get("expr", "")))
            if lhs:
                forbidden.add(_norm_varname(lhs))

    required: Set[str] = set()
    for _, f in selected_formulas.items():
        if not isinstance(f, dict):
            continue
        expr = str(f.get("expr", ""))
        if "=" in expr:
            rhs = expr.split("=", 1)[1]
        else:
            rhs = expr
        for tok in _extract_identifiers(rhs):
            if _norm_varname(tok) not in forbidden:
                required.add(tok)

    # remove empties
    required = {t for t in required if _norm_varname(t)}
    return required


def _chunk_list(items: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [items]
    return [items[i : i + n] for i in range(0, len(items), n)]


def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    return json.loads(s)


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from LLM text.
    Supports ```json ...``` fenced blocks, or the first {...} object.
    """
    if not text:
        raise ValueError("Empty LLM response")

    m = _JSON_BLOCK_RE.search(text)
    if m:
        return _safe_json_loads(m.group(1))

    # fallback: find first { ... } by balancing braces (simple scan)
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return _safe_json_loads(candidate)

    raise ValueError("Unbalanced JSON braces in LLM response")


def _coerce_unit(u: Any) -> str | None:
    if u is None:
        return None
    us = str(u).strip()
    if not us or us in ("1", "-", "none", "None", "null"):
        return None
    return us


def _coerce_value(v: Any) -> str:
    if v is None:
        return ""
    vs = str(v).strip()
    return vs


# -------------------------
# LLM prompt
# -------------------------

_PROMPT_TEMPLATE = """
你是航空航天/飞行器设计领域的工程师。请为下列“输入变量”生成**合理的典型值**，并给出**变量中文名称(name)**与**单位(unit)**。

[硬性要求]
- 必须覆盖 required 列表里的每一个变量 id，不能遗漏。
- 只输出 JSON（不要解释、不要多余文字）。
- unit 使用常见国际单位制（SI）或工程常用单位；如果变量是无量纲，unit 置为 null。
- value 尽量输出可解析的数字字符串（可以是小数或科学计数法）。
- 绝对不要在输出中包含 targets 中列出的变量（即使你认为它们合理也不要输出）。

[targets（禁止输出）]
{targets}

[required（必须输出）]
{required_block}

[输出格式]
```json
{{
  "items": {{
    "变量id": {{
      "name": "中文名称",
      "value": "数值字符串",
      "unit": "单位或null"
    }}
  }}
}}
```

[上下文]
- category: {category}
- formulaKey: {formula_key}
""".strip()


def _build_required_block(required: List[str]) -> str:
    return "\n".join([f"- {rid}" for rid in required])


def _llm_generate_items(
    client: LLMClient,
    *,
    required: List[str],
    targets_norm: Set[str],
    category: str,
    formula_key: str | None,
    temperature: float = 0.2,
    retries: int = 2,
) -> Dict[str, Dict[str, Any]]:
    required_block = _build_required_block(required)
    prompt = _PROMPT_TEMPLATE.format(
        required_block=required_block,
        category=category,
        formula_key=formula_key,
        targets=", ".join(sorted(targets_norm)) if targets_norm else "",
    )

    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            llm_text = client.completion_text(user_prompt=prompt, temperature=temperature)
            obj = _extract_json_obj(llm_text)
            items = obj.get("items") or obj.get("known_inputs") or obj.get("variables") or {}
            if not isinstance(items, dict):
                raise ValueError("LLM JSON: items is not a dict")
            # normalize to dict[varid] -> {name,value,unit}
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in items.items():
                if _norm_varname(k) in targets_norm:
                    continue
                if isinstance(v, dict):
                    out[str(k)] = v
                else:
                    # allow "value unit" string, best-effort parse
                    raw = str(v).strip()
                    out[str(k)] = {"name": "", "value": raw, "unit": None}
            return out
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("LLM generation failed")


# -------------------------
# Public APIs
# -------------------------

def generate_known_input_items_from_payload(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
    targets: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return dict: var_id -> {"name": str, "value": str, "unit": str|None}

    强制裁剪：
    - 只保留 required 变量
    - 严格剔除 targets
    """
    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    selected = params.get("selectedFormulas", {}) if isinstance(params, dict) else {}
    formula_key = params.get("formulaKey") if isinstance(params, dict) else None

    cat = (category or DEFAULT_CATEGORY).strip()

    selected_norm: Dict[str, Dict[str, Any]] = {}
    for k, v in (selected or {}).items():
        if isinstance(v, dict):
            selected_norm[str(k)] = v
    if not selected_norm:
        raise ValueError("selectedFormulas is empty or invalid")

    required = sorted(_collect_required_inputs(selected_norm))

    targets_norm: Set[str] = set(_norm_varname(t) for t in (targets or []))
    # also forbid the targets implied by selected formulas
    targets_norm |= set(_norm_varname(k) for k in selected_norm.keys())
    for _, f in selected_norm.items():
        lhs = _lhs_identifier(str(f.get("expr", "")))
        if lhs:
            targets_norm.add(_norm_varname(lhs))

    # re-filter required
    required = [r for r in required if _norm_varname(r) not in targets_norm]

    llm_config = llm_cfg or LLMConfig()
    client = LLMClient(llm_config)

    batch_size = int(os.getenv("LLM_KNOWN_INPUTS_BATCH_SIZE", "8"))
    retries = int(os.getenv("LLM_KNOWN_INPUTS_RETRIES", "2"))

    items: Dict[str, Dict[str, Any]] = {}
    for batch in _chunk_list(required, batch_size):
        got = _llm_generate_items(
            client,
            required=batch,
            targets_norm=targets_norm,
            category=cat,
            formula_key=formula_key,
            temperature=0.2,
            retries=retries,
        )
        items.update(got)

    # ✅ 强制裁剪：只保留 required，并剔除 targets（双保险）
    required_norm = set(_norm_varname(x) for x in required)
    items = {k: v for k, v in items.items() if _norm_varname(k) in required_norm and _norm_varname(k) not in targets_norm}

    # fill missing required with minimal placeholders (仍然不查库)
    for rid in required:
        if rid not in items:
            items[rid] = {"name": "", "value": "1", "unit": None}

    return items


def generate_known_inputs_from_payload(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
    targets: List[str] | None = None,
) -> Dict[str, str]:
    """
    Backward-compatible API: return dict var_id -> "<value> <unit>" or "<value>" if unit is null.
    """
    items = generate_known_input_items_from_payload(
        payload,
        category=category,
        llm_cfg=llm_cfg,
        targets=targets,
    )
    out: Dict[str, str] = {}
    for rid, spec in items.items():
        v = _coerce_value(spec.get("value"))
        u = _coerce_unit(spec.get("unit"))
        out[rid] = f"{v} {u}".strip() if u else v
    return out


def generate_known_inputs_response(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
    targets: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Return the original response shape (保持格式不变):

    {
      "variables": [
        {"quantity_id","symbol","name","value","unit","context","source"},
        ...
      ],
      "status": "ok",
      "category": <str>
    }
    """
    cat = (category or DEFAULT_CATEGORY).strip()

    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    selected = params.get("selectedFormulas", {}) if isinstance(params, dict) else {}
    selected_norm: Dict[str, Dict[str, Any]] = {}
    for k, v in (selected or {}).items():
        if isinstance(v, dict):
            selected_norm[str(k)] = v

    # forbidden targets (external + implied)
    forbidden_norm: Set[str] = set(_norm_varname(t) for t in (targets or []))
    forbidden_norm |= set(_norm_varname(k) for k in selected_norm.keys())
    for _, f in selected_norm.items():
        lhs = _lhs_identifier(str(f.get("expr", "")))
        if lhs:
            forbidden_norm.add(_norm_varname(lhs))

    items = generate_known_input_items_from_payload(
        payload,
        category=cat,
        llm_cfg=llm_cfg,
        targets=list(forbidden_norm),
    )

    variables: List[Dict[str, Any]] = []
    for rid, spec in items.items():
        qid = str(rid)

        # ✅ 最终双保险：绝不输出 targets
        if _norm_varname(qid) in forbidden_norm:
            continue

        name = str(spec.get("name", "") or "")
        value = _coerce_value(spec.get("value"))
        unit = _coerce_unit(spec.get("unit"))
        variables.append(
            {
                "quantity_id": qid,
                "symbol": f"${qid}$",
                "name": name,
                "value": value,
                "unit": unit,
                "context": "",
                "source": "大模型生成",
            }
        )

    return {
        "variables": variables,
        "status": "ok",
        "category": cat,
    }


__all__ = [
    "generate_known_inputs_from_payload",
    "generate_known_input_items_from_payload",
    "generate_known_inputs_response",
]
