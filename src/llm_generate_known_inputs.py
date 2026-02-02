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
- selectedFormulas 中每个公式块的 target 属于“未知量/目标量”，
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

from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

try:
    import yaml
except Exception:
    yaml = None


EXPERT_QUANTITIES_FILE = Path("/data/se42/extraction_test/data/quantities/expert/quantities.yaml")
THESIS_QUANTITIES_DIR = Path("/data/se42/extraction_test/data/quantities/thesis")


@dataclass
class QuantityMeta:
    quantity_id: str
    name_zh: str | None
    unit: str | None
    source: str  # "expert" or thesis filename stem


def _safe_load_yaml(path: Path):
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_quantity_items(doc) -> list[dict]:
    """
    兼容几种常见结构：
    - list[dict]
    - dict 包含 "quantities"/"items"/"variables"/"data" -> list[dict]
    """
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    if isinstance(doc, dict):
        for k in ("quantities", "items", "variables", "data"):
            v = doc.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _norm_unit(u):
    if u is None:
        return None
    us = str(u).strip()
    if not us or us.lower() in ("null", "none", "1", "-"):
        return None
    return us


def _thesis_source_from_filename(fp: Path) -> str:
    # e.g. "tilt_rotor_quantities.yaml" -> "tilt_rotor"
    name = fp.stem  # remove .yaml
    suffix = "_quantities"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


@lru_cache(maxsize=1)
def build_quantity_index() -> dict[str, QuantityMeta]:
    """
    返回：quantity_id -> QuantityMeta
    （你当前需求只需要按 quantity_id 精确匹配即可）
    """
    idx: dict[str, QuantityMeta] = {}

    # 1) expert/quantities.yaml -> source="专家知识"
    if EXPERT_QUANTITIES_FILE.exists():
        doc = _safe_load_yaml(EXPERT_QUANTITIES_FILE)
        for item in _extract_quantity_items(doc):
            qid = str(item.get("quantity_id") or item.get("id") or "").strip()
            if not qid:
                continue
            meta = QuantityMeta(
                quantity_id=qid,
                name_zh=str(item.get("name_zh") or item.get("quantity_name_zh") or item.get("name") or "").strip() or None,
                unit=_norm_unit(item.get("unit")),
                source="专家知识",
            )
            idx[qid] = meta

    # 2) thesis/*.yaml -> source=filename(without _quantities.yaml)
    if THESIS_QUANTITIES_DIR.exists():
        for fp in sorted(list(THESIS_QUANTITIES_DIR.glob("*.yml")) + list(THESIS_QUANTITIES_DIR.glob("*.yaml"))):
            doc = _safe_load_yaml(fp)
            src = _thesis_source_from_filename(fp)
            for item in _extract_quantity_items(doc):
                qid = str(item.get("quantity_id") or item.get("id") or "").strip()
                if not qid:
                    continue
                # expert 优先：thesis 不覆盖已存在的 expert
                if qid in idx:
                    continue
                meta = QuantityMeta(
                    quantity_id=qid,
                    name_zh=str(item.get("name_zh") or item.get("quantity_name_zh") or item.get("name") or "").strip() or None,
                    unit=_norm_unit(item.get("unit")),
                    source=src,
                )
                idx[qid] = meta

    return idx


DEFAULT_CATEGORY = "llm"  # 仅用于输出占位

def _format_source_label(source: str, src_file: str | None = None) -> str:
    if source == "expert":
        return "专家知识"
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


def _collect_required_inputs(selected_formulas: Dict[str, Dict[str, Any]]) -> Set[str]:
    """
    Collect required input variables from formula expressions.

    Rules:
    - Use the whole expr (do NOT infer by left/right side).
    - Exclude targets: keys of selected_formulas.
    """
    targets_norm = set(_norm_varname(k) for k in selected_formulas.keys())

    required: Set[str] = set()
    for _, f in selected_formulas.items():
        if not isinstance(f, dict):
            continue
        expr = str(f.get("expr", ""))
        for tok in _extract_identifiers(expr):
            if _norm_varname(tok) not in targets_norm:
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

def _collect_var_formula_context(selected_formulas: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Return: var -> [context_str, ...]
    context_str 例： "expr: L = q * CL * S (mentions: q, CL, S)"
    """
    ctx: Dict[str, List[str]] = {}
    for _, f in (selected_formulas or {}).items():
        if not isinstance(f, dict):
            continue
        expr = str(f.get("expr", "") or "")
        if not expr:
            continue

        # Use all identifiers in the full expression; do not infer by side.
        toks = sorted(set(_extract_identifiers(expr)))

        # 为每个 token 记录它在哪个 expr 出现
        for t in toks:
            ctx.setdefault(t, []).append(f"expr: {expr}")
    return ctx

def _build_required_block_with_context(required: List[str], var_ctx_map: Dict[str, List[str]], max_expr: int = 3) -> str:
    lines = []
    for rid in required:
        exprs = var_ctx_map.get(rid, [])[:max_expr]
        if exprs:
            lines.append(f"- {rid}\n  - " + "\n  - ".join(exprs))
        else:
            lines.append(f"- {rid}\n  - (no expr context found)")
    return "\n".join(lines)

def _ensure_distinct_fallback(items: Dict[str, Dict[str, Any]]) -> None:
    dups = _find_duplicate_name_context(items)
    for group in dups:
        # 从第二个开始加后缀，保证不同
        for idx, k in enumerate(group):
            if idx == 0:
                continue
            name = str(items[k].get("name", "") or "")
            ctx = str(items[k].get("context", "") or "")
            # 用 quantity_id 做最小差异化
            items[k]["name"] = f"{name}（{k}）" if name else f"{k}"
            items[k]["context"] = f"{ctx}（变量id={k}，需结合公式进一步区分）" if ctx else f"变量id={k}，需结合公式进一步区分。"


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
- context 用中文，1~2 句话，尽量解释“是什么/代表什么/在飞行器设计里用来干嘛”，不要写成一大段科普。
- **同一批输出中：不同变量id 不允许出现完全相同的 (name + context)。**
  如果两个变量物理意义接近，也必须在 name 或 context 中点明差异来源（例如：属于哪个部件/哪个比值/哪个系数/在公式中与哪些量关联）。

[targets（禁止输出）]
{targets}

[required（必须输出,含公式上下文）]
{required_block}

 [输出格式]
 ```json
 {{
   "items": {{
     "变量id": {{
       "name": "中文名称",
+      "context": "一句话定义/物理意义/使用场景（中文，尽量精炼）",
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

_PROMPT_TEMPLATE_FOUND = """
你是航空航天/飞行器设计领域的工程师。下列变量的中文名与单位已由物理量库确定，请你仅生成：
- 合理的典型值(value)：尽量可解析的数字字符串
- 简短中文说明(context)：1~2 句话，说明物理意义/用途（结合给出的公式上下文）

[硬性要求]
- 必须覆盖 required 列表里的每一个变量 id，不能遗漏。
- 只输出 JSON（不要解释、不要多余文字）。
- 绝对不要输出 targets 中列出的变量。
- 不要输出 name/unit/source 字段（已经确定，不需要你生成）。

[targets（禁止输出）]
{targets}

[required（必须输出,含公式上下文）]
{required_block}

[输出格式]
```json
{{
  "items": {{
    "变量id": {{
      "value": "数值字符串",
      "context": "一句话定义/物理意义/使用场景（中文，尽量精炼）"
    }}
  }}
}}
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
    var_ctx_map: Dict[str, List[str]] | None = None,
    temperature: float = 0.2,
    retries: int = 2,
    prompt_template: str = _PROMPT_TEMPLATE,
) -> Dict[str, Dict[str, Any]]:
    required_block = _build_required_block_with_context(required, var_ctx_map)
    prompt = prompt_template.format(
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

def _find_duplicate_name_context(items: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """
    找出 (name, context) 完全相同但 key 不同的组
    Return: [["lambdah","lambdaht"], ...]
    """
    buckets: Dict[Tuple[str, str], List[str]] = {}
    for k, v in items.items():
        name = str((v or {}).get("name", "") or "").strip()
        ctx = str((v or {}).get("context", "") or "").strip()
        key = (name, ctx)
        buckets.setdefault(key, []).append(k)
    return [ks for (name, ctx), ks in buckets.items() if len(ks) > 1 and (name or ctx)]

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
    Return dict: var_id -> {"name": str, "value": str, "unit": str|None, "context": str}

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

    var_ctx_map = _collect_var_formula_context(selected_norm)

    required = sorted(_collect_required_inputs(selected_norm))

    targets_norm: Set[str] = set(_norm_varname(t) for t in (targets or []))
    # also forbid the targets implied by selected formulas
    targets_norm |= set(_norm_varname(k) for k in selected_norm.keys())

    # re-filter required
    required = [r for r in required if _norm_varname(r) not in targets_norm]

    llm_config = llm_cfg or LLMConfig()
    client = LLMClient(llm_config)

    batch_size = int(os.getenv("LLM_KNOWN_INPUTS_BATCH_SIZE", "8"))
    retries = int(os.getenv("LLM_KNOWN_INPUTS_RETRIES", "2"))

    # ==========================================================
    # ✅ 第四步：分流（库内 found / 库外 missing）+ 两套提示词
    # ==========================================================
    meta_index = build_quantity_index()  # quantity_id -> meta(含name_zh/unit/source)

    found_required = [r for r in required if str(r) in meta_index]
    missing_required = [r for r in required if str(r) not in meta_index]

    items: Dict[str, Dict[str, Any]] = {}

    # A) 库内：只生成 value/context（不要 name/unit/source）
    for batch in _chunk_list(found_required, batch_size):
        got = _llm_generate_items(
            client,
            required=batch,
            targets_norm=targets_norm,
            category=cat,
            formula_key=formula_key,
            var_ctx_map=var_ctx_map,
            temperature=0.2,
            retries=retries,
            prompt_template=_PROMPT_TEMPLATE_FOUND,  # ✅ 新增的“库内 prompt”
        )
        items.update(got)

    # B) 库外：用你现有逻辑生成 name/unit/value/context
    for batch in _chunk_list(missing_required, batch_size):
        got = _llm_generate_items(
            client,
            required=batch,
            targets_norm=targets_norm,
            category=cat,
            formula_key=formula_key,
            var_ctx_map=var_ctx_map,
            temperature=0.2,
            retries=retries,
            prompt_template=_PROMPT_TEMPLATE,  # ✅ 你原来的模板
        )
        items.update(got)

    # ✅ 强制裁剪：只保留 required，并剔除 targets（双保险）
    required_norm = set(_norm_varname(x) for x in required)
    items = {
        k: v
        for k, v in items.items()
        if _norm_varname(k) in required_norm and _norm_varname(k) not in targets_norm
    }

    # fill missing required with minimal placeholders
    for rid in required:
        if rid not in items:
            # 库内缺失：先给空壳，value/context 后续也可再补一次
            if str(rid) in meta_index:
                items[rid] = {"value": "1", "context": ""}  # name/unit/source 不在这里填
            else:
                items[rid] = {"name": "", "context": "", "value": "1", "unit": None}

    # ==========================================================
    # ✅ 修复缩进：dups 不要写在 for 循环内部（你原来缩进有问题）:contentReference[oaicite:2]{index=2}
    # 并且：建议只对“库外 missing”做 name/context 去重，避免污染库内 name
    # ==========================================================
    dups = _find_duplicate_name_context(items)  # :contentReference[oaicite:3]{index=3}
    if dups:
        conflict_vars = sorted({x for group in dups for x in group})

        # 只对库外量做重生成（可选但强烈建议）
        conflict_vars_missing = [x for x in conflict_vars if str(x) not in meta_index]
        if conflict_vars_missing:
            got2 = _llm_generate_items(
                client,
                required=conflict_vars_missing,
                targets_norm=targets_norm,
                category=cat,
                formula_key=formula_key,
                var_ctx_map=var_ctx_map,
                temperature=0.1,
                retries=retries,
                prompt_template=_PROMPT_TEMPLATE,  # 仍然用库外模板
            )
            for k in conflict_vars_missing:
                if k in got2 and isinstance(got2[k], dict):
                    items[k] = got2[k]

    # 仅对库外量做 fallback 去重（避免改库内 name）
    missing_set = set(str(x) for x in missing_required)
    items_missing_view = {k: v for k, v in items.items() if str(k) in missing_set}
    _ensure_distinct_fallback(items_missing_view)
    for k, v in items_missing_view.items():
        items[k] = v

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

    # 生成 items（第四步已分流：库内只 value/context，库外全套）
    items = generate_known_input_items_from_payload(
        payload,
        category=cat,
        llm_cfg=llm_cfg,
        targets=list(forbidden_norm),
    )

    # ✅ 第五步新增：加载物理量库索引，用于补齐 name/unit/source
    meta_index = build_quantity_index()  # quantity_id -> QuantityMeta(name_zh, unit, source)

    variables: List[Dict[str, Any]] = []
    for rid, spec in items.items():
        qid = str(rid)

        # ✅ 最终双保险：绝不输出 targets
        if _norm_varname(qid) in forbidden_norm:
            continue

        meta = meta_index.get(qid)

        # value/context：始终来自 items（LLM）
        value = _coerce_value(spec.get("value"))
        context = str(spec.get("context", "") or "")

        if meta:
            # ✅ 库内量：name/unit/source 用库（按你给的 expert/thesis 规则）
            name = meta.name_zh or ""  # 库里没中文名则空
            unit = meta.unit  # 库里没有单位就 None
            source = meta.source  # "expert" 或 thesis 文件名去后缀
        else:
            # ✅ 库外量：name/unit 用 LLM 输出；source 标记大模型生成
            name = str(spec.get("name", "") or "")
            unit = _coerce_unit(spec.get("unit"))
            source = "大模型生成"

        variables.append(
            {
                "quantity_id": qid,
                "symbol": f"${qid}$",
                "name": name,
                "value": value,
                "unit": unit,
                "context": context,
                "source": source,
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
