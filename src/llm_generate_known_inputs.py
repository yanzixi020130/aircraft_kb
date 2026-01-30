#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate plausible known_inputs for formulas via LLM.

Given a payload containing selected formulas (each with an expr string), this
module will:
1) Collect all input variable identifiers (excluding the unknown targets).
2) Match them against quantities.yaml (including aliases) to obtain units.
3) Ask the configured LLM to propose reasonable values with units.
4) Fallback to simple defaults if the LLM output is missing/invalid.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import yaml

from llm_client import LLMClient, LLMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CATEGORY = "expert"

_GREEK_LATEX_TO_UNICODE = {
    r"\\Alpha": "Α", r"\\Beta": "Β", r"\\Gamma": "Γ", r"\\Delta": "Δ", r"\\Epsilon": "Ε", r"\\Zeta": "Ζ", r"\\Eta": "Η", r"\\Theta": "Θ", r"\\Iota": "Ι", r"\\Kappa": "Κ", r"\\Lambda": "Λ", r"\\Mu": "Μ", r"\\Nu": "Ν", r"\\Xi": "Ξ", r"\\Omicron": "Ο", r"\\Pi": "Π", r"\\Rho": "Ρ", r"\\Sigma": "Σ", r"\\Tau": "Τ", r"\\Upsilon": "Υ", r"\\Phi": "Φ", r"\\Chi": "Χ", r"\\Psi": "Ψ", r"\\Omega": "Ω",
    r"\\alpha": "α", r"\\beta": "β", r"\\gamma": "γ", r"\\delta": "δ", r"\\epsilon": "ε", r"\\zeta": "ζ", r"\\eta": "η", r"\\theta": "θ", r"\\iota": "ι", r"\\kappa": "κ", r"\\lambda": "λ", r"\\mu": "μ", r"\\nu": "ν", r"\\xi": "ξ", r"\\omicron": "ο", r"\\pi": "π", r"\\rho": "ρ", r"\\sigma": "σ", r"\\tau": "τ", r"\\upsilon": "υ", r"\\phi": "φ", r"\\chi": "χ", r"\\psi": "ψ", r"\\omega": "ω"
}

def filter_latex_unicode(s: str) -> str:
    # 替换希腊字母
    for latex, uni in _GREEK_LATEX_TO_UNICODE.items():
        s = re.sub(latex + r'(?![a-zA-Z])', uni, s)
    # 去除 \text{...}、\left、\right
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = s.replace(r'\left', '').replace(r'\right', '')
    return s

# Minimal list of math helper names to exclude when scanning identifiers.
MATH_FUNCS = {
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "sqrt",
    "exp",
    "abs",
    "pow",
}

# Detect strings that are pure numeric values (optionally in scientific notation)
_NUMERIC_ONLY_RE = re.compile(r"^[\s\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?\s*$")


def _load_quantities(category: str) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Load quantities under data/quantities/<category> from ALL yaml files.

    Merges every *.yaml inside the category folder (recursively). Later files
    override earlier ones for the same id.
    """

    cat_dir = DATA_DIR / "quantities" / category
    yaml_files = sorted(cat_dir.rglob("*.yaml")) if cat_dir.exists() else []
    if not yaml_files:
        raise FileNotFoundError(f"No yaml files found under category '{category}' at {cat_dir}")

    quantities: Dict[str, Dict] = {}
    aliases: Dict[str, str] = {}

    for yf in yaml_files:
        try:
            with yf.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            continue

        for item in data.get("quantities", []) or []:
            qid = item.get("id")
            if not qid:
                continue
            quantities[qid] = item
            for alias in item.get("aliases", []) or []:
                if alias not in aliases:
                    aliases[alias] = qid

    return quantities, aliases


def _extract_identifiers(expr: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return tokens


def _collect_required_inputs(selected_formulas: Dict[str, Dict]) -> Set[str]:
    unknowns = set(selected_formulas.keys())
    required: Set[str] = set()
    for key, formula in selected_formulas.items():
        expr = str(formula.get("expr", ""))
        for tok in _extract_identifiers(expr):
            if tok in unknowns:
                continue
            if tok in MATH_FUNCS:
                continue
            if tok.replace("_", "").lower() in {"e", "pi"}:
                continue
            # Skip pure numbers (already filtered by regex), retain others
            required.add(tok)
    return required


def _ensure_units(
    known_inputs: Dict[str, str],
    quantities: Dict[str, Dict],
    unit_overrides: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Attach canonical units to numeric-only values when unit is known.

    If a value already contains unit text or the quantity is dimensionless (unit "1"),
    it is returned as-is. This safeguards cases where the LLM omits units.
    """

    hydrated: Dict[str, str] = {}
    unit_overrides = unit_overrides or {}
    for rid, raw_val in known_inputs.items():
        text = str(raw_val).strip()
        if rid in quantities:
            unit = quantities[rid].get("unit", "1")
            if unit not in (None, "", "1") and _NUMERIC_ONLY_RE.match(text):
                hydrated[rid] = f"{text} {unit}"
                continue
        elif rid in unit_overrides:
            unit = unit_overrides[rid]
            if unit not in (None, "", "1") and _NUMERIC_ONLY_RE.match(text):
                hydrated[rid] = f"{text} {unit}"
                continue
        hydrated[rid] = text
    return hydrated


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]




# ==================== PROMPT TEMPLATE ====================
_PROMPT_TEMPLATE = """
你是航空航天/飞行器设计领域的工程师，负责为公式中的输入物理量生成合理的典型值（带单位）。

[任务要求]
- 必须为每个 required 变量生成合理的典型值，不能遗漏。
- 结果必须包含单位，单位优先使用 quantities.yaml 中定义的单位，否则选择合适的国际单位制（SI）。
- 希腊字母等符号请直接用Unicode字符（如Λ、τ、φ、Δ、α等），不要用LaTeX转义。
- 不要使用 \\text、\\left、\\right 等LaTeX修饰符。
- 数值使用阿拉伯数字，尽量 2~3 位有效数字，保持各物理量量纲一致且数量级合理。
- 只输出 JSON，不要输出解释文字。

[输入物理量列表]
{required_block}

[输出格式]
```
{{
  "known_inputs": {{
    "变量id": "<value> <unit>",
    ...
  }}
}}
```

[上下文]
- category: {category}
- formulaKey: {formula_key}
""".strip()

# 用于生成 required_block 的辅助函数
def _build_required_block(required: list[str], quantities: dict) -> str:
    lines = []
    for rid in required:
        if rid in quantities:
            meta = quantities[rid]
            lines.append(f"- {rid}: {meta.get('name_zh', '')}，单位: {meta.get('unit', '')}")
        else:
            lines.append(f"- {rid}: 单位请结合常识选择合适的国际单位制")
    return "\n".join(lines)




def _build_metadata_prompt(
    *,
    required: List[str],
    quantities: Dict[str, Dict] | None = None,
) -> str:
    lines = []
    lines.append("You are generating metadata for physical variables.")
    lines.append("For each variable id, return:")
    lines.append("- name_zh: Chinese name (must include Chinese characters).")
    lines.append("- context: Chinese definition in the exact format '<name_zh>：<definition>'.")
    lines.append("- symbol_latex: LaTeX symbol, e.g. \\alpha, \\mu, V_{tip}.")
    lines.append("- unit: SI unit or '1' for dimensionless.")
    lines.append("Do NOT use placeholders such as '变量', '未知', '未识别', '未提供', '待补充'.")
    lines.append("If a name_zh is provided below, you MUST use it exactly in both name_zh and context prefix.")
    lines.append("Return pure JSON only.")
    lines.append('Format: {"meta": {"rid": {"name_zh": "...", "context": "...", "symbol_latex": "...", "unit": "..."}}}')
    lines.append("")
    quantities = quantities or {}
    for rid in required:
        name_zh = ""
        if rid in quantities:
            name_zh = str(quantities[rid].get("name_zh") or "").strip()
        if name_zh:
            lines.append(f"- {rid}: name_zh={name_zh} (must use exactly)")
        else:
            lines.append(f"- {rid}: name_zh unknown")
    return "\n".join(lines)


def _parse_metadata_from_llm(text: str) -> Dict[str, Dict[str, str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    obj = json.loads(text[start : end + 1])
    if isinstance(obj, dict) and isinstance(obj.get("meta"), dict):
        out: Dict[str, Dict[str, str]] = {}
        for k, v in obj["meta"].items():
            if isinstance(v, dict):
                out[k] = {kk: str(vv) for kk, vv in v.items()}
        return out
    if isinstance(obj, dict):
        out: Dict[str, Dict[str, str]] = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = {kk: str(vv) for kk, vv in v.items()}
        return out
    raise ValueError("Invalid metadata JSON")


def _infer_symbol_latex_from_id(rid: str) -> str:
    s = str(rid)
    greek_map = {
        "alpha": r"\\alpha",
        "beta": r"\\beta",
        "gamma": r"\\gamma",
        "delta": r"\\delta",
        "epsilon": r"\\epsilon",
        "zeta": r"\\zeta",
        "eta": r"\\eta",
        "theta": r"\\theta",
        "iota": r"\\iota",
        "kappa": r"\\kappa",
        "lambda": r"\\lambda",
        "mu": r"\\mu",
        "nu": r"\\nu",
        "xi": r"\\xi",
        "omicron": "o",
        "pi": r"\\pi",
        "rho": r"\\rho",
        "sigma": r"\\sigma",
        "tau": r"\\tau",
        "upsilon": r"\\upsilon",
        "phi": r"\\phi",
        "chi": r"\\chi",
        "psi": r"\\psi",
        "omega": r"\\omega",
    }
    lower = s.lower()
    for name, latex in greek_map.items():
        if lower.startswith(name):
            rest = s[len(name):]
            if rest:
                rest = rest.lstrip("_")
                return f"{latex}_{{{rest}}}"
            return latex
    return s



def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


_BAD_TEXT_SNIPPETS = (
    "变量",
    "未识别",
    "未知",
    "未提供",
    "未给出",
    "无定义",
    "无描述",
    "不确定",
    "无法",
    "待补充",
    "未说明",
)


def _is_valid_name_zh(text: str) -> bool:
    s = str(text or "").strip()
    if not _has_chinese(s):
        return False
    return not any(bad in s for bad in _BAD_TEXT_SNIPPETS)


def _is_valid_context(name_zh: str, context: str) -> bool:
    name = str(name_zh or "").strip()
    s = str(context or "").strip()
    if not name or not _has_chinese(s):
        return False
    if any(bad in s for bad in _BAD_TEXT_SNIPPETS):
        return False
    if not re.match(rf"^{re.escape(name)}\s*[:：]", s):
        return False
    remainder = re.sub(rf"^{re.escape(name)}\s*[:：]\s*", "", s)
    return _has_chinese(remainder) and len(remainder) >= 4


def _parse_units_from_llm(text: str) -> Dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    obj = json.loads(text[start : end + 1])
    if isinstance(obj, dict) and isinstance(obj.get("units"), dict):
        return {k: str(v) for k, v in obj["units"].items()}
    if isinstance(obj, dict):
        return {k: str(v) for k, v in obj.items()}
    raise ValueError("Invalid units JSON")


def _parse_known_inputs_from_llm(text: str) -> Dict[str, str]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON is not an object")
        def _filter_dict(d):
            return {k: filter_latex_unicode(str(v)) for k, v in d.items()}
        if "known_inputs" in obj and isinstance(obj["known_inputs"], dict):
            return _filter_dict(obj["known_inputs"])
        return _filter_dict(obj)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Failed to parse LLM output: {exc}")


def _fallback_values(required: List[str], quantities: Dict[str, Dict]) -> Dict[str, str]:
    defaults = {
        # Aerodynamic basics
        "rhoc": "1.225 kg/m^3",
        "Vc": "80 m/s",
        "Swing": "20 m^2",
        "CL": "0.5",
        "W0": "3000 kg",
        "g": "9.81 m/s^2",
        "b": "10.2 m",
        "mu": "1.8e-5 N·s/m",
        "thick": "0.3 m",
        "c": "1.5 m",

        # Plane design block
        "Lfus": "6.5 m",
        "Lambda_quarter_chord": "12 deg",
        "i": "3 deg",

        # Overall parameter extraction
        "AR": "7.0",
        "S": "6.5 m^2",
        "W_to": "4200 kg",
        "W_empty": "2084 kg",
        "eta_battery": "0.284",
        "TWR": "1.1",
        "P": "1000 kW",
        "R_rotor": "5.0 m",
        "sigma": "0.085",
        "c_rotor": "0.213 m",
        "Vtip_heli": "230 m/s",
        "Vtip_fixed": "180 m/s",

        # Requirement analysis
        "Vmax": "600 km/h",
        "Vmin_heli": "185 km/h",
        "Vmin_plane": "220 km/h",
        "R": "750 km",
        "T_endurance": "2.5 h",
        "H_service": "7600 m",
        "ROC_max": "8.5 m/s",
        "T_maintenance": "600 h",
        "m_payload": "600 kg",
        "C_acquisition": "8 M$",
        "C_operational": "1200 $/h",
        "Vto_heli": "95 km/h",
        "Vto_plane": "130 km/h",
        "Vld_heli": "85 km/h",
        "Vld_plane": "120 km/h",
        "n_max": "3.5",
        "n_min": "-1.0",
        "S_to": "450 m",
        "S_ld": "400 m",
        "zeta_lat": "0.7",
        "tau_long": "1.5 s",
    }
    out: Dict[str, str] = {}
    for rid in required:
        if rid in defaults:
            out[rid] = defaults[rid]
        elif rid in quantities:
            unit = quantities[rid].get("unit", "1")
            out[rid] = f"1 {unit}" if unit != "1" else "1"
        else:
            out[rid] = "1"
    return out


def generate_known_inputs_from_payload(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
) -> Dict[str, str]:
    """Main entry: parse payload, call LLM, return known_inputs dict."""
    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    selected = params.get("selectedFormulas", {}) if isinstance(params, dict) else {}
    formula_key = params.get("formulaKey") if isinstance(params, dict) else None

    # Decide category; default to expert unless explicitly given
    cat = (category or DEFAULT_CATEGORY).strip()

    quantities, aliases = _load_quantities(cat)

    # Normalize selected formulas dict values to ensure we have expr
    selected_norm: Dict[str, Dict] = {}
    for k, v in selected.items():
        if isinstance(v, dict):
            selected_norm[str(k)] = v
    if not selected_norm:
        raise ValueError("selectedFormulas is empty or invalid")

    required_raw = _collect_required_inputs(selected_norm)

    # Resolve aliases to canonical ids
    resolved: Set[str] = set()
    for tok in required_raw:
        if tok in quantities:
            resolved.add(tok)
        elif tok in aliases:
            resolved.add(aliases[tok])
        else:
            resolved.add(tok)

    required = sorted(resolved)

    llm_config = llm_cfg or LLMConfig()
    client = LLMClient(llm_config)

    batch_size = int(os.getenv("LLM_KNOWN_INPUTS_BATCH_SIZE", "8"))
    retries = int(os.getenv("LLM_KNOWN_INPUTS_RETRIES", "2"))


    # 直接合并为一条大提示词，仿照llm_fill_missing_quantities.py
    known_inputs: Dict[str, str] = {}
    for batch in _chunk_list(required, batch_size):
        required_block = _build_required_block(batch, quantities)
        prompt = _PROMPT_TEMPLATE.format(
            required_block=required_block,
            category=cat,
            formula_key=formula_key,
        )
        for _ in range(retries + 1):
            try:
                llm_text = client.completion_text(user_prompt=prompt, temperature=0.2)
                known_inputs.update(_parse_known_inputs_from_llm(llm_text))
                break
            except Exception:
                continue

    # Fallback for any missing items
    missing = [rid for rid in required if rid not in known_inputs or not str(known_inputs[rid]).strip()]
    if missing:
        known_inputs.update(_fallback_values(missing, quantities))

    # Ensure all required keys present
    for rid in required:
        if rid not in known_inputs or not str(known_inputs[rid]).strip():
            known_inputs[rid] = _fallback_values([rid], quantities)[rid]

    # Attach canonical units if the LLM returned numeric-only values
    known_inputs = _ensure_units(known_inputs, quantities)

    return known_inputs


__all__ = ["generate_known_inputs_from_payload"]
def _split_value_unit(text: str) -> Tuple[str, str | None]:
    """Split a value string into (value, unit).

    Heuristics:
    - If there's at least one space, take the last token as unit.
    - Otherwise, treat as pure numeric and unit=None.
    """
    s = str(text).strip()
    if not s:
        return "", None
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return s, None


def _format_source_label(source: str, source_file: str | None = None) -> str:
    raw = str(source or "")
    norm = raw.strip().lower()
    if norm == "expert":
        return "专家知识"
    if norm == "llm":
        return "大模型生成"
    if norm == "thesis":
        return source_file or raw or "thesis"
    return raw


def generate_known_inputs_response(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
) -> Dict[str, Any]:
    """API-style response builder that returns variables with quantity_id."""

    known_inputs = generate_known_inputs_from_payload(payload, category=category, llm_cfg=llm_cfg)

    cat = (category or DEFAULT_CATEGORY).strip()
    quantities, aliases = _load_quantities(cat)

    meta_map: Dict[str, Dict[str, str]] = {}
    if known_inputs:
        batch_size = int(os.getenv("LLM_KNOWN_INPUTS_BATCH_SIZE", "8"))
        retries = int(os.getenv("LLM_KNOWN_INPUTS_RETRIES", "2"))
        client = LLMClient(llm_cfg or LLMConfig())
        request_ids: List[str] = []
        for rid in known_inputs.keys():
            qid = aliases.get(rid, rid)
            request_ids.append(qid)
        seen_ids: Set[str] = set()
        request_ids = [x for x in request_ids if not (x in seen_ids or seen_ids.add(x))]
        for batch in _chunk_list(request_ids, batch_size):
            prompt_meta = _build_metadata_prompt(required=batch, quantities=quantities)
            for _ in range(retries + 1):
                try:
                    meta_text = client.completion_text(user_prompt=prompt_meta, temperature=0.1)
                    meta_map.update(_parse_metadata_from_llm(meta_text))
                    break
                except Exception:
                    continue

        # Retry only the ids that still lack valid name/context (especially for unknown quantities).
        missing_meta: List[str] = []
        for rid in known_inputs.keys():
            qid = aliases.get(rid, rid)
            spec = quantities.get(qid, {})
            name_zh = spec.get("name_zh") or meta_map.get(qid, {}).get("name_zh") or ""
            if not _is_valid_name_zh(name_zh):
                missing_meta.append(qid)
                continue
            context = meta_map.get(qid, {}).get("context", "")
            if not _is_valid_context(name_zh, context):
                missing_meta.append(qid)

        seen_ids = set()
        missing_meta = [x for x in missing_meta if not (x in seen_ids or seen_ids.add(x))]
        for batch in _chunk_list(missing_meta, batch_size):
            if not batch:
                continue
            prompt_meta = _build_metadata_prompt(required=batch, quantities=quantities)
            for _ in range(retries + 1):
                try:
                    meta_text = client.completion_text(user_prompt=prompt_meta, temperature=0.1)
                    meta_map.update(_parse_metadata_from_llm(meta_text))
                    break
                except Exception:
                    continue

    variables: List[Dict[str, Any]] = []
    for rid, raw in known_inputs.items():
        qid = aliases.get(rid, rid)
        spec = quantities.get(qid, {})
        meta = meta_map.get(qid, {})

        name_zh = spec.get("name_zh") or meta.get("name_zh") or ""
        if not _is_valid_name_zh(name_zh):
            name_zh = ""

        symbol_latex = spec.get("symbol_latex") or spec.get("symbol") or meta.get("symbol_latex") or _infer_symbol_latex_from_id(qid)
        symbol_out = f"${symbol_latex}$" if not str(symbol_latex).startswith("$") else str(symbol_latex)

        val_str, unit_str = _split_value_unit(raw)
        canonical_unit = spec.get("unit", meta.get("unit", "1"))
        if canonical_unit == "1":
            unit_str = None

        context = meta.get("context") if _is_valid_context(name_zh, meta.get("context", "")) else ""

        variables.append(
            {
                "quantity_id": qid,
                "symbol": symbol_out,
                "name": name_zh,
                "value": val_str,
                "unit": unit_str,
                "context": context,
                "source": _format_source_label((spec and cat) or "llm"),
            }
        )

    return {
        "variables": variables,
        "status": "ok",
        "category": cat,
    }


__all__ = [
    "generate_known_inputs_from_payload",
    "generate_known_inputs_response",
]
