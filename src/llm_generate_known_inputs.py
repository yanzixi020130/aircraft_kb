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
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from llm_client import LLMClient, LLMConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CATEGORY = "expert"

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


def _ensure_units(known_inputs: Dict[str, str], quantities: Dict[str, Dict]) -> Dict[str, str]:
    """Attach canonical units to numeric-only values when unit is known.

    If a value already contains unit text or the quantity is dimensionless (unit "1"),
    it is returned as-is. This safeguards cases where the LLM omits units.
    """

    hydrated: Dict[str, str] = {}
    for rid, raw_val in known_inputs.items():
        text = str(raw_val).strip()
        if rid in quantities:
            unit = quantities[rid].get("unit", "1")
            if unit not in (None, "", "1") and _NUMERIC_ONLY_RE.match(text):
                hydrated[rid] = f"{text} {unit}"
                continue
        hydrated[rid] = text
    return hydrated


def _build_prompt(
    *,
    required: List[str],
    quantities: Dict[str, Dict],
    category: str,
    formula_key: str | None,
) -> str:
    lines = []
    lines.append("你是一位航空航天领域的工程师，请参考 data/pdf 中的相关设计文献，为下列物理量生成合理的典型值（带单位）。")
    lines.append("请用 JSON 返回，不要额外说明。")
    lines.append("")
    lines.append("【物理量列表（含单位）】")
    for rid in required:
        if rid in quantities:
            meta = quantities[rid]
            lines.append(f"- {rid}: {meta.get('name_zh', '')}，单位: {meta.get('unit', '')}")
        else:
            lines.append(f"- {rid}: 单位请结合常识选择合适的国际单位制")
    lines.append("")
    lines.append("【输出格式】")
    lines.append('{"known_inputs": {"rid": "<value> <unit>", ...}}')
    lines.append("要求：数值使用阿拉伯数字，尽量 2~3 位有效数字，保持各物理量量纲一致且数量级合理。")
    lines.append(f"类别: {category}, formulaKey: {formula_key}")
    return "\n".join(lines)


def _parse_known_inputs_from_llm(text: str) -> Dict[str, str]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("Top-level JSON is not an object")
        if "known_inputs" in obj and isinstance(obj["known_inputs"], dict):
            # normalize to str values
            return {k: str(v) for k, v in obj["known_inputs"].items()}
        # Fallback: maybe the dict itself is known_inputs
        return {k: str(v) for k, v in obj.items()}
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

    # Build prompt
    prompt = _build_prompt(required=required, quantities=quantities, category=cat, formula_key=formula_key)

    llm_config = llm_cfg or LLMConfig()
    client = LLMClient(llm_config)

    try:
        llm_text = client.completion_text(user_prompt=prompt, temperature=0.2)
        known_inputs = _parse_known_inputs_from_llm(llm_text)
    except Exception:
        known_inputs = _fallback_values(required, quantities)

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


def generate_known_inputs_response(
    payload: Dict,
    *,
    category: str | None = None,
    llm_cfg: LLMConfig | None = None,
) -> Dict[str, Any]:
    """API-style response builder that returns variables with quantity_id.

    Input payload example (abridged):
    {
      "type": "checkpoint.reply",
      "params": {
        "formulaKey": "plane_design",
        "selectedFormulas": {
          "q": {"expr": "L = q * Swing * CL", ...},
          "Wload": {"expr": "Wload = kload * W0", ...}
        }
      }
    }

    Output shape:
    {
      "variables": [
        {"quantity_id": "CL", "symbol": "$C_{L}$", "name": "升力系数", "value": "0.5", "unit": null, "context": "变量：升力系数", "source": "expert"},
        ...
      ],
      "status": "ok",
      "category": "expert"
    }
    """

    # Generate known_inputs first
    known_inputs = generate_known_inputs_from_payload(payload, category=category, llm_cfg=llm_cfg)

    # Load quantities + aliases to resolve metadata
    cat = (category or DEFAULT_CATEGORY).strip()
    quantities, aliases = _load_quantities(cat)

    variables: List[Dict[str, Any]] = []
    for rid, raw in known_inputs.items():
        qid = rid
        # Map alias back to canonical id if needed
        if qid in aliases:
            qid = aliases[qid]

        spec = quantities.get(qid, {})
        name_zh = spec.get("name_zh") or qid
        # Prefer LaTeX symbol if available; else fallback to id itself
        symbol_latex = spec.get("symbol_latex") or spec.get("symbol") or qid
        symbol_out = f"${symbol_latex}$" if not str(symbol_latex).startswith("$") else str(symbol_latex)

        val_str, unit_str = _split_value_unit(raw)
        # If canonical unit is dimensionless ('1'), suppress unit in output
        canonical_unit = spec.get("unit", "1")
        if canonical_unit == "1":
            unit_str = None

        variables.append(
            {
                "quantity_id": qid,
                "symbol": symbol_out,
                "name": name_zh,
                "value": val_str,
                "unit": unit_str,
                "context": f"变量：{name_zh}",
                # Default to category as source; if not found in YAML, mark as llm
                "source": (spec and cat) or "llm",
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
