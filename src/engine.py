from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

from pathlib import Path
import hashlib
from llm_generate_known_inputs import _format_source_label

import yaml
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

try:
    import pint
    _HAS_PINT = True
except Exception:
    _HAS_PINT = False


# =========================
# Pint UnitRegistry factory (singleton)
def _pint_ureg():
    if not _HAS_PINT:
        raise ImportError("pint is not installed or failed to import.")
    if not hasattr(_pint_ureg, "_instance"):
        _pint_ureg._instance = pint.UnitRegistry()
    return _pint_ureg._instance
# Global settings
# =========================
KB_ROOT = r"/data/se42/extraction_yjx/data"
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_RESERVED_SYMBOL_MAP = {"lambda": "lambda_"}

_QUANTITIES_DIR = os.path.join(KB_ROOT, "quantities")
_FORMULAS_DIR = os.path.join(KB_ROOT, "formulas")
_RAW_DIR = os.path.join(KB_ROOT, "raw")
_MD_DIR = os.path.join(KB_ROOT, "md")


class LatexDict(dict):
    def __repr__(self) -> str:
        return _latex_safe_repr(self)

    __str__ = __repr__


def _latex_safe_repr(value: Any) -> str:
    if isinstance(value, LatexDict):
        value = dict(value)
    if isinstance(value, dict):
        items = [f"{_latex_safe_repr(k)}: {_latex_safe_repr(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_latex_safe_repr(v) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_latex_safe_repr(v) for v in value)
        return "(" + inner + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    return repr(value)


def _wrap_latex(value: Any) -> Any:
    if isinstance(value, LatexDict):
        return value
    if isinstance(value, dict):
        return LatexDict({k: _wrap_latex(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap_latex(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap_latex(v) for v in value)
    return value


# =========================
# Data models
# =========================
@dataclass(frozen=True)
class QuantitySpec:
    id: str
    symbol: str
    symbol_latex: str
    name_zh: str
    unit: str


@dataclass(frozen=True)
class FormulaSpec:
    id: str
    expr: str  # e.g. "L = q * S * CL"
    name_zh: str
    category: str
    extractid: List[str]
    source: str = ""  # expert|thesis|llm (llm is filled at response time)


# =========================
# IO: load YAML
# =========================
def _iter_yaml_files(path_or_dir: str) -> List[Path]:
    p = Path(path_or_dir)
    if p.is_file():
        return [p]
    if not p.exists() or not p.is_dir():
        return []
    # Recursively discover YAML files (support nested folders like expert/, thesis/)
    return sorted([f for f in p.rglob("*.yaml") if f.is_file()])


def _infer_base_from_filename(path: Path, *, suffix: str) -> str:
    stem = path.stem
    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
        return base or stem
    return stem


def _ensure_extractid(value, fallback: str) -> List[str]:
    """Normalize extractid field (accept legacy extractids) to a non-empty list of strings."""
    ids: List[str] = []
    if isinstance(value, str) and value.strip():
        ids = [value.strip()]
    elif isinstance(value, list):
        ids = [str(v).strip() for v in value if str(v).strip()]
    if not ids and fallback:
        ids = [fallback]
    return ids


def _infer_source_from_yaml_path(yaml_path: Path, *, formulas_root: Path) -> str:
    """Infer formula source label from YAML file path.

    Expected directory structure:
      data/formulas/expert/*.yaml -> expert
      data/formulas/thesis/*.yaml -> thesis

    If no known bucket is present, default to 'thesis'.
    """
    try:
        rel = yaml_path.resolve().relative_to(formulas_root.resolve())
        parts = [p.lower() for p in rel.parts]
    except Exception:
        parts = [p.lower() for p in yaml_path.parts]

    if "expert" in parts:
        return "expert"
    if "thesis" in parts:
        return "thesis"
    return "thesis"


def load_all_quantities(dir_path: str = _QUANTITIES_DIR) -> Dict[str, QuantitySpec]:
    """Load and merge quantities from all YAML files under a directory."""
    qmap: Dict[str, QuantitySpec] = {}
    for yf in _iter_yaml_files(dir_path):
        try:
            obj = yaml.safe_load(open(yf, "r", encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict) or "quantities" not in obj:
            continue
        for item in obj.get("quantities") or []:
            if not isinstance(item, dict) or "id" not in item:
                continue
            q = QuantitySpec(
                id=item["id"],
                symbol=item.get("symbol", item["id"]),
                symbol_latex=item.get("symbol_latex", item.get("symbol", item["id"])),
                name_zh=item.get("name_zh", ""),
                unit=str(item.get("unit", "1")),
            )
            qmap[q.id] = q
    return qmap


def _normalize_quantity_key(value: str) -> str:
    s = str(value or "")
    s = s.replace("\\", "").replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    return s.strip().lower()


def _build_quantity_index(quantities: Dict[str, QuantitySpec]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for q in quantities.values():
        for raw in (q.name_zh, q.symbol, q.symbol_latex):
            key = _normalize_quantity_key(raw)
            if key:
                index.setdefault(key, []).append(q.id)
    return index


def _split_value_unit(raw: str) -> Tuple[str, str | None]:
    s = str(raw).strip()
    if not s:
        return "", None
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return s, None


def _build_temp_quantities(
    known_inputs: Dict[str, str],
    known_inputs_meta: Dict[str, Dict[str, Any]] | None = None,
    existing: Dict[str, QuantitySpec] | None = None,
) -> Dict[str, QuantitySpec]:
    """Build temporary QuantitySpec entries for known inputs missing from the library."""
    existing = existing or {}
    known_inputs_meta = known_inputs_meta or {}
    temp: Dict[str, QuantitySpec] = {}
    for raw_key, raw_val in (known_inputs or {}).items():
        key = str(raw_key).strip()
        if not key or key in existing:
            continue
        meta = known_inputs_meta.get(key, {}) if isinstance(known_inputs_meta, dict) else {}
        name_zh = str(meta.get("name_zh") or meta.get("name") or "")
        symbol = str(meta.get("symbol") or key)
        symbol_latex = str(meta.get("symbol_latex") or symbol)
        unit = str(meta.get("unit") or "")
        if not unit:
            _, unit_guess = _split_value_unit(raw_val)
            unit = unit_guess or "1"
        if not name_zh and re.search(r"[\u4e00-\u9fff]", key):
            name_zh = key
        if not name_zh:
            name_zh = key
        temp[key] = QuantitySpec(
            id=key,
            symbol=symbol,
            symbol_latex=symbol_latex,
            name_zh=name_zh,
            unit=unit,
        )
    return temp


def _resolve_quantity_inputs(
    inputs: List[str],
    quantities: Dict[str, QuantitySpec],
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, List[str]]]:
    """Resolve input quantity identifiers or Chinese names to quantity ids.

    Returns:
      resolved_ids: list of quantity ids (deduped, order-preserving)
      unresolved_inputs: list of raw inputs not resolved to a unique id
      resolved_map: mapping of raw input -> resolved id (only for resolved)
      ambiguous_map: mapping of raw input -> list of matching ids (name_zh duplicates)
    """
    name_index = _build_quantity_index(quantities)

    resolved_ids: List[str] = []
    unresolved_inputs: List[str] = []
    resolved_map: Dict[str, str] = {}
    ambiguous_map: Dict[str, List[str]] = {}
    seen: set[str] = set()

    for raw in inputs:
        raw_str = str(raw).strip()
        if not raw_str:
            continue
        if raw_str in quantities:
            if raw_str not in seen:
                resolved_ids.append(raw_str)
                seen.add(raw_str)
            resolved_map[raw_str] = raw_str
            continue
        key = _normalize_quantity_key(raw_str)
        matches = name_index.get(key, [])
        if len(matches) == 1:
            rid = matches[0]
            if rid not in seen:
                resolved_ids.append(rid)
                seen.add(rid)
            resolved_map[raw_str] = rid
        elif len(matches) > 1:
            ambiguous_map[raw_str] = matches
            unresolved_inputs.append(raw_str)
        else:
            unresolved_inputs.append(raw_str)

    return resolved_ids, unresolved_inputs, resolved_map, ambiguous_map


def load_all_formulas(dir_path: str = _FORMULAS_DIR) -> List[FormulaSpec]:
    """Load formulas from all YAML files under a directory.

    If a formula entry lacks category/extractid, infer both from filename:
    - '<base>_formulas.yaml' -> category=<base>, extractid=<base>
    """
    out: List[FormulaSpec] = []
    formulas_root = Path(dir_path) if Path(dir_path).is_dir() else Path(dir_path).parent
    for yf in _iter_yaml_files(dir_path):
        inferred_base = _infer_base_from_filename(yf, suffix="_formulas")
        inferred_source = _infer_source_from_yaml_path(yf, formulas_root=formulas_root)
        try:
            obj = yaml.safe_load(open(yf, "r", encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict) or "formulas" not in obj:
            continue
        for f in obj.get("formulas") or []:
            if not isinstance(f, dict) or "id" not in f or "expr" not in f:
                continue
            extractid = _ensure_extractid(f.get("extractid", f.get("extractids", "")), inferred_base)
            out.append(
                FormulaSpec(
                    id=f["id"],
                    expr=f["expr"],
                    name_zh=f.get("name_zh", ""),
                    category=f.get("category", "") or inferred_base,
                    extractid=extractid,
                    source=f.get("source", "") or inferred_source,
                )
            )
    return out


def load_quantities(path: str) -> Dict[str, QuantitySpec]:
    obj = yaml.safe_load(open(path, "r", encoding="utf-8"))
    qmap: Dict[str, QuantitySpec] = {}
    for item in obj["quantities"]:
        q = QuantitySpec(
            id=item["id"],
            symbol=item.get("symbol", item["id"]),
            symbol_latex=item.get("symbol_latex", item.get("symbol", item["id"])),
            name_zh=item.get("name_zh", ""),
            unit=str(item.get("unit", "1")),
        )
        qmap[q.id] = q
    return qmap


def load_formulas(path: str) -> List[FormulaSpec]:
    obj = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return [
        FormulaSpec(
            id=f["id"],
            expr=f["expr"],
            name_zh=f.get("name_zh", ""),
            category=f.get("category", ""),
            extractid=_ensure_extractid(f.get("extractid", f.get("extractids", "")), f.get("category", "")),
            source=f.get("source", ""),
        )
        for f in obj["formulas"]
    ]


# =========================
# SymPy helpers
# =========================
def _safe_sym_name(name: str) -> str:
    return _RESERVED_SYMBOL_MAP.get(name, name)


def _sanitize_expr(expr: str) -> str:
    out = expr
    for raw, safe in _RESERVED_SYMBOL_MAP.items():
        out = re.sub(rf"\b{re.escape(raw)}\b", safe, out)
    return out


def _symbol_latex_map(
    quantities: Dict[str, QuantitySpec],
    symtab: Dict[str, sp.Symbol],
) -> Dict[sp.Symbol, str]:
    return {symtab[qid]: quantities[qid].symbol_latex for qid in quantities if qid in symtab}


def expr_to_latex(
    *,
    expr: str,
    quantities: Dict[str, QuantitySpec],
) -> str:
    """
    Convert an expression string to LaTeX using symbol_latex, wrapped with $...$.
    """
    symtab = _mk_symbols(list(quantities.keys()))
    eq = _parse_equation(expr, symtab)
    symbol_names = _symbol_latex_map(quantities, symtab)
    return f"${sp.latex(eq, symbol_names=symbol_names)}$"


def expr_to_latex_from_quantities(
    *,
    expr: str,
) -> Dict[str, Any]:
    """
    Convert expr to LaTeX using quantities.yaml (with symbol_latex).
    """
    if not _iter_yaml_files(_QUANTITIES_DIR):
        return _wrap_latex({
            "status": "error",
            "error_code": "QUANTITIES_NOT_FOUND",
            "message": f"[quantities not found] {_QUANTITIES_DIR}",
        })
    quantities = load_all_quantities(_QUANTITIES_DIR)
    return _wrap_latex({
        "status": "ok",
        "latex": expr_to_latex(expr=expr, quantities=quantities),
    })


def _mk_symbols(quantity_ids: List[str]) -> Dict[str, sp.Symbol]:
    return {qid: sp.Symbol(_safe_sym_name(qid), real=True) for qid in quantity_ids}


def _parse_equation(expr: str, symtab: Dict[str, sp.Symbol]) -> sp.Eq:
    """
    Parse "a = b" into sympy.Eq, with support for '^' as power.
    """
    if "=" not in expr:
        raise ValueError(f"Equation must contain '=': {expr}")

    left, right = expr.split("=", 1)
    left = _sanitize_expr(left.strip().replace("^", "**"))
    right = _sanitize_expr(right.strip().replace("^", "**"))

    parse_symtab = {_safe_sym_name(k): v for k, v in symtab.items()}
    lhs = parse_expr(left, local_dict=parse_symtab, transformations=TRANSFORMS)
    rhs = parse_expr(right, local_dict=parse_symtab, transformations=TRANSFORMS)
    return sp.Eq(lhs, rhs)


def _symbols_in_equation(eq: sp.Eq, symtab: Dict[str, sp.Symbol]) -> List[str]:
    """
    Return variable ids (strings) that appear in eq and are in our symtab.
    """
    symset = set(symtab.values())
    sym_to_id = {sym: qid for qid, sym in symtab.items()}
    return [sym_to_id[s] for s in eq.free_symbols if s in symset]


def _parse_known_inputs_to_magnitudes(
    ureg: "pint.UnitRegistry",
    quantities: Dict[str, QuantitySpec],
    known_inputs: Dict[str, str],
) -> Dict[str, float]:
    """
    known_inputs: {"q":"5000 Pa", "CL":"0.5", ...}
    -> returns magnitudes in canonical units defined in quantities.yaml
    """
    known_vals: Dict[str, float] = {}
    for k, s in known_inputs.items():
        k = str(k).strip()
        if not k:
            continue
        qid = k
        canonical_unit = quantities.get(qid).unit if qid in quantities else "1"
        s = str(s).strip()

        # If user provided unit text, parse directly; else attach canonical unit.
        if any(ch.isalpha() for ch in s) or "/" in s or "^" in s:
            qv = ureg(s)
        else:
            qv = float(s) * ureg(canonical_unit)

        # Handle conversion: only convert if canonical_unit is not "1"
        if canonical_unit != "1":
            qv = qv.to(ureg(canonical_unit))
            known_vals[qid] = qv.magnitude
        else:
            # For dimensionless quantities, just use the magnitude
            if hasattr(qv, 'magnitude'):
                known_vals[qid] = qv.magnitude
            else:
                known_vals[qid] = float(qv)

    return known_vals


# =========================
# Solvers
# =========================
def solve_one(
    eq: sp.Eq,
    known: Dict[str, float],
    target: str,
    symtab: Dict[str, sp.Symbol],
) -> float:
    """
    Solve equation for 'target' given known numeric values (unitless magnitudes).
    """
    if target not in symtab:
        raise KeyError(f"Unknown target: {target}")

    subs = {symtab[k]: v for k, v in known.items() if k in symtab}
    eq_sub = sp.Eq(eq.lhs.subs(subs), eq.rhs.subs(subs))

    sols = sp.solve(eq_sub, symtab[target], dict=True)
    if not sols:
        raise ValueError(
            f"No solution for target={target}. known={list(known.keys())}, equation={eq}"
        )

    # MVP: pick the first solution
    val = sp.N(sols[0][symtab[target]])
    if val.has(sp.I):
        raise ValueError(f"Complex solution for {target}: {val}")
    return float(val)


def _residual(eq: sp.Eq, symtab: Dict[str, sp.Symbol], values: Dict[str, float]) -> float:
    subs = {symtab[k]: v for k, v in values.items() if k in symtab}
    return float(sp.N((eq.lhs - eq.rhs).subs(subs)))


# =========================
# Category loader + cache
# =========================
def _dir_fingerprint(dir_path: str) -> str:
    h = hashlib.sha256()
    for f in _iter_yaml_files(dir_path):
        try:
            st = f.stat()
        except Exception:
            continue
        h.update(f.name.encode("utf-8"))
        h.update(str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


def _dir_fingerprint_ext(dir_path: str, *, patterns: Tuple[str, ...]) -> str:
    """Fingerprint a directory based on matching files.

    Used to cache LLM results that depend on data/raw PDFs and/or data/md.
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return ""
    h = hashlib.sha256()
    for pat in patterns:
        for f in sorted([x for x in p.rglob(pat) if x.is_file()]):
            try:
                st = f.stat()
            except Exception:
                continue
            h.update(str(f.relative_to(p)).encode("utf-8", errors="ignore"))
            h.update(str(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


def _collect_llm_formula_context(
    *,
    category: str,
    extractid: str | None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    formulas_all = load_all_formulas(_FORMULAS_DIR)

    if extractid:
        formulas = [f for f in formulas_all if f.category == category and extractid in f.extractid]
    else:
        formulas = [f for f in formulas_all if f.category == category]

    if len(formulas) < 3:
        formulas = [f for f in formulas_all if f.category == category]

    allowed_formula_ids = [f.id for f in formulas if isinstance(f.id, str) and f.id]
    if len(allowed_formula_ids) < 5:
        seen_ids = set(allowed_formula_ids)
        for f in formulas_all:
            if f.category == category and isinstance(f.id, str) and f.id and f.id not in seen_ids:
                allowed_formula_ids.append(f.id)
                seen_ids.add(f.id)
                if len(allowed_formula_ids) >= 8:
                    break

    examples = [
        {
            "formula_id": f.id,
            "formula_name_zh": f.name_zh,
            "expr": f.expr,
            "source": f.source,
        }
        for f in formulas[:20]
    ]

    return allowed_formula_ids, examples


@lru_cache(maxsize=128)

def _llm_fill_missing_quantities_cached(
    category: str,
    extractid: str,
    missing_name_zh: Tuple[str, ...],
    raw_fp: str,
    md_fp: str,
) -> Dict[str, Dict[str, Any]]:
    """
    分批（每批最多5个）调用 generate_missing_quantities，最后合并所有结果。
    """
    _ = raw_fp, md_fp

    from llm_fill_missing_quantities import generate_missing_quantities

    batch_size = 5
    names = list(missing_name_zh)
    all_result: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        result = generate_missing_quantities(
            category=category,
            extractid=extractid,
            missing_quantity_name_zh=batch,
            raw_dir=_RAW_DIR,
            md_dir=_MD_DIR,
            key_mode="name_zh",
        )
        if isinstance(result, dict):
            all_result.update(result)
    return all_result


@lru_cache(maxsize=128)
def _llm_fill_missing_formulas_cached(
    category: str,
    extractid: str,
    quantity_specs_json: str,
    formulas_fp: str,
    raw_fp: str,
    md_fp: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    分批（每批最多5个物理量）调用 generate_missing_formulas，最后合并所有结果。
    """
    _ = formulas_fp, raw_fp, md_fp

    from llm_fill_missing_formulas import generate_missing_formulas

    quantities = json.loads(quantity_specs_json) if quantity_specs_json else []
    allowed_formula_ids, examples = _collect_llm_formula_context(
        category=category,
        extractid=extractid,
    )

    batch_size = 5
    all_result: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(0, len(quantities), batch_size):
        batch = quantities[i:i+batch_size]
        result = generate_missing_formulas(
            category=category,
            extractid=extractid,
            quantities=batch,
            existing_formula_examples=examples,
            raw_dir=_RAW_DIR,
            md_dir=_MD_DIR,
            allowed_formula_ids=allowed_formula_ids,
        )
        if isinstance(result, dict):
            for k, v in result.items():
                all_result[k] = v
    return all_result


@lru_cache(maxsize=32)
def _load_category_context_cached(
    category: str,
    quantities_fp: str,
    formulas_fp: str,
) -> Tuple[Dict[str, QuantitySpec], Dict[str, sp.Eq], Dict[str, sp.Symbol]]:
    """
    Load and parse quantities + formulas for a category, cached.
    Returns: (quantities_dict, eqs_dict, symtab)
    """
    _ = quantities_fp, formulas_fp
    if not _iter_yaml_files(_QUANTITIES_DIR):
        raise FileNotFoundError(f"[quantities not found] {_QUANTITIES_DIR}")
    if not _iter_yaml_files(_FORMULAS_DIR):
        raise FileNotFoundError(f"[formulas not found] {_FORMULAS_DIR}")

    quantities = load_all_quantities(_QUANTITIES_DIR)
    formulas_all = load_all_formulas(_FORMULAS_DIR)
    formulas = [f for f in formulas_all if f.category == category]
    if not formulas:
        available = sorted({f.category for f in formulas_all if f.category})
        raise KeyError(
            f"category '{category}' not found in formulas.yaml. "
            f"Available: {available}"
        )

    symtab = _mk_symbols(list(quantities.keys()))
    eqs = {
        f.id: {
            "eq": _parse_equation(f.expr, symtab),
            "name_zh": f.name_zh,
            "expr": f.expr,
            "category": f.category,
            "extractid": list(f.extractid),
        }
        for f in formulas
    }

    return quantities, eqs, symtab


def _load_category_context(category: str) -> Tuple[Dict[str, QuantitySpec], Dict[str, sp.Eq], Dict[str, sp.Symbol]]:
    """Wrapper to auto-bust cache when YAML files change."""
    return _load_category_context_cached(
        category,
        _dir_fingerprint(_QUANTITIES_DIR),
        _dir_fingerprint(_FORMULAS_DIR),
    )


# =========================
# Main API 1: single-step solve
# =========================
def solve_with_units(
    *,
    category: str,
    formula_id: str,
    known_inputs: Dict[str, str],
    target: str,
) -> Dict[str, Any]:
    """
    Single equation solve. Auto-loads YAML by category.
    Return: (value, unit, residual)
    """
    quantities, eqs, symtab = _load_category_context(category)

    if formula_id not in eqs:
        raise KeyError(
            f"formula_id '{formula_id}' not found in category '{category}'. "
            f"Available: {sorted(eqs.keys())}"
        )
    if target not in quantities:
        raise KeyError(
            f"target '{target}' not found in quantities of category '{category}'. "
            f"Available: {sorted(quantities.keys())}"
        )

    ureg = _pint_ureg()
    known_vals = _parse_known_inputs_to_magnitudes(ureg, quantities, known_inputs)

    eq = eqs[formula_id]["eq"]
    formula_name_zh = eqs[formula_id]["name_zh"]
    x = solve_one(eq, known_vals, target, symtab)

    # residual after substituting known + solved target
    vals2 = dict(known_vals)
    vals2[target] = x
    residual = _residual(eq, symtab, vals2)

    return {
        "value": x,
        "unit": quantities[target].unit,
        "residual": residual,
        "formula_id": formula_id,
        "formula_name_zh": formula_name_zh,
        "formula_category": eqs[formula_id]["category"],
        "formula_extractid": eqs[formula_id]["extractid"],
    }


# =========================
# Main API 2: forward-chain multi-step solve (路线A)
# =========================
def _forward_chain_all(
    *,
    category: str,
    known_inputs: Dict[str, str],
    max_steps: int = 50,
) -> Dict[str, Any]:
    """
    Forward-chaining once, try to derive as many variables as possible.
    Returns:
      {
        "known_vals": {var: float},           # magnitudes in canonical units
        "path": [ {formula_id, formula_name_zh, target, value, unit, residual}, ... ]
      }
    """
    quantities, eqs, symtab = _load_category_context(category)
    ureg = _pint_ureg()
    known_vals = _parse_known_inputs_to_magnitudes(ureg, quantities, known_inputs)

    path: List[Dict[str, Any]] = []

    steps = 0
    while steps < max_steps:
        progressed = False

        for fid, info in eqs.items():
            eq = info["eq"]
            vars_in_eq = _symbols_in_equation(eq, symtab)
            if not vars_in_eq:
                continue

            for target in vars_in_eq:
                if target in known_vals:
                    continue

                required = [v for v in vars_in_eq if v != target]
                if all(v in known_vals for v in required):
                    try:
                        x = solve_one(eq, known_vals, target, symtab)
                    except Exception:
                        continue

                    known_vals[target] = x
                    res = _residual(eq, symtab, {**{k: known_vals[k] for k in required}, target: x})
                    residual_ok = abs(res) < 1e-9

                    path.append(
                        {
                            "formula_id": fid,
                            "formula_name_zh": eqs[fid]["name_zh"],
                            "formula_expr": eqs[fid]["expr"],
                            "formula_latex": expr_to_latex(expr=eqs[fid]["expr"], quantities=quantities),
                            "formula_category": eqs[fid]["category"],
                            "formula_extractid": eqs[fid]["extractid"],
                            "target": target,
                            "value": x,
                            "unit": quantities[target].unit,
                            "residual": res,
                            "residual_ok": residual_ok,
                        }
                    )
                    progressed = True
                    break

            if progressed:
                break

        if not progressed:
            break

        steps += 1

    return {"known_vals": known_vals, "path": path}

def solve_goal_with_units(
    *,
    category: str,
    known_inputs: Dict[str, str],
    goal: str,
    max_steps: int = 50,
) -> Dict[str, Any]:
    quantities, eqs, symtab = _load_category_context(category)

    if goal not in quantities:
        raise KeyError(
            f"goal '{goal}' not found in quantities of category '{category}'. "
            f"Available: {sorted(quantities.keys())}"
        )

    out = _forward_chain_all(category=category, known_inputs=known_inputs, max_steps=max_steps)
    known_vals = out["known_vals"]
    path = out["path"]

    return {
        "goal": goal,
        "solved": goal in known_vals,
        "values": {k: (known_vals[k], quantities[k].unit) for k in known_vals},
        "path": path,
    }


def _candidate_formulas_for_target(eqs, symtab, target: str):
    """
    Return list of (formula_id, eq, vars_in_eq) that structurally can solve target.
    """
    candidates = []
    for fid, info in eqs.items():
        eq =info["eq"]
        vars_in_eq = _symbols_in_equation(eq, symtab)
        if target in vars_in_eq:
            try:
                # test if sympy can solve for target symbolically
                sp.solve(eq, symtab[target])
                candidates.append((fid, eq, vars_in_eq))
            except Exception:
                continue
    return candidates

def _build_missing_chain(
    target: str,
    known_vals: Dict[str, float],
    quantities,
    eqs,
    symtab,
):
    """
    Build reverse missing chain starting from target.

    Scheme 2 (graph-safe):
    - Prevent cycles by:
      1) visited_targets: each target expanded once
      2) used_formulas: avoid using the same formula repeatedly in the chain
      3) parent_target: avoid immediate backtracking (A -> ... -> A)
    """
    chain: List[Dict[str, Any]] = []
    visited_targets: set[str] = set()
    used_formulas: set[str] = set()

    def pick_best_formula(cur_target: str, parent_target: str | None):
        """
        Choose the best candidate formula for cur_target with loop-safe constraints.
        Scoring:
          - minimize missing count
          - tie-breaker: minimize required count
        Constraints:
          - do not reuse formulas already used in chain
          - avoid immediate backtracking: if parent_target is missing, prefer not selecting
        """
        candidates = _candidate_formulas_for_target(eqs, symtab, cur_target)
        if not candidates:
            return None  # no formula can solve this target

        scored = []
        for fid, eq, vars_in_eq in candidates:
            if fid in used_formulas:
                continue  # prevent same formula being used multiple times in chain

            required = sorted(set(vars_in_eq) - {cur_target})
            missing = sorted(v for v in required if v not in known_vals)

            # avoid immediate backtracking if possible
            backtrack_penalty = 1 if (parent_target is not None and parent_target in missing) else 0

            scored.append(
                (
                    len(missing),              # primary: fewer missing
                    backtrack_penalty,         # secondary: avoid backtracking
                    len(required),             # tertiary: fewer required
                    fid,
                    eq,
                    vars_in_eq,
                    required,
                    missing,
                )
            )

        if not scored:
            return None

        scored.sort()
        _, _, _, fid, eq, vars_in_eq, required, missing = scored[0]
        return fid, eq, vars_in_eq, required, missing

    def dfs(cur_target: str, parent_target: str | None = None):
        if cur_target in visited_targets:
            return
        visited_targets.add(cur_target)

        picked = pick_best_formula(cur_target, parent_target)
        if picked is None:
            return

        fid, eq, vars_in_eq, required, missing = picked
        used_formulas.add(fid)

        chain.append(
            {
                "target": cur_target,
                "formula_id": fid,
                "formula_name_zh": eqs[fid]["name_zh"],
                "formula_expr": eqs[fid]["expr"],
                "formula_latex": expr_to_latex(expr=eqs[fid]["expr"], quantities=quantities),
                "required": required,
                "missing": missing,
            }
        )

        # recurse on missing vars
        for m in missing:
            dfs(m, parent_target=cur_target)

    dfs(target, parent_target=None)
    return chain

def solve_target_auto(
    *,
    category: str,
    known_inputs: Dict[str, str],
    target: str,
    formula_overrides: Dict[str, str] | None = None,
):
    """
    Unified business-level solver:
    - No formula_id required
    - Automatically decides single-step or multi-step
    - Returns clear error if cannot solve
    """
    quantities, eqs, symtab = _load_category_context(category)

    # Rule 0: target exists?
    if target not in quantities:
        return _wrap_latex({
            "status": "error",
            "error_code": "TARGET_NOT_FOUND",
            "message": f"target '{target}' not found in quantities.yaml",
            "details": {
                "available": sorted(quantities.keys())
            },
        })

    # Parse known inputs to magnitudes
    ureg = _pint_ureg()
    known_vals = _parse_known_inputs_to_magnitudes(
        ureg, quantities, known_inputs
    )

    # Rule 0.5: if formula override specified for target, enforce it
    if formula_overrides and target in formula_overrides:
        fid = formula_overrides[target]
        if fid not in eqs:
            return _wrap_latex({
                "status": "error",
                "error_code": "FORMULA_ID_NOT_FOUND",
                "message": f"Formula '{fid}' not found in category '{category}'",
            })

        eq = eqs[fid]["eq"]
        vars_in_eq = _symbols_in_equation(eq, symtab)
        if target not in vars_in_eq:
            return _wrap_latex({
                "status": "error",
                "error_code": "FORMULA_CANNOT_SOLVE_TARGET",
                "message": f"Formula '{fid}' cannot solve target '{target}'",
            })

        required = sorted(set(vars_in_eq) - {target})
        missing = sorted(v for v in required if v not in known_vals)
        if missing:
            return _wrap_latex({
                "status": "error",
                "error_code": "INSUFFICIENT_INPUTS",
                "message": f"Formula '{fid}' cannot solve target '{target}' due to missing inputs",
                "details": {"missing": missing},
            })

        try:
            x = solve_one(eq, known_vals, target, symtab)
        except Exception as exc:
            return _wrap_latex({
                "status": "error",
                "error_code": "FORMULA_NO_SOLUTION",
                "message": f"Formula '{fid}' failed to solve target '{target}'",
                "details": {"error": str(exc)},
            })

        residual = _residual(eq, symtab, {**known_vals, target: x})
        residual_ok = abs(residual) < 1e-9
        return _wrap_latex({
            "status": "ok",
            "mode": "single-step",
            "target": target,
            "value": x,
            "unit": quantities[target].unit,
            "used_formula": fid,
            "used_formula_name_zh": eqs[fid]["name_zh"],
            "used_formula_expr": eqs[fid]["expr"],
            "used_formula_latex": expr_to_latex(expr=eqs[fid]["expr"], quantities=quantities),
            "used_formula_category": eqs[fid]["category"],
            "used_formula_extractid": eqs[fid]["extractid"],
            "residual": residual,
            "residual_ok": residual_ok,
        })

    # Rule 1: any formula can solve target?
    candidates = _candidate_formulas_for_target(eqs, symtab, target)
    if not candidates:
        return _wrap_latex({
            "status": "error",
            "error_code": "FORMULA_FOR_TARGET_NOT_FOUND",
            "message": f"No formula can solve target '{target}'",
        })

    # Rule 2: try single-step first
    for fid, eq, vars_in_eq in candidates:
        required = set(vars_in_eq) - {target}
        if all(v in known_vals for v in required):
            try:
                x = solve_one(eq, known_vals, target, symtab)
                residual = _residual(eq, symtab, {**known_vals, target: x})
                residual_ok = abs(residual) < 1e-9
                return _wrap_latex({
                    "status": "ok",
                    "mode": "single-step",
                    "target": target,
                    "value": x,
                    "unit": quantities[target].unit,
                    "used_formula": fid,
                    "used_formula_name_zh": eqs[fid]["name_zh"],
                    "used_formula_expr": eqs[fid]["expr"],
                    "used_formula_latex": expr_to_latex(expr=eqs[fid]["expr"], quantities=quantities),
                    "used_formula_category": eqs[fid]["category"],
                    "used_formula_extractid": eqs[fid]["extractid"],
                    "residual": residual,
                    "residual_ok": residual_ok,
                })
            except Exception:
                pass

    # Rule 3: multi-step forward chaining
    out = _forward_chain_all(category=category, known_inputs=known_inputs)
    known_vals2 = out["known_vals"]
    path = out["path"]

    if target in known_vals2:
        val = known_vals2[target]
        return _wrap_latex({
            "status": "ok",
            "mode": "multi-step",
            "target": target,
            "value": val,
            "unit": quantities[target].unit,
            "path": path,
        })

    # Rule 4: build reverse missing chain
    missing_chain = _build_missing_chain(
        target, known_vals2, quantities, eqs, symtab
    )

    return _wrap_latex({
        "status": "error",
        "error_code": "INSUFFICIENT_INPUTS",
        "message": f"Cannot solve target '{target}' due to missing inputs",
        "missing_chain": missing_chain,
    })

def solve_targets_auto(
    *,
    category: str,
    extractid: Optional[str] = None,
    known_inputs: List[Dict[str, Any]] | Dict[str, str],
    targets: List[str],
    formula_overrides: Dict[str, Any] | None = None,
    max_steps: int = 50,
    known_inputs_meta: Dict[str, Dict[str, Any]] | None = None,
    taskid: str | None = None,
    formula_key: str | None = None,
) -> Dict[str, Any]:
    _ = known_inputs_meta, taskid, formula_key

    def _extract_identifiers(expr: str) -> List[str]:
        return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)

    math_funcs = {"sin", "cos", "tan", "log", "ln", "sqrt", "exp", "abs", "pow"}

    def _fmt_quantity_value(v: Any) -> str:
        """Return numeric string without unit."""
        try:
            fval = float(v)
            return f"{fval:.4f}"
        except Exception:
            try:
                if hasattr(v, "magnitude"):
                    fval = float(v.magnitude)
                    return f"{fval:.4f}"
            except Exception:
                pass
        return str(v)

    def _parse_numeric_value(raw: Any) -> float | None:
        text = str(raw).strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            try:
                val = parse_expr(text, transformations=TRANSFORMS)
                return float(sp.N(val))
            except Exception:
                return None

    def _clean_key(raw: str) -> str:
        return str(raw).replace("\\", "").replace("{", "").replace("}", "").replace("$", "").strip()

    def _norm_key(raw: str) -> str:
        return _normalize_quantity_key(raw)

    def _norm_key_no_us(raw: str) -> str:
        return _normalize_quantity_key(raw).replace("_", "")

    exprs: Dict[str, str] = {}
    tokens: set[str] = set()
    formula_overrides = formula_overrides or {}
    for qid, override in formula_overrides.items():
        expr_str = ""
        if isinstance(override, dict):
            expr_str = str(override.get("expr") or "")
        else:
            expr_str = str(override or "")
        if expr_str:
            exprs[qid] = expr_str
            for tok in _extract_identifiers(expr_str):
                if tok in math_funcs:
                    continue
                if tok.lower() in {"e", "pi"}:
                    continue
                tokens.add(tok)
        tokens.add(str(qid))

    symtab = _mk_symbols(sorted(tokens))

    def _add_unique(mapping: Dict[str, str | None], key: str, token: str) -> None:
        if key in mapping and mapping[key] != token:
            mapping[key] = None
        else:
            mapping[key] = token

    token_map: Dict[str, str | None] = {}
    token_map_no_us: Dict[str, str | None] = {}
    for tok in tokens:
        _add_unique(token_map, _norm_key(tok), tok)
        _add_unique(token_map_no_us, _norm_key_no_us(tok), tok)

    known_vals: Dict[str, float] = {}
    if isinstance(known_inputs, dict):
        known_items: List[Dict[str, Any]] = [{"quantity_id": k, "value": v} for k, v in known_inputs.items()]
    else:
        known_items = []
        for item in known_inputs or []:
            if isinstance(item, dict):
                known_items.append(item)
                continue
            if hasattr(item, "model_dump"):
                known_items.append(item.model_dump())
                continue
            if hasattr(item, "dict"):
                known_items.append(item.dict())

    for item in known_items:
        raw_key = item.get("quantity_id") or item.get("symbol") or item.get("name")
        if raw_key is None:
            continue
        raw_key_str = str(raw_key).strip()
        if not raw_key_str:
            continue
        raw_val = item.get("value")
        val = _parse_numeric_value(raw_val)
        if val is None:
            continue
        clean_key = _clean_key(raw_key_str)
        token = None
        if clean_key in tokens:
            token = clean_key
        else:
            norm = _norm_key(raw_key_str)
            token = token_map.get(norm)
            if token is None:
                token = token_map_no_us.get(_norm_key_no_us(raw_key_str))
        if token:
            known_vals[token] = val

    eq_map: Dict[str, sp.Eq] = {}
    vars_map: Dict[str, List[str]] = {}
    target_map: Dict[str, str] = {} 
    for qid, expr_str in exprs.items():
        try:
            eq = _parse_equation(expr_str, symtab)
        except Exception:
            continue
        eq_map[qid] = eq
        vars_map[qid] = _symbols_in_equation(eq, symtab)

    solved: Dict[str, float] = {}      # display_key -> value（用于回填 quantity_value）
    pending = set(eq_map.keys())

    for _ in range(max_steps):
        progressed = False

        for display_key in list(pending):
            target_var = (target_map.get(display_key) or "").strip()

            # 没有 target 的，直接跳过（你也可以在这里兜底用 lhs 变量）
            if not target_var:
                pending.remove(display_key)
                continue

            vars_in_eq = vars_map.get(display_key, [])

            # target 必须出现在方程变量里，否则说明这个公式块和 target 不匹配
            if target_var not in vars_in_eq:
                pending.remove(display_key)
                continue

            required = set(vars_in_eq) - {target_var}
            if not required.issubset(known_vals.keys()):
                continue

            try:
                known_for_solve = {k: v for k, v in known_vals.items() if k != target_var}
                val = solve_one(eq_map[display_key], known_for_solve, target_var, symtab)
            except Exception:
                continue

            solved[display_key] = val             # 回填到展示 key
            known_vals[target_var] = val          # 链式：写回真实变量名
            pending.remove(display_key)
            progressed = True

        if not progressed:
            break

    target_keys: List[str] = []
    seen_keys: set[str] = set()
    for qid in targets or []:
        if qid not in seen_keys:
            target_keys.append(qid)
            seen_keys.add(qid)
    for qid in formula_overrides.keys():
        if qid not in seen_keys:
            target_keys.append(qid)
            seen_keys.add(qid)

    quantity_results: Dict[str, List[Dict[str, Any]]] = {qid: [] for qid in target_keys}
    for qid in target_keys:
        if qid not in formula_overrides:
            continue
        override = formula_overrides.get(qid)
        if isinstance(override, dict):
            out = dict(override)
        else:
            out = {"expr": str(override)}
        val = solved.get(qid)
        out["quantity_value"] = _fmt_quantity_value(val) if val is not None else ""
        quantity_results[qid] = [out]

    missing_formula_qids = [qid for qid, items in quantity_results.items() if not items]
    results = {category: {extractid: quantity_results}}

    return _wrap_latex({
        "status": "ok",
        "filters": {
            "category": category,
            "extractid": extractid,
            "extractid_matched": True,
        },
        "formulas_path": _FORMULAS_DIR,
        "quantity_ids": target_keys,
        "missing_quantity_ids": list(missing_formula_qids),
        "missing_quantity_inputs": [],
        "missing_formula_quantity_ids": list(missing_formula_qids),
        "quantity_id_resolved": {},
        "quantity_id_ambiguous": {},
        "results": results,
    })


# =========================
# Formula lookup by quantity
# =========================
def find_formulas_by_quantity(
    *,
    category: str | None = None,
    extractid: str | None = None,
    quantity_name_zh: str | None = None,
) -> Dict[str, Any]:
    """
    Find formulas in a category that contain the given quantity (by 中文名).
    """
    if not any([category, extractid, quantity_name_zh]):
        return _wrap_latex({
            "status": "error",
            "error_code": "MISSING_FILTER",
            "message": "At least one of category, extractid, quantity_name_zh must be provided",
        })

    if not _iter_yaml_files(_QUANTITIES_DIR):
        return _wrap_latex({
            "status": "error",
            "error_code": "QUANTITIES_NOT_FOUND",
            "message": f"[quantities not found] {_QUANTITIES_DIR}",
        })
    if not _iter_yaml_files(_FORMULAS_DIR):
        return _wrap_latex({
            "status": "error",
            "error_code": "FORMULAS_NOT_FOUND",
            "message": f"[formulas not found] {_FORMULAS_DIR}",
        })

    quantities = load_all_quantities(_QUANTITIES_DIR)
    # 支持通过 quantity_name_zh 查找 id
    quantity_id = None
    if quantity_name_zh is not None:
        for qid, qspec in quantities.items():
            if qspec.get("name_zh") == quantity_name_zh:
                quantity_id = qid
                break
        if quantity_id is None:
            return _wrap_latex({
                "status": "error",
                "error_code": "QUANTITY_NOT_FOUND",
                "message": f"quantity_name_zh '{quantity_name_zh}' not found in quantities.yaml",
                "details": {"available": [q.get("name_zh") for q in quantities.values()]},
            })

    formulas_all = load_all_formulas(_FORMULAS_DIR)
    formulas = formulas_all
    if category is not None:
        formulas = [f for f in formulas if f.category == category]
        if not formulas:
            return _wrap_latex({
                "status": "ok",
                "category": category,
                "extractid": extractid,
                "formulas_path": _FORMULAS_DIR,
                "quantity": quantity_name_zh,
                "categories": {},
                "message": f"category '{category}' not found in formulas.yaml",
            })
    if extractid is not None:
        formulas_before_extractid = formulas
        formulas = [f for f in formulas if extractid in f.extractid]
        extractid_matched = bool(formulas)
        if not formulas:
            # Fallback: search in the overall formulas for the current category.
            formulas = formulas_before_extractid
    else:
        extractid_matched = True
    symtab = _mk_symbols(list(quantities.keys()))

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for f in formulas:
        try:
            eq = _parse_equation(f.expr, symtab)
        except Exception:
            continue
        vars_in_eq = _symbols_in_equation(eq, symtab)
        if quantity_id is not None and quantity_id not in vars_in_eq:
            continue

        cat_key = f.category or ""
        ext_key = ",".join(f.extractid) if f.extractid else ""
        grouped.setdefault(cat_key, {}).setdefault(ext_key, []).append(
            {
                "formula_id": f.id,
                "formula_name_zh": f.name_zh,
                "expr": f.expr,
                "latex": expr_to_latex(expr=f.expr, quantities=quantities),
            }
        )

    return _wrap_latex({
        "status": "ok",
        "filters": {"category": category, "extractid": extractid},
        "extractid_matched": extractid_matched,
        "formulas_path": _FORMULAS_DIR,
        "quantity": quantity_name_zh,
        "categories": grouped,
    })


def find_formulas_by_quantities(
    *,
    category: str,
    extractid: Optional[str] = None,
    quantity_name_zh: List[str],
) -> Dict[str, Any]:
    """
    Find formulas for multiple quantities in a category with specified extractid.
    Returns results organized by category -> extractid -> quantity display key.
    """
    if not _iter_yaml_files(_QUANTITIES_DIR):
        return _wrap_latex({
            "status": "error",
            "error_code": "QUANTITIES_NOT_FOUND",
            "message": f"[quantities not found] {_QUANTITIES_DIR}",
        })
    if not _iter_yaml_files(_FORMULAS_DIR):
        return _wrap_latex({
            "status": "error",
            "error_code": "FORMULAS_NOT_FOUND",
            "message": f"[formulas not found] {_FORMULAS_DIR}",
        })

    quantities = load_all_quantities(_QUANTITIES_DIR)

    # Build name_zh index (only name_zh matching)
    name_index: Dict[str, List[str]] = {}
    for q in quantities.values():
        key = _normalize_quantity_key(q.name_zh)
        if key:
            name_index.setdefault(key, []).append(q.id)

    existing_qids: List[str] = []
    existing_name_zh: List[str] = []
    missing_name_zh: List[str] = []
    resolved_map: Dict[str, str] = {}
    ambiguous_map: Dict[str, List[str]] = {}
    seen_qids: set[str] = set()
    seen_names: set[str] = set()

    for raw in quantity_name_zh:
        name = str(raw).strip()
        if not name:
            continue
        key = _normalize_quantity_key(name)
        matches = name_index.get(key, [])
        if not matches:
            if name not in seen_names:
                missing_name_zh.append(name)
                seen_names.add(name)
            continue
        if len(matches) > 1:
            ambiguous_map[name] = matches
        qid = matches[0]
        resolved_map[name] = qid
        if name not in seen_names:
            existing_name_zh.append(name)
            seen_names.add(name)
        if qid not in seen_qids:
            existing_qids.append(qid)
            seen_qids.add(qid)

    if not existing_qids and not missing_name_zh:
        return _wrap_latex({
            "status": "error",
            "error_code": "QUANTITY_NOT_FOUND",
            "message": "None of the requested quantity_name_zh were found in quantities.yaml",
            "details": {"available": sorted(quantities.keys()), "missing": missing_name_zh},
        })

    formulas_all = load_all_formulas(_FORMULAS_DIR)
    formulas = [f for f in formulas_all if f.category == category]
    if not formulas:
        return _wrap_latex({
            "status": "error",
            "error_code": "CATEGORY_NOT_FOUND",
            "message": f"category '{category}' not found in formulas.yaml",
        })

    extractid_matched = True
    if extractid:
        formulas_exact = [f for f in formulas if extractid in f.extractid]
        extractid_matched = bool(formulas_exact)
        formulas = formulas_exact if formulas_exact else formulas

    symtab = _mk_symbols(list(quantities.keys()))

    def _display_key(qid: str) -> str:
        q = quantities.get(qid)
        if not q:
            return qid
        return q.symbol_latex or q.symbol or q.id

    display_key_map = {qid: _display_key(qid) for qid in existing_qids}
    quantity_results: Dict[str, List[Dict[str, Any]]] = {
        display_key_map[qid]: [] for qid in existing_qids
    }
    qid_has_formula: Dict[str, bool] = {qid: False for qid in existing_qids}
    existing_set = set(existing_qids)

    for f in formulas:
        try:
            eq = _parse_equation(f.expr, symtab)
        except Exception:
            continue
        vars_in_eq = _symbols_in_equation(eq, symtab)

        for qid in vars_in_eq:
            if qid not in existing_set:
                continue
            # source字段映射
            src_file = None
            src_val = f.source or "thesis"
            if src_val == "thesis":
                if hasattr(f, 'extractid') and f.extractid:
                    src_file = str(f.extractid[0]).replace('_quantities', '').replace('_formulas', '')
            mapped_source = _format_source_label(src_val, src_file)
            quantity_results[display_key_map[qid]].append({
                "quantity_name_zh": quantities.get(qid).name_zh if qid in quantities else "",
                "quantity_value": "",
                "formula_id": f.id,
                "formula_name_zh": f.name_zh,
                "expr": f.expr,
                "latex": expr_to_latex(expr=f.expr, quantities=quantities),
                "source": mapped_source,
                "extractid": list(f.extractid),
                "target": qid,
            })
            qid_has_formula[qid] = True

    missing_formula_qids = [qid for qid, ok in qid_has_formula.items() if not ok]

    def _normalize_llm_quantity_spec(name_zh: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec = spec or {}
        _id = str(spec.get("id") or spec.get("symbol") or name_zh).strip()
        symbol = str(spec.get("symbol") or _id).strip()
        symbol_latex = str(spec.get("symbol_latex") or symbol).strip()
        zh = str(spec.get("name_zh") or "").strip()
        if not zh:
            if re.search(r"[一-鿿]", str(name_zh)):
                zh = str(name_zh)
            else:
                zh = _id
        unit = str(spec.get("unit") or "1").strip() or "1"
        return {
            "id": _id,
            "symbol": symbol,
            "symbol_latex": symbol_latex,
            "name_zh": zh,
            "unit": unit,
        }

    # LLM fill: missing quantities + missing formulas
    llm_quantity_specs: Dict[str, Dict[str, Any]] = {}
    if (missing_name_zh or missing_formula_qids) and os.getenv("DISABLE_LLM_MISSING_FILL", "0") != "1":
        try:
            if missing_name_zh:
                llm_quantity_specs = _llm_fill_missing_quantities_cached(
                    category,
                    extractid or "",
                    tuple(missing_name_zh),
                    _dir_fingerprint_ext(_RAW_DIR, patterns=("*.pdf",)),
                    _dir_fingerprint_ext(_MD_DIR, patterns=("*.md",)),
                )

            quantity_specs: List[Dict[str, Any]] = []
            for qid in missing_formula_qids:
                q = quantities.get(qid)
                if not q:
                    continue
                quantity_specs.append({
                    "id": q.id,
                    "symbol": q.symbol,
                    "symbol_latex": q.symbol_latex,
                    "name_zh": q.name_zh,
                    "unit": q.unit,
                })

            for name in missing_name_zh:
                spec = llm_quantity_specs.get(name) if isinstance(llm_quantity_specs, dict) else None
                quantity_specs.append(_normalize_llm_quantity_spec(name, spec or {}))

            quantity_specs_sorted = sorted(quantity_specs, key=lambda x: str(x.get("id") or ""))
            quantity_specs_json = json.dumps(quantity_specs_sorted, ensure_ascii=False, sort_keys=True)

            llm_formulas = _llm_fill_missing_formulas_cached(
                category,
                extractid or "",
                quantity_specs_json,
                _dir_fingerprint(_FORMULAS_DIR),
                _dir_fingerprint_ext(_RAW_DIR, patterns=("*.pdf",)),
                _dir_fingerprint_ext(_MD_DIR, patterns=("*.md",)),
            )

            for spec in quantity_specs_sorted:
                qid = str(spec.get("id") or "").strip()
                if not qid:
                    continue
                display_key = spec.get("symbol_latex") or spec.get("symbol") or qid
                items = llm_formulas.get(qid, []) if isinstance(llm_formulas, dict) else []
                if display_key not in quantity_results:
                    quantity_results[display_key] = []
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    # source字段映射
                    src_file = None
                    src_val = it.get("source") or "llm"
                    if src_val == "thesis":
                        if extractid:
                            src_file = str(extractid).replace('_quantities', '').replace('_formulas', '')
                    mapped_source = _format_source_label(src_val, src_file)
                    quantity_results[display_key].append({
                        "quantity_name_zh": spec.get("name_zh", ""),
                        "quantity_value": "",
                        "formula_id": it.get("formula_id"),
                        "formula_name_zh": it.get("formula_name_zh"),
                        "expr": it.get("expr"),
                        "latex": it.get("latex"),
                        "source": mapped_source,
                        "target": qid,
                    })
        except Exception:
            for qid in missing_formula_qids:
                quantity_results.setdefault(display_key_map.get(qid, qid), [])
            for name in missing_name_zh:
                quantity_results.setdefault(name, [])
    else:
        for qid in missing_formula_qids:
            quantity_results.setdefault(display_key_map.get(qid, qid), [])
        for name in missing_name_zh:
            quantity_results.setdefault(name, [])

    # Deduplicate: for each quantity's formulas, if multiple entries share the same expr,
    # keep the one with source == "expert"; otherwise keep the first.
    def _norm_expr_dedup(s: str) -> str:
        return re.sub(r"\s+", "", str(s).replace("^", "**"))

    for qid, flist in list(quantity_results.items()):
        if not isinstance(flist, list) or not flist:
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in flist:
            expr_str = item.get("expr", "")
            key = _norm_expr_dedup(expr_str)
            grouped.setdefault(key, []).append(item)

        deduped: List[Dict[str, Any]] = []
        for key, items in grouped.items():
            chosen = None
            for it in items:
                if str(it.get("source", "")).lower() == "expert":
                    chosen = it
                    break
            if chosen is None:
                chosen = items[0]
            deduped.append(chosen)

        # Second pass: deduplicate by identical LaTeX, prefer expert
        if len(deduped) > 1:
            latex_groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in deduped:
                lx = str(it.get("latex", ""))
                latex_groups.setdefault(lx, []).append(it)
            deduped2: List[Dict[str, Any]] = []
            for lx, items in latex_groups.items():
                chosen = None
                for it in items:
                    if str(it.get("source", "")).lower() == "expert":
                        chosen = it
                        break
                if chosen is None:
                    chosen = items[0]
                deduped2.append(chosen)
            deduped = deduped2

        quantity_results[qid] = deduped

    results = {category: {extractid: quantity_results}}

    llm_new_ids = []
    if isinstance(llm_quantity_specs, dict):
        for spec in llm_quantity_specs.values():
            if isinstance(spec, dict):
                qid = str(spec.get("id") or "").strip()
                if qid:
                    llm_new_ids.append(qid)

    return _wrap_latex({
        "status": "ok",
        "filters": {
            "category": category,
            "extractid": extractid,
            "extractid_matched": extractid_matched,
        },
        "formulas_path": _FORMULAS_DIR,
        "existing_name_zh": existing_name_zh,
        "missing_name_zh": missing_name_zh,
        "results": results,
    })
