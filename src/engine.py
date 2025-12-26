from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Any

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
# Global settings
# =========================
KB_ROOT = r"C:\Project\aircraft_kb\data"
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


# =========================
# Data models
# =========================
@dataclass(frozen=True)
class QuantitySpec:
    id: str
    symbol: str
    name_zh: str
    unit: str


@dataclass(frozen=True)
class FormulaSpec:
    id: str
    expr: str  # e.g. "L = q * S * CL"
    name_zh: str


# =========================
# IO: load YAML
# =========================
def load_quantities(path: str) -> Dict[str, QuantitySpec]:
    obj = yaml.safe_load(open(path, "r", encoding="utf-8"))
    qmap: Dict[str, QuantitySpec] = {}
    for item in obj["quantities"]:
        q = QuantitySpec(
            id=item["id"],
            symbol=item.get("symbol", item["id"]),
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
            name_zh=f.get("name_zh", "")
        )
        for f in obj["formulas"]
    ]


# =========================
# SymPy helpers
# =========================
def _mk_symbols(quantity_ids: List[str]) -> Dict[str, sp.Symbol]:
    return {qid: sp.Symbol(qid, real=True) for qid in quantity_ids}


def _parse_equation(expr: str, symtab: Dict[str, sp.Symbol]) -> sp.Eq:
    """
    Parse "a = b" into sympy.Eq, with support for '^' as power.
    """
    if "=" not in expr:
        raise ValueError(f"Equation must contain '=': {expr}")

    left, right = expr.split("=", 1)
    left = left.strip().replace("^", "**")
    right = right.strip().replace("^", "**")

    lhs = parse_expr(left, local_dict=symtab, transformations=TRANSFORMS)
    rhs = parse_expr(right, local_dict=symtab, transformations=TRANSFORMS)
    return sp.Eq(lhs, rhs)


def _symbols_in_equation(eq: sp.Eq, symtab: Dict[str, sp.Symbol]) -> List[str]:
    """
    Return variable ids (strings) that appear in eq and are in our symtab.
    """
    symset = set(symtab.values())
    ids = []
    for s in eq.free_symbols:
        if s in symset:
            ids.append(s.name)
    # stable order: by name
    return sorted(set(ids))


# =========================
# Unit system (pint)
# =========================
def _pint_ureg() -> "pint.UnitRegistry":
    if not _HAS_PINT:
        raise RuntimeError("pint is not installed. Please: pip install pint")
    return pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


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
        if k not in quantities:
            raise KeyError(f"known key '{k}' not in quantities. Available: {sorted(quantities.keys())}")

        canonical_unit = quantities[k].unit
        s = str(s).strip()

        # If user provided unit text, parse directly; else attach canonical unit.
        if any(ch.isalpha() for ch in s) or "/" in s or "^" in s:
            qv = ureg(s)
        else:
            qv = float(s) * ureg(canonical_unit)

        # Handle conversion: only convert if canonical_unit is not "1"
        if canonical_unit != "1":
            qv = qv.to(ureg(canonical_unit))
            known_vals[k] = qv.magnitude
        else:
            # For dimensionless quantities, just use the magnitude
            if hasattr(qv, 'magnitude'):
                known_vals[k] = qv.magnitude
            else:
                known_vals[k] = float(qv)

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
@lru_cache(maxsize=32)
def _load_category_context(category: str) -> Tuple[Dict[str, QuantitySpec], Dict[str, sp.Eq], Dict[str, sp.Symbol]]:
    """
    Load and parse quantities + formulas for a category, cached.
    Returns: (quantities_dict, eqs_dict, symtab)
    """
    q_path = os.path.join(KB_ROOT, "quantities", category, "quantities.yaml")
    f_path = os.path.join(KB_ROOT, "formulas", category, "fomulas.yaml")  # keep your current spelling

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"[quantities not found] {q_path}")
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"[formulas not found] {f_path}")

    quantities = load_quantities(q_path)
    formulas = load_formulas(f_path)

    symtab = _mk_symbols(list(quantities.keys()))
    eqs = {
        f.id: {
            "eq": _parse_equation(f.expr, symtab),
            "name_zh": f.name_zh
        }
        for f in formulas
    }

    return quantities, eqs, symtab


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
        return {
            "status": "error",
            "error_code": "TARGET_NOT_FOUND",
            "message": f"target '{target}' not found in quantities.yaml",
            "details": {
                "available": sorted(quantities.keys())
            },
        }

    # Parse known inputs to magnitudes
    ureg = _pint_ureg()
    known_vals = _parse_known_inputs_to_magnitudes(
        ureg, quantities, known_inputs
    )

    # Rule 1: any formula can solve target?
    candidates = _candidate_formulas_for_target(eqs, symtab, target)
    if not candidates:
        return {
            "status": "error",
            "error_code": "FORMULA_FOR_TARGET_NOT_FOUND",
            "message": f"No formula can solve target '{target}'",
        }

    # Rule 2: try single-step first
    for fid, eq, vars_in_eq in candidates:
        required = set(vars_in_eq) - {target}
        if all(v in known_vals for v in required):
            try:
                x = solve_one(eq, known_vals, target, symtab)
                residual = _residual(eq, symtab, {**known_vals, target: x})
                residual_ok = abs(residual) < 1e-9
                return {
                    "status": "ok",
                    "mode": "single-step",
                    "target": target,
                    "value": x,
                    "unit": quantities[target].unit,
                    "used_formula": fid,
                    "used_formula_name_zh": eqs[fid]["name_zh"],
                    "residual": residual,
                    "residual_ok": residual_ok,
                }
            except Exception:
                pass

    # Rule 3: multi-step forward chaining
    out = _forward_chain_all(category=category, known_inputs=known_inputs)
    known_vals2 = out["known_vals"]
    path = out["path"]

    if target in known_vals2:
        val = known_vals2[target]
        return {
            "status": "ok",
            "mode": "multi-step",
            "target": target,
            "value": val,
            "unit": quantities[target].unit,
            "path": path,
        }

    # Rule 4: build reverse missing chain
    missing_chain = _build_missing_chain(
        target, known_vals2, quantities, eqs, symtab
    )

    return {
        "status": "error",
        "error_code": "INSUFFICIENT_INPUTS",
        "message": f"Cannot solve target '{target}' due to missing inputs",
        "missing_chain": missing_chain,
    }

def solve_targets_auto(
    *,
    category: str,
    known_inputs: Dict[str, str],
    targets: List[str],
    max_steps: int = 50,
) -> Dict[str, Any]:
    quantities, eqs, symtab = _load_category_context(category)

    # 初始已知（仅来自输入）
    ureg = _pint_ureg()
    known_vals_initial = _parse_known_inputs_to_magnitudes(ureg, quantities, known_inputs)

    # 一次性前向链推导（复用中间量）
    out = _forward_chain_all(category=category, known_inputs=known_inputs, max_steps=max_steps)
    known_vals_all = out["known_vals"]
    path_all = out["path"]

    def path_until(t: str) -> List[Dict[str, Any]]:
        """返回推导路径中从开始到首次得到 t 的子路径；如果没得到 t 返回空列表。"""
        for i, step in enumerate(path_all):
            if step["target"] == t:
                return path_all[: i + 1]
        return []

    results: Dict[str, Any] = {}

    for t in targets:
        # Rule 0
        if t not in quantities:
            results[t] = {
                "status": "error",
                "error_code": "TARGET_NOT_FOUND",
                "message": f"target '{t}' not found in quantities.yaml",
                "details": {"available": sorted(quantities.keys())},
            }
            continue

        # Rule 1：是否存在可解 t 的公式
        candidates = _candidate_formulas_for_target(eqs, symtab, t)
        if not candidates:
            results[t] = {
                "status": "error",
                "error_code": "FORMULA_FOR_TARGET_NOT_FOUND",
                "message": f"No formula can solve target '{t}'",
            }
            continue

        # 已知就直接返回（可选：如果你不需要这个分支可以删）
        if t in known_vals_initial:
            results[t] = {
                "status": "ok",
                "mode": "given",
                "target": t,
                "value": known_vals_initial[t],
                "unit": quantities[t].unit,
            }
            continue

        # 先判断：是否能单步（基于 initial known，不用推导中间量）
        single_ok = None
        for fid, eq, vars_in_eq in candidates:
            required = set(vars_in_eq) - {t}
            if all(v in known_vals_initial for v in required):
                try:
                    x = solve_one(eq, known_vals_initial, t, symtab)
                    residual = _residual(eq, symtab, {**known_vals_initial, t: x})
                    residual_ok = abs(residual) < 1e-9
                    single_ok = {
                        "status": "ok",
                        "mode": "single-step",
                        "target": t,
                        "value": x,
                        "unit": quantities[t].unit,
                        "used_formula": fid,
                        "used_formula_name_zh": eqs[fid]["name_zh"],
                        "residual": residual,
                        "residual_ok": residual_ok,
                    }
                    break
                except Exception:
                    pass

        if single_ok is not None:
            results[t] = single_ok
            continue

        # 再看：多步推导是否已得到 t
        if t in known_vals_all:
            results[t] = {
                "status": "ok",
                "mode": "multi-step",
                "target": t,
                "value": known_vals_all[t],
                "unit": quantities[t].unit,
                "path": path_until(t),   # 只返回到 t 为止的路径，避免太长
            }
            continue

        # 多步也失败：反向缺参链（用已推导的 known_vals_all 更准确）
        missing_chain = _build_missing_chain(t, known_vals_all, quantities, eqs, symtab)
        results[t] = {
            "status": "error",
            "error_code": "INSUFFICIENT_INPUTS",
            "message": f"Cannot solve target '{t}' due to missing inputs",
            "missing_chain": missing_chain,
        }

    return {
        "status": "ok",
        "category": category,
        "targets": targets,
        "results": results,
    }
