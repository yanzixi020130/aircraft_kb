from engine import solve_with_units, solve_goal_with_units, solve_target_auto, solve_targets_auto, find_formulas_by_quantity, find_formulas_by_quantities


def main():
    category = "tilt_totor"  # 换类别就改这里

    print("=== Demo A: 单公式求解 (F_dynamic_pressure 解 q) ===")
    result = solve_with_units(
        category=category,
        formula_id="F_dynamic_pressure",
        known_inputs={"rhoc": "1.225 kg/m^3", "Vc": "80 m/s"},
        target="q",
    )
    print(f"F_dynamic_pressure ({result['formula_name_zh']}) solve q = {result['value']} {result['unit']} residual={result['residual']}")

    print("\n=== Demo B: 前向链多步求解 rhoc,Vc,Swing,CL -> L (先解 q 再解 L) ===")
    result = solve_goal_with_units(
        category=category,
        known_inputs={"rhoc": "1.225 kg/m^3", "Vc": "80 m/s", "Swing": "20 m^2", "CL": "0.5"},
        goal="L",
    )

    print("Solved:", result["solved"])
    print("Goal:", result["goal"])

    print("\n[PATH]")
    for step in result["path"]:
        print(
            f"  {step['formula_id']} ({step['formula_name_zh']}) -> solve {step['target']} = {step['value']} {step['unit']}"
            f"  (residual={step['residual']})"
        )

    if result["solved"]:
        v, u = result["values"]["L"]
        print("\nResult L =", v, u)

    print("=== 成功案例：rhoc,Vc,Swing,CL -> L ===")
    res = solve_target_auto(
        category=category,
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5",
        },
        target="L",
    )
    print(res)

    print("\n=== 失败案例：缺 rhoc ===")
    res = solve_target_auto(
        category=category,
        known_inputs={
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5",
        },
        target="L",
    )
    print(res)

    print("\n=== 缺少求解 target 的公式 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "q": "5000 Pa",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        target="Pfc"   # quantities 有 Pfc，但没有 Pfc 的公式
    )
    print(res)

    print("\n=== 单步成功（F_lift） ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "q": "5000 Pa",
            "Swing": "20 m^2",
            "L": "50000 N"
        },
        target="CL"
    )
    print(res)    

    print("\n=== 单步不行 → 自动多步推导成功 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        target="L"
    )
    print(res)   

    print("\n=== 多步失败，返回缺参链 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        target="L"
    )
    print(res)  

    print("\n=== 目标已在 known_inputs 中（隐含规则，但非常重要） ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "L": "30000 N"
        },
        target="L"
    )
    print(res)  

    print("\n=== 指定部分公式（公式覆盖）示例 ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        targets=["q", "L"],
        formula_overrides={
            "q": "F_dynamic_pressure",
            "L": "F_lift"
        }
    )
    print(res)

    print("\n=== 正常多目标 ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        targets=["q", "L", "CL"],
    )
    print(res)

    print("\n=== 缺输入导致部分失败 ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "Vc": "80 m/s",
            "Swing": "20 m^2",
        },
        targets=["q", "L"],
    )
    print(res)

    print("\n=== 部分公式覆盖 + 其它自动 ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        targets=["q", "L", "CD"],
        formula_overrides={
            "q": "F_dynamic_pressure",
        }
    )
    print(res)

    print("\n=== 不存在目标 ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5"
        },
        targets=["NotExistA", "NotExistB"],
    )
    print(res)

    print("\n=== 目标已在输入中（given） ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "L": "30000 N",
            "CL": "0.4"
        },
        targets=["L", "CL"],
    )
    print(res)

    print("\n=== 多目标求解成功案例：rhoc,Vc,Swing -> L,CD ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={"rhoc": "1.225 kg/m^3", "Vc": "80 m/s", "Swing": "20 m^2", "CL": "0.5"},
        targets=["q", "L", "K", "NotExist"],
    )
    print(res)

    print("\n=== 单目标返回 LaTeX 公式 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5",
        },
        target="L",
    )
    print(res.get("used_formula_latex"))

    print("\n=== 多目标返回 LaTeX 公式（q, L） ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={
            "rhoc": "1.225 kg/m^3",
            "Vc": "80 m/s",
            "Swing": "20 m^2",
            "CL": "0.5",
        },
        targets=["q", "L"],
    )
    for k, v in res.get("results", {}).items():
        print(k, v.get("used_formula_latex"))

    print("\n=== 查询包含指定物理量的公式（按类别） ===")
    res = find_formulas_by_quantity(
        category="tilt_totor",
        quantity_id="q",
    )
    print(res)
    if res.get("status") == "ok":
        print("\n[LaTeX]")
        for f in res.get("formulas", []):
            print(f"  {f['formula_id']}: {f['latex']}")

        print("\n=== 查询包含指定多个物理量的公式（按类别） ===")
    res = find_formulas_by_quantities(
        category="tilt_totor",
        quantity_ids=["q", "CL"],
    )
    print(res)
    if res.get("status") == "ok":
        print("\n[LaTeX]")
        for f in res.get("formulas", []):
            print(f"  {f['formula_id']}: {f['latex']}")
    
if __name__ == "__main__":
    main()
