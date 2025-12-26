from engine import solve_with_units, solve_goal_with_units, solve_target_auto, solve_targets_auto


def main():
    category = "tilt_totor"  # 换类别就改这里

    print("=== Demo A: 单公式求解 (F4 解 q) ===")
    result = solve_with_units(
        category=category,
        formula_id="F4",
        known_inputs={"rho": "1.225 kg/m^3", "v": "80 m/s"},
        target="q",
    )
    print(f"F4 ({result['formula_name_zh']}) solve q = {result['value']} {result['unit']} residual={result['residual']}")

    print("\n=== Demo B: 前向链多步求解 rho,v,S,CL -> L (先解 q 再解 L) ===")
    result = solve_goal_with_units(
        category=category,
        known_inputs={"rho": "1.225 kg/m^3", "v": "80 m/s", "S": "20 m^2", "CL": "0.5"},
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

    print("=== 成功案例：rho,v,S,CL -> L ===")
    res = solve_target_auto(
        category=category,
        known_inputs={
            "rho": "1.225 kg/m^3",
            "v": "80 m/s",
            "S": "20 m^2",
            "CL": "0.5",
        },
        target="L",
    )
    print(res)

    print("\n=== 失败案例：缺 rho ===")
    res = solve_target_auto(
        category=category,
        known_inputs={
            "v": "80 m/s",
            "S": "20 m^2",
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
            "S": "20 m^2",
            "CL": "0.5"
        },
        target="CD"   # quantities 有 CD，但没有 CD 的公式
    )
    print(res)

    print("\n=== 单步成功（F1） ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "q": "5000 Pa",
            "S": "20 m^2",
            "L": "50000 N"
        },
        target="CL"
    )
    print(res)    

    print("\n=== 单步不行 → 自动多步推导成功 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "rho": "1.225 kg/m^3",
            "v": "80 m/s",
            "S": "20 m^2",
            "CL": "0.5"
        },
        target="L"
    )
    print(res)   

    print("\n=== 多步失败，返回缺参链 ===")
    res = solve_target_auto(
        category="tilt_totor",
        known_inputs={
            "v": "80 m/s",
            "S": "20 m^2",
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

    print("\n=== 多目标求解成功案例：rho,v,S -> L,CD ===")
    res = solve_targets_auto(
        category="tilt_totor",
        known_inputs={"rho": "1.225 kg/m^3", "v": "80 m/s", "S": "20 m^2", "CL": "0.5"},
        targets=["q", "L", "K", "NotExist"],
    )
    print(res)
    
if __name__ == "__main__":
    main()
