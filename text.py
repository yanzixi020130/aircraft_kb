#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))


from llm_fill_missing_quantities import generate_missing_quantities
from llm_fill_missing_formulas import generate_missing_formulas
from llm_client import LLMClient


def main():

    # 测试 llm_fill_missing_formulas.py
    quantities = [
        {
            "id": "V_tip_h",
            "symbol": "V_tip_h",
            "symbol_latex": "V_{tip,h}",
            "name_zh": "旋翼桨尖速度（直升机模式）",
            "unit": "m/s"
        },
        {
            "id": "V_tip_f",
            "symbol": "V_tip_f",
            "symbol_latex": "V_{tip,f}",
            "name_zh": "旋翼桨尖速度（固定翼模式）",
            "unit": "m/s"
        }
    ]
    out = generate_missing_formulas(
        category="tilt_rotor",
        extractid="Overall_Parameter_Extraction_Parameters",
        quantities=quantities,
        existing_formula_examples=[],
        raw_dir=os.path.join(BASE_DIR, "data", "raw"),
        md_dir=os.path.join(BASE_DIR, "data", "md"),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

    llm = LLMClient()
    print(llm.completion_text(user_prompt="你好", system_prompt="你是助手"))


if __name__ == "__main__":
    main()
