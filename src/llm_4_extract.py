#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Formula Extraction Module
从 fixed JSONL 读取清洗后的公式，调用 LLM 生成 formulas.yaml 和 quantities.yaml
"""

import asyncio
from pathlib import Path
import json
import re
from typing import List, Tuple, Optional, Iterable
import yaml
from llm_client import LLMClient, LLMConfig

# ==================== Prompt 模板 ====================
PROMPT_TEMPLATE_FIXED = """
你是一位专业的物理公式解析专家。给定一组已经清洗好的公式，请仅基于这些公式提取并生成 formulas.yaml 与 quantities.yaml。

【输入文件】{file_name}
【已清洗公式列表】
{context}

## 📋 输出格式规范

### 1. formulas.yaml 格式：
```yaml
formulas:
    - id: F_公式英文描述
        name_zh: "公式的中文名称"
        expr: "公式的 SymPy 可识别形式"
        extractid: [阶段1, 阶段2]
```

### 2. quantities.yaml 格式：
```yaml
quantities:
  - id: 变量名
    symbol: 变量符号
    symbol_latex: LaTeX 格式
    name_zh: 物理量中文名称
    unit: 国际标准单位
```

## 要求：
1. expr 必须是合法的 Python 表达式，乘法用 `*`，幂用 `**`，变量名用下划线不含特殊字符
2. 变量在 quantities.yaml 中去重，单位使用 SI，无量纲用 '1'
3. 输出格式缩进 2 空格
4. 提取 `extractid`（列表，可多值）。可选枚举仅限四个：`Flight_Performance_Analysis_Extraction_Parameters`、`plane_design`、`Overall_Parameter_Extraction_Parameters`、`Others`。优先匹配前三类；只有无法归入前三类时才使用 `Others`。
5. 若同一公式跨阶段适用，可在 `extractid` 列出多个阶段。

## ✅ 变量校验与清洗规则（必须遵守）
1. `quantities.yaml` 的 `id` 集合必须与所有 `expr` 中出现的变量集合完全一致；不得新增或遗漏任何变量。
2. 变量命名仅允许字母、数字、下划线，禁止空格与反斜杠空格；如遇 `LGr\ cw` 等，规范化为 `LGrcw`。
3. `symbol` 与 `symbol_latex` 不允许出现 `\text{{...}}` 等任何修饰符号；`symbol_latex` 仅用标准 LaTeX（希腊字母如 `\chi` 允许，下标形如 `X_{{sub}}`）。
4. 若遇 OCR/清洗误差（如 `\chi_{{LGr\ cw}}`、`Lambda_w0.25`），请规范为合法标识（示例：`\chi_{{LGrcw}}`、`Lambda_w0_25`）；无法确定则不要输出该变量，并避免在 `expr` 中使用不合法变量。
5. 乘法用 `*`，除法用 `/`，幂用 `**`，确保 SymPy 可解析；变量名不要使用花体命令或空格。
6. 单位用 SI，无量纲用 `'1'`；`name_zh` 使用简短准确的中文名。
7. 系数命名统一：凡属无量纲系数、权重、修正系数，一律使用 `K_*` 命名（`id/symbol`: `K_xxx`；`symbol_latex`: `K_{{xxx}}`），不要使用希腊字母命名系数。

## ✅ ID 复用规则（必须严格遵守）
你必须严格复用知识库中已有的 id，不得新增任何新的 id：
1) formulas.yaml 中每条公式的 `id` 必须从下面【允许的 formula_id 列表】中选择，不得编造、不得改写、不得加后缀。
2) quantities.yaml 中每条物理量的 `id` 必须从下面【允许的 quantity_id 列表】中选择，不得新增。
3) 若遇到同义/同符号变量（例如 `C_L`/`CL`、`S wing`/`Swing`），必须映射到列表中已有的最匹配 id。

【允许的 formula_id 列表（只能从这里选）】
{allowed_formula_ids}

【允许的 quantity_id 列表（只能从这里选）】
{allowed_quantity_ids}

## 📤 输出格式（必须严格遵守）：

### formulas.yaml
```yaml
formulas:
  - id: F_example
    name_zh: "示例公式"
        expr: "F = m * a"
        extractid: [plane_design]
```

### quantities.yaml
```yaml
quantities:
  - id: F
    symbol: F
    symbol_latex: F
    name_zh: 力
    unit: N
```

现在请按照上述格式解析公式：
"""


def _iter_yaml_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.rglob("*.yaml") if p.is_file()])


def _bucket_dirs(base_dir: Path) -> List[Path]:
    """Return which buckets to use.

    If thesis has any YAML files, use [expert, thesis]; otherwise only [expert].
    """
    expert = base_dir / "expert"
    thesis = base_dir / "thesis"

    thesis_has_files = any(_iter_yaml_files(thesis))
    if thesis_has_files:
        return [expert, thesis]
    return [expert]


def _collect_ids_from_kb(*, kind: str, project_root: Path, limit: int = 600) -> List[str]:
    """Collect existing ids from KB.

    kind:
      - 'formulas': collect formulas[].id under data/formulas/{expert,thesis}
      - 'quantities': collect quantities[].id under data/quantities/{expert,thesis}
    """
    data_dir = project_root / "data"
    if kind == "formulas":
        base_dir = data_dir / "formulas"
        key = "formulas"
    elif kind == "quantities":
        base_dir = data_dir / "quantities"
        key = "quantities"
    else:
        raise ValueError(f"unknown kind: {kind}")

    ids: List[str] = []
    seen = set()
    for d in _bucket_dirs(base_dir):
        for yf in _iter_yaml_files(d):
            try:
                obj = yaml.safe_load(open(yf, "r", encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            items = obj.get(key) or []
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                _id = it.get("id")
                if isinstance(_id, str):
                    _id = _id.strip()
                if not _id:
                    continue
                if _id in seen:
                    continue
                seen.add(_id)
                ids.append(_id)
                if len(ids) >= limit:
                    return ids
    return ids

llm_cfg_default = LLMConfig(system_prompt="你是一位专业的物理公式解析专家，精通数学符号、LaTeX 格式和 YAML 格式。")

# ---- IO helpers ----
def read_fixed_jsonl(jsonl_path: str) -> List[str]:
    """从 fixed JSONL 中读取清洗后的公式"""
    formulas: List[str] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fixed_list = obj.get("fixed", []) or []
            for formula in fixed_list:
                if isinstance(formula, str) and formula.strip():
                    formulas.append(formula.strip())
    return formulas

def chunk_list(seq: List[str], size: int) -> Iterable[List[str]]:
    """分块列表"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def extract_yaml_sections(llm_output: str) -> Tuple[Optional[str], Optional[str]]:
    """从 LLM 输出中提取 YAML 内容"""
    formulas_match = re.search(r"###\s*formulas\.yaml\s*```yaml\s*(.*?)\s*```", llm_output, re.DOTALL)
    quantities_match = re.search(r"###\s*quantities\.yaml\s*```yaml\s*(.*?)\s*```", llm_output, re.DOTALL)
    
    if not formulas_match:
        formulas_match = re.search(r"```yaml\s*(formulas:.*?)```", llm_output, re.DOTALL)
    if not quantities_match:
        quantities_match = re.search(r"```yaml\s*(quantities:.*?)```", llm_output, re.DOTALL)
    
    return (
        formulas_match.group(1) if formulas_match else None,
        quantities_match.group(1) if quantities_match else None,
    )

# ---- Main function ----
async def extract_formulas_to_yaml(
    input_jsonl: str = None,
    output_dir: str = None,
    api_key: str = "ximu-llm-api-key",
    base_url: str = "http://www.science42.vip:40200/v1/chat/completions",
    batch_size: int = 20,
    verbose: bool = False,
    temperature: float = 0.2,
    llm_cfg: Optional[LLMConfig] = None,
) -> str:
    """
    从 fixed JSONL 提取公式并生成 YAML 文件
    
    参数：
        input_jsonl: 输入 JSONL 文件路径
        output_dir: 输出目录
        api_key: LLM API 密钥
        base_url: LLM API 基础 URL
        batch_size: 批处理大小
        verbose: 是否打印详细信息
        temperature: LLM 温度参数
    
    返回：
        输出目录路径
    """
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _DATA_DIR = _PROJECT_ROOT / "data"
    jsonl_path = input_jsonl or str((_DATA_DIR / "fixed_latex" / "tilt_rotor" / "tilt_rotor_fixed.jsonl").resolve())
    output_base = output_dir or str((_DATA_DIR / "Q_and_F").resolve())
    
    # 初始化 LLM
    llm_config = llm_cfg or LLMConfig(
        api_key=api_key,
        base_url=base_url,
        system_prompt="你是一位专业的物理公式解析专家，精通数学符号、LaTeX 格式和 YAML 格式。",
    )
    llm = LLMClient(llm_config)
    
    # 读取清洗后的公式
    formulas = read_fixed_jsonl(jsonl_path)
    if not formulas:
        raise SystemExit(f"输入为空: {jsonl_path}")

    formulas_acc: List[str] = []
    quantities_acc: List[str] = []
    file_name = Path(jsonl_path).name

    # KB id whitelist (strict reuse)
    allowed_formula_ids = _collect_ids_from_kb(kind="formulas", project_root=_PROJECT_ROOT)
    allowed_quantity_ids = _collect_ids_from_kb(kind="quantities", project_root=_PROJECT_ROOT)

    # 创建输出目录和文件
    out_path = Path(output_base)
    out_path.mkdir(parents=True, exist_ok=True)
    formulas_file = out_path / "formulas.yaml"
    quantities_file = out_path / "quantities.yaml"
    
    with open(formulas_file, "w", encoding="utf-8") as f:
        f.write("formulas:\n")
    with open(quantities_file, "w", encoding="utf-8") as f:
        f.write("quantities:\n")

    # 分批处理公式
    for batch_no, batch in enumerate(chunk_list(formulas, batch_size), start=1):
        print(f"\n📦 处理批次 {batch_no}/{(len(formulas) + batch_size - 1) // batch_size}")
        print(f"   批次大小: {len(batch)} 个公式")
        
        lines = [f"[{i}] {expr}" for i, expr in enumerate(batch, start=1)]
        context = "\n".join(lines)

        prompt = PROMPT_TEMPLATE_FIXED.format(
            file_name=f"{file_name} (batch {batch_no})",
            context=context,
            allowed_formula_ids=json.dumps(allowed_formula_ids, ensure_ascii=False),
            allowed_quantity_ids=json.dumps(allowed_quantity_ids, ensure_ascii=False),
        )

        print(f"   正在请求 LLM...")
        result_text = await llm.acompletion_text(
            user_prompt=prompt,
            system_prompt=llm_config.system_prompt,
            temperature=temperature,
            max_tokens=4096,
            model=llm_config.model,
            timeout=200,
        )
        
        if not result_text:
            print(f"⚠️ 批次 {batch_no} 返回为空，跳过")
            continue
        
        print(f"   ✓ 收到响应 ({len(result_text)} 字符)")

        if verbose and result_text:
            print(f"\n---- 批次 {batch_no} 响应 ----\n{result_text}\n")

        formulas_section, quantities_section = extract_yaml_sections(result_text)
        if not formulas_section:
            print(f"⚠️ 批次 {batch_no} 未检测到 formulas.yaml 代码块")
        else:
            print(f"   ✓ 提取到 formulas 内容")
        if not quantities_section:
            print(f"⚠️ 批次 {batch_no} 未检测到 quantities.yaml 代码块")
        else:
            print(f"   ✓ 提取到 quantities 内容")

        def collect_body(section: Optional[str], header: str) -> List[str]:
            if not section:
                return []
            lines = section.splitlines()
            body: List[str] = []
            seen_header = False
            for ln in lines:
                if not seen_header and ln.strip().startswith(header):
                    seen_header = True
                    continue
                if seen_header:
                    body.append(ln.rstrip())
            if not seen_header:
                body = [ln.rstrip() for ln in lines]
            return body

        formulas_body = collect_body(formulas_section, "formulas:")
        quantities_body = collect_body(quantities_section, "quantities:")

        formulas_acc.extend(formulas_body)
        quantities_acc.extend(quantities_body)

        # 逐批写入文件
        if formulas_body:
            with open(formulas_file, "a", encoding="utf-8") as f:
                for line in formulas_body:
                    line_to_write = line if line.startswith(" ") else f"  {line}"
                    f.write(f"{line_to_write}\n")
            print(f"   ✓ 写入 {len(formulas_body)} 行到 formulas.yaml")
        if quantities_body:
            with open(quantities_file, "a", encoding="utf-8") as f:
                for line in quantities_body:
                    line_to_write = line if line.startswith(" ") else f"  {line}"
                    f.write(f"{line_to_write}\n")
            print(f"   ✓ 写入 {len(quantities_body)} 行到 quantities.yaml")

    print(f"\n✅ 完成！")
    print(f"   formulas.yaml: {formulas_file}")
    print(f"   quantities.yaml: {quantities_file}")
    
    return str(output_base)

if __name__ == "__main__":
    asyncio.run(extract_formulas_to_yaml())
