# ALPHA-PrecisePhysicalVariableDefinition 项目文档

## 项目概述

`aircraft_kb` 是一个**单位感知的符号化工程计算引擎**，专门用于解决空气动力学和机械工程问题 [1](#0-0) 。该系统结合了符号数学（SymPy）、单位转换（Pint）和前向链逻辑，能够从已知输入自动推导目标物理量 [2](#0-1) 。

### 核心特性

- **声明式知识库定义**：通过YAML文件定义物理量（含单位）和公式（方程式），按类别组织（如 `tilt_totor`、`fix_wing`） [3](#0-2) 
- **自动单位处理**：接受任何兼容单位的输入（如 `"80 m/s"` 或 `"160 knots"`），引擎自动转换到标准单位，符号化求解，返回带正确单位的结果 [4](#0-3) 
- **多步自动推导**：当目标量无法直接计算时，引擎使用前向链推导中间量直至达到目标 [5](#0-4) 
- **智能错误诊断**：当输入不足时，系统执行反向依赖分析，精确报告缺失的物理量 [6](#0-5) 

## 系统架构

系统采用**三层分层架构**：

| 层级 | 组件 | 职责 |
|------|------|------|
| **用户界面层** | `run_demo.py`、外部Python脚本 | 调用求解器的入口点 |
| **核心引擎层** | `engine.py` 及其4个公共API + 6个内部函数 | 符号求解、单位转换、前向链、错误诊断 |
| **知识库层** | `data/` 目录中基于类别的YAML文件 | 领域知识（物理量、公式）和参考PDF |

### 核心组件

#### 引擎模块 (`engine.py`)

提供四个公共API函数 [7](#0-6) ：

| 函数 | 用途 | 使用场景 |
|------|------|----------|
| `solve_with_units()` | 单公式求解 | 当你知道确切的公式ID并拥有所有必需输入时 |
| `solve_goal_with_units()` | 多步前向链 | 当你想通过自动公式选择推导目标时 |
| `solve_target_auto()` | 智能求解器（推荐） | 自动选择最佳策略的推荐入口点 |
| `solve_targets_auto()` | 批处理 | 高效解决多个目标，共享前向链推导 |

#### 知识库 (`data/`)

知识库按**类别**组织领域知识，采用三目录模式：

```
data/
├── quantities/
│   └── tilt_totor/
│       └── quantities.yaml      # 9个物理量：CL, CD, L, D, K, q, rho, S, v
├── formulas/
│   └── tilt_totor/
│       └── fomulas.yaml         # 4个公式：F1-F4
└── raw/
    └── tilt_totor/
        └── tilt_totor.pdf       # 领域文档
```

**物理量定义模式** (`quantities.yaml`) [3](#0-2) ：
```yaml
quantities:
  - id: L                   # 唯一标识符
    symbol: L               # 数学符号
    name_zh: 升力           # 中文名称
    unit: N                 # 标准单位（Pint兼容）
```

**公式定义模式** (`formulas.yaml`)：
```yaml
formulas:
  - id: F1                  # 唯一标识符
    name_zh: "升力公式"     # 中文名称
    expr: "L = q * S * CL"  # 方程（支持 ^, *, /, +, -）
```

## 使用示例

### 倾转旋翼机空气动力学示例

`tilt_totor` 类别展示了系统能力 [8](#0-7) ：

**物理量**（共9个）：
- `L`（升力，N）、`D`（阻力，N）、`K`（升阻比，无量纲）
- `q`（动压，Pa）、`rho`（空气密度，kg/m³）、`v`（空速，m/s）
- `S`（机翼面积，m²）、`CL`（升力系数，无量纲）、`CD`（阻力系数，无量纲）

**公式**（共4个）：
- **F1**: `L = q * S * CL`（升力公式）
- **F2**: `D = q * S * CD`（阻力公式）
- **F3**: `K = L / D`（升阻比）
- **F4**: `q = 0.5 * rho * v^2`（动压）

### 快速开始

```python
from src.engine import solve_target_auto

# 从密度、速度、机翼面积和升力系数求解升力
result = solve_target_auto(
    category="tilt_totor",
    known_inputs={
        "rho": "1.225 kg/m^3",  # 空气密度
        "v": "80 m/s",           # 空速
        "S": "20 m^2",           # 机翼面积
        "CL": "0.5"              # 升力系数
    },
    target="L"  # 计算升力
)

# 结果：
# {
#   'status': 'ok',
#   'mode': 'multi-step',
#   'target': 'L',
#   'value': 39200.0,
#   'unit': 'N',
#   'path': [
#       {'formula_id': 'F4', 'target': 'q', 'value': 3920.0, 'unit': 'Pa', ...},
#       {'formula_id': 'F1', 'target': 'L', 'value': 39200.0, 'unit': 'N', ...}
#   ]
# }
```

## 技术栈

系统依赖三个外部库 [9](#0-8) ：

| 库 | 版本 | 用途 | 使用者 |
|----|------|------|--------|
| **SymPy** | 任意 | 符号数学、方程解析、代数求解 | `_parse_equation()`、`solve_one()` |
| **Pint** | 任意 | 单位注册表、转换、解析 | `_parse_known_inputs_to_magnitudes()` |
| **PyYAML** | 任意 | YAML解析 | `load_quantities()`、`load_formulas()` |

## 关键设计模式

1. **延迟加载与缓存**：`_load_category_context()` 使用 `@lru_cache(maxsize=32)` 按类别加载和解析YAML文件 [10](#0-9) 
2. **两阶段单位处理**：Pint转换输入到标准单位 → SymPy用无量纲数值求解 → Pint为单位附加输出 [4](#0-3) 
3. **策略模式**：`solve_target_auto()` 先尝试单步，回退到多步，再到诊断模式 [11](#0-10) 

## Notes

- 项目支持中英双语，YAML模式中包含 `name_zh` 字段用于国际化团队
- 系统使用不可变数据类确保类型安全：`QuantitySpec` 和 `FormulaSpec` 在 `src/engine.py:33-46` 定义
- 完整的演示和测试用例请参考 `src/run_demo.py`，包含成功和失败案例
- 知识库支持类别隔离，每个类别独立管理，避免交叉污染

