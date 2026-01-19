#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formula Extraction Pipeline - 本地文件处理
整合四个处理步骤：
  1. PDF to Markdown (pdf_to_md)
  2. Markdown to LaTeX JSONL (md_to_latex)
  3. Formula Division and Fixing (devide_and_fix)
  4. Formula to YAML (llm_4_extract)

本文件仅包含本地文件处理逻辑，FastAPI 接口请使用 main.py
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

from llm_client import LLMConfig

# 导入四个模块的主函数
from pdf_to_md import extract_pdf_to_md_async
from md_to_latex import extract_md_to_latex
from devide_and_fix import process_and_fix_formulas
from llm_4_extract import extract_formulas_to_yaml

# ==================== 配置 ====================
class Config:
    """全局配置（基于仓库根目录）"""
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _DATA_DIR = _PROJECT_ROOT / "data"
    # PDF 输入目录
    PDF_INPUT_DIR = str((_DATA_DIR / "raw").resolve())
    # Markdown 输出目录
    MD_OUTPUT_DIR = str((_DATA_DIR / "md").resolve())
    # LaTeX JSONL 输出目录
    LATEX_OUTPUT_DIR = str((_DATA_DIR / "latex").resolve())
    # Fixed LaTeX 输出目录
    FIXED_LATEX_OUTPUT_DIR = str((_DATA_DIR / "fixed_latex").resolve())
    # YAML 输出目录
    FORMULAS_OUTPUT_DIR = str((_DATA_DIR / "formulas" / "thesis").resolve())
    QUANTITIES_OUTPUT_DIR = str((_DATA_DIR / "quantities" / "thesis").resolve())
    RUNTIME_DIR = str((_DATA_DIR / "runtime").resolve())
    # MinerU API 地址
    MINERU_API_BASE = "http://www.science42.vip:40093"
    # LLM API 配置
    LLM_API_KEY = "ximu-llm-api-key"
    LLM_API_BASE = "http://www.science42.vip:40200/v1/chat/completions"
    # 上传文件临时目录（放在 data/uploads）
    UPLOAD_DIR = str((_DATA_DIR / "uploads").resolve())


# 统一的 LLM 配置实例
LLM_CFG = LLMConfig(api_key=Config.LLM_API_KEY, base_url=Config.LLM_API_BASE)

# ==================== 本地文件输入流程 ====================
async def process_local_pdf(
    pdf_path: str,
    output_suffix: str = None,
) -> dict:
    """
    处理本地 PDF 文件的完整流程
    
    参数：
        pdf_path: PDF 文件路径
        output_suffix: 输出文件名后缀（可选，用于区分不同的处理结果）
    
    返回：
        包含处理结果的字典
    """
    print("\n" + "="*60)
    print("开始处理本地 PDF 文件")
    print("="*60)
    
    try:
        def _finalize_yaml_outputs(source_stem: str, yaml_dir: str) -> Tuple[str, str]:
            """Move formulas.yaml & quantities.yaml into fixed dirs with requested names."""
            yaml_base = Path(yaml_dir)
            src_formulas = yaml_base / "formulas.yaml"
            src_quantities = yaml_base / "quantities.yaml"

            formulas_out_dir = Path(Config.FORMULAS_OUTPUT_DIR)
            quantities_out_dir = Path(Config.QUANTITIES_OUTPUT_DIR)
            formulas_out_dir.mkdir(parents=True, exist_ok=True)
            quantities_out_dir.mkdir(parents=True, exist_ok=True)

            dst_formulas = formulas_out_dir / f"{source_stem}_formulas.yaml"
            dst_quantities = quantities_out_dir / f"{source_stem}_quantities.yaml"

            if not src_formulas.exists() or not src_quantities.exists():
                raise FileNotFoundError(
                    f"未找到期望的 YAML: {src_formulas} / {src_quantities}"
                )

            shutil.move(str(src_formulas), str(dst_formulas))
            shutil.move(str(src_quantities), str(dst_quantities))
            return str(dst_formulas), str(dst_quantities)

        # 步骤 1: PDF 转 Markdown
        print("\n[步骤 1/4] PDF 转 Markdown...")
        md_path = await extract_pdf_to_md_async(
            input_path=pdf_path,
            output_dir=Config.MD_OUTPUT_DIR,
            docker_url=Config.MINERU_API_BASE,
        )
        if not md_path:
            return {"success": False, "error": "PDF 转 Markdown 失败"}
        print(f"✅ Markdown 文件: {md_path}")
        
        # 步骤 2: Markdown 转 LaTeX JSONL
        print("\n[步骤 2/4] Markdown 转 LaTeX JSONL...")
        jsonl_path = extract_md_to_latex(
            input_md=str(md_path),
            output_dir=Config.LATEX_OUTPUT_DIR
        )
        print(f"✅ JSONL 文件: {jsonl_path}")
        
        # 步骤 3: 公式清洗与修复
        print("\n[步骤 3/4] 公式清洗与修复...")
        md_context_path = str(md_path)
        fixed_jsonl_path = await process_and_fix_formulas(
            input_jsonl=jsonl_path,
            output_dir=Config.FIXED_LATEX_OUTPUT_DIR,
            md_file=md_context_path,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_API_BASE,
            llm_cfg=LLM_CFG,
        )
        print(f"✅ 清洗后的 JSONL: {fixed_jsonl_path}")
        
        # 步骤 4: 公式提取为 YAML
        print("\n[步骤 4/4] 公式提取为 YAML...")

        source_stem = Path(pdf_path).stem
        runtime_root = Path(Config.RUNTIME_DIR)
        runtime_root.mkdir(parents=True, exist_ok=True)
        tmp_yaml_dir = tempfile.mkdtemp(prefix=f"qf_{source_stem}_", dir=str(runtime_root))

        yaml_output_dir = await extract_formulas_to_yaml(
            input_jsonl=fixed_jsonl_path,
            output_dir=tmp_yaml_dir,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_API_BASE,
            llm_cfg=LLM_CFG,
        )
        formulas_file, quantities_file = _finalize_yaml_outputs(source_stem, yaml_output_dir)
        print(f"✅ 公式表: {formulas_file}")
        print(f"✅ 物理量表: {quantities_file}")
        
        print("\n" + "="*60)
        print("✅ 处理完成！")
        print("="*60)
        
        return {
            "success": True,
            "md_path": str(md_path),
            "jsonl_path": jsonl_path,
            "fixed_jsonl_path": fixed_jsonl_path,
            "formulas_file": formulas_file,
            "quantities_file": quantities_file,
        }
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return {"success": False, "error": str(e)}

# ==================== 命令行接口 ====================
async def main_cli():
    """命令行入口"""
    import sys
    
    print("\n" + "="*60)
    print("Formula Extraction Pipeline - 本地文件处理")
    print("="*60)
    print("\n使用方法：")
    print("  python src/offline_extract.py <PDF文件路径>")
    print("\n注意：")
    print("  FastAPI 接口服务请使用 main.py")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n❌ 请指定 PDF 文件路径！")
        print("示例: python src/offline_extract.py C:\\path\\to\\file.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return
    
    result = await process_local_pdf(pdf_path)
    
    # 打印结果
    print("\n📋 处理结果：")
    for key, value in result.items():
        print(f"  {key}: {value}")

# ==================== 示例函数 ====================
async def example_pipeline():
    """示例：完整流程处理"""
    pdf_file = str((Config._DATA_DIR / "raw" / "tilt_rotor" / "tilt_rotor.pdf").resolve())
    
    print("\n示例：处理 tilt_rotor.pdf")
    result = await process_local_pdf(pdf_file)
    
    if result["success"]:
        print("\n📋 处理成功，生成的文件：")
        print(f"  Markdown: {result['md_path']}")
        print(f"  JSONL: {result['jsonl_path']}")
        print(f"  Fixed JSONL: {result['fixed_jsonl_path']}")
        print(f"  YAML 目录: {result['yaml_output_dir']}")
    else:
        print(f"\n❌ 处理失败: {result['error']}")

if __name__ == "__main__":
    import sys
    
    # 如果有命令行参数，使用 CLI 模式
    if len(sys.argv) > 1:
        asyncio.run(main_cli())
    else:
        # 否则运行示例
        print("运行示例流程...")
        asyncio.run(example_pipeline())
