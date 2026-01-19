#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similar Extract Pipeline
完整流程：上传PDF + 提问query + topk参数，串联所有模块进行相似度检索与公式提取

流程：
1. PDF → MD（pdf_to_md）
2. MD → chunks（llm_retrieve）
3. chunks → embeddings（embedding）
4. query → embedding（embedding）
5. 相似度计算取topK（query_rerank）
6. 仅对topK chunks做公式清洗/修复与提取（devide_and_fix + llm_4_extract）
7. 输出：topK结果 + 公式提取文件 + 参数提取文件

依赖：
- src/llm_retrieve.py (process_markdown_to_chunks)
- src/embedding.py (process_jsonl_to_embeddings)
- src/query_rerank/query_rerank.py (余弦相似度计算函数)
- devide_and_fix、llm_4_extract（通过调用其主逻辑）
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import shutil

from llm_client import LLMConfig

# 导入内部模块
sys.path.insert(0, str(Path(__file__).parent))

# ==================== 配置 ====================
class Config:
    # 基础路径（仓库根）
    ROOT = Path(__file__).resolve().parents[1]
    RUNTIME_DIR = ROOT / "data" / "runtime"
    FORMULAS_DIR = ROOT / "data" / "formulas" / "thesis"
    QUANTITIES_DIR = ROOT / "data" / "quantities" / "thesis"
    
    # MinerU API
    MINERU_API_BASE = "http://www.science42.vip:40093"
    
    # LLM 配置
    LLM_API_KEY = "ximu-llm-api-key"
    LLM_API_BASE = "http://www.science42.vip:40200/v1/chat/completions"
    
    # SSH 配置（用于 embedding）
    SSH_PASS = os.getenv("SSH_PASS", "")


# 统一的 LLM 配置实例（可通过环境变量覆盖默认）
LLM_CFG = LLMConfig(api_key=Config.LLM_API_KEY, base_url=Config.LLM_API_BASE)


def create_session_id() -> str:
    """生成会话ID：session_YYYYMMDD_HHMMSS_6位随机"""
    import random
    import string
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"session_{ts}_{rand}"


def create_session_dir(session_id: str) -> Path:
    """创建会话目录"""
    Config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = Config.RUNTIME_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


async def extract_formulas_and_quantities_from_topk(
    topk_chunks: List[Dict[str, Any]],
    session_dir: Path,
    source_stem: str,
) -> Tuple[str, str]:
    """
    仅对 topK chunks 做公式清洗/修复与参数提取。
    
    参考 devide_and_fix.py 和 llm_4_extract.py 的思路：
    - 把 topK chunks 的 content 转成 LaTeX JSONL
    - 调用 devide_and_fix 清洗/修复公式
    - 调用 llm_4_extract 提取参数
    
    输出：
    - formulas_file: 公式提取文件路径
    - quantities_file: 参数提取文件路径
    """
    try:
        # 动态导入（避免循环依赖）
        from offline_extract import process_local_pdf
        from devide_and_fix import process_and_fix_formulas
        from llm_4_extract import extract_formulas_to_yaml
        
        # 步骤 1: 构建 LaTeX JSONL（模仿 md_to_latex 的输出格式）
        latex_jsonl_path = session_dir / "topk_latex.jsonl"
        with latex_jsonl_path.open("w", encoding="utf-8") as f:
            for idx, chunk in enumerate(topk_chunks):
                content = chunk.get("content", "")
                # 简化的 LaTeX 记录（保留原 content）
                record = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": content,
                    "formulas": [],  # 占位符
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 步骤 2: 公式清洗/修复（使用 devide_and_fix 逻辑）
        fixed_latex_path = session_dir / "topk_latex_fixed.jsonl"
        try:
            fixed_latex_path = await process_and_fix_formulas(
                input_jsonl=str(latex_jsonl_path),
                output_dir=str(session_dir),
                md_file=None,
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE,
                llm_cfg=LLM_CFG,
            )
        except Exception as e:
            print(f"[WARNING] 公式修复失败，使用原始文件: {e}")
            fixed_latex_path = latex_jsonl_path
        
        # 步骤 3: 参数提取（使用 llm_4_extract 逻辑）
        yaml_output_dir = await extract_formulas_to_yaml(
            input_jsonl=str(fixed_latex_path),
            output_dir=str(session_dir),
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_API_BASE,
            llm_cfg=LLM_CFG,
        )

        # 将输出移动到固定目录，并按“原始文件名_*.yaml”命名
        src_dir = Path(yaml_output_dir)
        src_formulas = src_dir / "formulas.yaml"
        src_quantities = src_dir / "quantities.yaml"

        Config.FORMULAS_DIR.mkdir(parents=True, exist_ok=True)
        Config.QUANTITIES_DIR.mkdir(parents=True, exist_ok=True)

        dst_formulas = Config.FORMULAS_DIR / f"{source_stem}_formulas.yaml"
        dst_quantities = Config.QUANTITIES_DIR / f"{source_stem}_quantities.yaml"

        if not src_formulas.exists() or not src_quantities.exists():
            raise FileNotFoundError(f"未找到期望的 YAML: {src_formulas} / {src_quantities}")

        shutil.move(str(src_formulas), str(dst_formulas))
        shutil.move(str(src_quantities), str(dst_quantities))

        return str(dst_formulas), str(dst_quantities)
    
    except Exception as e:
        print(f"[ERROR] 公式与参数提取失败: {e}")
        # 返回空文件名或占位符
        return "", ""


async def process_pipeline(
    pdf_path: str,
    query: str,
    topk: int = 10,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    完整流程：PDF → chunks → embeddings → 相似度 → topK → 公式提取
    
    参数：
        pdf_path: 上传的 PDF 文件路径
        query: 用户提问
        topk: 返回的 topK 数量
        session_id: 会话ID（若为空则生成）
    
    返回：
        {
            "session_id": str,
            "topk": int,
            "results": [{score, chunk_id, content, ...}],
            "outputs": {formulas_file, quantities_file},
            "artifacts": {md_path, chunks_path, embeddings_path}
        }
    """
    if not session_id:
        session_id = create_session_id()
    
    session_dir = create_session_dir(session_id)
    
    print(f"\n{'='*70}")
    print(f"[会话] {session_id}")
    print(f"[工作目录] {session_dir}")
    print(f"{'='*70}\n")
    
    try:
        source_stem = Path(pdf_path).stem

        # ========== Step 1: PDF → MD ==========
        print("[Step 1/6] PDF → MD...")
        from pdf_to_md import extract_pdf_to_md_async
        md_path = await extract_pdf_to_md_async(
            input_path=pdf_path,
            output_dir=str(session_dir),
            docker_url=Config.MINERU_API_BASE,
        )
        if not md_path:
            return {"success": False, "error": "PDF 转 MD 失败"}
        print(f"✅ MD: {md_path}")
        
        # ========== Step 2: MD → chunks ==========
        print("\n[Step 2/6] MD → chunks...")
        from llm_retrieve import process_markdown_to_chunks
        chunks_path = process_markdown_to_chunks(
            input_md=md_path,
            out_dir=session_dir,
            verbose=True
        )
        print(f"✅ chunks: {chunks_path}")
        
        # ========== Step 3: chunks → embeddings ==========
        print("\n[Step 3/6] chunks → embeddings...")
        from embedding import process_jsonl_to_embeddings
        embedded_path = process_jsonl_to_embeddings(
            input_jsonl=chunks_path,
            out_dir=session_dir,
            batch_size=32,
            verbose=True
        )
        print(f"✅ embeddings: {embedded_path}")
        
        # ========== Step 4: query embedding + 相似度计算 ==========
        print("\n[Step 4/6] 相似度计算...")
        from query_rerank import read_embedded_jsonl, fetch_embeddings_over_ssh, _cosine, SSHConfig
        
        # 读取已嵌入的 chunks
        records = read_embedded_jsonl(Path(embedded_path))
        
        # query 向量化（与 embedding.py 相同的服务）
        from embedding import escape_backslashes
        cfg = SSHConfig()
        texts = [escape_backslashes(query)]
        q_emb = fetch_embeddings_over_ssh(texts, batch_size=32, cfg=cfg)[0]
        
        # 计算相似度并排序
        scored = []
        for rec in records:
            emb = rec.get("embedding")
            if isinstance(emb, list):
                score = _cosine(q_emb, emb)
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 取 topK
        topk = max(1, min(topk, len(scored)))
        topk_results = [(score, rec) for score, rec in scored[:topk]]
        print(f"✅ 找到 {len(scored)} 条记录，返回 topK={topk}")
        
        # ========== Step 5: 仅对 topK 做公式提取 ==========
        print("\n[Step 5/6] 对 topK chunks 做公式/参数提取...")
        topk_chunks = [rec for _, rec in topk_results]
        formulas_file, quantities_file = await extract_formulas_and_quantities_from_topk(
            topk_chunks,
            session_dir,
            source_stem,
        )
        print(f"✅ 公式文件: {formulas_file}")
        print(f"✅ 参数文件: {quantities_file}")
        
        # ========== Step 6: 组织返回结果 ==========
        print("\n[Step 6/6] 组织返回结果...")
        response = {
            "success": True,
            "session_id": session_id,
            "topk": topk,
            "results": [
                {
                    "score": float(score),
                    "chunk_id": rec.get("chunk_id"),
                    "content": rec.get("content"),
                    "page_hint": rec.get("page_hint"),
                    "section_path": rec.get("section_path"),
                }
                for score, rec in topk_results
            ],
            "outputs": {
                "formulas_file": formulas_file,
                "quantities_file": quantities_file,
            },
            "artifacts": {
                "md_path": str(md_path),
                "chunks_path": str(chunks_path),
                "embeddings_path": str(embedded_path),
                "session_dir": str(session_dir),
            }
        }
        
        print("\n" + "="*70)
        print("✅ 处理完成")
        print("="*70 + "\n")
        
        return response
    
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "session_id": session_id}


# ==================== 命令行入口 ==========
async def main_cli():
    """命令行测试入口"""
    pdf = str((Config.ROOT / "data" / "raw" / "tilt_rotor" / "tilt_rotor.pdf").resolve())
    query = "旋翼需用功率如何计算？"
    topk = 3
    
    result = await process_pipeline(pdf, query, topk)
    
    if result["success"]:
        print("\n📋 返回结果摘要：")
        print(f"  session_id: {result['session_id']}")
        print(f"  topk: {result['topk']}")
        print(f"  结果条数: {len(result['results'])}")
        print(f"  公式文件: {result['outputs']['formulas_file']}")
        print(f"  参数文件: {result['outputs']['quantities_file']}")
        
        # 打印 topK 结果
        print("\n📌 TopK 搜索结果：")
        for i, item in enumerate(result['results'], 1):
            print(f"\n  [{i}] score={item['score']:.4f} chunk_id={item['chunk_id']}")
            print(f"      content: {item['content'][:100]}...")
    else:
        print(f"\n❌ 失败: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main_cli())
