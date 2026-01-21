#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formula Extraction API
FastAPI 接口服务：
- /process-pdf: 上传 PDF 并处理（四步流程）
- /process-local-pdf: 本地 PDF 处理
- /similar-extract: 完整相似度检索 + 公式提取（query + PDF + topK）
"""

import uvicorn
import os
import sys
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, root_validator

# Allow importing from src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from engine import find_formulas_by_quantity, find_formulas_by_quantities, solve_targets_auto, load_all_quantities  # noqa: E402
from llm_generate_known_inputs import generate_known_inputs_from_payload  # noqa: E402
from offline_extract import Config, process_local_pdf  # noqa: E402
from similar_extract import process_pipeline  # noqa: E402

app = FastAPI(title="Formula Extraction API", version="1.0.0")


class FindFormulasByQuantityRequest(BaseModel):
    category: str | None = None
    extractid: str | None = None
    quantity_id: str | None = None

    @root_validator(pre=True)
    def at_least_one(cls, values):
        if not any(values.get(k) for k in ("category", "extractid", "quantity_id")):
            raise ValueError("At least one of category/extractid/quantity_id is required")
        return values


class FindFormulasByQuantitiesRequest(BaseModel):
    category: str
    extractid: str
    quantity_ids: list[str]


@app.post("/formulas/by-quantity")
def api_find_formulas_by_quantity(
    payload: FindFormulasByQuantityRequest,
) -> Dict[str, Any]:
    return find_formulas_by_quantity(
        category=payload.category,
        extractid=payload.extractid,
        quantity_id=payload.quantity_id,
    )


@app.post("/formulas/by-quantities")
def api_find_formulas_by_quantities(
    payload: FindFormulasByQuantitiesRequest,
) -> Dict[str, Any]:
    """Batch query formulas grouped by category -> extractid -> quantity_id."""
    return find_formulas_by_quantities(
        category=payload.category,
        extractid=payload.extractid,
        quantity_ids=payload.quantity_ids,
    )


class SolveTargetsAutoRequest(BaseModel):
    category: str
    extractid: str | None = None
    known_inputs: Dict[str, str]
    targets: list[str]
    formula_overrides: Dict[str, Any] | None = None
    max_steps: int = 50


class SelectedFormula(BaseModel):
    formula_id: str | None = None
    formula_name_zh: str | None = None
    latex: str | None = None
    expr: str
    source: str | None = None
    quantity_name_zh: str | None = None


class GenerateKnownInputsParams(BaseModel):
    formulaKey: str | None = None
    selectedFormulas: Dict[str, SelectedFormula]


class GenerateKnownInputsRequest(BaseModel):
    type: str | None = None
    taskid: str | None = None
    params: GenerateKnownInputsParams
    approved: bool | None = None


@app.post("/solve/targets-auto")
def api_solve_targets_auto(
    payload: SolveTargetsAutoRequest,
) -> Dict[str, Any]:
    return solve_targets_auto(
        category=payload.category,
        extractid=payload.extractid,
        known_inputs=payload.known_inputs,
        targets=payload.targets,
        formula_overrides=payload.formula_overrides,
        max_steps=payload.max_steps,
    )


@app.post("/generate-known-inputs")
def api_generate_known_inputs(payload: GenerateKnownInputsRequest) -> Dict[str, Any]:
    """根据 selectedFormulas 的 expr 解析所需输入，调用大模型生成变量列表。

    输出结构：
    {
        "variables": [
            {"quantity_id": "...", "symbol": "$...$", "name": "...", "value": "...", "unit": "...|null", "context": "...", "source": "expert|thesis|llm"}
        ],
        "status": "ok",
        "category": "expert"
    }
    """
    category = "expert"
    try:
        # Generate known inputs using LLM or fallback
        known_inputs = generate_known_inputs_from_payload(payload.dict(), category=category)

        # Load quantities from expert and thesis to determine metadata and source
        quantities_expert = load_all_quantities(os.path.join(BASE_DIR, "data", "quantities", "expert"))
        quantities_thesis = load_all_quantities(os.path.join(BASE_DIR, "data", "quantities", "thesis"))

        def split_value(val: str) -> tuple[str, str | None]:
            text = str(val).strip()
            # Try to split into numeric and unit by first space
            if " " in text:
                num, unit = text.split(" ", 1)
                return num, unit.strip() or None
            return text, None

        variables = []
        # Try to enrich context from incoming payload's selectedFormulas
        selected = (payload.params.selectedFormulas if payload and payload.params else {})

        for rid, raw in known_inputs.items():
            # Determine source and quantity spec (prefer expert)
            qspec = None
            source = "llm"
            if rid in quantities_expert:
                qspec = quantities_expert[rid]
                source = "expert"
            elif rid in quantities_thesis:
                qspec = quantities_thesis[rid]
                source = "thesis"

            num, unit = split_value(raw)
            # Normalize unit: if quantities has canonical unit and current unit is None, use canonical (except dimensionless "1")
            if qspec is not None:
                canonical_unit = qspec.unit
                if (unit is None) and canonical_unit and canonical_unit != "1":
                    unit = canonical_unit

            symbol_latex = qspec.symbol_latex if qspec is not None else rid
            name_zh = qspec.name_zh if qspec is not None else rid

            # Context: prefer SelectedFormula.quantity_name_zh if provided
            context_text = ""
            if isinstance(selected, dict) and rid in selected:
                sf = selected[rid]
                context_text = getattr(sf, "quantity_name_zh", None) or ""
            if not context_text:
                context_text = f"变量：{name_zh}"

            variables.append({
                "quantity_id": rid,
                "symbol": f"${symbol_latex}$",
                "name": name_zh,
                "value": num,
                "unit": unit,
                "context": context_text,
                "source": source,
            })

        return {"variables": variables, "status": "ok", "category": category}
    except Exception as e:  # pragma: no cover - defensive
        return {"status": "error", "message": str(e)}


@app.post("/process-pdf")
async def process_pdf_upload(file: UploadFile = File(...)) -> dict:
    """
    上传 PDF 文件进行处理。

    流程：保存上传文件 -> 调用本地处理管线 -> 返回结果。
    """
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

    # Keep uploads (no deletion). Store under a unique subfolder but keep original filename.
    original_name = os.path.basename(file.filename)
    upload_id = uuid4().hex
    upload_dir = os.path.join(Config.UPLOAD_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, original_name)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"✅ 文件已上传: {file_path}")
    except Exception as e:  # pragma: no cover - I/O guard
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"文件保存失败: {str(e)}"},
        )

    try:
        result = await process_local_pdf(file_path)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - pipeline guard
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
    finally:
        # Intentionally keep uploaded files in uploads/.
        pass


@app.get("/health")
async def health_check() -> dict:
    """健康检查端点。"""
    return {"status": "healthy", "service": "Formula Extraction API"}


@app.post("/similar-extract")
async def similar_extract(
    file: UploadFile = File(...),
    query: str = Form(...),
    topk: int = Form(10),
) -> dict:
    """
    完整相似度检索 + 公式提取接口。

    上传 PDF 与查询文本，返回检索与公式提取结果。
    """
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

    # Keep uploads (no deletion). Store under a unique subfolder but keep original filename.
    original_name = os.path.basename(file.filename)
    upload_id = uuid4().hex
    upload_dir = os.path.join(Config.UPLOAD_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, original_name)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"✅ 文件已上传: {file_path}")
    except Exception as e:  # pragma: no cover - I/O guard
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"文件保存失败: {str(e)}"},
        )

    try:
        result = await process_pipeline(
            pdf_path=file_path,
            query=query,
            topk=topk,
        )
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - pipeline guard
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
    finally:
        # Intentionally keep uploaded files in uploads/.
        pass


@app.get("/process-local-pdf")
async def process_local_pdf_endpoint(pdf_path: str) -> dict:
    """
    处理本地 PDF 文件的 REST 接口。
    """
    if not os.path.exists(pdf_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": f"文件不存在: {pdf_path}"},
        )

    try:
        result = await process_local_pdf(pdf_path)
        return JSONResponse(content=result)
    except Exception as e:  # pragma: no cover - pipeline guard
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )

if __name__ == "__main__":
    uvicorn.run(
        app='main:app', 
        host="0.0.0.0", 
        port=1420, 
        reload=False,
    )
