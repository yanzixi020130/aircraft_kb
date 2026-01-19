#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI service: receive a query, embed it using the same remote embedding
service as embedding.py, compute cosine similarity against embeddings stored in
an embedded JSONL (e.g., data/embedding/tilt_rotor_embedded.jsonl), and
return the top-K most similar records.

Endpoints
- POST /query_rerank : body {query: str, top_k?: int, embedding_path?: str}

Behavior
- Load all records from the embedded JSONL; each line contains an "embedding"
  vector and other fields (chunk_id, content, ...).
- Call remote embedding service over SSH to get the query embedding.
- Compute cosine similarity(query_emb, record_emb) and sort desc.
- Return topK records with scores.

Run
    uvicorn src.query_rerank.query_rerank:app --host 0.0.0.0 --port 8000

Dependencies
    pip install fastapi uvicorn requests paramiko
    # 并设置 SSH_PASS 环境变量用于 SSH 登录远端主机
"""

from __future__ import annotations

import json
import os
import math
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import paramiko
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------- Embedding over SSH ----------------------------
class SSHConfig(BaseModel):
    Host: str = "192.168.0.112"
    Port: int = 22
    User: str = "ubuntu"
    PasswordEnv: str = "SSH_PASS"


def _escape_backslashes(text: str) -> str:
    return text.replace("\\", "\\\\")


def _build_remote_curl_command(payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(payload, ensure_ascii=False)
    data_arg = shlex.quote(payload_str)
    return (
        "curl -sS -X POST http://127.0.0.1:40291/embed "
        "-H 'Content-Type: application/json' "
        f"-d {data_arg}"
    )


def fetch_embeddings_over_ssh(texts: List[str], batch_size: int, cfg: SSHConfig) -> List[List[float]]:
    password = os.environ.get(cfg.PasswordEnv)
    if not password:
        raise RuntimeError(
            f"请先设置环境变量 {cfg.PasswordEnv}（用于 SSH 登录）"
        )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(cfg.Host, port=cfg.Port, username=cfg.User, password=password, timeout=15)

    all_embs: List[List[float]] = []
    try:
        payload = {"texts": texts, "batch_size": max(1, batch_size)}
        cmd = _build_remote_curl_command(payload)
        _stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if err.strip():
            raise RuntimeError("Remote error:\n" + err)
        try:
            resp = json.loads(out)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"远端返回不是合法 JSON：{e}\nRaw:\n{out}") from e
        embs = resp.get("embeddings")
        if not isinstance(embs, list):
            raise RuntimeError("远端响应缺少 embeddings 字段或类型不对")
        all_embs.extend(embs)
    finally:
        ssh.close()

    if len(all_embs) != len(texts):
        raise RuntimeError(
            f"embeddings 数量不匹配：texts={len(texts)} embeddings={len(all_embs)}"
        )
    return all_embs


# ---------------------------- Configuration ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDED_JSONL = BASE_DIR / "data" / "embedding" / "tilt_rotor_embedded.jsonl"


# ---------------------------- Models ----------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    top_k: int = Field(5, ge=1, description="Top-K results to return")
    embedding_path: Optional[str] = Field(
        None, description="Path to *_embedded.jsonl; default to data/embedding/tilt_rotor_embedded.jsonl"
    )


class RankedItem(BaseModel):
    score: float
    record: Dict[str, Any]


class RankedResponse(BaseModel):
    total: int
    top_k: int
    items: List[RankedItem]


# ---------------------------- Helpers ----------------------------
def read_embedded_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedded JSONL not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                # 记录错误并跳过
                continue
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb:
                records.append(obj)
    if not records:
        raise RuntimeError(f"No records with embedding found in {path}")
    return records


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------- FastAPI App ----------------------------
app = FastAPI(title="Query Rerank Service", version="1.0.0")


@app.post("/query_rerank", response_model=RankedResponse)
def query_rerank(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    embedded_path = Path(req.embedding_path) if req.embedding_path else DEFAULT_EMBEDDED_JSONL

    # 1) 读取已嵌入的记录
    try:
        records = read_embedded_jsonl(embedded_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 2) 查询向量化（与 embedding.py 相同的远端服务）
    try:
        cfg = SSHConfig()
        texts = [_escape_backslashes(req.query)]
        q_emb = fetch_embeddings_over_ssh(texts, batch_size=32, cfg=cfg)[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding error: {exc}")

    # 3) 相似度计算并排序
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec in records:
        emb = rec.get("embedding")
        if isinstance(emb, list):
            score = _cosine(q_emb, emb)
            scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)

    topk = max(1, req.top_k)
    items = [RankedItem(score=score, record=rec) for score, rec in scored[:topk]]
    return RankedResponse(total=len(scored), top_k=topk, items=items)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.query_rerank.query_rerank:app", host="0.0.0.0", port=8000, reload=False)
