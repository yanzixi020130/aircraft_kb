#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remote embedding service client over SSH."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import paramiko


@dataclass(frozen=True)
class SSHConfig:
    Host: str = "192.168.0.112"
    Port: int = 22
    User: str = "ubuntu"
    PasswordEnv: str = "SSH_PASS"


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_remote_curl_command(payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(payload, ensure_ascii=False)
    data_arg = shlex.quote(payload_str)
    return (
        "curl -sS -X POST http://www.science42.vip:40291/embed "
        "-H 'Content-Type: application/json' "
        f"-d {data_arg}"
    )


def fetch_embeddings_over_ssh(
    texts: List[str],
    batch_size: int,
    cfg: SSHConfig,
    verbose: bool = True,
) -> List[List[float]]:
    password = os.environ.get(cfg.PasswordEnv)
    if not password:
        raise RuntimeError(
            f"请先设置环境变量 {cfg.PasswordEnv} 用于 SSH 登录，例如: $env:{cfg.PasswordEnv}='your_password'"
        )

    embeddings_all: List[List[float]] = []
    if verbose:
        print(f"[INFO] SSH 连接 {cfg.User}@{cfg.Host}:{cfg.Port}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(cfg.Host, port=cfg.Port, username=cfg.User, password=password, timeout=15)
        batches = list(iter_batches(texts, batch_size))
        for idx, batch in enumerate(batches, 1):
            if verbose:
                print(f"[INFO] 批次 {idx}/{len(batches)} 大小={len(batch)}")
            payload = {"texts": batch, "batch_size": batch_size}
            cmd = build_remote_curl_command(payload)
            _stdin, stdout, stderr = ssh.exec_command(cmd)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                raise RuntimeError(f"远端错误:\n{err}")
            try:
                resp = json.loads(out)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"远端返回非法 JSON: {e}\nRaw:\n{out}") from e
            emb = resp.get("embeddings")
            if not isinstance(emb, list):
                raise RuntimeError(f"远端响应缺少 embeddings，响应 keys={list(resp.keys())}")
            embeddings_all.extend(emb)
    finally:
        ssh.close()
        if verbose:
            print("[INFO] SSH 连接已关闭")

    if len(embeddings_all) != len(texts):
        raise RuntimeError(
            f"embeddings 数量不匹配：texts={len(texts)} vs embeddings={len(embeddings_all)}"
        )
    return embeddings_all


__all__ = ["SSHConfig", "build_remote_curl_command", "fetch_embeddings_over_ssh"]