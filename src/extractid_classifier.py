#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify formulas' extractid (list) via LLM.

Reads a formulas YAML file, asks the in-project LLM to assign extractid
(list, from four allowed stages), and writes a new YAML to data/test/ by default.

Allowed extractid values (stored in a list, can contain multiple stages):
- Flight_Performance_Analysis_Extraction_Parameters
- plane_design
- Overall_Parameter_Extraction_Parameters
- Others (only when none of the above apply)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import yaml

from llm_client import LLMClient, LLMConfig


EXTRACTID_DESCRIPTIONS = {
    "Flight_Performance_Analysis_Extraction_Parameters": "飞行性能分析阶段：气动、升阻、推重比、功率、性能评估相关公式。",
    "plane_design": "方案/构型设计阶段：几何、尺寸、布局、经验估算、构型参数相关公式。",
    "Overall_Parameter_Extraction_Parameters": "总体参数/需求提取阶段：需求指标、任务剖面、总质量/载荷/续航/航程等上层约束与分配相关公式。",
    "Others": "仅当无法归入前三类时使用。",
}

ALLOWED_EXTRACTID_VALUES = list(EXTRACTID_DESCRIPTIONS.keys())

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


PROMPT_JSON_TEMPLATE = """
你是航空航天领域的公式标注专家。你的任务：只为每条公式生成 extractid 列表。
你必须严格输出【纯 JSON】，不得输出任何解释、注释、Markdown 代码块、前后缀文字。

阶段枚举（可多选）：
{extractid_guides}

输入（JSON 数组，每项包含 id, name_zh, expr, category）：
{batch_json}

输出（纯 JSON 数组，顺序不限，但必须包含所有输入 id；不得新增 id）：
[
  {{"id":"...","extractid":["...","..."]}},
  ...
]
规则：
- extractid 必须是数组，元素只能来自四个枚举
- 若同时符合多个阶段，必须输出多个（不要只选一个）
- 如果输入已给出 extractid，必须保留这些已有值；如判断还适用于其他阶段，只能补充除已有值和 Others 之外的其他阶段
- 只有在没有已有 extractid 且无法归入前三类时才用 Others
"""

import json

def build_prompt_for_json(batch_items: list[dict]) -> str:
    guides = "\n".join([f"- {k}: {v}" for k, v in EXTRACTID_DESCRIPTIONS.items()])
    batch_json = json.dumps(
        [
            {"id": x["id"], "name_zh": x.get("name_zh"), "expr": x.get("expr"), "category": x.get("category")}
            for x in batch_items
        ],
        ensure_ascii=False
    )
    return PROMPT_JSON_TEMPLATE.format(
        extractid_guides=guides,
        batch_json=batch_json
    )

def parse_llm_json(resp_text: str) -> dict[str, list[str]]:
    import json
    # 尽量只提取第一个 [ ... ]
    start = resp_text.find('[')
    end = resp_text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM JSON输出格式错误，未找到完整数组")
    try:
        data = json.loads(resp_text[start:end+1])
    except Exception as exc:
        raise ValueError(f"解析LLM JSON失败: {exc}\n原始响应: {resp_text[:1000]}")
    if not isinstance(data, list):
        raise ValueError("LLM JSON不是数组")
    mp = {}
    for it in data:
        if not isinstance(it, dict):
            continue
        fid = it.get("id")
        xs = it.get("extractid", [])
        if isinstance(xs, str):
            xs = [xs]
        if fid:
            mp[fid] = [v for v in xs if v in ALLOWED_EXTRACTID_VALUES] or ["Others"]
    return mp

def merge_by_id(batch_items: list[dict], id2extract: dict[str, list[str]]) -> list[dict]:
    merged = []
    for base in batch_items:
        fid = base["id"]
        xs = id2extract.get(fid)
        if not xs:
            xs = base.get("extractid") or ["Others"]
        merged.append({**base, "extractid": xs})
    return merged

def classify_extractid(
    input_path: Path,
    output_path: Path | None = None,
    *,
    llm_cfg: LLMConfig | None = None,
    temperature: float = 0.1,
    batch_size: int | None = None,
    retries: int = 2,
) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input YAML not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    # Parse original YAML to allow fallback/merge when LLM output is incomplete
    try:
        _orig_obj = yaml.safe_load(text) or {}
    except Exception:
        _orig_obj = {}
    _orig_formulas = _orig_obj.get("formulas") if isinstance(_orig_obj, dict) else None
    if not isinstance(_orig_formulas, list):
        _orig_formulas = []
    def _norm_list(v):
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    def _norm_expr(expr: str | None) -> str | None:
        if not isinstance(expr, str):
            return None
        # Normalize for dedup: remove whitespace, unify power symbol, lowercase.
        return (
            expr.replace("^", "**")
            .replace(" ", "")
            .replace("\t", "")
            .replace("\n", "")
            .lower()
        )
    _orig_list: list[dict] = []
    for it in _orig_formulas:
        if not isinstance(it, dict):
            continue
        fid = it.get("id")
        if not fid:
            continue
        _orig_list.append(
            {
                "id": fid,
                "name_zh": it.get("name_zh"),
                "expr": it.get("expr"),
                "category": it.get("category"),
                # accept legacy 'extractids' in input
                "extractid": _norm_list(it.get("extractid", it.get("extractids", []))),
            }
        )
    def _extract_yaml_text(raw: str) -> str:
        """Prefer fenced yaml block if present; otherwise return raw."""
        if "```" not in raw:
            return raw
        parts = raw.split("```")
        # Odd indices are inside fences; pick the first fenced block.
        for i in range(1, len(parts), 2):
            block = parts[i]
            lines = block.splitlines()
            if lines and lines[0].strip().lower().startswith("yaml"):
                lines = lines[1:]
            block_text = "\n".join(lines).strip()
            if block_text:
                return block_text
        return raw

    client = LLMClient(llm_cfg or LLMConfig())

    def _merge_one_response(resp_text: str, batch_items: list[dict]) -> list[dict]:
        id2extract = parse_llm_json(resp_text)
        return merge_by_id(batch_items, id2extract)

    def _process_batch(batch, retries_left, cur_depth=0):
        prompt = build_prompt_for_json(batch)
        client_retries = retries_left
        last_err = None
        for attempt in range(client_retries + 1):
            resp = client.completion_text(user_prompt=prompt, temperature=temperature)
            if not resp:
                last_err = "empty response"
                continue
            try:
                merged = _merge_one_response(resp, batch)
                return merged
            except Exception as exc:
                last_err = exc
                if attempt < client_retries:
                    print(f"⚠ batch parse failed, retry {attempt + 1}/{client_retries}")
                    continue
        # 如果失败且batch大于1，自动二分递归
        if len(batch) > 1:
            mid = len(batch) // 2
            print(f"⚠ batch parse failed after retries, auto split ({len(batch)}→{mid}+{len(batch)-mid}): {last_err}")
            left = _process_batch(batch[:mid], retries_left, cur_depth+1)
            right = _process_batch(batch[mid:], retries_left, cur_depth+1)
            return left + right
        # fallback: 单条也失败，直接用原始
        print(f"⚠ single item parse failed, fallback to original: {last_err}")
        base = batch[0]
        xs = base.get("extractid")
        if isinstance(xs, str):
            xs = [xs]
        if not isinstance(xs, list) or not xs:
            xs = ["Others"]
        xs = [v for v in xs if v in ALLOWED_EXTRACTID_VALUES] or ["Others"]
        return [{**base, "extractid": xs}]

    out_path = output_path or Path("/data/se42/extraction_yjx/data/test") / input_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_items(fh, items: list[dict], seen_expr: set[str]):
        wrote = 0
        for it in items:
            norm = _norm_expr(it.get("expr"))
            if norm and norm in seen_expr:
                continue
            if norm:
                seen_expr.add(norm)
            block = yaml.dump([it], allow_unicode=True, sort_keys=False, Dumper=NoAliasDumper)
            indented = "\n".join(["  " + line if line else "" for line in block.splitlines()]) + "\n"
            fh.write(indented)
            wrote += 1
        return wrote

    total_items = len(_orig_list)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("formulas:\n")
        seen_expr: set[str] = set()

        if batch_size and batch_size > 0:
            total = (total_items + batch_size - 1) // batch_size
            for start in range(0, total_items, batch_size):
                batch = _orig_list[start : start + batch_size]
                merged = _process_batch(batch, retries)
                _write_items(fh, merged, seen_expr)
                batch_idx = start // batch_size + 1
                print(f"✔ processed batch {batch_idx}/{total} (items: {len(batch)})")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Classify formulas extractid (list) via LLM")
    parser.add_argument("input", type=str, help="Input formulas YAML path")
    parser.add_argument("--output", type=str, help="Output path (default: data/test/<basename>)")
    parser.add_argument("--temp", type=float, default=0, help="LLM temperature（默认 0，建议保持 0 提高稳定性）")
    parser.add_argument("--batch-size", type=int, default=30, help="Batch size for processing（默认 30，自动二分缩小，建议不超过30）")
    parser.add_argument("--retries", type=int, default=3, help="LLM 重试次数（解析失败时重试，默认 3 次）")
    args = parser.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve() if args.output else None

    out_path = classify_extractid(
        inp,
        output_path=out,
        temperature=args.temp,
        batch_size=args.batch_size,
        retries=args.retries,
    )
    print(f"✅ wrote classified YAML to {out_path}")


if __name__ == "__main__":
    main()
