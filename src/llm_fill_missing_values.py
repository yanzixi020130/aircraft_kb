#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM helper: generate numeric values for missing targets.

Returns a JSON object mapping target "key" -> value (number or null).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm_client import LLMClient, LLMConfig


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_obj(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")

    m = _JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON braces in LLM response")


_PROMPT_TEMPLATE = (
    "You are a domain expert in aerospace/vehicle physics modeling.\n"
    "Task: estimate numeric values for the targets using the provided formulas and known inputs.\n\n"
    "Rules:\n"
    "- Output ONLY a JSON object.\n"
    "- Keys MUST match the `key` field from the targets list.\n"
    "- Values MUST be JSON numbers or null.\n"
    "- Do NOT include units or any extra text.\n"
    "- If the information is insufficient, output null.\n"
    "- Use the provided unit as the reference unit if available.\n\n"
    "category: {category}\n"
    "extractid: {extractid}\n\n"
    "known_inputs (JSON):\n"
    "{known_inputs_json}\n\n"
    "targets (JSON):\n"
    "{targets_json}\n"
)


def generate_missing_values(
    *,
    category: str,
    extractid: str | None,
    targets: List[Dict[str, Any]],
    known_inputs: List[Dict[str, Any]],
    llm_cfg: LLMConfig | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retries: int = 1,
) -> Dict[str, Any]:
    if not targets:
        return {}

    targets_json = json.dumps(targets, ensure_ascii=False)
    known_inputs_json = json.dumps(known_inputs or [], ensure_ascii=False)

    prompt = _PROMPT_TEMPLATE.format(
        category=category,
        extractid=extractid or "",
        targets_json=targets_json,
        known_inputs_json=known_inputs_json,
    )

    client = LLMClient(llm_cfg or LLMConfig())

    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            text = client.completion_text(
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            obj = _extract_json_obj(text)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_err = exc
            continue

    if last_err is not None:
        raise RuntimeError(f"LLM value generation failed: {last_err}") from last_err
    return {}


__all__ = ["generate_missing_values"]
