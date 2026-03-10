#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple API test for /formulas/by-quantity.by-quantities"""

import json
import urllib.error
import urllib.request


def main() -> None:
    url = "http://36.103.203.113:1411/formulas/by-quantity"
    quantities = [
        "机身长度",
        "1/4后掠角",
        "机翼扭转角",
        "机翼跨度",
        "翼根弦长",
        "翼梢弦长",
        "机翼上反角",
        "平尾四分之一弦线后掠角",
        "平尾扭转角",
        "平尾跨度",
        "平尾翼根弦",
        "平尾翼梢弦长",
        "平尾下反角",
        "垂尾四分之一弦线后掠角",
        "垂尾扭转角",
        "垂尾跨度",
        "垂尾翼根弦",
        "垂尾翼梢弦长",
        "垂尾下反角",
    ]

    for idx, quantity_name in enumerate(quantities, start=1):
        payload = {
            "category": "tilt_rotor",
            "quantity_name_zh": quantity_name,
            "extractid": "plane_design",
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(f"\n[{idx}/{len(quantities)}] quantity_name_zh={quantity_name}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                print(f"status: {resp.status}")
                try:
                    print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"status: {e.code}")
            try:
                print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(body)
        except urllib.error.URLError as e:
            print("request failed:", e)


if __name__ == "__main__":
    main()
