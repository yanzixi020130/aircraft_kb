#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple API test for /formulas/by-quantity.by-quantities"""

import json
import urllib.error
import urllib.request


def main() -> None:
    url = "http://36.103.203.113:1411/formulas/by-quantity"
    payload = {
        "category": "tilt_rotor",
        "quantity_name_zh": "翼根弦长",
        "extractid": "plane_design",
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

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
