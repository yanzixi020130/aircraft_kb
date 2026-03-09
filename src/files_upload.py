#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_API_URL = "http://36.103.203.113:1411/files/pdf"


def upload_pdf(pdf_path: Path, api_url: str, timeout: int = 120) -> Dict[str, Any]:
	try:
		with pdf_path.open("rb") as file_handle:
			files = {
				"file": (pdf_path.name, file_handle, "application/pdf")
			}
			response = requests.post(api_url, files=files, timeout=timeout)

		try:
			payload = response.json()
		except ValueError:
			payload = {"error": response.text}

		if response.status_code == 200 and isinstance(payload, dict) and payload.get("url"):
			return {
				"name": pdf_path.name,
				"url": payload["url"],
				"status": "success",
			}

		return {
			"name": pdf_path.name,
			"url": None,
			"error": payload.get("error") if isinstance(payload, dict) else str(payload),
			"status_code": response.status_code,
			"status": "failed",
		}
	except Exception as exc:
		return {
			"name": pdf_path.name,
			"url": None,
			"error": str(exc),
			"status": "failed",
		}


def build_tree_and_upload(directory: Path, api_url: str, dry_run: bool = False) -> Optional[Dict[str, Any]]:
	children: List[Dict[str, Any]] = []

	for item in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
		if item.is_dir():
			subtree = build_tree_and_upload(item, api_url, dry_run=dry_run)
			if subtree and subtree.get("children"):
				children.append(subtree)
		elif item.is_file() and item.suffix.lower() == ".pdf":
			if dry_run:
				children.append({
					"name": item.name,
					"url": "",
				})
			else:
				children.append(upload_pdf(item, api_url))

	if not children:
		return None

	return {
		"name": directory.name,
		"children": children,
	}


def fill_urls_from_manifest(
		node: Dict[str, Any],
		current_path: Path,
		api_url: str,
		skip_existing: bool = True,
	) -> Tuple[int, int, int]:
	"""
	根据既有 children JSON 结构递归上传 PDF 并回填 url。
	返回: (成功数, 失败数, 跳过数)
	"""
	success = 0
	failed = 0
	skipped = 0

	children = node.get("children")
	if isinstance(children, list):
		for child in children:
			if isinstance(child, dict):
				child_name = child.get("name")
				if not child_name:
					continue
				child_path = current_path / child_name
				s, f, k = fill_urls_from_manifest(
					child,
					child_path,
					api_url=api_url,
					skip_existing=skip_existing,
				)
				success += s
				failed += f
				skipped += k
		return success, failed, skipped

	file_name = node.get("name")
	if not file_name:
		return success, failed, skipped

	if current_path.suffix.lower() != ".pdf":
		return success, failed, skipped

	if skip_existing and node.get("url"):
		return success, failed, skipped + 1

	if not current_path.exists() or not current_path.is_file():
		node["url"] = ""
		node["error"] = f"文件不存在: {current_path}"
		return success, failed + 1, skipped

	upload_result = upload_pdf(current_path, api_url)
	if upload_result.get("status") == "success" and upload_result.get("url"):
		node["url"] = upload_result["url"]
		node.pop("error", None)
		return success + 1, failed, skipped

	node["url"] = ""
	node["error"] = upload_result.get("error", "上传失败")
	return success, failed + 1, skipped


def main() -> None:
	parser = argparse.ArgumentParser(description="递归上传目录内 PDF 并输出 children 树形 JSON")
	parser.add_argument(
		"--root",
		default="Asset_Allocation_Center",
		help="待上传的根目录（默认: Asset_Allocation_Center）",
	)
	parser.add_argument(
		"--api-url",
		default=DEFAULT_API_URL,
		help="上传接口地址",
	)
	parser.add_argument(
		"--output",
		default="asset_allocation_center_upload_result.json",
		help="输出 JSON 文件路径",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="仅生成 children JSON 结构，不执行上传",
	)
	parser.add_argument(
		"--manifest",
		default="",
		help="已有 children JSON 路径；传入后会在原结构上回填 url",
	)
	parser.add_argument(
		"--no-skip-existing",
		action="store_true",
		help="回填模式下不跳过已有 url（默认跳过）",
	)
	args = parser.parse_args()

	root_dir = Path(args.root).resolve()
	if not root_dir.exists() or not root_dir.is_dir():
		raise FileNotFoundError(f"目录不存在: {root_dir}")

	if args.manifest:
		manifest_path = Path(args.manifest).resolve()
		if not manifest_path.exists() or not manifest_path.is_file():
			raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

		result = json.loads(manifest_path.read_text(encoding="utf-8"))
		success, failed, skipped = fill_urls_from_manifest(
			node=result,
			current_path=root_dir,
			api_url=args.api_url,
			skip_existing=not args.no_skip_existing,
		)
		print(f"上传完成: success={success}, failed={failed}, skipped={skipped}")
	else:
		result = build_tree_and_upload(root_dir, args.api_url, dry_run=args.dry_run)
		if result is None:
			result = {
				"name": root_dir.name,
				"children": [],
			}

	output_path = Path(args.output).resolve()
	output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(result, ensure_ascii=False, indent=2))
	print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
	main()
