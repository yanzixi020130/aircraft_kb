# from __future__ import annotations
#
# import argparse
# import os
# import re
# import shutil
# import subprocess
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Tuple
# import logging
#
# DEFAULT_MINERU_URL = "http://www.science42.vip:40093"
#
#
# class MinerUParser:
#     """
#     MinerU CLI wrapper (fixed HTTP backend), no extra post-processing.
#     """
#
#     def __init__(self, docker_url: str = None):
#         self.docker_url = docker_url or DEFAULT_MINERU_URL
#         self.logger = logging.getLogger(__name__)
#
#     def _resolve_mineru_bin(self) -> str:
#         env_bin = os.getenv("MINERU_BIN")
#         if env_bin:
#             return env_bin
#         which_bin = shutil.which("mineru")
#         if which_bin:
#             return which_bin
#         raise RuntimeError("mineru executable not found; set MINERU_BIN or install mineru")
#
#     def _collect_markdown_files(self, output_dir: str, input_path: str) -> Optional[list[Path]]:
#         pdf_stem = Path(input_path).stem
#         vlm_dir = Path(output_dir) / pdf_stem / "vlm"
#         if not vlm_dir.exists():
#             self.logger.error("VLM directory not found: %s", vlm_dir)
#             return None
#         md_files = list(vlm_dir.glob("*.md"))
#         if not md_files:
#             self.logger.error("No markdown files found in: %s", vlm_dir)
#             return None
#         md_files.sort(key=self._markdown_sort_key)
#         return md_files
#
#     def _markdown_sort_key(self, path: Path) -> tuple:
#         numbers = re.findall(r"\d+", path.stem)
#         if numbers:
#             return (0, int(numbers[0]), path.name)
#         return (1, path.name)
#
#     def _run_mineru_cli(
#         self,
#         input_path: str,
#         output_dir: str,
#         docker_url: str,
#         start_page: Optional[int] = None,
#         end_page: Optional[int] = None,
#         timeout: int = 1800,
#         formula: bool = True,
#         table: bool = True,
#     ) -> bool:
#         mineru_bin = self._resolve_mineru_bin()
#         cmd = [
#             mineru_bin,
#             "-p",
#             str(Path(input_path).resolve()),
#             "-o",
#             str(Path(output_dir).resolve()),
#             "-b",
#             "vlm-http-client",
#             "-u",
#             docker_url,
#             "--http-timeout",
#             str(timeout),
#         ]
#         if start_page is not None:
#             cmd.extend(["-s", str(start_page)])
#         if end_page is not None:
#             cmd.extend(["-e", str(end_page)])
#         if formula:
#             cmd.extend(["--formula", "true"])
#         if table:
#             cmd.extend(["--table", "true"])
#
#         env = os.environ.copy()
#         env["MINERU_FORMULA_ENABLE"] = "true" if formula else "false"
#         env["MINERU_VLM_FORMULA_ENABLE"] = "true" if formula else "false"
#
#         self.logger.info("Running mineru CLI: %s", " ".join(cmd))
#         result = subprocess.run(cmd, env=env, capture_output=True, text=True)
#         if result.stdout:
#             self.logger.info(result.stdout.strip())
#         if result.stderr:
#             self.logger.error(result.stderr.strip())
#         return result.returncode == 0
#
#     async def parse_file(
#         self,
#         file_path: str,
#         output_dir: str,
#         max_pages_per_batch: int = 10,
#         cleanup: bool = False,
#         keep_images: bool = True,
#     ) -> Tuple[bool, Optional[str], Optional[str]]:
#         """
#         CLI-only parse: run mineru once for the whole file and return Markdown content.
#         """
#         _ = max_pages_per_batch, cleanup, keep_images
#
#         abs_output_dir = os.path.abspath(output_dir)
#         os.makedirs(abs_output_dir, exist_ok=True)
#
#         ok = self._run_mineru_cli(
#             input_path=file_path,
#             output_dir=abs_output_dir,
#             docker_url=self.docker_url,
#         )
#         if not ok:
#             return False, None, None
#
#         md_files = self._collect_markdown_files(abs_output_dir, file_path)
#         if not md_files:
#             return False, None, None
#         if len(md_files) == 1:
#             md_content = md_files[0].read_text(encoding="utf-8")
#         else:
#             self.logger.info("Found %d markdown files, merging", len(md_files))
#             md_content = "\n\n".join(p.read_text(encoding="utf-8") for p in md_files)
#         images_dir = Path(abs_output_dir) / Path(file_path).stem / "vlm" / "images"
#         images_dir_path = str(images_dir) if images_dir.exists() else None
#
#         return True, md_content, images_dir_path
#
#
# def _main() -> int:
#     parser = argparse.ArgumentParser(description="MinerU CLI wrapper: PDF/image -> Markdown")
#     parser.add_argument("input_path", help="PDF file path")
#     parser.add_argument("output_dir", help="Output directory")
#     parser.add_argument("--docker-url", default=DEFAULT_MINERU_URL, help="MinerU API URL")
#     parser.add_argument("--start", type=int, default=None, help="Start page (1-based)")
#     parser.add_argument("--end", type=int, default=None, help="End page (1-based)")
#     parser.add_argument("--timeout", type=int, default=1800, help="HTTP timeout seconds")
#     parser.add_argument("--no-formula", dest="formula", action="store_false")
#     parser.add_argument("--no-table", dest="table", action="store_false")
#     parser.set_defaults(formula=True, table=True)
#     args = parser.parse_args()
#
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     parser_obj = MinerUParser(docker_url=args.docker_url)
#     ok = parser_obj._run_mineru_cli(
#         input_path=args.input_path,
#         output_dir=args.output_dir,
#         docker_url=args.docker_url,
#         start_page=args.start,
#         end_page=args.end,
#         timeout=args.timeout,
#         formula=args.formula,
#         table=args.table,
#     )
#     return 0 if ok else 1
#
#
# if __name__ == "__main__":
#     raise SystemExit(_main())

import os
import re
import asyncio
import subprocess
import shutil
import hashlib
from typing import Optional, Tuple
from pathlib import Path
import logging

class MinerUParser:
    """
    新版minerU Docker服务解析器
    负责调用minerU Docker服务解析PDF和图片文件
    """

    def __init__(self, docker_url: str = None):
        # 如果没有传入docker_url，则从环境变量读取
        if docker_url is None:
            docker_url = os.getenv('MINERU_DOCKER_URL', 'http://www.science42.vip:40093')

        self.docker_url = docker_url
        self.logger = logging.getLogger(__name__)

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除特殊字符和空格
        """
        # 保留文件扩展名
        name, ext = os.path.splitext(filename)
        # 移除特殊字符，只保留字母、数字、下划线和连字符
        cleaned_name = re.sub(r'[^\w\-]', '_', name)
        # 移除连续的下划线
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # 移除开头和结尾的下划线
        cleaned_name = cleaned_name.strip('_')

        return f"{cleaned_name}{ext}"

    async def _run_mineru_command(self, input_path: str, output_dir: str) -> bool:
        """
        异步执行minerU命令 - 使用绝对路径确保路径正确
        """
        # 转换为绝对路径
        abs_input_path = os.path.abspath(input_path)
        abs_output_dir = os.path.abspath(output_dir)

        cmd = [
            "mineru",
            "-p", abs_input_path,
            "-o", abs_output_dir,
            "-b", "vlm-sglang-client",
            "-u", self.docker_url
        ]

        try:
            self.logger.info(f"执行minerU命令: {' '.join(cmd)}")
            self.logger.info(f"输入文件绝对路径: {abs_input_path}")
            self.logger.info(f"输出目录绝对路径: {abs_output_dir}")

            # 异步执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.logger.info(f"minerU解析成功: {abs_input_path}")
                return True
            else:
                self.logger.error(f"minerU解析失败: {abs_input_path}")
                self.logger.error(f"错误输出: {stderr.decode()}")
                return False

        except Exception as e:
            self.logger.error(f"执行minerU命令异常: {e}")
            return False

    def _find_markdown_file(self, result_dir: str, original_filename: str) -> Optional[str]:
        """
        在解析结果目录中查找markdown文件 - 修复路径查找逻辑
        """
        # 使用绝对路径
        abs_result_dir = os.path.abspath(result_dir)
        name_without_ext = os.path.splitext(original_filename)[0]

        # minerU输出的目录结构: output_dir/filename_without_ext/vlm/filename.md
        vlm_dir = os.path.join(abs_result_dir, name_without_ext, "vlm")

        self.logger.info(f"查找VLM目录: {vlm_dir}")

        if not os.path.exists(vlm_dir):
            # 如果标准路径不存在，尝试列出实际的目录结构
            self.logger.warning(f"VLM目录不存在: {vlm_dir}")

            # 列出实际存在的目录结构用于调试
            if os.path.exists(abs_result_dir):
                self.logger.info(f"实际结果目录内容: {os.listdir(abs_result_dir)}")

                # 查找任何包含文件名的目录
                for item in os.listdir(abs_result_dir):
                    item_path = os.path.join(abs_result_dir, item)
                    if os.path.isdir(item_path) and name_without_ext in item:
                        self.logger.info(f"找到相关目录: {item_path}")
                        self.logger.info(f"目录内容: {os.listdir(item_path)}")

                        # 查找vlm子目录
                        vlm_path = os.path.join(item_path, "vlm")
                        if os.path.exists(vlm_path):
                            vlm_dir = vlm_path
                            self.logger.info(f"找到VLM目录: {vlm_dir}")
                            break

            if not os.path.exists(vlm_dir):
                return None

        # 查找.md文件
        md_files = [f for f in os.listdir(vlm_dir) if f.endswith('.md')]

        if not md_files:
            self.logger.warning(f"未找到markdown文件: {vlm_dir}")
            self.logger.info(f"VLM目录内容: {os.listdir(vlm_dir)}")
            return None

        if len(md_files) > 1:
            self.logger.warning(f"找到多个markdown文件，使用第一个: {md_files}")

        md_file_path = os.path.join(vlm_dir, md_files[0])
        self.logger.info(f"找到markdown文件: {md_file_path}")
        return md_file_path

    def _cleanup_temp_files(self, result_dir: str, original_filename: str, keep_images: bool = True):
        """
        清理临时文件，只保留markdown文件
        """
        abs_result_dir = os.path.abspath(result_dir)
        name_without_ext = os.path.splitext(original_filename)[0]
        vlm_dir = os.path.join(abs_result_dir, name_without_ext, "vlm")

        if not os.path.exists(vlm_dir):
            return

        # 需要保留的文件/目录
        keep_items = set()

        # 保留所有.md文件
        for item in os.listdir(vlm_dir):
            if item.endswith('.md'):
                keep_items.add(item)

        # 删除images目录和其他所有文件
        for item in os.listdir(vlm_dir):
            if item not in keep_items:
                item_path = os.path.join(vlm_dir, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        self.logger.debug(f"删除文件: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        self.logger.debug(f"删除目录: {item_path}")
                except Exception as e:
                    self.logger.warning(f"删除失败 {item_path}: {e}")

    async def parse_file(
        self,
        file_path: str,
        output_dir: str,
        cleanup: bool = False,
        keep_images: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        解析单个文件

        Args:
            file_path: 输入文件路径
            output_dir: 输出目录
            cleanup: 是否清理临时文件
            keep_images: 是否保留images目录

        Returns:
            (success, markdown_content, images_dir_path)
        """
        # 创建输出目录
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)

        # 获取清理后的文件名
        original_filename = os.path.basename(file_path)
        sanitized_filename = self._sanitize_filename(original_filename)

        # 如果文件名有变化，复制到新位置
        if sanitized_filename != original_filename:
            sanitized_path = os.path.join(os.path.dirname(file_path), sanitized_filename)
            shutil.copy2(file_path, sanitized_path)
            input_path = sanitized_path
            should_cleanup_input = True
        else:
            input_path = file_path
            should_cleanup_input = False

        try:
            # 执行minerU解析
            success = await self._run_mineru_command(input_path, abs_output_dir)

            if not success:
                return False, None, None

            # 查找markdown文件
            md_path = self._find_markdown_file(abs_output_dir, sanitized_filename)
            if not md_path:
                return False, None, None

            # 读取markdown内容
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # 获取images目录路径
            name_without_ext = os.path.splitext(sanitized_filename)[0]
            images_dir = os.path.join(abs_output_dir, name_without_ext, "vlm", "images")
            images_dir_path = images_dir if os.path.exists(images_dir) else None

            # 清理临时文件
            if cleanup:
                self._cleanup_temp_files(abs_output_dir, sanitized_filename, keep_images)

            return True, md_content, images_dir_path

        except Exception as e:
            self.logger.error(f"解析文件异常 {file_path}: {e}")
            return False, None, None

        finally:
            # 清理临时输入文件
            if should_cleanup_input and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception as e:
                    self.logger.warning(f"清理临时输入文件失败 {input_path}: {e}")

    def estimate_content_length(self, content: str) -> int:
        """
        估算内容长度（字符数）
        """
        return len(content.strip())

    def should_use_rag(self, content: str, threshold: int = 8000) -> bool:
        """
        判断是否需要使用RAG流程

        Args:
            content: markdown内容
            threshold: 长度阈值，超过则使用RAG
        """
        return self.estimate_content_length(content) > threshold
