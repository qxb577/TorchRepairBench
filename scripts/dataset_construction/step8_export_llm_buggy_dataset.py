"""
Step8: 导出可直接发送给 LLM 的 PyTorch bug-fix 数据

输入:
  - defects4c_pytorch_multi_func_valid.jsonl
  - annotation_results_multi.jsonl

输出:
  - llm_buggy_samples_multi.jsonl

导出规则:
  1. 仅保留人工确认:
     - Q1 = y
     - Q2 = y
  2. 为每条样本补充:
     - buggy 版本源码（parent commit）
     - bug reveal test 代码
     - gold patch diff
     - 可直接发给 LLM 的 prompt
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PYTORCH_GIT_DIR = Path("pytorch")
VALID_FILE = Path("./defects4c_pytorch_multi_func_valid.jsonl")
ANNOTATION_FILE = Path("./annotation_results_multi.jsonl")
OUTPUT_FILE = Path("./llm_buggy_samples_multi.jsonl")


@dataclass
class CodeBlock:
    qualname: str
    name: str
    start_lineno: int
    end_lineno: int
    code: str


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_show(rev: str, file_path: str) -> str:
    res = subprocess.run(
        ["git", "show", f"{rev}:{file_path}"],
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res.returncode != 0:
        return ""
    return res.stdout


def git_diff(parent_sha: str, commit_sha: str, file_path: str) -> str:
    res = subprocess.run(
        ["git", "diff", f"{parent_sha}..{commit_sha}", "--", file_path],
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res.returncode != 0:
        return ""
    return res.stdout


class CodeIndexer(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.class_stack: list[str] = []
        self.func_stack: list[str] = []
        self.blocks: list[CodeBlock] = []

    def _add_block(self, node: ast.AST, name: str) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            return
        qual_parts = [*self.class_stack, *self.func_stack, name]
        qualname = ".".join(qual_parts)
        code = "".join(self.lines[start - 1 : end])
        self.blocks.append(
            CodeBlock(
                qualname=qualname,
                name=name,
                start_lineno=start,
                end_lineno=end,
                code=code,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_block(node, node.name)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_block(node, node.name)
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_block(node, node.name)
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()


def build_code_index(source: str) -> list[CodeBlock]:
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    indexer = CodeIndexer(source)
    indexer.visit(tree)
    return indexer.blocks


def normalize_func_label(label: str) -> str:
    if label.startswith("def:"):
        return label[len("def:") :]
    if label.startswith("class:"):
        return label[len("class:") :]
    return label


def select_matching_blocks(blocks: list[CodeBlock], func_labels: Iterable[str]) -> list[dict]:
    selected: list[dict] = []
    seen = set()
    for label in func_labels:
        target = normalize_func_label(label)
        exact = [b for b in blocks if b.qualname == target]
        suffix = [b for b in blocks if b.qualname.endswith("." + target)]
        by_name = [b for b in blocks if b.name == target.split(".")[-1]]
        matches = exact or suffix or by_name
        if not matches:
            selected.append(
                {
                    "requested": label,
                    "matched_qualname": None,
                    "start_lineno": None,
                    "end_lineno": None,
                    "code": "",
                }
            )
            continue
        block = sorted(matches, key=lambda b: (len(b.qualname), b.start_lineno))[0]
        key = (block.qualname, block.start_lineno, block.end_lineno)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "requested": label,
                "matched_qualname": block.qualname,
                "start_lineno": block.start_lineno,
                "end_lineno": block.end_lineno,
                "code": block.code,
            }
        )
    return selected


def get_test_snippet(commit_sha: str, test_ref: str) -> dict:
    if "::" not in test_ref:
        return {"test_ref": test_ref, "file": "", "func": "", "code": ""}
    test_file, func_name = test_ref.split("::", 1)
    source = git_show(commit_sha, test_file)
    blocks = build_code_index(source)
    matches = [b for b in blocks if b.name == func_name or b.qualname.endswith("." + func_name)]
    block = sorted(matches, key=lambda b: (len(b.qualname), b.start_lineno))[0] if matches else None
    return {
        "test_ref": test_ref,
        "file": test_file,
        "func": func_name,
        "matched_qualname": block.qualname if block else None,
        "start_lineno": block.start_lineno if block else None,
        "end_lineno": block.end_lineno if block else None,
        "code": block.code if block else "",
    }


def build_buggy_sources(sample: dict) -> list[dict]:
    out = []
    for file_detail in sample.get("source_file_details", []):
        file_path = file_detail["file"]
        parent_source = git_show(sample["parent_sha"], file_path)
        blocks = build_code_index(parent_source)
        snippets = select_matching_blocks(blocks, file_detail.get("modified_func_names", []))
        out.append(
            {
                "file": file_path,
                "modified_func_names": file_detail.get("modified_func_names", []),
                "buggy_functions": snippets,
                "parent_source_available": bool(parent_source),
            }
        )
    return out


def build_gold_patches(sample: dict) -> list[dict]:
    patches = []
    for file_path in sample["source_files"]:
        patches.append(
            {
                "file": file_path,
                "diff": git_diff(sample["parent_sha"], sample["commit_id"], file_path),
            }
        )
    return patches


def build_prompt(sample: dict, buggy_sources: list[dict], test_snippets: list[dict]) -> str:
    parts = []
    parts.append("You are given a real bug-fix task from PyTorch.")
    parts.append("Please understand the bug and produce a patch for the buggy code.")
    parts.append("")
    parts.append("Bug summary:")
    parts.append(sample["commit_message"].strip())
    parts.append("")
    parts.append(f"Bug category: {sample['bug_category']}")
    parts.append(f"Modified files: {', '.join(sample['source_files'])}")
    parts.append(f"Modified functions: {', '.join(sample.get('modified_func_names', []))}")
    parts.append("")
    parts.append("Buggy code (from parent commit):")
    for item in buggy_sources:
        parts.append(f"\n### File: {item['file']}")
        for fn in item["buggy_functions"]:
            parts.append(f"\n#### Function: {fn['requested']}")
            if fn["code"]:
                parts.append("```python")
                parts.append(fn["code"].rstrip())
                parts.append("```")
            else:
                parts.append("[function source not found]")
    parts.append("")
    parts.append("Bug-revealing tests:")
    for test in test_snippets:
        parts.append(f"\n### Test: {test['test_ref']}")
        if test["code"]:
            parts.append("```python")
            parts.append(test["code"].rstrip())
            parts.append("```")
        else:
            parts.append("[test source not found]")
    parts.append("")
    parts.append("Return only the patch or the corrected code changes.")
    return "\n".join(parts).strip()


def build_export_rows(limit: int | None = None) -> list[dict]:
    valid_rows = {row["commit_id"]: row for row in load_jsonl(VALID_FILE)}
    ann_rows = {row["commit_id"]: row for row in load_jsonl(ANNOTATION_FILE)}

    exported = []
    for commit_id, ann in ann_rows.items():
        if not (ann.get("q1_is_bugfix") == "y" and ann.get("q2_test_relevant") == "y"):
            continue
        sample = valid_rows.get(commit_id)
        if sample is None:
            continue

        buggy_sources = build_buggy_sources(sample)
        test_snippets = [get_test_snippet(sample["commit_id"], t) for t in sample.get("bug_reveal_tests", [])]
        gold_patches = build_gold_patches(sample)

        row = {
            "commit_id": sample["commit_id"],
            "parent_sha": sample["parent_sha"],
            "commit_message": sample["commit_message"],
            "bug_category": ann["bug_category"],
            "source_files": sample["source_files"],
            "modified_func_names": sample.get("modified_func_names", []),
            "bug_reveal_tests": sample.get("bug_reveal_tests", []),
            "buggy_sources": buggy_sources,
            "test_snippets": test_snippets,
            "gold_patches": gold_patches,
            "llm_prompt": build_prompt(
                {
                    **sample,
                    "bug_category": ann["bug_category"],
                },
                buggy_sources,
                test_snippets,
            ),
            "annotation": {
                "q1_is_bugfix": ann["q1_is_bugfix"],
                "q2_test_relevant": ann["q2_test_relevant"],
                "bug_category": ann["bug_category"],
                "note": ann.get("note", ""),
            },
        }
        exported.append(row)
        if limit is not None and len(exported) >= limit:
            break

    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only export first N confirmed samples")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output jsonl path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_export_rows(limit=args.limit)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} samples -> {args.output}")


if __name__ == "__main__":
    main()
