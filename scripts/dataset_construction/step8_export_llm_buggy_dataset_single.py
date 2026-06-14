"""
Step8-Single: 导出单函数版本的可直接发送给 LLM 的 PyTorch bug-fix 数据

输入:
  - defects4c_pytorch_single_func.jsonl
  - annotation_results.jsonl

输出:
  - llm_buggy_samples_single.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from step8_export_llm_buggy_dataset import (
    build_code_index,
    build_gold_patches,
    build_prompt,
    get_test_snippet,
    git_show,
    load_jsonl,
    select_matching_blocks,
)


VALID_FILE = Path("./defects4c_pytorch_single_func.jsonl")
ANNOTATION_FILE = Path("./annotation_results.jsonl")
OUTPUT_FILE = Path("./llm_buggy_samples_single.jsonl")


def build_buggy_source(sample: dict) -> list[dict]:
    parent_source = git_show(sample["parent_sha"], sample["file"])
    blocks = build_code_index(parent_source)
    snippets = select_matching_blocks(blocks, [sample["single_func_name"]])
    return [
        {
            "file": sample["file"],
            "modified_func_names": [sample["single_func_name"]],
            "buggy_functions": snippets,
            "parent_source_available": bool(parent_source),
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only export first N confirmed samples")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output jsonl path")
    return parser.parse_args()


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

        buggy_sources = build_buggy_source(sample)
        test_snippets = [get_test_snippet(sample["commit_id"], t) for t in sample.get("bug_reveal_tests", [])]
        gold_patches = build_gold_patches(
            {
                "parent_sha": sample["parent_sha"],
                "commit_id": sample["commit_id"],
                "source_files": [sample["file"]],
            }
        )

        row = {
            "commit_id": sample["commit_id"],
            "parent_sha": sample["parent_sha"],
            "commit_message": sample["commit_message"],
            "bug_category": ann["bug_category"],
            "source_files": [sample["file"]],
            "modified_func_names": [sample["single_func_name"]],
            "bug_reveal_tests": sample.get("bug_reveal_tests", []),
            "buggy_sources": buggy_sources,
            "test_snippets": test_snippets,
            "gold_patches": gold_patches,
            "llm_prompt": build_prompt(
                {
                    **sample,
                    "source_files": [sample["file"]],
                    "modified_func_names": [sample["single_func_name"]],
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


def main() -> None:
    args = parse_args()
    rows = build_export_rows(limit=args.limit)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"exported {len(rows)} samples -> {args.output}")


if __name__ == "__main__":
    main()
