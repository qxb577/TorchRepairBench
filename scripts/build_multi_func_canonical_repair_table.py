#!/usr/bin/env python3
"""Build canonical repair tables for multi-function runs.

One row represents one paper-facing experiment unit:
commit_id + model + agent + strategy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import glob
import os
from collections import Counter
from pathlib import Path


BASE = Path("benchmark_pilot/multi_func_160")


CONFIGS = [
    {
        "model": "deepseek",
        "agent": "agentless",
        "main": ["agentless_searchreplace_v4pro_3000_editregion_pilot_records.jsonl"],
        "retry": ["agentless_searchreplace_v4pro_10000_retry_3000_nopatch_editregion_records.jsonl"],
        "strategy": "search-replace-known-location-editregion",
    },
    {
        "model": "deepseek",
        "agent": "aider",
        "main": ["aider_searchreplace_short_v4pro_3000_editregion_pilot_records.jsonl"],
        "retry": [
            "aider_searchreplace_short_v4pro_10000_retry_3000_nopatch_editregion_records.jsonl",
            "aider_searchreplace_short_v4pro_10000_retry_3000_nopatch_editregion_from76_records.jsonl",
        ],
        "strategy": "search-replace-short-known-location-editregion",
    },
    {
        "model": "deepseek",
        "agent": "autocoderover",
        "main": ["autocoderover_acrpatch_short_v4pro_3000_editregion_pilot_records.jsonl"],
        "retry": ["autocoderover_acrpatch_short_v4pro_10000_retry_3000_nopatch_editregion_records.jsonl"],
        "strategy": "original-patched-short-known-location-editregion",
    },
    {
        "model": "deepseek",
        "agent": "swe",
        "main": ["swe_toolcall_short_v4pro_3000_editregion_pilot_records.jsonl"],
        "retry": ["swe_toolcall_short_v4pro_10000_retry_3000_nopatch_editregion_records.jsonl"],
        "strategy": "tool-call-short-known-location-editregion",
    },
    {
        "model": "gpt",
        "agent": "agentless",
        "main": [
            "agentless_openai_gpt4o_searchreplace_10000_first10_editregion_records.jsonl",
            "agentless_openai_gpt4o_searchreplace_10000_remaining_from11_editregion_records.jsonl",
        ],
        "retry": [],
        "strategy": "search-replace-known-location-editregion",
    },
    {
        "model": "gpt",
        "agent": "aider",
        "main": [
            "aider_openai_gpt4o_searchreplace_short_10000_first10_editregion_records.jsonl",
            "aider_openai_gpt4o_searchreplace_short_10000_remaining_from11_editregion_records.jsonl",
        ],
        "retry": ["aider_openai_gpt4o_searchreplace_short_10000_retry_0090_0109_editregion_records.jsonl"],
        "strategy": "search-replace-short-known-location-editregion",
    },
    {
        "model": "gpt",
        "agent": "autocoderover",
        "main": [
            "autocoderover_openai_gpt4o_acrpatch_short_10000_first10_editregion_records.jsonl",
            "autocoderover_openai_gpt4o_acrpatch_short_10000_remaining_from11_editregion_records.jsonl",
        ],
        "retry": ["autocoderover_openai_gpt4o_acrpatch_short_10000_retry_0083_0107_editregion_records.jsonl"],
        "strategy": "original-patched-short-known-location-editregion",
    },
    {
        "model": "gpt",
        "agent": "swe",
        "main": [
            "swe_openai_gpt4o_toolcall_short_10000_first10_editregion_records.jsonl",
            "swe_openai_gpt4o_toolcall_short_10000_remaining_from11_editregion_records.jsonl",
        ],
        "retry": [],
        "strategy": "tool-call-short-known-location-editregion",
    },
]


def load_records(filename: str) -> list[dict]:
    path = BASE / filename
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("instance_id"):
                row["_source_record"] = filename
                rows.append(row)
    return rows


def bug_key(row: dict) -> str:
    return row.get("commit_id") or row["instance_id"]


def patch_path(row: dict) -> str:
    if row.get("patch_path"):
        return row["patch_path"]
    if row.get("output_dir"):
        return str(Path(row["output_dir"]) / "patch.diff")
    return ""


def patch_hash(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return "missing_file"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def has_patch(row: dict | None) -> bool:
    if not row or row.get("patch_generated") is not True:
        return False
    path = patch_path(row)
    return bool(path and Path(path).exists())


def pick_by_bug(files: list[str]) -> dict[str, dict]:
    picked = {}
    for filename in files:
        for row in load_records(filename):
            picked.setdefault(bug_key(row), row)
    return picked


def validated_patch_paths() -> set[str]:
    paths: set[str] = set()
    for raw in glob.glob(str(BASE / "*validation*.csv")):
        path = Path(raw)
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    patch = row.get("patch_path")
                    if patch:
                        paths.add(str(Path(patch)))
        except Exception:
            continue
    return paths


def choose_record(
    main_row: dict | None,
    retry_row: dict | None,
    validated: set[str],
) -> tuple[dict | None, str, str]:
    candidates: list[tuple[dict, str]] = []
    if main_row is not None:
        candidates.append((main_row, "main"))
    if retry_row is not None:
        candidates.append((retry_row, "no_patch_retry_fill"))

    patched = [(row, role) for row, role in candidates if has_patch(row)]
    for row, role in patched:
        if str(Path(patch_path(row))) in validated:
            return row, role, f"{role}_selected_validated_first"

    if has_patch(main_row):
        return main_row, "main", "main_selected"
    if retry_row is not None:
        return retry_row, "no_patch_retry_fill", "main_no_patch_or_missing_retry_selected"
    if main_row is not None:
        return main_row, "main", "main_selected_no_patch"
    return None, "", ""


def main() -> None:
    out_rows = []
    decision_rows = []
    validated = validated_patch_paths()

    for cfg in CONFIGS:
        main_by_bug = pick_by_bug(cfg["main"])
        retry_by_bug = pick_by_bug(cfg["retry"])
        for key in sorted(set(main_by_bug) | set(retry_by_bug)):
            main_row = main_by_bug.get(key)
            retry_row = retry_by_bug.get(key)
            selected, run_role, reason = choose_record(main_row, retry_row, validated)
            if selected is None:
                continue

            pp = patch_path(selected)
            out_rows.append(
                {
                    "commit_id": selected.get("commit_id") or key,
                    "selected_instance_id": selected.get("instance_id", ""),
                    "model": cfg["model"],
                    "agent": cfg["agent"],
                    "strategy": cfg["strategy"],
                    "run_role": run_role,
                    "selection_reason": reason,
                    "source_record": selected.get("_source_record", ""),
                    "main_source_records": "|".join(cfg["main"]),
                    "retry_source_records": "|".join(cfg["retry"]),
                    "patch_generated": selected.get("patch_generated", ""),
                    "apply_status": selected.get("apply_status", ""),
                    "patch_path": pp,
                    "patch_hash": patch_hash(pp),
                    "total_tokens": selected.get("total_tokens", ""),
                    "prompt_tokens": selected.get("prompt_tokens", ""),
                    "completion_tokens": selected.get("completion_tokens", ""),
                    "elapsed_seconds": selected.get("elapsed_seconds", ""),
                    "model_elapsed_seconds": selected.get("model_elapsed_seconds", ""),
                    "finish_reason": selected.get("finish_reason", ""),
                    "response_chars": selected.get("response_chars", ""),
                    "reasoning_chars": selected.get("reasoning_chars", ""),
                    "parent_sha": selected.get("parent_sha", ""),
                    "bug_category": selected.get("bug_category", ""),
                    "source_files": ";".join(selected.get("source_files", []) or []),
                    "modified_func_names": ";".join(selected.get("modified_func_names", []) or []),
                    "bug_reveal_tests": ";".join(selected.get("bug_reveal_tests", []) or []),
                }
            )
            decision_rows.append(
                {
                    "commit_id": selected.get("commit_id") or key,
                    "selected_instance_id": selected.get("instance_id", ""),
                    "model": cfg["model"],
                    "agent": cfg["agent"],
                    "main_present": main_row is not None,
                    "main_patch_generated": has_patch(main_row),
                    "main_patch_validated": bool(
                        main_row and patch_path(main_row) and str(Path(patch_path(main_row))) in validated
                    ),
                    "retry_present": retry_row is not None,
                    "retry_patch_generated": has_patch(retry_row),
                    "retry_patch_validated": bool(
                        retry_row and patch_path(retry_row) and str(Path(patch_path(retry_row))) in validated
                    ),
                    "selected_role": run_role,
                    "selection_reason": reason,
                    "selected_source_record": selected.get("_source_record", ""),
                }
            )

    runs_csv = BASE / "canonical_multi_func_repair_runs.csv"
    decisions_csv = BASE / "canonical_multi_func_selection_decisions.csv"
    summary_csv = BASE / "canonical_multi_func_repair_summary.csv"
    generation_csv = BASE / "canonical_multi_func_patch_generation_by_agent.csv"

    with runs_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    with decisions_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(decision_rows[0].keys()))
        writer.writeheader()
        writer.writerows(decision_rows)

    summary_rows = []
    for cfg in CONFIGS:
        rows = [r for r in out_rows if r["model"] == cfg["model"] and r["agent"] == cfg["agent"]]
        patch_rows = [
            r
            for r in rows
            if str(r["patch_generated"]) == "True" and r["patch_path"] and r["patch_hash"] != "missing_file"
        ]
        roles = Counter(r["run_role"] for r in patch_rows)
        bug_units = len({r["commit_id"] for r in rows})
        patch_bugs = len({r["commit_id"] for r in patch_rows})
        summary_rows.append(
            {
                "model": cfg["model"],
                "agent": cfg["agent"],
                "strategy": cfg["strategy"],
                "bug_units": bug_units,
                "patch_generated_bugs": patch_bugs,
                "no_patch_bugs": bug_units - patch_bugs,
                "main_patch_bugs": roles.get("main", 0),
                "retry_fill_patch_bugs": roles.get("no_patch_retry_fill", 0),
                "patch_generation_rate": f"{patch_bugs / bug_units:.4f}" if bug_units else "",
                "total_tokens": sum(int(r["total_tokens"] or 0) for r in rows),
                "avg_elapsed_seconds": round(
                    sum(float(r["elapsed_seconds"] or 0) for r in rows) / len(rows), 3
                )
                if rows
                else "",
            }
        )

    for path in [summary_csv, generation_csv]:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"wrote {runs_csv}")
    print(f"wrote {decisions_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {generation_csv}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
