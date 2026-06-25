#!/usr/bin/env python3
"""Build the paper-facing canonical repair table for single-function runs.

The raw experiment folder contains exploratory runs, full runs, split runs, and
no-patch retries. This script keeps one formal result per
instance/model/agent, while preserving where that row came from.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


BASE = Path("benchmark_pilot/single_func_180")


CONFIGS = [
    {
        "model": "deepseek",
        "agent": "agentless",
        "main": ["known_location_direct_v4_pro_searchreplace_3000_run_records.jsonl"],
        "retry": ["known_location_direct_v4_pro_searchreplace_6000_retry_nopatch_run_records.jsonl"],
        "strategy": "search-replace-known-location",
    },
    {
        "model": "deepseek",
        "agent": "aider",
        "main": ["aider_repair_only_v4pro_searchreplace_3000_run_records.jsonl"],
        "retry": ["aider_repair_only_v4pro_searchreplace_short_10000_retry_nopatch_run_records.jsonl"],
        "strategy": "search-replace-known-location",
    },
    {
        "model": "deepseek",
        "agent": "autocoderover",
        "main": ["autocoderover_repair_only_v4pro_acrpatch_3000_run_records.jsonl"],
        "retry": ["autocoderover_repair_only_v4pro_acrpatch_10000_retry_nopatch_run_records.jsonl"],
        "strategy": "original-patched-known-location",
    },
    {
        "model": "deepseek",
        "agent": "swe",
        "main": ["swe_repair_only_v4pro_toolcall_3000_run_records.jsonl"],
        "retry": ["swe_repair_only_v4pro_toolcall_10000_retry_nopatch_run_records.jsonl"],
        "strategy": "tool-call-known-location",
    },
    {
        "model": "gpt",
        "agent": "agentless",
        "main": ["agentless_openai_gpt4o_searchreplace_10000_full_records.jsonl"],
        "retry": ["agentless_openai_gpt4o_searchreplace_10000_first10_nopatch_run_records.jsonl"],
        "strategy": "search-replace-known-location",
    },
    {
        "model": "gpt",
        "agent": "aider",
        "main": ["aider_openai_gpt4o_searchreplace_10000_full_records.jsonl"],
        "retry": ["aider_openai_gpt4o_searchreplace_10000_first10_nopatch_retry2_run_records.jsonl"],
        "strategy": "search-replace-known-location",
    },
    {
        "model": "gpt",
        "agent": "autocoderover",
        "main": ["autocoderover_openai_gpt4o_acrpatch_10000_remaining_from11_records.jsonl"],
        "retry": [
            "autocoderover_repair_only_openai_gpt4o_acrpatch_10000_first10_nopatch_retry2_run_records.jsonl",
            "autocoderover_openai_gpt4o_acrpatch_10000_missing5_records.jsonl",
        ],
        "strategy": "original-patched-known-location",
    },
    {
        "model": "gpt",
        "agent": "swe",
        "main": ["swe_openai_gpt4o_toolcall_10000_remaining_from11_records.jsonl"],
        "retry": [
            "swe_openai_gpt4o_toolcall_10000_first10_nopatch_retry2_run_records.jsonl",
            "swe_openai_gpt4o_toolcall_10000_missing2_records.jsonl",
        ],
        "strategy": "tool-call-known-location",
    },
]


def load_records(filename: str) -> list[dict]:
    path = BASE / filename
    if not path.exists():
        return []
    records = []
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
                records.append(row)
    return records


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


def bug_key(row: dict) -> str:
    return row.get("commit_id") or row["instance_id"]


def pick_by_bug(files: list[str]) -> dict[str, dict]:
    picked: dict[str, dict] = {}
    for filename in files:
        for row in load_records(filename):
            inst = bug_key(row)
            # Keep the first record from the configured priority order.
            picked.setdefault(inst, row)
    return picked


def has_patch(row: dict | None) -> bool:
    if not row or row.get("patch_generated") is not True:
        return False
    path = patch_path(row)
    return bool(path and Path(path).exists())


def latest_cumulative() -> Path | None:
    best: tuple[Path | None, int] = (None, 0)
    for path in BASE.glob("allruns_patch_test_validation_corrected*.csv"):
        match = re.search(r"corrected(\d+)\.csv$", path.name)
        if match and int(match.group(1)) > best[1]:
            best = (path, int(match.group(1)))
    return best[0]


def validated_patch_paths() -> set[str]:
    latest = latest_cumulative()
    if not latest:
        return set()
    paths: set[str] = set()
    with latest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patch = row.get("patch_path")
            if patch:
                paths.add(str(Path(patch)))
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
        main_by_inst = pick_by_bug(cfg["main"])
        retry_by_inst = pick_by_bug(cfg["retry"])
        instances = sorted(set(main_by_inst) | set(retry_by_inst))
        for inst in instances:
            main_row = main_by_inst.get(inst)
            retry_row = retry_by_inst.get(inst)
            selected, run_role, selection_reason = choose_record(main_row, retry_row, validated)

            if selected is None:
                continue

            pp = patch_path(selected)
            out_rows.append(
                {
                    "instance_id": inst,
                    "selected_instance_id": selected.get("instance_id", ""),
                    "model": cfg["model"],
                    "agent": cfg["agent"],
                    "strategy": cfg["strategy"],
                    "run_role": run_role,
                    "selection_reason": selection_reason,
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
                    "commit_id": selected.get("commit_id", ""),
                    "bug_category": selected.get("bug_category", ""),
                    "source_files": ";".join(selected.get("source_files", []) or []),
                    "modified_func_names": ";".join(selected.get("modified_func_names", []) or []),
                    "bug_reveal_tests": ";".join(selected.get("bug_reveal_tests", []) or []),
                }
            )
            decision_rows.append(
                {
                    "instance_id": inst,
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
                    "selection_reason": selection_reason,
                    "selected_source_record": selected.get("_source_record", ""),
                }
            )

    out_csv = BASE / "canonical_single_func_repair_runs.csv"
    decisions_csv = BASE / "canonical_single_func_selection_decisions.csv"
    summary_csv = BASE / "canonical_single_func_repair_summary.csv"

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    with decisions_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(decision_rows[0].keys()))
        writer.writeheader()
        writer.writerows(decision_rows)

    summary_rows = []
    for cfg in CONFIGS:
        rows = [r for r in out_rows if r["model"] == cfg["model"] and r["agent"] == cfg["agent"]]
        roles = Counter(r["run_role"] for r in rows)
        summary_rows.append(
            {
                "model": cfg["model"],
                "agent": cfg["agent"],
                "strategy": cfg["strategy"],
                "selected_units": len(rows),
                "patch_generated": sum(str(r["patch_generated"]) == "True" for r in rows),
                "main_selected": roles.get("main", 0),
                "no_patch_retry_fill_selected": roles.get("no_patch_retry_fill", 0),
                "total_tokens": sum(int(r["total_tokens"] or 0) for r in rows),
                "avg_elapsed_seconds": round(
                    sum(float(r["elapsed_seconds"] or 0) for r in rows) / len(rows), 3
                )
                if rows
                else "",
            }
        )
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {out_csv}")
    print(f"wrote {decisions_csv}")
    print(f"wrote {summary_csv}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
