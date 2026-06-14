#!/usr/bin/env python3
"""Summarize validation progress for canonical single-function patches."""

from __future__ import annotations

import csv
import glob
import os
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("benchmark_pilot/single_func_180")

APPLICABLE_STATUSES = {
    "apply_check_ok",
    "apply_recount_check_ok",
    "search_replace_applied",
    "acr_original_patched_applied",
    "swe_tool_call_applied",
    "swe_str_replace_applied",
}


def latest_validation_csv() -> Path:
    files = glob.glob(str(BASE / "allruns_patch_test_validation_corrected*.csv"))
    if not files:
        raise SystemExit("no corrected validation CSV found")
    return Path(max(files, key=os.path.getmtime))


def main() -> None:
    canonical_path = BASE / "canonical_single_func_repair_runs.csv"
    validation_path = latest_validation_csv()

    canonical_rows = list(csv.DictReader(canonical_path.open(encoding="utf-8")))
    validation_rows = list(csv.DictReader(validation_path.open(encoding="utf-8")))

    latest_by_path = {}
    for row in validation_rows:
        patch_path = row.get("patch_path")
        if patch_path:
            latest_by_path[patch_path] = row

    by_bug = defaultdict(list)
    for row in canonical_rows:
        if row.get("patch_generated") != "True":
            continue
        if row.get("apply_status") not in APPLICABLE_STATUSES:
            continue
        if not row.get("patch_path") or row.get("patch_hash") == "missing_file":
            continue
        by_bug[row["commit_id"] or row["instance_id"]].append(row)

    progress_rows = []
    missing_rows = []
    for bug_key, rows in sorted(by_bug.items()):
        statuses = Counter()
        validated = 0
        instance_ids = sorted({r["selected_instance_id"] or r["instance_id"] for r in rows})
        for row in rows:
            result = latest_by_path.get(row["patch_path"])
            if result:
                validated += 1
                statuses[result.get("test_status", "")] += 1
            else:
                statuses["unvalidated"] += 1
                missing_rows.append(
                    {
                        "commit_id": bug_key,
                        "selected_instance_id": row["selected_instance_id"],
                        "model": row["model"],
                        "agent": row["agent"],
                        "run_role": row["run_role"],
                        "source_record": row["source_record"],
                        "patch_path": row["patch_path"],
                    }
                )
        total = len(rows)
        progress_rows.append(
            {
                "commit_id": bug_key,
                "selected_instance_ids": "|".join(instance_ids),
                "canonical_patch_count": total,
                "validated_patch_count": validated,
                "missing_patch_count": total - validated,
                "complete": "yes" if validated == total else "no",
                "pass": statuses.get("pass", 0),
                "fail_or_env_error": statuses.get("fail_or_env_error", 0),
                "not_run": statuses.get("not_run", 0),
                "unvalidated": statuses.get("unvalidated", 0),
            }
        )

    progress_path = BASE / "canonical_single_func_validation_progress.csv"
    missing_path = BASE / "canonical_single_func_missing_validation_patches.csv"
    summary_path = BASE / "canonical_single_func_validation_progress_summary.csv"

    with progress_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(progress_rows[0].keys()))
        writer.writeheader()
        writer.writerows(progress_rows)

    with missing_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "commit_id",
            "selected_instance_id",
            "model",
            "agent",
            "run_role",
            "source_record",
            "patch_path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    summary = {
        "validation_csv": str(validation_path),
        "canonical_patch_bugs": len(progress_rows),
        "complete_bugs": sum(r["complete"] == "yes" for r in progress_rows),
        "partial_bugs": sum(
            r["complete"] == "no" and int(r["validated_patch_count"]) > 0
            for r in progress_rows
        ),
        "not_started_bugs": sum(int(r["validated_patch_count"]) == 0 for r in progress_rows),
        "canonical_patch_rows": sum(int(r["canonical_patch_count"]) for r in progress_rows),
        "validated_patch_rows": sum(int(r["validated_patch_count"]) for r in progress_rows),
        "missing_patch_rows": sum(int(r["missing_patch_count"]) for r in progress_rows),
        "pass_rows": sum(int(r["pass"]) for r in progress_rows),
        "fail_or_env_error_rows": sum(int(r["fail_or_env_error"]) for r in progress_rows),
        "not_run_rows": sum(int(r["not_run"]) for r in progress_rows),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"wrote {progress_path}")
    print(f"wrote {missing_path}")
    print(f"wrote {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
