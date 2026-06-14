#!/usr/bin/env python3
"""Validate missing canonical single-function patches for existing builds.

This script is the guardrail for the paper-facing validation loop: for each
bug, only patches selected in canonical_single_func_repair_runs.csv are treated
as required. A bug is complete when every generated canonical patch for that bug
has a validation row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SINGLE_DIR = ROOT / "benchmark_pilot" / "single_func_180"
CANONICAL = SINGLE_DIR / "canonical_single_func_repair_runs.csv"
DATASET = ROOT / "llm_buggy_samples_single.jsonl"
VALIDATION_ROOT = Path("/tmp/validation_builds")
PYTHON = Path(os.environ.get("VALIDATION_PYTHON", "python"))
SUMMARY = SINGLE_DIR / "missing_canonical_patch_validation_summary.csv"

DETAIL_FIELDS = [
    "method",
    "instance_id",
    "parent_sha",
    "commit_id",
    "source_files",
    "modified_func_names",
    "bug_reveal_tests",
    "bug_category",
    "status",
    "patch_generated",
    "patch_chars",
    "apply_status",
    "replace_applied",
    "finish_reason",
    "response_chars",
    "reasoning_chars",
    "elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "output_dir",
    "patch_path",
]

APPLICABLE_STATUSES = {
    "apply_check_ok",
    "apply_recount_check_ok",
    "search_replace_applied",
    "acr_original_patched_applied",
    "swe_tool_call_applied",
    "swe_str_replace_applied",
}


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged, check=True)


def latest_cumulative() -> tuple[Path | None, int]:
    best: tuple[Path | None, int] = (None, 0)
    for path in SINGLE_DIR.glob("allruns_patch_test_validation_corrected*.csv"):
        match = re.search(r"corrected(\d+)\.csv$", path.name)
        if match and int(match.group(1)) > best[1]:
            best = (path, int(match.group(1)))
    return best


def merge_cumulative(new_result: Path) -> Path:
    prev, idx = latest_cumulative()
    out = SINGLE_DIR / f"allruns_patch_test_validation_corrected{idx + 1}.csv"
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in [prev, new_result]:
        if not path:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    if fieldnames is None:
        raise RuntimeError("no cumulative CSV found")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def validated_patch_paths() -> set[str]:
    latest, _ = latest_cumulative()
    if not latest:
        return set()
    paths: set[str] = set()
    with latest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patch_path = row.get("patch_path")
            if patch_path:
                paths.add(str(Path(patch_path)))
    return paths


def method_name(row: dict[str, str]) -> str:
    return "_".join(
        part
        for part in [
            row.get("model", ""),
            row.get("agent", ""),
            row.get("run_role", ""),
            Path(row.get("patch_path", "")).parent.name,
        ]
        if part
    ).replace("-", "_")


def load_canonical_patches() -> dict[str, dict[str, dict[str, str]]]:
    by_instance: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with CANONICAL.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("patch_generated") != "True":
                continue
            if row.get("apply_status") not in APPLICABLE_STATUSES:
                continue
            if not row.get("patch_path") or row.get("patch_hash") == "missing_file":
                continue
            patch_path = Path(row["patch_path"])
            if not patch_path.exists():
                continue
            instance_id = row.get("selected_instance_id") or row.get("instance_id")
            if not instance_id:
                continue
            record = {
                "method": method_name(row),
                "instance_id": instance_id,
                "parent_sha": row.get("parent_sha", ""),
                "commit_id": row.get("commit_id", ""),
                "source_files": row.get("source_files", ""),
                "modified_func_names": row.get("modified_func_names", ""),
                "bug_reveal_tests": row.get("bug_reveal_tests", ""),
                "bug_category": row.get("bug_category", ""),
                "status": "completed",
                "patch_generated": row.get("patch_generated", ""),
                "patch_chars": "",
                "apply_status": row.get("apply_status", ""),
                "replace_applied": "",
                "finish_reason": row.get("finish_reason", ""),
                "response_chars": row.get("response_chars", ""),
                "reasoning_chars": row.get("reasoning_chars", ""),
                "elapsed_seconds": row.get("elapsed_seconds", ""),
                "prompt_tokens": row.get("prompt_tokens", ""),
                "completion_tokens": row.get("completion_tokens", ""),
                "total_tokens": row.get("total_tokens", ""),
                "output_dir": str(patch_path.parent),
                "patch_path": str(patch_path),
            }
            by_instance[instance_id][str(patch_path)] = record
    return by_instance


def worktree_for_instance(instance_id: str) -> Path:
    return VALIDATION_ROOT / instance_id.replace("pytorch_single_full_", "pytorch_single_")


def is_built_worktree(worktree: Path) -> bool:
    if not worktree.exists():
        return False
    if (worktree / "torch" / "lib" / "libtorch_cpu.so").exists():
        return True
    return any((worktree / "torch").glob("_C*.so"))


def available_same_commit_worktrees(
    records: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str], list[tuple[str, Path]]]:
    """Return available builds keyed by exact parent/commit.

    Some earlier runs placed canonical patches for the same PyTorch bug under
    different selected-instance folders.  If the parent commit is identical, an
    already-built worktree can safely validate those split patches too.
    """
    available: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    canonical = SINGLE_DIR / "canonical_single_func_repair_runs.csv"
    if canonical.exists():
        with canonical.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                instance_id = row.get("selected_instance_id") or row.get("instance_id")
                if not instance_id:
                    continue
                worktree = worktree_for_instance(instance_id)
                if not is_built_worktree(worktree):
                    continue
                key = (row.get("parent_sha", ""), row.get("commit_id", ""))
                if not all(key):
                    continue
                marker = (key[0], key[1], instance_id)
                if marker in seen:
                    continue
                seen.add(marker)
                available[key].append((instance_id, worktree))
    for instance_id, by_patch in records.items():
        worktree = worktree_for_instance(instance_id)
        if not is_built_worktree(worktree):
            continue
        for row in by_patch.values():
            key = (row.get("parent_sha", ""), row.get("commit_id", ""))
            if not all(key):
                continue
            marker = (key[0], key[1], instance_id)
            if marker in seen:
                continue
            seen.add(marker)
            available[key].append((instance_id, worktree))
    return available


def choose_worktree(
    instance_id: str,
    rows: list[dict[str, str]],
    same_commit_worktrees: dict[tuple[str, str], list[tuple[str, Path]]],
) -> tuple[Path | None, str, str]:
    exact = worktree_for_instance(instance_id)
    if is_built_worktree(exact):
        return exact, instance_id, "exact_instance_worktree"
    if not rows:
        return None, "", "missing_worktree"
    key = (rows[0].get("parent_sha", ""), rows[0].get("commit_id", ""))
    for donor_instance_id, donor_worktree in same_commit_worktrees.get(key, []):
        if is_built_worktree(donor_worktree):
            return donor_worktree, donor_instance_id, "same_commit_worktree"
    return None, "", "missing_worktree"


def write_details(instance_id: str, rows: list[dict[str, str]]) -> Path:
    short_id = instance_id.split("_full_")[1].split("_", 1)[0]
    path = SINGLE_DIR / f"tmp_missing_canonical_patches_{short_id}.csv"
    seen: dict[str, int] = {}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            method = out["method"]
            seen[method] = seen.get(method, 0) + 1
            if seen[method] > 1:
                out["method"] = f"{method}_{seen[method]}"
            writer.writerow({field: out.get(field, "") for field in DETAIL_FIELDS})
    return path


def validate(instance_id: str, details: Path, worktree: Path) -> Path:
    short_id = instance_id.split("_full_")[1].split("_", 1)[0]
    out = SINGLE_DIR / f"unvalidated_canonical_patch_validation_{short_id}.csv"
    run(
        [
            str(PYTHON),
            "agent_deploy/scripts/validate_four_method_patches.py",
            "--details",
            str(details),
            "--dataset",
            str(DATASET),
            "--out",
            str(out),
            "--output-root",
            str(SINGLE_DIR / "instances"),
            "--worktree",
            str(worktree),
            "--preserve-build",
            "--pytest-timeout",
            "600",
            "--fresh",
        ],
        cwd=ROOT,
        env={"VALIDATION_PYTHON": str(PYTHON), "PYTHONPATH": str(worktree)},
    )
    return out


def result_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def write_progress() -> None:
    run([str(PYTHON), "agent_deploy/scripts/summarize_canonical_validation_progress.py"], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument(
        "--instances",
        nargs="*",
        default=[],
        help="Only validate these selected_instance_id values.",
    )
    args = parser.parse_args()
    wanted_instances = set(args.instances)

    records = load_canonical_patches()
    validated = validated_patch_paths()
    same_commit_worktrees = available_same_commit_worktrees(records)
    candidates: list[tuple[int, str, list[dict[str, str]], Path, str, str]] = []
    summary_rows: list[dict[str, str]] = []

    total_missing = 0
    canonical_patch_count = sum(len(v) for v in records.values())
    validated_canonical_count = 0
    for instance_id, by_patch in records.items():
        if wanted_instances and instance_id not in wanted_instances:
            continue
        validated_canonical_count += sum(1 for patch_path in by_patch if patch_path in validated)
        missing = [row for patch_path, row in by_patch.items() if patch_path not in validated]
        if not missing:
            continue
        total_missing += len(missing)
        short = int(instance_id.split("_full_")[1].split("_", 1)[0])
        worktree, worktree_source, worktree_status = choose_worktree(instance_id, missing, same_commit_worktrees)
        if worktree is None or not is_built_worktree(worktree):
            for row in missing:
                summary_rows.append(
                    {
                        "instance_id": instance_id,
                        "method": row["method"],
                        "patch_path": row["patch_path"],
                        "status": "skipped_missing_worktree",
                        "worktree": str(worktree_for_instance(instance_id)),
                        "worktree_source_instance_id": "",
                        "output": "",
                    }
                )
            continue
        candidates.append((short, instance_id, missing, worktree, worktree_source, worktree_status))

    print(
        json.dumps(
            {
                "canonical_patch_paths": canonical_patch_count,
                "already_validated_canonical_patch_paths": validated_canonical_count,
                "missing_canonical_patch_paths": total_missing,
                "instances_with_existing_build": len(candidates),
                "patches_with_missing_worktree": sum(
                    1 for row in summary_rows if row["status"] == "skipped_missing_worktree"
                ),
                "instances_using_same_commit_build": sum(
                    1 for *_, worktree_status in candidates if worktree_status == "same_commit_worktree"
                ),
                "patches_using_same_commit_build": sum(
                    len(rows)
                    for _, _, rows, _, _, worktree_status in candidates
                    if worktree_status == "same_commit_worktree"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.audit_only:
        write_progress()
        return 0

    if args.max_instances:
        candidates = candidates[: args.max_instances]

    for _, instance_id, rows, worktree, worktree_source, worktree_status in sorted(candidates):
        details = write_details(instance_id, rows)
        result = validate(instance_id, details, worktree)
        if result_row_count(result) == 0:
            for row in rows:
                summary_rows.append(
                    {
                        "instance_id": instance_id,
                        "method": row["method"],
                        "patch_path": row["patch_path"],
                        "status": "skipped_no_applicable_validation_row",
                        "worktree": str(worktree),
                        "worktree_source_instance_id": worktree_source,
                        "output": str(result),
                    }
                )
            print(f"skipped merge for {instance_id}; validator produced 0 rows", flush=True)
            continue
        cumulative = merge_cumulative(result)
        for row in rows:
            summary_rows.append(
                {
                    "instance_id": instance_id,
                    "method": row["method"],
                    "patch_path": row["patch_path"],
                    "status": f"validated_{worktree_status}",
                    "worktree": str(worktree),
                    "worktree_source_instance_id": worktree_source,
                    "output": str(result),
                }
            )
        print(f"validated canonical patches for {instance_id}; cumulative={cumulative}", flush=True)

    exists = SUMMARY.exists()
    with SUMMARY.open("a", newline="", encoding="utf-8") as f:
        fields = [
            "instance_id",
            "method",
            "patch_path",
            "status",
            "worktree",
            "worktree_source_instance_id",
            "output",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(summary_rows)

    write_progress()
    print(
        json.dumps(
            {
                "validated_rows": sum(1 for row in summary_rows if row["status"] == "validated"),
                "validated_rows_any_worktree": sum(
                    1 for row in summary_rows if row["status"].startswith("validated")
                ),
                "skipped_missing_worktree": sum(1 for row in summary_rows if row["status"] == "skipped_missing_worktree"),
                "summary": str(SUMMARY),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
