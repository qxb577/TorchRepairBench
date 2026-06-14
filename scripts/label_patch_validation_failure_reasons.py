#!/usr/bin/env python3
"""Add paper-style failure reason labels to patch validation results."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "benchmark_pilot" / "single_func_180" / "allruns_patch_test_validation_corrected44.csv"
DEFAULT_OUT = ROOT / "benchmark_pilot" / "single_func_180" / "allruns_patch_failure_reason_labeled_corrected44.csv"
DEFAULT_SUMMARY = ROOT / "benchmark_pilot" / "single_func_180" / "allruns_patch_failure_reason_summary_corrected44.csv"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def compact_evidence(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    priority = [
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "IndentationError",
        "NameError",
        "AttributeError",
        "TypeError",
        "RuntimeError",
        "AssertionError",
        "FAILED ",
        "ERROR collecting",
        "found no collectors",
    ]
    picked = []
    for marker in priority:
        for line in lines:
            if marker in line and line not in picked:
                picked.append(line)
                break
        if len(picked) >= 3:
            break
    if not picked:
        picked = lines[-3:]
    return " | ".join(picked)[:500]


def patch_paths(patch_path: str) -> set[str]:
    path = Path(patch_path)
    text = read_text(path)
    paths: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(parts[2].removeprefix("a/"))
                paths.add(parts[3].removeprefix("b/"))
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            paths.add(line.split(maxsplit=1)[1][2:])
    return {p for p in paths if p and p != "/dev/null"}


def classify(row: dict[str, str]) -> dict[str, str]:
    status = row.get("test_status", "")
    notes = row.get("notes", "")
    returncode = row.get("test_returncode", "")
    val_dir = Path(row.get("validation_dir", ""))
    pytest_log = read_text(val_dir / "pytest.log")
    apply_log = read_text(val_dir / "apply.log")
    evidence_text = pytest_log or apply_log or notes

    result_group = "Incorrect Patch"
    stage = "Strategy&Logic"
    reason = "test_behavior_failed"
    confidence = "medium"
    needs_manual_review = "yes"
    diagnosis_step = "Patch Comparison"
    action = "Compare generated patch against gold patch and inspect failed assertion."

    if status == "pass":
        return {
            "result_group": "Correct Patch",
            "five_stage_label": "Correct Patch",
            "failure_reason": "none",
            "confidence": "high",
            "needs_manual_review": "no",
            "diagnosis_step": "Official Test Result",
            "recommended_action": "Count as correct patch.",
            "evidence": "",
        }

    if row.get("patch_applies") != "y" or "git apply failed" in notes:
        result_group = "Invalid Patch"
        stage = "Implementation&Execution"
        reason = "patch_apply_failed"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Implementation Check"
        action = "Record as protocol/apply failure; do not treat as semantic repair failure."
    elif status == "timeout":
        stage = "Implementation&Execution"
        reason = "execution_timeout"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Trajectory Attribution"
        action = "Record timeout and inspect whether patch introduced non-termination."
    elif "Failed to load PyTorch C extensions" in pytest_log:
        result_group = "Invalid Evaluation Attempt"
        stage = "Validation&Harness Constraints"
        reason = "stale_or_missing_built_extension"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Harness Diagnosis"
        action = "Restore built extension or rerun validation; do not count as patch incorrect."
    elif "unsupported operand type(s) for |" in pytest_log:
        result_group = "Invalid Evaluation Attempt"
        stage = "Validation&Harness Constraints"
        reason = "python_version_mismatch"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Harness Diagnosis"
        action = "Rerun with the intended pytorch_dataset Python interpreter."
    elif (
        "ERROR collecting" in pytest_log
        or "found no collectors" in pytest_log
        or "ImportError while importing test module" in pytest_log
        or "ImportError while loading conftest" in pytest_log
        or "ModuleNotFoundError" in pytest_log
    ):
        result_group = "Invalid Evaluation Attempt"
        stage = "Validation&Harness Constraints"
        reason = "test_collection_or_import_error"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Harness Diagnosis"
        action = "Fix evaluation environment/test collection and rerun before counting correctness."
    elif "SyntaxError" in pytest_log or "IndentationError" in pytest_log:
        stage = "Implementation&Execution"
        reason = "syntax_error_in_patch"
        confidence = "high"
        needs_manual_review = "no"
        diagnosis_step = "Implementation Check"
        action = "Count as implementation failure."
    elif re.search(r"\b(NameError|AttributeError|TypeError)\b", pytest_log):
        stage = "Implementation&Execution"
        reason = "runtime_api_or_type_error"
        confidence = "medium"
        needs_manual_review = "yes"
        diagnosis_step = "Trajectory Attribution"
        action = "Inspect traceback to decide whether this is a coding bug or uncovered original behavior."
    elif "AssertionError" in pytest_log or "FAILED " in pytest_log:
        stage = "Strategy&Logic"
        reason = "failed_bug_revealing_assertion"
        confidence = "medium"
        needs_manual_review = "yes"
        diagnosis_step = "Patch Comparison"
        action = "Compare with gold patch to separate wrong strategy from incomplete implementation."
    elif "RuntimeError" in pytest_log:
        stage = "Strategy&Logic"
        reason = "runtime_behavior_mismatch"
        confidence = "medium"
        needs_manual_review = "yes"
        diagnosis_step = "Context Reconstruction"
        action = "Inspect failing runtime path and compare against expected invariant."
    else:
        stage = "Validation&Harness Constraints" if returncode == "4" else "Strategy&Logic"
        reason = "unknown_needs_manual_diagnosis"
        confidence = "low"
        needs_manual_review = "yes"
        diagnosis_step = "Trajectory Attribution"
        action = "Read pytest log, patch diff, and agent trajectory manually."

    source_files = {p for p in row.get("source_files", "").split(";") if p}
    touched = patch_paths(row.get("patch_path", ""))
    if (
        status != "pass"
        and touched
        and source_files
        and not (touched & source_files)
        and stage not in {"Validation&Harness Constraints", "Implementation&Execution"}
    ):
        stage = "Localization"
        reason = "patch_touched_non_target_file"
        confidence = "medium"
        needs_manual_review = "yes"
        diagnosis_step = "Patch Comparison"
        action = "Check whether the agent repaired the wrong file or a legitimate dependency."

    return {
        "result_group": result_group,
        "five_stage_label": stage,
        "failure_reason": reason,
        "confidence": confidence,
        "needs_manual_review": needs_manual_review,
        "diagnosis_step": diagnosis_step,
        "recommended_action": action,
        "evidence": compact_evidence(evidence_text),
    }


def write_summary(rows: list[dict[str, str]], summary_path: Path) -> None:
    counters = []
    for group_name, key_fn in [
        ("overall_by_stage_reason", lambda r: (r["five_stage_label"], r["failure_reason"])),
        ("by_method_stage_reason", lambda r: (r["method"], r["five_stage_label"], r["failure_reason"])),
        ("by_result_group", lambda r: (r["result_group"],)),
        ("manual_review", lambda r: (r["needs_manual_review"], r["five_stage_label"])),
    ]:
        counts = Counter(key_fn(row) for row in rows)
        for key, count in sorted(counts.items()):
            counters.append({"summary_group": group_name, "key": " | ".join(key), "count": str(count)})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["summary_group", "key", "count"])
        writer.writeheader()
        writer.writerows(counters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fields = reader.fieldnames or []
        rows = []
        for row in reader:
            row.update(classify(row))
            rows.append(row)

    extra_fields = [
        "result_group",
        "five_stage_label",
        "failure_reason",
        "confidence",
        "needs_manual_review",
        "diagnosis_step",
        "recommended_action",
        "evidence",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, args.summary)

    print({"rows": len(rows), "out": str(args.out), "summary": str(args.summary)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
