#!/usr/bin/env python3
"""Summarize latest single-function canonical repair and manual-review rates."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


BASE = Path("benchmark_pilot/single_func_180")
SEM = Path("benchmark_pilot/semantic_review")

CANONICAL = BASE / "canonical_single_func_repair_runs.csv"
VALIDATION = BASE / "allruns_patch_test_validation_corrected566.csv"
SEMANTIC = SEM / "single_passed_patches_bug_fix_author_review_canonical.csv"

RATE_OUT = BASE / "single_repair_rate_by_model_agent.csv"
SEM_RATE_OUT = SEM / "single_confirmed_repair_rate_by_model_agent_canonical.csv"
DATASET_OUT = SEM / "single_confirmed_repair_rate_dataset_summary_canonical.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "0.0%"


def norm_path(path: str) -> str:
    return str(Path(path)) if path else ""


def main() -> None:
    canonical_rows = read_csv(CANONICAL)
    validation_rows = read_csv(VALIDATION)
    semantic_rows = read_csv(SEMANTIC)

    validation_by_patch = {norm_path(r["patch_path"]): r for r in validation_rows if r.get("patch_path")}
    semantic_by_patch = {norm_path(r["patch_path"]): r for r in semantic_rows if r.get("patch_path")}

    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in canonical_rows:
        key = (row["model"], row["agent"])
        g = grouped[key]
        g["attempts"] += 1
        generated = row.get("patch_generated") == "True" and bool(row.get("patch_path"))
        if not generated:
            g["no_patch"] += 1
            continue
        g["generated"] += 1
        val = validation_by_patch.get(norm_path(row["patch_path"]))
        if not val:
            # Some canonical runs produced a patch file but failed the local
            # protocol/whitespace applicability check before validation.
            # They are generated-but-not-applicable, not missing validation.
            apply_status = row.get("apply_status", "")
            if apply_status and not apply_status.endswith("_applied") and apply_status != "applied":
                g["not_applicable"] += 1
            else:
                g["unvalidated_generated"] += 1
            continue
        if val.get("patch_applies") != "y":
            g["not_applicable"] += 1
            continue
        g["applied"] += 1
        if val.get("test_status") == "pass":
            g["passed"] += 1
        else:
            g["failed_test"] += 1

        sem = semantic_by_patch.get(norm_path(row["patch_path"]))
        if sem:
            g["manually_reviewed"] += 1
            if sem.get("confirmed_bug_fix_for_rate") in {"True", "true", "1", "yes", "confirmed"}:
                g["confirmed_bug_fix"] += 1
            else:
                g["not_confirmed_bug_fix"] += 1

    rate_fields = [
        "dataset",
        "model",
        "agent",
        "attempts",
        "generated",
        "generated_rate",
        "applied",
        "applied_rate",
        "passed",
        "test_passing_repair_rate",
        "failed_test",
        "not_applicable",
        "no_patch",
        "unvalidated_generated",
    ]
    rate_rows = []
    sem_rows = []
    for model, agent in sorted(grouped):
        g = grouped[(model, agent)]
        attempts = g["attempts"]
        rate_rows.append(
            {
                "dataset": "single",
                "model": model,
                "agent": agent,
                "attempts": attempts,
                "generated": g["generated"],
                "generated_rate": f"{g['generated'] / attempts:.4f}",
                "applied": g["applied"],
                "applied_rate": f"{g['applied'] / attempts:.4f}",
                "passed": g["passed"],
                "test_passing_repair_rate": f"{g['passed'] / attempts:.4f}",
                "failed_test": g["failed_test"],
                "not_applicable": g["not_applicable"],
                "no_patch": g["no_patch"],
                "unvalidated_generated": g["unvalidated_generated"],
            }
        )
        sem_rows.append(
            {
                "dataset": "single",
                "model": model,
                "agent": agent,
                "attempts": attempts,
                "test_passing_patches": g["passed"],
                "manually_reviewed_patches": g["manually_reviewed"],
                "confirmed_bug_fix": g["confirmed_bug_fix"],
                "not_confirmed_bug_fix": g["not_confirmed_bug_fix"],
                "confirmed_rate_among_test_passing": pct(g["confirmed_bug_fix"], g["passed"]),
                "confirmed_rate_among_attempts": f"{g['confirmed_bug_fix'] / attempts:.4f}",
            }
        )

    with RATE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rate_fields)
        writer.writeheader()
        writer.writerows(rate_rows)

    sem_fields = [
        "dataset",
        "model",
        "agent",
        "attempts",
        "test_passing_patches",
        "manually_reviewed_patches",
        "confirmed_bug_fix",
        "not_confirmed_bug_fix",
        "confirmed_rate_among_test_passing",
        "confirmed_rate_among_attempts",
    ]
    with SEM_RATE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sem_fields)
        writer.writeheader()
        writer.writerows(sem_rows)

    totals = defaultdict(int)
    for g in grouped.values():
        for field in [
            "attempts",
            "generated",
            "applied",
            "passed",
            "failed_test",
            "not_applicable",
            "no_patch",
            "unvalidated_generated",
            "manually_reviewed",
            "confirmed_bug_fix",
            "not_confirmed_bug_fix",
        ]:
            totals[field] += g[field]

    dataset_fields = [
        "dataset",
        "attempts",
        "generated",
        "applied",
        "test_passing_patches",
        "manually_reviewed_test_passing_patches",
        "confirmed_bug_fix",
        "not_confirmed_bug_fix",
        "test_passing_rate_among_attempts",
        "confirmed_rate_among_attempts",
        "confirmed_rate_among_test_passing",
        "unvalidated_generated",
    ]
    dataset_row = {
        "dataset": "single",
        "attempts": totals["attempts"],
        "generated": totals["generated"],
        "applied": totals["applied"],
        "test_passing_patches": totals["passed"],
        "manually_reviewed_test_passing_patches": totals["manually_reviewed"],
        "confirmed_bug_fix": totals["confirmed_bug_fix"],
        "not_confirmed_bug_fix": totals["not_confirmed_bug_fix"],
        "test_passing_rate_among_attempts": f"{totals['passed'] / totals['attempts']:.4f}",
        "confirmed_rate_among_attempts": f"{totals['confirmed_bug_fix'] / totals['attempts']:.4f}",
        "confirmed_rate_among_test_passing": pct(totals["confirmed_bug_fix"], totals["passed"]),
        "unvalidated_generated": totals["unvalidated_generated"],
    }
    with DATASET_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dataset_fields)
        writer.writeheader()
        writer.writerow(dataset_row)

    print(f"wrote {RATE_OUT}")
    print(f"wrote {SEM_RATE_OUT}")
    print(f"wrote {DATASET_OUT}")
    print(dataset_row)


if __name__ == "__main__":
    main()
