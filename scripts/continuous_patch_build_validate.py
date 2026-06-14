#!/usr/bin/env python3
"""Continuously build PyTorch validation worktrees and test generated patches.

This script automates the manual loop used in the repair experiment:

1. Pick the next unvalidated single-function bug with applicable patches.
2. Create/hydrate a validation worktree under /tmp/validation_builds.
3. Copy known local third-party submodule caches from previous successful builds.
4. Run the CPU-only PyTorch build.
5. Validate all applicable patches for that bug.
6. Merge the result into the next corrected cumulative CSV.
7. Reset the worktree and continue to the next bug.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTORCH_REPO = Path(os.environ.get("PYTORCH_REPO", "pytorch"))
PYTHON = Path(os.environ.get("VALIDATION_PYTHON", "python"))
SINGLE_DIR = ROOT / "benchmark_pilot" / "single_func_180"
DATASET = ROOT / "llm_buggy_samples_single.jsonl"
VALIDATION_ROOT = Path("/tmp/validation_builds")
BUILD_FAILURES = SINGLE_DIR / "build_failed_instances.csv"

METHOD_FILES = [
    (
        "agentless_deepseek_v4pro_6000_retry",
        SINGLE_DIR / "known_location_direct_v4_pro_searchreplace_6000_retry_nopatch_run_records.jsonl",
    ),
    (
        "autocoderover_deepseek_v4pro_3000",
        SINGLE_DIR / "autocoderover_repair_only_v4pro_acrpatch_3000_run_records.jsonl",
    ),
    (
        "autocoderover_deepseek_v4pro_10000_retry",
        SINGLE_DIR / "autocoderover_repair_only_v4pro_acrpatch_10000_retry_nopatch_run_records.jsonl",
    ),
    (
        "aider_deepseek_v4pro_10000_retry",
        SINGLE_DIR / "aider_repair_only_v4pro_searchreplace_short_10000_retry_nopatch_run_records.jsonl",
    ),
    (
        "agentless_openai_gpt4o_10000",
        SINGLE_DIR / "agentless_openai_gpt4o_searchreplace_10000_full_records.jsonl",
    ),
    (
        "swe_openai_gpt4o_10000_first10",
        SINGLE_DIR / "swe_openai_gpt4o_toolcall_10000_first10_nopatch_retry2_run_records.jsonl",
    ),
    (
        "swe_openai_gpt4o_10000_remaining",
        SINGLE_DIR / "swe_openai_gpt4o_toolcall_10000_remaining_from11_records.jsonl",
    ),
    (
        "autocoderover_openai_gpt4o_10000_remaining",
        SINGLE_DIR / "autocoderover_openai_gpt4o_acrpatch_10000_remaining_from11_records.jsonl",
    ),
    (
        "aider_openai_gpt4o_10000",
        SINGLE_DIR / "aider_openai_gpt4o_searchreplace_10000_full_records.jsonl",
    ),
]

APPLIED_STATUSES = {
    "search_replace_applied",
    "swe_tool_call_applied",
    "acr_original_patched_applied",
    "apply_check_ok",
    "apply_recount_check_ok",
}

COPY_SUBMODULES = [
    "third_party/aiter",
    "third_party/composable_kernel",
    "third_party/fbgemm/external/asmjit",
    "third_party/fbgemm/external/composable_kernel",
    "third_party/fbgemm/external/cpuinfo",
    "third_party/fbgemm/external/cutlass",
    "third_party/fbgemm/external/googletest",
    "third_party/fbgemm/external/hipify_torch",
    "third_party/fbgemm/external/json",
    "third_party/fbgemm/third_party/cutlass",
    "third_party/fbgemm/third_party/hipify_torch",
    "third_party/flash-attention",
    "third_party/breakpad",
    "third_party/kineto/libkineto/third_party/dynolog",
    "third_party/kineto/libkineto/third_party/fmt",
    "third_party/kineto/libkineto/third_party/googletest",
    "third_party/opentelemetry-cpp",
    "third_party/onnx",
    "third_party/foxi",
    "third_party/protobuf",
    "third_party/python-peachpy",
    "third_party/tensorpipe/third_party/googletest",
    "third_party/tensorpipe/third_party/libnop",
    "third_party/tensorpipe/third_party/libuv",
    "third_party/tensorpipe/third_party/pybind11",
    "third_party/x86-simd-sort",
]

SUBMODULE_MARKERS = (
    "CMakeLists.txt",
    "Makefile",
    "setup.py",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
)

BUILD_ENV = {
    "PATH": f"{PYTHON.parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONPATH": str(ROOT / "agent_deploy" / "shims"),
    "CCACHE_DIR": "/tmp/ccache",
    "USE_CUDA": "0",
    "USE_ROCM": "0",
    "BUILD_TEST": "0",
    "USE_DISTRIBUTED": "0",
    "USE_NCCL": "0",
    "USE_KINETO": "0",
    "USE_BREAKPAD": "0",
    "USE_MKLDNN": "0",
    "USE_NNPACK": "0",
    "USE_PYTORCH_QNNPACK": "0",
    "USE_XNNPACK": "0",
    "USE_FBGEMM": "0",
    "USE_X86_SIMD_SORT": "0",
    "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
}


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=check)


def prune_worktrees() -> None:
    run(["git", "worktree", "prune"], cwd=PYTORCH_REPO, check=False)


def add_worktree_with_retry(worktree: Path, parent_sha: str) -> None:
    run(["git", "cat-file", "-e", f"{parent_sha}^{{commit}}"], cwd=PYTORCH_REPO)
    prune_worktrees()
    first = run(
        ["git", "worktree", "add", "--detach", "-f", str(worktree), parent_sha],
        cwd=PYTORCH_REPO,
        check=False,
    )
    if first.returncode == 0:
        return
    prune_worktrees()
    run(["git", "worktree", "add", "--detach", "-f", str(worktree), parent_sha], cwd=PYTORCH_REPO)


def latest_cumulative() -> tuple[Path | None, int]:
    files = sorted(SINGLE_DIR.glob("allruns_patch_test_validation_corrected*.csv"))
    best: tuple[Path | None, int] = (None, 0)
    for path in files:
        match = re.search(r"corrected(\d+)\.csv$", path.name)
        if match and int(match.group(1)) > best[1]:
            best = (path, int(match.group(1)))
    return best


def load_validated_ids(cumulative: Path | None) -> set[str]:
    if not cumulative or not cumulative.exists():
        return set()
    with cumulative.open(newline="", encoding="utf-8") as f:
        return {row["instance_id"] for row in csv.DictReader(f) if row.get("instance_id")}


def load_build_failed_ids() -> set[str]:
    if not BUILD_FAILURES.exists():
        return set()
    with BUILD_FAILURES.open(newline="", encoding="utf-8") as f:
        return {row["instance_id"] for row in csv.DictReader(f) if row.get("instance_id")}


def instance_order(instance_id: str) -> int:
    try:
        return int(instance_id.split("_full_")[1].split("_")[0])
    except Exception:
        return 9999


def patch_path_for(row: dict[str, object]) -> Path | None:
    raw = row.get("patch_path")
    if raw:
        return Path(str(raw))
    outdir = row.get("output_dir")
    if outdir:
        return Path(str(outdir)) / "patch.diff"
    return None


def collect_candidates(
    validated_ids: set[str],
    min_patches: int,
    allowed_ids: set[str] | None = None,
) -> tuple[str, list[dict[str, object]]] | None:
    by_instance: dict[str, list[dict[str, object]]] = {}
    for method, path in METHOD_FILES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if not instance_id or instance_id in validated_ids:
                    continue
                if allowed_ids is not None and str(instance_id) not in allowed_ids:
                    continue
                status = str(row.get("apply_status") or "")
                applied = status in APPLIED_STATUSES or status.endswith("_applied")
                if not (row.get("patch_generated") and applied):
                    continue
                patch_path = patch_path_for(row)
                if not patch_path or not patch_path.exists():
                    continue
                record = {
                    key: row.get(key, "")
                    for key in [
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
                    ]
                }
                record["method"] = method
                record["patch_path"] = str(patch_path)
                by_instance.setdefault(str(instance_id), []).append(record)

    choices = [
        (instance_order(instance_id), -len(rows), instance_id)
        for instance_id, rows in by_instance.items()
        if len(rows) >= min_patches
    ]
    if not choices:
        return None
    _, _, instance_id = sorted(choices)[0]
    rows = by_instance[instance_id]
    seen: dict[str, int] = {}
    for row in rows:
        method = str(row["method"])
        seen[method] = seen.get(method, 0) + 1
        if seen[method] > 1:
            row["method"] = f"{method}_{seen[method]}"
    return instance_id, rows


def join_if_list(value: object) -> str:
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return str(value or "")


def write_details(instance_id: str, rows: list[dict[str, object]]) -> Path:
    path = SINGLE_DIR / f"tmp_unvalidated_allruns_{instance_id}.csv"
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: join_if_list(row.get(field, "")) for field in fields})
    return path


def worktree_for(instance_id: str) -> Path:
    return VALIDATION_ROOT / instance_id.replace("pytorch_single_full_", "pytorch_single_")


def cache_roots(current: Path) -> list[Path]:
    roots = []
    if VALIDATION_ROOT.exists():
        roots.extend(path for path in VALIDATION_ROOT.iterdir() if path.is_dir() and path != current)
    local_validation_roots = ROOT / "benchmark_pilot" / "repos" / "validation_builds"
    if local_validation_roots.exists():
        roots.extend(path for path in local_validation_roots.iterdir() if path.is_dir())
    local_worktree_roots = ROOT / "benchmark_pilot" / "repos" / "pytorch_worktrees"
    if local_worktree_roots.exists():
        roots.extend(path for path in local_worktree_roots.iterdir() if path.is_dir())
    if PYTORCH_REPO.exists():
        roots.append(PYTORCH_REPO)
    return sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)


def has_submodule_marker(path: Path) -> bool:
    return any((path / marker).exists() for marker in SUBMODULE_MARKERS)


def usable_submodule_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and has_submodule_marker(path)


def copy_cached_submodules(worktree: Path) -> None:
    roots = cache_roots(worktree)
    for rel in COPY_SUBMODULES:
        dest = worktree / rel
        if usable_submodule_dir(dest):
            continue
        for root in roots:
            src = root / rel
            if usable_submodule_dir(src):
                dest.mkdir(parents=True, exist_ok=True)
                run(["cp", "-a", f"{src}/.", str(dest)])
                print(f"copied {rel} from {root.name}", flush=True)
                break


def ensure_disabled_submodule_markers(worktree: Path) -> None:
    """Add harmless markers for optional submodules disabled in BUILD_ENV.

    Some PyTorch revisions check for a marker file in every declared submodule
    before honoring USE_* switches.  For offline validation we cannot hydrate
    missing optional submodules such as x86-simd-sort, but with the feature
    disabled an empty marker is enough to let setup continue.
    """
    optional_placeholders = []
    if BUILD_ENV.get("USE_X86_SIMD_SORT") == "0":
        optional_placeholders.append(("third_party/x86-simd-sort", "USE_X86_SIMD_SORT=0"))
    optional_placeholders.append(("third_party/breakpad", "offline validation"))

    for rel, reason in optional_placeholders:
        target = worktree / rel
        if not has_submodule_marker(target):
            target.mkdir(parents=True, exist_ok=True)
            (target / "CMakeLists.txt").write_text(
                f"# Offline validation placeholder; {reason}.\n",
                encoding="utf-8",
            )
            print(f"created placeholder marker for {rel}", flush=True)


def apply_old_pytorch_compatibility_fixes(worktree: Path) -> None:
    """Patch temporary worktrees for host-tool compatibility.

    These edits are applied only inside /tmp validation worktrees.  Some older
    PyTorch commits call PyYAML's removed default loader API during codegen,
    which prevents us from reaching patch validation on modern environments.
    A few old commits also initialize PyTypeObject using pre-Python-3.8 field
    comments; Python 3.8 renamed tp_print to tp_vectorcall_offset and expects
    a Py_ssize_t there, so nullptr fails to compile.
    """
    yaml_loader_files = [
        worktree / "aten" / "src" / "ATen" / "cwrap_parser.py",
        worktree / "tools" / "cwrap" / "cwrap.py",
    ]
    for cwrap_parser in yaml_loader_files:
        if not cwrap_parser.exists():
            continue
        text = cwrap_parser.read_text(encoding="utf-8")
        fixed = text.replace(
            "yaml.load('\\n'.join(declaration_lines))",
            "yaml.safe_load('\\n'.join(declaration_lines))",
        )
        fixed = fixed.replace(
            'yaml.load("\\n".join(declaration_lines))',
            'yaml.safe_load("\\n".join(declaration_lines))',
        )
        fixed = fixed.replace("yaml.load(", "yaml.safe_load(")
        if fixed != text:
            cwrap_parser.write_text(fixed, encoding="utf-8")
            print(f"patched old PyYAML yaml.load call in {cwrap_parser.relative_to(worktree)}", flush=True)

    torch_csrc = worktree / "torch" / "csrc"
    pytype_files = sorted(torch_csrc.rglob("*.cpp")) if torch_csrc.exists() else []
    for pytype_cpp in pytype_files:
        text = pytype_cpp.read_text(encoding="utf-8", errors="replace")
        fixed = re.sub(
            r"(?m)^(\s*)nullptr,(\s*/\*\s*tp_print\s*\*/)",
            r"\g<1>0,\g<2>",
            text,
        )
        if fixed != text:
            pytype_cpp.write_text(fixed, encoding="utf-8")
            print(f"patched old PyTypeObject tp_print slot in {pytype_cpp.relative_to(worktree)}", flush=True)


def reset_incompatible_cmake_cache(worktree: Path) -> None:
    cache = worktree / "build" / "CMakeCache.txt"
    if not cache.exists():
        return
    text = cache.read_text(encoding="utf-8", errors="replace")
    if BUILD_ENV.get("USE_BREAKPAD") == "0" and "USE_BREAKPAD:BOOL=ON" in text:
        shutil.rmtree(worktree / "build", ignore_errors=True)
        print("removed stale build cache with USE_BREAKPAD=ON", flush=True)
        return
    if str(PYTHON) not in text and ("PYTHON_EXECUTABLE:FILEPATH=" in text or "Python_EXECUTABLE:FILEPATH=" in text):
        shutil.rmtree(worktree / "build", ignore_errors=True)
        print(f"removed stale build cache with different Python executable; expected {PYTHON}", flush=True)


def build_worktree(worktree: Path, parent_sha: str) -> None:
    if not worktree.exists():
        add_worktree_with_retry(worktree, parent_sha)
    else:
        run(["git", "checkout", "--detach", "-f", parent_sha], cwd=worktree)
    run(["git", "reset", "--hard", parent_sha], cwd=worktree)
    hydrate = run(
        [str(PYTHON), "agent_deploy/scripts/hydrate_local_submodules.py", "--worktree", str(worktree)],
        cwd=ROOT,
        check=False,
    )
    if hydrate.returncode != 0:
        print(
            f"hydrate exited with {hydrate.returncode}; continuing with cached submodule copy",
            flush=True,
        )
    copy_cached_submodules(worktree)
    ensure_disabled_submodule_markers(worktree)
    apply_old_pytorch_compatibility_fixes(worktree)
    reset_incompatible_cmake_cache(worktree)
    run([str(PYTHON), "setup.py", "develop"], cwd=worktree, env=BUILD_ENV)


def validate(instance_id: str, details: Path, worktree: Path) -> Path:
    short_id = instance_id.split("_full_")[1].split("_", 1)[0]
    out = SINGLE_DIR / f"unvalidated_allruns_patch_validation_{short_id}.csv"
    env = {
        "VALIDATION_PYTHON": str(PYTHON),
        "PYTHONPATH": str(worktree),
    }
    run(
        [
            str(PYTHON),
            "agent_deploy/scripts/validate_four_method_patches.py",
            "--details",
            str(details),
            "--dataset",
            "llm_buggy_samples_single.jsonl",
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
        env=env,
    )
    return out


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
        raise RuntimeError("no CSV fieldnames found")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def summarize_csv(path: Path) -> tuple[int, int, Counter[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows), len({row["instance_id"] for row in rows}), Counter(row["test_status"] for row in rows)


def append_logs(instance_id: str, details: Path, worktree: Path, result: Path, cumulative: Path) -> None:
    rows = list(csv.DictReader(result.open(newline="", encoding="utf-8")))
    counts = Counter(row["test_status"] for row in rows)
    total_rows, instances, total_counts = summarize_csv(cumulative)
    methods = ", ".join(f"{row['method']}={row['test_status']}" for row in rows)
    block = (
        f"\n### Continuous Patch Validation: {instance_id}\n\n"
        f"- details: `{details}`\n"
        f"- worktree: `{worktree}`\n"
        f"- validation output: `{result}`\n"
        f"- merged cumulative: `{cumulative}`\n"
        f"- attempted patches: `{len(rows)}`\n"
        f"- pass: `{counts.get('pass', 0)}`\n"
        f"- fail_or_env_error: `{counts.get('fail_or_env_error', 0)}`\n"
        f"- per-method: {methods}\n"
        f"- cumulative rows: `{total_rows}`\n"
        f"- cumulative instances: `{instances}`\n"
        f"- cumulative pass: `{total_counts.get('pass', 0)}`\n"
        f"- cumulative fail_or_env_error: `{total_counts.get('fail_or_env_error', 0)}`\n"
    )
    for log in [ROOT / "handoff_note.md", ROOT / "experiment_status_log.md"]:
        with log.open("a", encoding="utf-8") as f:
            f.write(block)


def record_build_failure(
    instance_id: str,
    parent_sha: str,
    details: Path,
    worktree: Path,
    error: subprocess.CalledProcessError,
) -> None:
    exists = BUILD_FAILURES.exists()
    fields = [
        "instance_id",
        "parent_sha",
        "details",
        "worktree",
        "returncode",
        "cmd",
        "failed_at",
        "notes",
    ]
    with BUILD_FAILURES.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "instance_id": instance_id,
                "parent_sha": parent_sha,
                "details": str(details),
                "worktree": str(worktree),
                "returncode": error.returncode,
                "cmd": " ".join(str(part) for part in error.cmd),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "notes": "build failed before patch validation; excluded from automatic retries",
            }
        )

    block = (
        f"\n### Continuous Patch Validation Build Failure: {instance_id}\n\n"
        f"- details: `{details}`\n"
        f"- worktree: `{worktree}`\n"
        f"- parent: `{parent_sha}`\n"
        f"- returncode: `{error.returncode}`\n"
        f"- command: `{' '.join(str(part) for part in error.cmd)}`\n"
        f"- decision: recorded in `{BUILD_FAILURES}` and skipped in future automatic rounds; "
        f"not counted as repair correctness.\n"
    )
    for log in [ROOT / "handoff_note.md", ROOT / "experiment_status_log.md"]:
        with log.open("a", encoding="utf-8") as f:
            f.write(block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=5, help="Number of bug parents to build and validate.")
    parser.add_argument("--min-patches", type=int, default=2, help="Only pick instances with at least this many patches.")
    parser.add_argument(
        "--retry-build-failed",
        action="store_true",
        help="Retry instances recorded in build_failed_instances.csv during this run.",
    )
    parser.add_argument(
        "--instance-ids",
        default="",
        help="Comma-separated instance ids to consider, in addition to the normal validation filters.",
    )
    args = parser.parse_args()
    allowed_ids = {part.strip() for part in args.instance_ids.split(",") if part.strip()} or None

    failed_this_run: set[str] = set()
    for round_idx in range(1, args.max_rounds + 1):
        cumulative, _ = latest_cumulative()
        validated_ids = load_validated_ids(cumulative) | failed_this_run
        if not args.retry_build_failed:
            validated_ids |= load_build_failed_ids()
        choice = collect_candidates(validated_ids, args.min_patches, allowed_ids=allowed_ids)
        if not choice:
            print("No unvalidated candidate found.", flush=True)
            return 0
        instance_id, rows = choice
        first = rows[0]
        parent_sha = str(first["parent_sha"])
        details = write_details(instance_id, rows)
        worktree = worktree_for(instance_id)
        print(
            f"\n=== round {round_idx}: {instance_id} patches={len(rows)} parent={parent_sha} ===",
            flush=True,
        )
        started = time.monotonic()
        try:
            build_worktree(worktree, parent_sha)
        except subprocess.CalledProcessError as error:
            record_build_failure(instance_id, parent_sha, details, worktree, error)
            failed_this_run.add(instance_id)
            print(
                f"build failed for {instance_id}; recorded and continuing to next candidate",
                flush=True,
            )
            continue
        result = validate(instance_id, details, worktree)
        cumulative_out = merge_cumulative(result)
        run(["git", "reset", "--hard", parent_sha], cwd=worktree)
        append_logs(instance_id, details, worktree, result, cumulative_out)
        print(
            f"completed {instance_id} in {time.monotonic() - started:.1f}s; cumulative={cumulative_out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
