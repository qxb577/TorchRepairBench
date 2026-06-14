#!/usr/bin/env python3
"""Restore submodule worktrees from an existing local PyTorch git cache.

This is intentionally offline-only.  It reads gitlinks from a checked-out
PyTorch worktree and checks out matching commits from .git/modules into the
submodule directories, including nested submodules such as onnx/benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MARKERS = (
    "CMakeLists.txt",
    "Makefile",
    "setup.py",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def gitlinks(git_dir: Path, commit: str) -> list[tuple[str, str]]:
    proc = run(["git", f"--git-dir={git_dir}", "ls-tree", "-r", commit])
    links: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0] == "160000":
            links.append((parts[2], parts[3]))
    return links


def has_marker(path: Path) -> bool:
    return any((path / marker).exists() for marker in MARKERS)


def hydrated_marker(path: Path) -> Path:
    return path / ".codex_hydrated_commit"


def checkout_submodule(git_dir: Path, worktree: Path, sha: str, force: bool) -> tuple[bool, str]:
    marker = hydrated_marker(worktree)
    if has_marker(worktree) and marker.exists() and marker.read_text(encoding="utf-8").strip() == sha and not force:
        return True, ""
    worktree.mkdir(parents=True, exist_ok=True)
    proc = run(
        ["git", f"--git-dir={git_dir}", f"--work-tree={worktree}", "checkout", "-f", sha, "--", "."],
        check=False,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    clean = run(["git", f"--git-dir={git_dir}", f"--work-tree={worktree}", "clean", "-fdx"], check=False)
    if clean.returncode != 0:
        return False, clean.stderr.strip()
    marker.write_text(f"{sha}\n", encoding="utf-8")
    return True, ""


def hydrate_repo(
    git_dir: Path,
    worktree: Path,
    commit: str,
    top_modules: Path,
    prefix: str,
    force: bool,
    restored: list[str],
    missing: list[str],
) -> None:
    for sha, rel in gitlinks(git_dir, commit):
        child_worktree = worktree / rel
        candidates = []
        if prefix:
            candidates.append(git_dir / "modules" / rel)
        else:
            candidates.append(top_modules / rel)
            name = Path(rel).name
            candidates.append(top_modules / "third_party" / "NNPACK_deps" / name)

        child_git_dir = next((p for p in candidates if p.is_dir()), None)
        display = f"{prefix}{rel}" if not prefix else f"{prefix}/{rel}"
        if child_git_dir is None:
            missing.append(display)
            continue

        ok, error = checkout_submodule(child_git_dir, child_worktree, sha, force)
        if not ok:
            missing.append(f"{display} checkout_failed {sha} {error}")
            continue
        restored.append(f"{display} {sha}")
        hydrate_repo(
            child_git_dir,
            child_worktree,
            sha,
            top_modules,
            display,
            force,
            restored,
            missing,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--modules-root", default=Path("pytorch/.git/modules"), type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root_git = run(["git", "-C", str(args.worktree), "rev-parse", "--git-dir"]).stdout.strip()
    root_git_dir = (args.worktree / root_git).resolve() if not root_git.startswith("/") else Path(root_git)
    commit = run(["git", "-C", str(args.worktree), "rev-parse", args.commit]).stdout.strip()

    restored: list[str] = []
    missing: list[str] = []
    hydrate_repo(root_git_dir, args.worktree, commit, args.modules_root, "", args.force, restored, missing)

    for item in restored:
        print(f"restored {item}")
    for item in missing:
        print(f"missing-cache {item}")
    print(f"summary restored={len(restored)} missing_cache={len(missing)}")


if __name__ == "__main__":
    main()
