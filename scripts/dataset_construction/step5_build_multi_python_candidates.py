"""
Step5b: 多源码文件 Python 候选生成（仅 Python）

目标：
  从 PyTorch commit backup 中重新筛一批“多函数/多源码文件”候选，
  作为后续 Python repair benchmark 的候选池。

和当前单函数主线的区别：
  - 当前主线只保留 1 个 Python 源文件改动
  - 这里允许 2~3 个 Python 源文件改动
  - 仍然要求有 Python 测试改动，并能从 patch 中提取到测试函数
  - 仍然限制为 Python-only，避免进入 C++/CUDA 环境复杂度

输出：
  1. candidates_multi_python.jsonl       原始多源码文件 Python 候选
  2. candidates_multi_python_recent.jsonl 进一步过滤后、适配当前环境的候选

后续你可以基于 recent 文件再做：
  - 多函数过滤
  - FAIL->PASS 验证
  - 人工标注
"""

import gzip
import json
import os
import re
import subprocess
from glob import glob

from tqdm import tqdm

PYTORCH_GIT_DIR = "pytorch"
PIP_TORCH_DIR = "path/to/site-packages/torch"
BACKUP_DIR = (
    "path/to/raw_collection_workspace"
    "/defects4c_bug/step2_crawler_api_github_commit/pytorch_commit_backup"
)

RAW_OUTPUT_FILE = "./candidates_multi_python.jsonl"
RECENT_OUTPUT_FILE = "./candidates_multi_python_recent.jsonl"

# 先做一个偏保守但比单函数宽很多的候选池
MIN_SRC_FILES = 2
MAX_SRC_FILES = 3
MAX_TOTAL_SRC_CHANGED_LINES = 160
MAX_TEST_FILES = 8
RECENT_SINCE = "2024-01"


def is_test_file(path):
    basename = os.path.basename(path)
    in_test_dir = "/test/" in path or "/tests/" in path
    return path.endswith(".py") and (in_test_dir or basename.startswith("test_"))


def is_py_src_file(path):
    # 这里只保留 torch 下的 Python 源文件，和你当前的 pip torch patch 流程一致
    if not (path.endswith(".py") and path.startswith("torch/") and not is_test_file(path)):
        return False
    if path.startswith("torch/testing/") or path.startswith("torch/testing/_internal/"):
        return False
    return True


def extract_test_funcs(patch):
    if not patch:
        return []

    funcs = set()
    added_funcs = set()

    for m in re.finditer(r"^\+\s+def (test_\w+)\s*\(", patch, re.MULTILINE):
        added_funcs.add(m.group(1))
    for m in re.finditer(r"^\+def (test_\w+)\s*\(", patch, re.MULTILINE):
        added_funcs.add(m.group(1))

    if added_funcs:
        return sorted(added_funcs)

    for m in re.finditer(r"@@ [^@]+ @@ def (test_\w+)", patch):
        funcs.add(m.group(1))
    return sorted(funcs)


def count_patch_changed_lines(patch):
    if not patch:
        return 0
    count = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def git_commit_date(sha):
    res = subprocess.run(
        ["git", "log", "-1", "--format=%ci", sha],
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if res.returncode == 0:
        return res.stdout.strip()[:7]
    return ""


def src_exists_in_pip(src_file):
    if not src_file.startswith("torch/"):
        return False
    pip_path = os.path.join(PIP_TORCH_DIR, src_file[len("torch/"):])
    return os.path.exists(pip_path)


def test_exists_in_git(test_file):
    return os.path.exists(os.path.join(PYTORCH_GIT_DIR, test_file))


def scan_backup_for_multi_candidates():
    backup_files = sorted(glob(os.path.join(BACKUP_DIR, "*.jsonl.gz")))
    if not backup_files:
        print(f"找不到 backup 文件: {BACKUP_DIR}")
        return []

    stats = {
        "total": 0,
        "no_src": 0,
        "too_many_or_too_few_src": 0,
        "no_test": 0,
        "too_many_tests": 0,
        "no_parent": 0,
        "no_funcs": 0,
        "too_large": 0,
        "ok": 0,
    }
    candidates = []

    for gz_file in tqdm(backup_files, desc="扫描多源码文件 Python 候选"):
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    cd = row.get("commit_data", {})
                    files = cd.get("files", [])
                    stats["total"] += 1

                    py_src = [
                        f for f in files
                        if is_py_src_file(f.get("filename", "")) and f.get("patch")
                    ]
                    py_test = [
                        f for f in files
                        if is_test_file(f.get("filename", "")) and f.get("patch")
                    ]

                    if len(py_src) == 0:
                        stats["no_src"] += 1
                        continue
                    if not (MIN_SRC_FILES <= len(py_src) <= MAX_SRC_FILES):
                        stats["too_many_or_too_few_src"] += 1
                        continue
                    if len(py_test) == 0:
                        stats["no_test"] += 1
                        continue
                    if len(py_test) > MAX_TEST_FILES:
                        stats["too_many_tests"] += 1
                        continue

                    parents = cd.get("parents", [])
                    if not parents or len(parents[0].get("sha", "")) < 40:
                        stats["no_parent"] += 1
                        continue

                    total_src_changed = sum(
                        count_patch_changed_lines(f.get("patch", "")) for f in py_src
                    )
                    if total_src_changed > MAX_TOTAL_SRC_CHANGED_LINES:
                        stats["too_large"] += 1
                        continue

                    test_files_info = []
                    for tf in py_test:
                        funcs = extract_test_funcs(tf.get("patch", ""))
                        if funcs:
                            test_files_info.append(
                                {
                                    "file": tf["filename"],
                                    "funcs": funcs,
                                }
                            )

                    if not test_files_info:
                        stats["no_funcs"] += 1
                        continue

                    candidate = {
                        "commit_id": cd.get("sha", ""),
                        "parent_sha": parents[0].get("sha", ""),
                        "commit_message": cd.get("commit", {}).get("message", ""),
                        "source_files": [f["filename"] for f in py_src],
                        "src_file_count": len(py_src),
                        "total_src_changed_lines": total_src_changed,
                        "test_files": test_files_info,
                    }
                    candidates.append(candidate)
                    stats["ok"] += 1
                except Exception:
                    continue

    print("\n原始候选扫描统计:")
    print(f"  总 commits:               {stats['total']}")
    print(f"  无 Python 源文件:         {stats['no_src']}")
    print(f"  源文件数不在范围内:       {stats['too_many_or_too_few_src']}")
    print(f"  无测试改动:               {stats['no_test']}")
    print(f"  测试文件过多:             {stats['too_many_tests']}")
    print(f"  无有效 parent:            {stats['no_parent']}")
    print(f"  无可提取测试函数:         {stats['no_funcs']}")
    print(f"  源码改动过大:             {stats['too_large']}")
    print(f"  ✅ 原始多文件候选:        {stats['ok']}")

    return candidates


def filter_recent_candidates(candidates):
    stats = {
        "skip_no_src_in_pip": 0,
        "skip_no_test_in_git": 0,
        "skip_old": 0,
        "kept": 0,
    }
    kept = []

    for c in tqdm(candidates, desc="过滤 recent 多文件候选"):
        if not all(src_exists_in_pip(src) for src in c["source_files"]):
            stats["skip_no_src_in_pip"] += 1
            continue

        if not all(test_exists_in_git(tf["file"]) for tf in c["test_files"]):
            stats["skip_no_test_in_git"] += 1
            continue

        date = git_commit_date(c["commit_id"])
        if not date or date < RECENT_SINCE:
            stats["skip_old"] += 1
            continue

        enriched = dict(c)
        enriched["commit_date"] = date
        kept.append(enriched)
        stats["kept"] += 1

    print("\nrecent 过滤统计:")
    print(f"  源文件不在 pip torch:      {stats['skip_no_src_in_pip']}")
    print(f"  测试文件不在 git 仓库:     {stats['skip_no_test_in_git']}")
    print(f"  commit 太旧:              {stats['skip_old']}")
    print(f"  ✅ recent 候选:           {stats['kept']}")

    return kept


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("=" * 70)
    print("  Step5b: 多源码文件 Python 候选生成（仅 Python）")
    print("=" * 70)
    print(f"  规则: 源文件数 {MIN_SRC_FILES}~{MAX_SRC_FILES}")
    print(f"        total_src_changed_lines <= {MAX_TOTAL_SRC_CHANGED_LINES}")
    print(f"        test_files <= {MAX_TEST_FILES}")
    print(f"        commit_date >= {RECENT_SINCE}")

    raw_candidates = scan_backup_for_multi_candidates()
    write_jsonl(RAW_OUTPUT_FILE, raw_candidates)
    print(f"\n已保存原始候选到: {RAW_OUTPUT_FILE}")

    recent_candidates = filter_recent_candidates(raw_candidates)
    write_jsonl(RECENT_OUTPUT_FILE, recent_candidates)
    print(f"已保存 recent 候选到: {RECENT_OUTPUT_FILE}")
