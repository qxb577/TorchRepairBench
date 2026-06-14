

"""
Defects4C复现 - Step5：Python层双版本测试验证

策略：
  - 不整体checkout pytorch仓库（太慢且有依赖问题）
  - 用 git show SHA:file.py 只提取改动的单个Python文件
  - 替换到pip安装的torch里运行测试，测完还原
  - 全程使用 pytorch_dataset conda环境（Python 3.11 + torch 2.11）
"""

import os
import json
import gzip
import re
import subprocess
import shutil
import tempfile
from glob import glob
from tqdm import tqdm

# ===================== 路径配置 =====================
PYTORCH_GIT_DIR  = "pytorch"
PIP_TORCH_DIR    = "path/to/site-packages/torch"
PYTHON_BIN       = "python"
BACKUP_DIR       = (
    "path/to/raw_collection_workspace"
    "/defects4c_bug/step2_crawler_api_github_commit/pytorch_commit_backup"
)
CANDIDATES_FILE  = "./candidates_recent.jsonl"
OUTPUT_DATASET   = "./defects4c_pytorch_python_valid.jsonl"
PROGRESS_FILE    = "./python_progress_recent.json"
FAIL_LOG_FILE    = "./python_failed_recent.log"


# ===================== 一次性：同步测试工具文件 =====================

def sync_test_utils():
    """
    将git HEAD的纯Python测试工具文件复制到pip torch里。
    这些文件只含新增符号（如TEST_WITH_MTIA），是向后兼容的，
    不影响FAIL→PASS判断逻辑。
    只需运行一次，脚本内部自动检测。
    """
    src_dir = os.path.join(PYTORCH_GIT_DIR, "torch", "testing", "_internal")
    dst_dir = os.path.join(PIP_TORCH_DIR, "testing", "_internal")
    marker  = os.path.join(dst_dir, ".synced_from_git")

    if os.path.exists(marker):
        return  # 已同步过

    print("⚙ 首次运行：同步git HEAD测试工具文件到pip torch...")
    for fname in os.listdir(src_dir):
        if not fname.endswith(".py"):
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)

    # 同步上一级的 torch/testing/*.py（如 __init__.py 等）
    src_test = os.path.join(PYTORCH_GIT_DIR, "torch", "testing")
    dst_test  = os.path.join(PIP_TORCH_DIR, "testing")
    for fname in os.listdir(src_test):
        if fname.endswith(".py"):
            shutil.copy2(os.path.join(src_test, fname),
                         os.path.join(dst_test, fname))

    open(marker, "w").close()
    print("  ✅ 同步完成")


# ===================== 工具函数 =====================

def log_failed(sha, reason):
    with open(FAIL_LOG_FILE, "a") as f:
        f.write(f"{sha}\t{reason}\n")


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed_shas": [], "valid_samples": []}


def save_progress(processed_shas, valid_samples):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "processed_shas": list(processed_shas),
            "valid_samples": valid_samples
        }, f)


# ===================== 文件类型判断 =====================

def is_test_file(fn):
    basename = os.path.basename(fn)
    in_test_dir = "/test/" in fn or "/tests/" in fn
    return fn.endswith(".py") and (in_test_dir or basename.startswith("test_"))


def is_py_src_file(fn):
    return (
        fn.endswith(".py")
        and not is_test_file(fn)
        and (fn.startswith("torch/") or fn.startswith("functorch/"))
    )


# ===================== 从patch提取测试函数名 =====================

def extract_test_funcs(patch):
    if not patch:
        return []
    funcs = set()
    added_funcs = set()

    # 优先：提取patch中新增的测试函数（包括类方法，有缩进）
    for m in re.finditer(r"^\+\s+def (test_\w+)\s*\(", patch, re.MULTILINE):
        added_funcs.add(m.group(1))
    # 新增的顶层函数（无缩进）
    for m in re.finditer(r"^\+def (test_\w+)\s*\(", patch, re.MULTILINE):
        added_funcs.add(m.group(1))

    if added_funcs:
        return list(added_funcs)

    # 备选：hunk header里的函数名（修改了已有函数体）
    for m in re.finditer(r"@@ [^@]+ @@ def (test_\w+)", patch):
        funcs.add(m.group(1))
    return list(funcs)


# ===================== 第一步：扫描backup =====================

def scan_backup_for_candidates():
    backup_files = sorted(glob(os.path.join(BACKUP_DIR, "*.jsonl.gz")))
    if not backup_files:
        print(f"❌ 找不到backup文件: {BACKUP_DIR}")
        return []

    candidates = []
    stats = {"total": 0, "ok": 0, "no_src": 0, "multi_src": 0,
             "no_test": 0, "no_funcs": 0}

    for gz in tqdm(backup_files, desc="扫描backup"):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cd = d.get("commit_data", {})
                    files = cd.get("files", [])
                    stats["total"] += 1

                    py_src  = [f for f in files
                               if is_py_src_file(f.get("filename", ""))
                               and f.get("patch")]
                    py_test = [f for f in files
                               if is_test_file(f.get("filename", ""))
                               and f.get("patch")]

                    if len(py_src) == 0:
                        stats["no_src"] += 1; continue
                    if len(py_src) > 1:
                        stats["multi_src"] += 1; continue
                    if len(py_test) == 0:
                        stats["no_test"] += 1; continue

                    parents = cd.get("parents", [])
                    if not parents:
                        continue
                    parent_sha = parents[0].get("sha", "")
                    if len(parent_sha) < 40:
                        continue

                    commit_sha = cd.get("sha", "")
                    commit_msg = cd.get("commit", {}).get("message", "")

                    test_files_info = []
                    for tf in py_test:
                        funcs = extract_test_funcs(tf.get("patch", ""))
                        if funcs:
                            test_files_info.append({
                                "file": tf["filename"],
                                "funcs": funcs
                            })

                    if not test_files_info:
                        stats["no_funcs"] += 1; continue

                    stats["ok"] += 1
                    candidates.append({
                        "commit_id":      commit_sha,
                        "parent_sha":     parent_sha,
                        "commit_message": commit_msg,
                        "file":           py_src[0]["filename"],
                        "test_files":     test_files_info,
                    })
                except Exception:
                    continue

    print(f"\n扫描统计: 总{stats['total']} | 无源码{stats['no_src']} "
          f"| 多文件{stats['multi_src']} | 无测试{stats['no_test']} "
          f"| 无函数{stats['no_funcs']} | ✅候选{stats['ok']}")
    return candidates


# ===================== 核心：文件替换测试 =====================

def git_show_file(sha, git_path):
    """从git仓库提取指定commit的某个文件内容"""
    res = subprocess.run(
        f"git show {sha}:{git_path}",
        cwd=PYTORCH_GIT_DIR, shell=True,
        capture_output=True, timeout=30
    )
    if res.returncode != 0:
        return None
    return res.stdout  # bytes


def apply_reverse_patch(pip_path, commit_sha, src_git_path):
    """
    对pip torch当前文件应用反向补丁（即revert某个commit的修改），
    制造兼容pip版本的buggy文件。
    返回 True 表示补丁成功应用，False 表示失败（跳过该候选）。
    """
    # 获取该commit在src_git_path上的diff
    res = subprocess.run(
        f"git diff {commit_sha}^..{commit_sha} -- {src_git_path}",
        cwd=PYTORCH_GIT_DIR, shell=True,
        capture_output=True, timeout=30
    )
    if res.returncode != 0 or not res.stdout.strip():
        return False
    diff = res.stdout

    # 用 patch --reverse 应用到pip_path（tmpdir中操作，避免损坏原文件）
    with tempfile.NamedTemporaryFile(suffix=".patch", delete=False) as pf:
        pf.write(diff)
        patch_file = pf.name

    try:
        # patch 需要文件在工作目录内，用 -i 直接修改
        res2 = subprocess.run(
            ["patch", "--reverse", "--force", "-i", patch_file, pip_path],
            capture_output=True, timeout=30
        )
        return res2.returncode == 0
    finally:
        os.unlink(patch_file)


def patch_and_run_tests_buggy(src_git_path, commit_sha, test_files_info):
    """
    对pip torch当前文件应用反向补丁制造buggy版本，运行测试，返回(failed, passed)。
    测试完毕后自动还原。
    """
    if not src_git_path.startswith("torch/"):
        return [], []
    pip_path = os.path.join(PIP_TORCH_DIR, src_git_path[len("torch/"):])
    if not os.path.exists(pip_path):
        tqdm.write(f"   [patch] pip中找不到: {pip_path}")
        return [], []

    backup_path = pip_path + ".bak"
    shutil.copy2(pip_path, backup_path)

    failed, passed = [], []
    try:
        ok = apply_reverse_patch(pip_path, commit_sha, src_git_path)
        if not ok:
            tqdm.write(f"   [patch] 反向补丁应用失败，跳过")
            return [], []
        failed, passed = run_tests(test_files_info)
    finally:
        shutil.copy2(backup_path, pip_path)
        os.remove(backup_path)

    return failed, passed


def run_fixed_tests(test_files_info):
    """用pip torch当前版本（即fixed版本）直接运行测试"""
    return run_tests(test_files_info)


def run_tests(test_files_info):
    """运行指定测试函数，返回(failed, passed)"""
    failed, passed = [], []

    # 构造干净的环境：
    # - 去掉pytorch根目录（会导致import本地未编译torch源码）
    # - 保留/加入pytorch/test/目录（conftest.py和pytest_shard_custom.py在这里）
    env = os.environ.copy()
    test_dir = os.path.join(PYTORCH_GIT_DIR, "test")
    pythonpath = [p for p in env.get("PYTHONPATH", "").split(":")
                  if p and p != PYTORCH_GIT_DIR]
    if test_dir not in pythonpath:
        pythonpath.insert(0, test_dir)
    env["PYTHONPATH"] = ":".join(pythonpath)

    for tf_info in test_files_info:
        test_file = tf_info["file"]   # 例如 "test/test_optim.py"
        funcs = tf_info["funcs"]

        # 测试文件在pip torch旁边的pytorch仓库里
        test_path = os.path.join(PYTORCH_GIT_DIR, test_file)
        if not os.path.exists(test_path):
            tqdm.write(f"   [test] 找不到测试文件: {test_file}")
            continue

        for func in funcs:
            try:
                res = subprocess.run(
                    [PYTHON_BIN, "-m", "pytest", test_path,
                     "-k", func,          # 用关键字匹配，兼容类方法（无需指定类名）
                     "-x", "-q", "--tb=no", "--no-header",
                     "--import-mode=importlib"],
                    capture_output=True, timeout=90,
                    cwd="/tmp",  # 关键：不用pytorch源码目录作为cwd，防止本地torch/覆盖pip torch
                    env=env
                )
                short_id = f"{test_file}::{func}"
                # returncode=5表示没收集到测试（函数名不存在），不算失败
                if res.returncode == 0:
                    passed.append(short_id)
                elif res.returncode != 5:
                    failed.append(short_id)
            except subprocess.TimeoutExpired:
                tqdm.write(f"   [test] 超时: {func}")
            except Exception as e:
                tqdm.write(f"   [test] 异常: {e}")

    return failed, passed


# ===================== 主流程 =====================

if __name__ == "__main__":
    print("=" * 70)
    print("  Defects4C复现 - Python层双版本测试验证")
    print("  策略: git show提取单文件 + pip torch运行测试（无需编译）")
    print("=" * 70)

    # ---------- 前置：同步测试工具文件 ----------
    sync_test_utils()

    # ---------- 第一步：扫描候选（有缓存则跳过）----------
    if os.path.exists(CANDIDATES_FILE):
        print(f"\n✅ 加载候选缓存: {CANDIDATES_FILE}")
        with open(CANDIDATES_FILE) as f:
            candidates = [json.loads(l) for l in f if l.strip()]
        print(f"   候选数: {len(candidates)}")
    else:
        print("\n第一步：扫描backup...")
        candidates = scan_backup_for_candidates()
        with open(CANDIDATES_FILE, "w") as f:
            for c in candidates:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"已缓存到: {CANDIDATES_FILE}")

    # ---------- 第二步：双版本验证 ----------
    progress = load_progress()
    processed_shas = set(progress["processed_shas"])
    valid_samples   = progress["valid_samples"]
    remaining = [c for c in candidates if c["commit_id"] not in processed_shas]

    print(f"\n第二步：双版本测试验证")
    print(f"  候选: {len(candidates)} | 已处理: {len(processed_shas)} "
          f"| 待处理: {len(remaining)} | 已有效: {len(valid_samples)}\n")

    for idx, sample in enumerate(tqdm(remaining, desc="验证进度")):
        commit_sha = sample["commit_id"]
        parent_sha = sample["parent_sha"]
        src_file   = sample["file"]

        tqdm.write(f"\n[{idx+1}/{len(remaining)}] {commit_sha[:8]} | {os.path.basename(src_file)}")

        try:
            # 1. 对pip torch应用反向补丁（制造buggy版本）并测试
            tqdm.write("  1. 测试buggy版本（反向补丁）...")
            failed_buggy, _ = patch_and_run_tests_buggy(
                src_file, commit_sha, sample["test_files"]
            )
            tqdm.write(f"     失败: {len(failed_buggy)}")

            if not failed_buggy:
                tqdm.write("  ⚠ buggy无失败，跳过")
                processed_shas.add(commit_sha)
                continue

            # 2. 用pip torch当前版本（fixed）直接测试
            tqdm.write("  2. 测试fixed版本（pip torch当前）...")
            _, passed_fixed = run_fixed_tests(sample["test_files"])
            tqdm.write(f"     通过: {len(passed_fixed)}")

            # 3. 核心：FAIL→PASS
            bug_reveal = list(set(failed_buggy) & set(passed_fixed))

            if not bug_reveal:
                tqdm.write("  ⚠ 无FAIL→PASS，跳过")
                processed_shas.add(commit_sha)
                continue

            tqdm.write(f"  🎯 找到 {len(bug_reveal)} 个bug暴露测试！")
            sample["bug_reveal_tests"] = bug_reveal
            valid_samples.append(sample)

            with open(OUTPUT_DATASET, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        except Exception as e:
            tqdm.write(f"  ❌ 异常: {str(e)[:100]}")
            log_failed(commit_sha, str(e)[:100])

        processed_shas.add(commit_sha)

        if len(processed_shas) % 20 == 0:
            save_progress(processed_shas, valid_samples)
            tqdm.write(f"  💾 已处理{len(processed_shas)}条，有效{len(valid_samples)}条")

    save_progress(processed_shas, valid_samples)

    print("\n" + "=" * 70)
    print("🎉 完成！")
    print(f"  有效样本: {len(valid_samples)}")
    print(f"  输出: {os.path.abspath(OUTPUT_DATASET)}")
    print("=" * 70)
