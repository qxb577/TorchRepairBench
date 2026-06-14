"""
从candidates中筛选：
1. 只保留2024年以后的commit（与torch 2.11兼容）
2. src文件在pip torch里存在
3. 测试文件在git仓库里存在
"""
import json
import subprocess
import os
from tqdm import tqdm

CANDIDATES_FILE = "./candidates_python.jsonl"
OUTPUT_FILE     = "./candidates_recent.jsonl"
PYTORCH_GIT_DIR = "pytorch"
PIP_TORCH_DIR   = "path/to/site-packages/torch"

def get_commit_date(sha):
    """用git log获取commit日期"""
    res = subprocess.run(
        f"git log -1 --format=%ci {sha}",
        cwd=PYTORCH_GIT_DIR, shell=True,
        capture_output=True, timeout=10
    )
    if res.returncode == 0:
        return res.stdout.decode().strip()[:7]  # 返回 "2024-03" 格式
    return ""

def src_exists_in_pip(src_file):
    """检查源文件是否在pip torch里存在"""
    if src_file.startswith("torch/"):
        pip_path = os.path.join(PIP_TORCH_DIR, src_file[len("torch/"):])
        return os.path.exists(pip_path)
    return False

def test_exists_in_git(test_file):
    """检查测试文件是否在当前git仓库里存在"""
    path = os.path.join(PYTORCH_GIT_DIR, test_file)
    return os.path.exists(path)

if __name__ == "__main__":
    with open(CANDIDATES_FILE) as f:
        candidates = [json.loads(l) for l in f if l.strip()]
    print(f"原始候选: {len(candidates)}")

    filtered = []
    skip_old = 0
    skip_no_src = 0
    skip_no_test = 0

    for c in tqdm(candidates, desc="过滤"):
        # 1. 检查src文件在pip torch存在
        if not src_exists_in_pip(c["file"]):
            skip_no_src += 1
            continue

        # 2. 检查所有测试文件在git仓库存在
        all_tests_exist = all(
            test_exists_in_git(tf["file"])
            for tf in c["test_files"]
        )
        if not all_tests_exist:
            skip_no_test += 1
            continue

        # 3. 检查commit日期（2024年以后）
        date = get_commit_date(c["commit_id"])
        if not date or date < "2024-01":
            skip_old += 1
            continue

        c["commit_date"] = date
        filtered.append(c)

    print(f"\n过滤结果:")
    print(f"  src文件不存在于pip torch: {skip_no_src}")
    print(f"  测试文件不存在于git仓库:  {skip_no_test}")
    print(f"  commit太旧(2024年前):     {skip_old}")
    print(f"  ✅ 保留候选:              {len(filtered)}")

    with open(OUTPUT_FILE, "w") as f:
        for c in filtered:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\n保存到: {OUTPUT_FILE}")
