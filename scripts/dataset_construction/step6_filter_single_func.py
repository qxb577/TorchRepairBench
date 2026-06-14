"""
Step6: 单函数修改过滤

对 defects4c_pytorch_python_valid.jsonl (291条) 补做论文 Defects4C 的
Single-Function Commit Filtering 步骤：

  论文逻辑: 只保留 fix commit 只修改了单个函数的样本。
  判断方法: 解析 git diff @@ hunk header 的上下文，提取函数/类名，
            若所有 hunk 指向同一个函数 → 单函数修改，保留。

输入:  defects4c_pytorch_python_valid.jsonl  (291条, 已通过FAIL→PASS验证)
输出:  defects4c_pytorch_single_func.jsonl   (单函数子集)
"""

import re
import os
import json
import subprocess
from tqdm import tqdm

PYTORCH_GIT_DIR = "pytorch"
INPUT_FILE  = "./defects4c_pytorch_python_valid.jsonl"
OUTPUT_FILE = "./defects4c_pytorch_single_func.jsonl"

# 匹配 @@ -a,b +c,d @@ context
HUNK_RE = re.compile(r'^@@ [^@]+ @@ ?(.*)$', re.MULTILINE)


def extract_func_name(context: str) -> str:
    """
    从 @@ hunk header 的上下文中提取最内层的函数/类名。
    优先提取 def xxx（方法/函数），其次 class xxx，最后原样返回。
    """
    ctx = context.strip()
    # def function_name( 或 def function_name:
    m = re.search(r'\bdef\s+(\w+)', ctx)
    if m:
        return "def:" + m.group(1)
    # class ClassName( 或 class ClassName:
    m = re.search(r'\bclass\s+(\w+)', ctx)
    if m:
        return "class:" + m.group(1)
    # 模块级（空 context）
    if not ctx:
        return "<module>"
    return "other:" + ctx[:60]


def get_diff(parent_sha: str, commit_sha: str, src_file: str) -> str:
    res = subprocess.run(
        f"git diff {parent_sha}..{commit_sha} -- {src_file}",
        cwd=PYTORCH_GIT_DIR, shell=True,
        capture_output=True, timeout=30
    )
    if res.returncode != 0:
        return ""
    return res.stdout.decode(errors="replace")


def is_single_func(diff: str) -> tuple[bool, set]:
    """
    判断 diff 是否为单函数修改。
    返回 (is_single, func_name_set)
    """
    if not diff.strip():
        return False, set()

    contexts = HUNK_RE.findall(diff)
    if not contexts:
        return False, set()

    names = set(extract_func_name(c) for c in contexts)
    return len(names) == 1, names


if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"输入样本数: {len(samples)}")

    kept = []
    skip_multi  = 0
    skip_no_diff = 0

    for s in tqdm(samples, desc="单函数过滤"):
        diff = get_diff(s["parent_sha"], s["commit_id"], s["file"])

        if not diff.strip():
            skip_no_diff += 1
            continue

        single, func_names = is_single_func(diff)
        if not single:
            skip_multi += 1
            continue

        # 记录识别到的函数名（方便后续分析）
        s["single_func_name"] = list(func_names)[0]
        kept.append(s)

    print(f"\n过滤结果:")
    print(f"  多函数修改 (丢弃): {skip_multi}")
    print(f"  无法获取diff (丢弃): {skip_no_diff}")
    print(f"  ✅ 单函数修改 (保留): {len(kept)}")

    with open(OUTPUT_FILE, "w") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n保存到: {OUTPUT_FILE}")
