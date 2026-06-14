"""
Step7-Multi: 多函数样本人工标注工具

对 defects4c_pytorch_multi_func_valid.jsonl 做人工验证。
逐条展示 commit message / 多文件 diff / 测试代码，标注三个问题：
  Q1: 是否真的是 bug fix？
  Q2: 测试是否真的暴露了 bug？
  Q3: bug 类型分类

支持断点续标（已标注的自动跳过）。
输出: annotation_results_multi.jsonl（逐条追加）
"""

import json
import os
import re
import subprocess

PYTORCH_GIT_DIR = "pytorch"
INPUT_FILE = "./defects4c_pytorch_multi_func_valid.jsonl"
OUTPUT_FILE = "./annotation_results_multi.jsonl"

BUG_CATEGORIES = {
    "1": "Logic        - 逻辑错误（条件/分支/算法错误）",
    "2": "Type         - 类型/dtype 错误",
    "3": "Shape        - 张量形状/维度错误",
    "4": "API          - API 使用错误或缺失处理",
    "5": "Numeric      - 数值精度/溢出/边界值",
    "6": "Concurrency  - 并发/异步相关",
    "7": "Other        - 其他",
}


def git_diff(parent_sha, commit_sha, src_file):
    res = subprocess.run(
        ["git", "diff", f"{parent_sha}..{commit_sha}", "--", src_file],
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        timeout=30,
    )
    return res.stdout.decode(errors="replace")


def get_test_code(test_file, func_name, commit_sha):
    res = subprocess.run(
        ["git", "show", f"{commit_sha}:{test_file}"],
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        timeout=30,
    )
    if res.returncode != 0:
        return "[测试文件不存在]"
    lines = res.stdout.decode(errors="replace").splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(
            rf"\s+def {re.escape(func_name)}\s*\(|^def {re.escape(func_name)}\s*\(",
            line,
        ):
            start = i
            break
    if start is None:
        return "[未找到测试函数]"
    indent = len(lines[start]) - len(lines[start].lstrip())
    code_lines = [lines[start]]
    for line in lines[start + 1 : start + 100]:
        stripped = line.lstrip()
        cur_indent = len(line) - len(stripped)
        if stripped and cur_indent <= indent and (
            stripped.startswith("def ") or stripped.startswith("class ")
        ):
            break
        code_lines.append(line)
    return "".join(code_lines)


def load_done():
    done = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["commit_id"]] = r
                except Exception:
                    pass
    return done


def ask(prompt, valid):
    while True:
        v = input(prompt).strip().lower()
        if v in valid:
            return v
        print(f"  请输入 {'/'.join(valid)}")


def display_sample(idx, total, sample, diffs):
    sep = "=" * 70
    print(f"\n{sep}")
    print(
        f"  [{idx}/{total}]  {sample['commit_id'][:12]}  |  files={sample.get('src_file_count', len(sample['source_files']))}"
    )
    print(f"  函数数: {sample.get('modified_func_count', 'n/a')}")
    print(f"  文件: {', '.join(os.path.basename(f) for f in sample['source_files'])}")
    print(sep)

    msg_lines = sample["commit_message"].strip().splitlines()
    print("\n【Commit Message】")
    for line in msg_lines[:20]:
        print(" ", line)
    if len(msg_lines) > 20:
        print(f"  ... (共{len(msg_lines)}行，已截断)")

    print("\n【修改函数】")
    for name in sample.get("modified_func_names", []):
        print(" ", name)

    for src_file in sample["source_files"]:
        diff = diffs[src_file]
        print(f"\n【Diff: {src_file}】")
        diff_lines = diff.splitlines()
        for line in diff_lines[:80]:
            print(line)
        if len(diff_lines) > 80:
            print(f"  ... (共{len(diff_lines)}行，已截断，按回车后可选择查看全文)")

    print("\n【Bug Reveal Tests】")
    for t in sample["bug_reveal_tests"]:
        print(" ", t)

    for t in sample["bug_reveal_tests"][:2]:
        parts = t.split("::")
        if len(parts) == 2:
            test_file, func = parts
            print(f"\n【测试代码: {func}】")
            print(get_test_code(test_file, func, sample["commit_id"]))


def annotate_sample(idx, total, sample):
    diffs = {
        src_file: git_diff(sample["parent_sha"], sample["commit_id"], src_file)
        for src_file in sample["source_files"]
    }
    display_sample(idx, total, sample, diffs)

    for src_file in sample["source_files"]:
        diff_lines = diffs[src_file].splitlines()
        if len(diff_lines) > 80:
            show = ask(f"\n  {src_file} diff 已截断，查看完整 diff？(y/n): ", ["y", "n"])
            if show == "y":
                for line in diff_lines[80:]:
                    print(line)

    print("\n" + "-" * 50)
    print("【标注】")

    q1 = ask("  Q1 是否真的是 bug fix（不是feature/重构/优化）？(y/n/s=跳过): ", ["y", "n", "s"])
    if q1 == "s":
        return None

    q2 = "n/a"
    category = "n/a"
    if q1 == "y":
        q2 = ask("  Q2 测试是否真的暴露了该 bug？(y/n): ", ["y", "n"])
        if q2 == "y":
            print("\n  Bug 类型：")
            for k, v in BUG_CATEGORIES.items():
                print(f"    {k}. {v}")
            category = ask("  Q3 选择 bug 类型 (1-7): ", list(BUG_CATEGORIES.keys()))
            category = BUG_CATEGORIES[category].split("-")[0].strip()

    note_input = input("  备注（可空，回车跳过；输入 v 从剪贴板粘贴）: ").strip()
    if note_input.lower() == "v":
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            note = root.clipboard_get().strip()
            root.destroy()
            print(f"  [剪贴板内容]: {note}")
        except Exception as e:
            print(f"  [警告] 无法读取剪贴板: {e}")
            note = ""
    else:
        note = note_input

    return {
        "commit_id": sample["commit_id"],
        "source_files": sample["source_files"],
        "modified_func_names": sample.get("modified_func_names", []),
        "modified_func_count": sample.get("modified_func_count", "n/a"),
        "q1_is_bugfix": q1,
        "q2_test_relevant": q2,
        "bug_category": category,
        "note": note,
    }


def print_stats(done):
    total = len(done)
    if total == 0:
        return
    y1 = sum(1 for r in done.values() if r["q1_is_bugfix"] == "y")
    y2 = sum(1 for r in done.values() if r["q2_test_relevant"] == "y")
    confirmed = sum(
        1
        for r in done.values()
        if r["q1_is_bugfix"] == "y" and r["q2_test_relevant"] == "y"
    )
    cats = {}
    for r in done.values():
        c = r.get("bug_category", "n/a")
        if c not in ("n/a",):
            cats[c] = cats.get(c, 0) + 1
    print(f"\n{'─' * 50}")
    print(f"  已标注: {total} | Q1确认bug: {y1} | Q2测试相关: {y2} | ✅最终确认: {confirmed}")
    if cats:
        print("  分类统计:", cats)
    print(f"{'─' * 50}")


if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        samples = [json.loads(l) for l in f if l.strip()]
    total = len(samples)

    done = load_done()
    remaining = [s for s in samples if s["commit_id"] not in done]

    print(f"\n{'=' * 70}")
    print("  PyTorch 多函数 Bug 数据集人工标注工具")
    print(f"  总计: {total} | 已标注: {len(done)} | 待标注: {len(remaining)}")
    print(f"  输出: {OUTPUT_FILE}")
    print(f"{'=' * 70}")
    print("  快捷键: y=是  n=否  s=跳过该条  Ctrl+C=保存退出")

    print_stats(done)

    try:
        for i, sample in enumerate(remaining, start=len(done) + 1):
            result = annotate_sample(i, total, sample)
            if result is None:
                print("  ⏭ 已跳过")
                continue
            done[sample["commit_id"]] = result
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print("  ✅ 已保存")
            print_stats(done)
    except KeyboardInterrupt:
        print(f"\n\n  中断退出，已保存 {len(done)} 条。下次运行自动续标。")

    print(f"\n标注完成！共 {len(done)} 条，保存在 {OUTPUT_FILE}")
    print_stats(done)
