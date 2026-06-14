"""
Step6c: 多文件/多函数 benchmark 候选过滤（仅 Python）

输入:
  candidates_multi_python_recent.jsonl

输出:
  defects4c_pytorch_multi_func_candidates.jsonl

设计目标：
  从 Step5b 生成的 1276 条 recent 多文件 Python 候选里，再筛出更适合做
  repair benchmark 的多函数候选，而不是保留所有“可能有关”的提交。

过滤规则（偏保守，但比单函数宽很多）：
  1. 仍然只处理 2~3 个 Python 源文件（Step5b 已保证）
  2. 测试文件数不宜太多（<= 2）
  3. 总改动行数不宜太大（<= 120）
  4. 每个源文件都必须能把改动行映射到真实函数/方法
  5. 总改动函数数控制在 2~6 个
  6. commit message 看起来要更像 bug fix，而不是 feature/support/cleanup

注意：
  这里是 benchmark 候选过滤，不是最终真值验证。
  后面仍建议继续做 FAIL->PASS 和人工核查。
"""

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Iterable, Optional, Union

from tqdm import tqdm

PYTORCH_GIT_DIR = "pytorch"
INPUT_FILE = "./candidates_multi_python_recent.jsonl"
OUTPUT_FILE = (
    "./defects4c_pytorch_multi_func_candidates.jsonl"
)

MAX_TEST_FILES = 2
MAX_TOTAL_CHANGED_LINES = 120
MIN_TOTAL_FUNC_COUNT = 2
MAX_TOTAL_FUNC_COUNT = 6


@dataclass(frozen=True)
class ChangedLines:
    old_lines: set
    new_lines: set
    hunk_count: int
    changed_line_count: int


@dataclass(frozen=True)
class FuncSpan:
    qualname: str
    lineno: int
    end_lineno: int
    depth: int


BUGFIX_STRONG_PATTERNS = [
    r"\bfix(es|ed)?\b",
    r"\bbug\b",
    r"\bcrash\b",
    r"\berror\b",
    r"\bassert(ion)?\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bfail(ure|ing|ed)?\b",
    r"\boverflow\b",
    r"\bbackward compatibility\b",
    r"\bcompatibility\b",
    r"\bdtype\b",
    r"\bshape\b",
]

FEATURE_LIKE_PREFIXES = (
    "add ",
    "[reland] support ",
    "support ",
    "allow ",
    "enable ",
    "re-enable ",
    "retire ",
    "move ",
    "decouple ",
    "sort ",
    "avoid double hash lookup",
    "not generate ",
)


def run_git(args):
    res = subprocess.run(
        args,
        cwd=PYTORCH_GIT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res.returncode != 0:
        return ""
    return res.stdout


def get_diff(parent_sha, commit_sha, src_file):
    return run_git(["git", "diff", "%s..%s" % (parent_sha, commit_sha), "--", src_file])


def get_file_at_commit(sha, src_file):
    return run_git(["git", "show", "%s:%s" % (sha, src_file)])


def looks_like_bugfix(commit_message):
    msg = (commit_message or "").strip().lower()
    first_line = msg.splitlines()[0] if msg else ""

    strong = any(re.search(p, msg) for p in BUGFIX_STRONG_PATTERNS)
    if strong:
        return True

    if first_line.startswith(FEATURE_LIKE_PREFIXES):
        return False

    # 没明显 feature 词，但也没明显 bug 词，默认不过滤进 benchmark 候选
    return False


def parse_unified_diff(diff):
    old_lines = set()
    new_lines = set()
    hunk_count = 0
    changed_line_count = 0

    old_lineno = None
    new_lineno = None

    for raw in diff.splitlines():
        if raw.startswith("@@ "):
            header = raw.split("@@")[1].strip()
            old_part, new_part = header.split()[:2]

            def parse_start(part):
                part = part[1:]
                if "," in part:
                    part = part.split(",", 1)[0]
                return int(part)

            old_lineno = parse_start(old_part)
            new_lineno = parse_start(new_part)
            hunk_count += 1
            continue

        if old_lineno is None or new_lineno is None:
            continue
        if not raw:
            old_lineno += 1
            new_lineno += 1
            continue
        if raw.startswith("\\"):
            continue
        if raw.startswith(" "):
            old_lineno += 1
            new_lineno += 1
        elif raw.startswith("-"):
            old_lines.add(old_lineno)
            old_lineno += 1
            changed_line_count += 1
        elif raw.startswith("+"):
            new_lines.add(new_lineno)
            new_lineno += 1
            changed_line_count += 1

    return ChangedLines(
        old_lines=old_lines,
        new_lines=new_lines,
        hunk_count=hunk_count,
        changed_line_count=changed_line_count,
    )


class FuncSpanCollector(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.spans = []

    def _push(self, name):
        self.stack.append(name)

    def _pop(self):
        self.stack.pop()

    def _qualname(self, node_name):
        if self.stack:
            return ".".join(self.stack + [node_name])
        return node_name

    def visit_ClassDef(self, node):
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    def visit_FunctionDef(self, node):
        self._add_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._add_func(node)

    def _add_func(self, node):
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            return
        self.spans.append(
            FuncSpan(
                qualname="def:" + self._qualname(node.name),
                lineno=node.lineno,
                end_lineno=end_lineno,
                depth=len(self.stack),
            )
        )
        self._push(node.name)
        self.generic_visit(node)
        self._pop()


def build_func_spans(source):
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    collector = FuncSpanCollector()
    collector.visit(tree)
    return collector.spans


def find_enclosing_func(line_no, spans):
    matches = [span for span in spans if span.lineno <= line_no <= span.end_lineno]
    if not matches:
        return None
    matches.sort(key=lambda s: (s.depth, s.lineno, -s.end_lineno), reverse=True)
    return matches[0].qualname


def extract_modified_funcs(parent_sha, commit_sha, src_file, diff_info):
    old_source = get_file_at_commit(parent_sha, src_file)
    new_source = get_file_at_commit(commit_sha, src_file)

    old_spans = build_func_spans(old_source)
    new_spans = build_func_spans(new_source)

    names = set()
    for line_no in diff_info.old_lines:
        name = find_enclosing_func(line_no, old_spans)
        if name is not None:
            names.add(name)
    for line_no in diff_info.new_lines:
        name = find_enclosing_func(line_no, new_spans)
        if name is not None:
            names.add(name)
    return names


if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print("=" * 70)
    print("  Step6c: 多文件/多函数 benchmark 候选过滤")
    print("=" * 70)
    print("  输入样本数:", len(samples))

    kept = []
    stats = {
        "skip_message": 0,
        "skip_test_files": 0,
        "skip_large_diff": 0,
        "skip_no_diff": 0,
        "skip_non_callable_file": 0,
        "skip_func_count": 0,
        "kept": 0,
    }

    for sample in tqdm(samples, desc="过滤 benchmark 候选"):
        if not looks_like_bugfix(sample.get("commit_message", "")):
            stats["skip_message"] += 1
            continue

        if len(sample["test_files"]) > MAX_TEST_FILES:
            stats["skip_test_files"] += 1
            continue

        if sample.get("total_src_changed_lines", 0) > MAX_TOTAL_CHANGED_LINES:
            stats["skip_large_diff"] += 1
            continue

        file_details = []
        all_funcs = set()
        no_callable_file = False

        for src_file in sample["source_files"]:
            diff = get_diff(sample["parent_sha"], sample["commit_id"], src_file)
            if not diff.strip():
                no_callable_file = True
                stats["skip_no_diff"] += 1
                break

            diff_info = parse_unified_diff(diff)
            funcs = extract_modified_funcs(
                sample["parent_sha"], sample["commit_id"], src_file, diff_info
            )
            if not funcs:
                no_callable_file = True
                stats["skip_non_callable_file"] += 1
                break

            all_funcs.update(funcs)
            file_details.append(
                {
                    "file": src_file,
                    "modified_func_names": sorted(funcs),
                    "modified_func_count": len(funcs),
                    "diff_changed_line_count": diff_info.changed_line_count,
                    "diff_hunk_count": diff_info.hunk_count,
                }
            )

        if no_callable_file:
            continue

        if not (MIN_TOTAL_FUNC_COUNT <= len(all_funcs) <= MAX_TOTAL_FUNC_COUNT):
            stats["skip_func_count"] += 1
            continue

        enriched = dict(sample)
        enriched["source_file_details"] = file_details
        enriched["modified_func_names"] = sorted(all_funcs)
        enriched["modified_func_count"] = len(all_funcs)
        kept.append(enriched)
        stats["kept"] += 1

    with open(OUTPUT_FILE, "w") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n过滤结果:")
    print("  commit message 不像 bug fix:", stats["skip_message"])
    print("  测试文件过多:", stats["skip_test_files"])
    print("  diff 过大:", stats["skip_large_diff"])
    print("  无法获取 diff:", stats["skip_no_diff"])
    print("  存在文件未映射到函数:", stats["skip_non_callable_file"])
    print("  总函数数不在范围内:", stats["skip_func_count"])
    print("  ✅ 保留候选:", stats["kept"])
    print("\n保存到:", OUTPUT_FILE)
