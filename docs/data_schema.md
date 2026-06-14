# Data Schema

## `tasks.jsonl`

Each line describes one benchmark task.

- `dataset`: dataset split.
- `task_id`: task identifier. For single-function tasks this is the fixing commit id; for
  multi-function tasks this is the commit id.
- `selected_instance_id`: local task instance id.
- `parent_sha`: buggy parent commit used for reconstruction.
- `commit_id`: human fixing commit id.
- `bug_category`: coarse bug category.
- `source_files`: semicolon-separated bug-related files.
- `modified_func_names`: semicolon-separated modified function names.
- `bug_reveal_tests`: semicolon-separated bug-revealing tests.

## `canonical_repair_runs.csv`

Each row is one canonical model-agent repair unit.

- `model`, `agent`, `strategy`: repair configuration.
- `run_role`: whether the row comes from a main run or a retry used to fill no-patch outputs.
- `patch_generated`: whether the agent generated a patch.
- `apply_status`: patch parsing/application status.
- `patch_path`: relative path to the released canonical patch, if available.
- `total_tokens`, `prompt_tokens`, `completion_tokens`: token usage when available.
- `elapsed_seconds`, `model_elapsed_seconds`: timing information.

## `validation_results.csv`

Each row is one executable validation result for an applicable canonical patch.

- `patch_applies`: whether the patch applies to the buggy version.
- `test_files_status`: whether target test files are available.
- `version_status`: whether the target version can be prepared.
- `test_status`: `pass` or `fail_or_env_error`.
- `test_returncode`: test command return code.
- `elapsed_seconds`: validation time.
- `notes`: short failure or environment note.

## `data/raw_index/candidate_filter_flow.csv`

Summarizes the data construction pipeline from raw PyTorch keyword commits to final benchmark tasks.

## `data/raw_index/raw_commit_index.csv`

A lightweight index of raw keyword-matched commits. It contains commit id, parent sha when available,
date, matched keyword category, and a shortened commit-message summary. It does not include full raw
GitHub API responses or full patches.
