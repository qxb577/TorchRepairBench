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

## `repair_rate_by_model_agent.csv`

Summarizes canonical repair outcomes by dataset, model, and agent.

- `attempts`: number of bug-level model-agent repair units.
- `generated`: number of units that produced a patch.
- `generated_rate`: patch generation rate over all attempts.
- `applied`: number of generated patches that were applicable.
- `applied_rate`: applicable-patch rate over all attempts.
- `passed`: number of patches that passed executable bug-revealing tests.
- `test_passing_repair_rate` or `repair_rate`: test-passing repair rate over all attempts.
- `failed_test`, `not_applicable`, `unvalidated_generated`: remaining outcome counts.

## `data/semantic_review/*.csv` and `data/single_func_180/semantic_review/*.csv`

These files provide an author-perspective semantic review of test-passing patches. Test-failed
patches are not included in the manual semantic-review denominator.

- `*_author_review.csv`: patch-level semantic review records.
- `*_agent_summary.csv`: confirmed bug-fix counts grouped by dataset, model, and agent.
- `*_dataset_summary.csv`: dataset-level confirmed bug-fix summary.
- `confirmed_bug_fix`: number of test-passing patches judged to repair the underlying bug.
- `not_confirmed` or `not_confirmed_bug_fix`: test-passing patches not confirmed as real bug fixes.
- `confirmed_repair_rate_by_model_agent.csv`: final single-function model-agent confirmed fix summary.
- `confirmed_repair_rate_dataset_summary.csv`: final single-function dataset-level confirmed fix summary.

## `data/raw_index/candidate_filter_flow.csv`

Summarizes the data construction pipeline from raw PyTorch keyword commits to final benchmark tasks.

## `data/raw_index/raw_commit_index.csv`

A lightweight index of raw keyword-matched commits. It contains commit id, parent sha when available,
date, matched keyword category, and a shortened commit-message summary. It does not include full raw
GitHub API responses or full patches.
