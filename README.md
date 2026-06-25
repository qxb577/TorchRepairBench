# TorchRepairBench

TorchRepairBench is a benchmark for evaluating LLM-agent repair capability on real PyTorch bugs.
It contains single-function and multi-function PyTorch repair tasks, canonical model-agent repair
runs, generated patches, and executable validation results.

## Contents

- `single_func_180`: tasks=180, canonical repair rows=1440, validation rows=1445, copied patch files=1108, validation source=`data/single_func_180/validation_results.csv`
- `multi_func_160`: tasks=160, canonical repair rows=1280, validation rows=977, copied patch files=958, validation source=`data/multi_func_160/validation_results.csv`

## Directory Layout

- `data/single_func_180/`: single-function PyTorch bug tasks and results.
- `data/multi_func_160/`: multi-function PyTorch bug tasks and current validation results.
- `data/*/patches/`: canonical generated patches selected for public release.
- `data/raw_index/`: lightweight raw commit index and filtering-flow statistics.
- `data/semantic_review/`: author-perspective semantic review of test-passing patches.
- `data/single_func_180/semantic_review/`: single-function confirmed bug-fix summaries under the final canonical test-passing review policy.
- `scripts/`: helper scripts used to construct canonical tables and summarize validation progress.
- `docs/`: schema and protocol documentation.

## Evaluation Unit

For each bug and each model-agent pair, TorchRepairBench keeps one canonical repair unit.
No-patch outputs are counted as repair failures, while generated and applicable canonical patches
are validated by rebuilding the target PyTorch version and running bug-revealing tests.

## Note

The released validation tables contain the complete canonical executable validation results for the
180 single-function tasks and 160 multi-function tasks. Semantic review tables cover only
test-passing patches and distinguish test-passing patches from patches that are confirmed as real bug
fixes under an author-review criterion.
