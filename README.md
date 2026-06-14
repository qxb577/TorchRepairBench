# TorchRepairBench

TorchRepairBench is a benchmark for evaluating LLM-agent repair capability on real PyTorch bugs.
It contains single-function and multi-function PyTorch repair tasks, canonical model-agent repair
runs, generated patches, and executable validation results.

## Contents

- `single_func_180`: tasks=180, canonical repair rows=1435, validation rows=1441, copied patch files=1108, validation source=`benchmark_pilot/single_func_180/allruns_patch_test_validation_corrected562.csv`
- `multi_func_160`: tasks=160, canonical repair rows=1280, validation rows=89, copied patch files=958, validation source=`benchmark_pilot/multi_func_160/allruns_multi_patch_test_validation_corrected22.csv`

## Directory Layout

- `data/single_func_180/`: single-function PyTorch bug tasks and results.
- `data/multi_func_160/`: multi-function PyTorch bug tasks and current validation results.
- `data/*/patches/`: canonical generated patches selected for public release.
- `data/raw_index/`: lightweight raw commit index and filtering-flow statistics.
- `scripts/`: helper scripts used to construct canonical tables and summarize validation progress.
- `docs/`: schema and protocol documentation.

## Evaluation Unit

For each bug and each model-agent pair, TorchRepairBench keeps one canonical repair unit.
No-patch outputs are counted as repair failures, while generated and applicable canonical patches
are validated by rebuilding the target PyTorch version and running bug-revealing tests.

## Note

The single-function validation is complete. The multi-function validation files reflect the current
validated state at release-package generation time and can be updated as additional builds finish.
