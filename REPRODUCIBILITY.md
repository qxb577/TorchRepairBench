# Reproducibility

The validation protocol reconstructs the buggy PyTorch version from `parent_sha`, applies one
canonical patch, builds PyTorch, and runs the corresponding bug-revealing tests.

Important environment choices used in our experiments include:

- CPU-only PyTorch builds.
- CUDA/ROCm disabled.
- Optional features such as distributed, MKLDNN, XNNPACK, QNNPACK, FBGEMM, and Kineto disabled
  where possible to reduce build cost.
- Validation organized at commit level: the same `parent_sha + commit_id` build can validate all
  canonical patches belonging to that commit, including patches stored under different selected
  instances.

Large build trees and raw local logs are intentionally not included in this release package.
The full raw GitHub API backup is not included; `data/raw_index/` provides a lightweight index and
filtering statistics.
