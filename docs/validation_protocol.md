# Validation Protocol

For each applicable canonical patch:

1. Reconstruct the buggy PyTorch version from `parent_sha`.
2. Restore required submodules from local PyTorch metadata when available.
3. Apply the canonical patch.
4. Build PyTorch in a CPU-only configuration.
5. Run the bug-revealing tests.
6. Mark the patch as correct only if the tests pass.

For multi-function tasks, validation is organized by `parent_sha + commit_id`. A single build can
be reused to validate all canonical patches belonging to the same commit, including patches stored
under different selected-instance directories.
