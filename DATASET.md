# Dataset Description

TorchRepairBench is built from PyTorch bug-fixing commits. Each task records the buggy parent
commit, fixing commit, related source files, modified function names, bug-revealing tests, and
agent-generated patches.

The benchmark contains:

- 180 single-function PyTorch bugs.
- 160 multi-function PyTorch bug commits.
- Canonical repair runs for DeepSeek and GPT backends paired with four repair agents:
  Agentless, Aider, AutoCodeRover, and SWE-agent.

The dataset is intended for research on automated program repair, LLM agent evaluation,
patch generation, patch applicability, executable validation, and repair failure analysis.
