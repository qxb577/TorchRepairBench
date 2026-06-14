# Security and Privacy

This release package excludes:

- API keys and local model credentials.
- Local environment files such as `agent_env.local`.
- Raw third-party agent repositories.
- PyTorch build worktrees.
- Large build logs and temporary validation directories.
- Private paper drafts and reference PDFs.

CSV paths are sanitized to avoid exposing local absolute paths.
