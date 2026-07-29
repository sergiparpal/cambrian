# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/sergiparpal/cambrian/security/advisories/new)
rather than opening a public issue.

Expect an initial response within 7 days. If a report is confirmed, the fix and the advisory
are published together.

## Supported versions

Only the latest release on `main` receives security fixes.

## Scope

This repository is a Claude Code plugin: a set of instructions plus a local, server-less Python
CLI. It runs on the user's own machine, exposes no network service, and holds no credentials.
The parts most worth scrutiny are therefore:

- **The provisioner** (`skills/ideate/scripts/bootstrap.py` and the `hooks/provision.*`
  launchers) — it creates a virtualenv and installs dependencies in the background on plugin
  load.
- **Dependency supply chain** — the runtime stack (`numpy`, `scikit-learn`, `model2vec`) and,
  for the opt-in `local` embedder, `sentence-transformers`.
- **Model download** — the default `static` embedder lazily fetches model weights from
  Hugging Face on first embed.
- **State handling** — the engine reads and writes `~/.cambrian/` (override: `CAMBRIAN_HOME`).

Out of scope: the quality, diversity, or content of generated ideas, and any behaviour of the
Claude model itself.
