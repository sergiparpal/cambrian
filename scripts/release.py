#!/usr/bin/env python3
"""Cut a release: bump, verify, commit, tag, push, publish — in one step.

Releases 0.6.0 and 0.6.1 both drifted because the version bump was folded into an
unrelated commit and the tag step was simply forgotten, twice. The fix is to make the
whole ritual one command, so there is no point at which a human has to remember the
next step. The versions through 0.5.x were cut as a standalone ``chore(release):``
commit plus an annotated tag; this script restores exactly that shape.

    python3 scripts/release.py 0.6.2 -m "0.6.2 — what changed"
    python3 scripts/release.py 0.6.2 -m "..." --notes-file notes.md
    python3 scripts/release.py 0.6.2 -m "..." --dry-run

The version is declared in three files (see tests/test_version_consistency.py). They
are rewritten by targeted line substitution rather than a parse/re-serialize round
trip, so formatting — key order, the single-line ``keywords`` array — is preserved
byte-for-byte apart from the version itself.

Ordering is deliberate: bump first, then run the full suite and the engine selftest
against the *bumped* tree (that is what ships), and only commit once they pass. If
they fail, the bump this script made is reverted automatically — nothing is committed,
tagged, or pushed. Everything after the first push is reported with recovery
instructions rather than auto-rolled-back, since by then the state is public.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (path, regex with a single capturing group around the version literal)
VERSION_SITES: list[tuple[Path, re.Pattern[str]]] = [
    (ROOT / ".claude-plugin/plugin.json",
     re.compile(r'("version":\s*")(\d+\.\d+\.\d+)(")')),
    (ROOT / "skills/ideate/scripts/pyproject.toml",
     re.compile(r'^(version = ")(\d+\.\d+\.\d+)(")', re.MULTILINE)),
    (ROOT / "skills/ideate/scripts/cambrian_engine/__init__.py",
     re.compile(r'^(__version__ = ")(\d+\.\d+\.\d+)(")', re.MULTILINE)),
]


class ReleaseError(RuntimeError):
    """A precondition failed. The message is user-facing."""


def run(*args: str, capture: bool = True, check: bool = True) -> str:
    proc = subprocess.run(
        args, cwd=ROOT, text=True, check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ReleaseError(f"`{' '.join(args)}` failed:\n{detail}")
    return (proc.stdout or "").strip()


def engine_python() -> str:
    """The dev venv interpreter if present, else whatever is running this script.

    Tests must run against the engine's own venv (the repo's standing rule), but the
    script stays usable from a bare checkout.
    """
    for rel in ("skills/ideate/.venv/bin/python", "skills/ideate/.venv/Scripts/python.exe"):
        candidate = ROOT / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check_preconditions(version: str, tag: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ReleaseError(f"{version!r} is not a X.Y.Z version.")

    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        raise ReleaseError(f"on branch {branch!r}; releases are cut from main.")

    if run("git", "status", "--porcelain"):
        raise ReleaseError("working tree is dirty; commit or stash first.")

    if run("git", "tag", "-l", tag):
        raise ReleaseError(f"tag {tag} already exists.")

    # A release built on a stale main would tag commits the remote does not have.
    run("git", "fetch", "origin", "main", check=False)
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "@{u}"):
        raise ReleaseError("local main and origin/main differ; pull/push first.")

    current = read_current_version()
    if current == version:
        raise ReleaseError(f"version is already {version}; nothing to bump.")


def read_current_version() -> str:
    path, pattern = VERSION_SITES[0]
    match = pattern.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ReleaseError(f"no version found in {path}")
    return match.group(2)


def revert_bump() -> None:
    """Undo only the three files this script rewrites.

    Surgical rather than ``git checkout -- .`` so it can never discard unrelated work,
    even though the clean-tree precondition means there should be none.
    """
    run("git", "checkout", "--", *(str(p) for p, _ in VERSION_SITES))


def bump(version: str) -> None:
    """Rewrite the version in all three sites, or raise without touching any."""
    edits: list[tuple[Path, str]] = []
    for path, pattern in VERSION_SITES:
        original = path.read_text(encoding="utf-8")
        updated, n = pattern.subn(rf"\g<1>{version}\g<3>", original)
        if n != 1:
            raise ReleaseError(f"expected exactly 1 version match in {path}, found {n}")
        edits.append((path, updated))
    for path, updated in edits:
        path.write_text(updated, encoding="utf-8")
        print(f"  bumped {path.relative_to(ROOT)}")


def verify() -> None:
    python = engine_python()
    print(f"\nVerifying with {python} ...")
    run(python, "-m", "pytest", "-q", capture=False)
    run(python, "-m", "cambrian_engine", "selftest", capture=False)


def publish(tag: str, message: str, notes_file: str | None) -> None:
    if not have_gh():
        print(f"\n! gh not available — tag pushed, but no GitHub Release created."
              f"\n  Create it with: gh release create {tag} --title '{message}' ...")
        return
    args = ["gh", "release", "create", tag, "--title", message, "--latest"]
    args += ["--notes-file", notes_file] if notes_file else ["--generate-notes"]
    url = run(*args)
    print(f"  published {url}")


def have_gh() -> bool:
    try:
        subprocess.run(["gh", "--version"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut a Cambrian release.")
    parser.add_argument("version", help="the new version, e.g. 0.6.2")
    parser.add_argument("-m", "--message",
                        help="tag/release title; defaults to 'vX.Y.Z — Cambrian'")
    parser.add_argument("--notes-file",
                        help="markdown file for the GitHub Release body "
                             "(default: --generate-notes)")
    parser.add_argument("--no-release", action="store_true",
                        help="tag and push, but skip the GitHub Release")
    parser.add_argument("--dry-run", action="store_true",
                        help="check preconditions and bump, then revert; no git writes")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    tag = f"v{version}"
    message = args.message or f"{tag} — Cambrian"
    if not message.startswith(tag):
        message = f"{tag} — {message}"

    try:
        check_preconditions(version, tag)
        print(f"Releasing {read_current_version()} -> {version} as {tag}\n")
        bump(version)

        if args.dry_run:
            try:
                verify()
            finally:
                revert_bump()   # a dry run leaves no residue, pass or fail
            print(f"\nDry run OK — bump reverted, nothing committed. Would tag {tag}.")
            return 0

        try:
            verify()
        except ReleaseError:
            revert_bump()
            print("\n! Verification failed — bump reverted, nothing committed.")
            raise

        run("git", "commit", "-am", f"chore(release): {version}")
        run("git", "tag", "-a", tag, "-m", message)
        print(f"\n  committed and tagged {tag}")

        # Past this point the state is public; report rather than auto-revert.
        try:
            run("git", "push", "origin", "main")
            run("git", "push", "origin", tag)
            print(f"  pushed main and {tag}")
        except ReleaseError:
            print(f"\n! Push failed. Local commit and {tag} exist. Retry with:"
                  f"\n    git push origin main && git push origin {tag}"
                  f"\n  or undo with:"
                  f"\n    git tag -d {tag} && git reset --hard HEAD~1")
            raise

        if not args.no_release:
            publish(tag, message, args.notes_file)

        print(f"\n{tag} released.")
        return 0

    except ReleaseError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
