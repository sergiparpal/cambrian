#!/usr/bin/env python3
"""Cut a release: bump, verify, PR, merge, tag, publish — in one step.

Releases 0.6.0 and 0.6.1 both drifted because the version bump was folded into an
unrelated commit and the tag step was simply forgotten, twice. The fix is to make the
whole ritual one command, so there is no point at which a human has to remember the
next step. The versions through 0.5.x were cut as a standalone ``chore(release):``
commit plus an annotated tag; this script restores exactly that shape.

Cutting v0.6.2 then exposed the flaw in the first version of that fix: it pushed the
release commit straight to ``main``, which the repo's ruleset rejects — ``pull_request``
is required with no bypass actors. The push failed *after* the local commit and tag
already existed, i.e. it produced the half-finished state the script exists to prevent.
So a release now travels the same road as every other change (the workflow CLAUDE.md
documents): branch, push, open a PR, wait for the required check, merge, and only then
tag the merged commit. That also makes the script correct on a repo *without* a
ruleset, so there is nothing to keep in sync with branch-protection settings.

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
branched, or pushed. Everything after the branch push is reported with recovery
instructions rather than auto-rolled-back, since by then the state is public.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The aggregating status check main's ruleset requires. It is one stable name on
# purpose (see CLAUDE.md): gating on the individual matrix legs would turn a dropped
# Python version into a required check that never reports again.
REQUIRED_CHECK = "ci-complete"
POLL_SECONDS = 20
CI_TIMEOUT_MINUTES = 30

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


def check_gh() -> None:
    """`gh` is now load-bearing, not a nice-to-have for the Release step.

    The whole merge path runs through it, so a missing or logged-out `gh` has to fail
    here — before the bump — rather than halfway through with a branch already pushed.
    """
    try:
        subprocess.run(["gh", "auth", "status"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise ReleaseError(
            "gh is not installed. The release is cut through a pull request, so gh is "
            "required: https://cli.github.com"
        ) from None
    except subprocess.CalledProcessError:
        raise ReleaseError("gh is not authenticated; run `gh auth login` first.") from None


def check_preconditions(version: str, tag: str, branch: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ReleaseError(f"{version!r} is not a X.Y.Z version.")

    check_gh()

    current = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if current != "main":
        raise ReleaseError(f"on branch {current!r}; releases are cut from main.")

    if run("git", "status", "--porcelain"):
        raise ReleaseError("working tree is dirty; commit or stash first.")

    if run("git", "tag", "-l", tag):
        raise ReleaseError(f"tag {tag} already exists.")

    # A half-finished earlier attempt would otherwise fail at `git checkout -b`, after
    # the bump and the whole test run.
    if run("git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False):
        raise ReleaseError(f"branch {branch} already exists locally; delete it first.")
    if run("git", "ls-remote", "--heads", "origin", branch, check=False):
        raise ReleaseError(f"branch {branch} already exists on origin; delete it first.")

    # A release built on a stale main would tag commits the remote does not have.
    run("git", "fetch", "origin", "main", check=False)
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "@{u}"):
        raise ReleaseError("local main and origin/main differ; pull/push first.")

    if read_current_version() == version:
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


def open_pr(version: str, tag: str, branch: str) -> int:
    body = (
        f"Cuts **{tag}**: bumps the three version declaration sites to `{version}`.\n\n"
        f"Opened by `scripts/release.py`, which verified the bumped tree with the full "
        f"pytest suite and `cambrian_engine selftest` before pushing. Once "
        f"`{REQUIRED_CHECK}` is green the script merges this PR, tags the merged commit "
        f"`{tag}`, and publishes the GitHub Release.\n"
    )
    run("git", "push", "-u", "origin", branch)
    url = run("gh", "pr", "create", "--base", "main", "--head", branch,
              "--title", f"chore(release): {version}", "--body", body)
    number = url.rstrip("/").rsplit("/", 1)[-1]
    if not number.isdigit():
        raise ReleaseError(f"could not parse a PR number out of {url!r}")
    print(f"  opened {url}")
    return int(number)


def check_conclusion(pr: int, name: str) -> str | None:
    """The named check's conclusion, or None while it is still pending.

    Two distinct shapes both mean "not finished", and both look like absence: gh
    reports an in-flight check with an EMPTY conclusion (not null), and a job that
    gates on others — ci-complete waits for the whole matrix — is missing from the
    rollup entirely until it starts. Reading either as a failure would abort every
    release seconds after opening the PR.
    """
    jq = rf'.statusCheckRollup[] | select(.name=="{name}") | .conclusion'
    return run("gh", "pr", "view", str(pr), "--json", "statusCheckRollup",
               "--jq", jq, check=False) or None


def progress_summary(pr: int) -> str:
    jq = (r'[.statusCheckRollup[] | if (.conclusion // "") == "" then "pending" '
          r'else .conclusion end] | group_by(.) | map("\(length) \(.[0])") | join(", ")')
    return run("gh", "pr", "view", str(pr), "--json", "statusCheckRollup",
               "--jq", jq, check=False) or "no checks reported yet"


def wait_for_check(pr: int, timeout_minutes: int) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    print(f"\nWaiting for {REQUIRED_CHECK} on PR #{pr} (timeout {timeout_minutes}m) ...")
    last = ""
    while time.monotonic() < deadline:
        conclusion = check_conclusion(pr, REQUIRED_CHECK)
        if conclusion == "SUCCESS":
            print(f"  {REQUIRED_CHECK}: SUCCESS")
            return
        if conclusion:
            # NEUTRAL/SKIPPED count as failure here: the ruleset wants this check to
            # have actually run, and merging on a check that opted out defeats the gate.
            raise ReleaseError(f"{REQUIRED_CHECK} concluded {conclusion}; not merging.")
        summary = progress_summary(pr)
        if summary != last:
            print(f"  {summary}")
            last = summary
        time.sleep(POLL_SECONDS)
    raise ReleaseError(
        f"timed out after {timeout_minutes}m waiting for {REQUIRED_CHECK} on PR #{pr}."
    )


def merge_pr(pr: int, version: str) -> None:
    """Merge, sync main, and confirm the bump actually landed before anything is tagged."""
    # Off the release branch first, so --delete-branch can clean up both copies.
    run("git", "checkout", "main")
    run("gh", "pr", "merge", str(pr), "--merge", "--delete-branch")
    # --delete-branch drops the branch on origin and locally but leaves this clone's
    # remote-tracking ref, so `git branch -a` keeps advertising a branch that is gone
    # (v0.6.2 and v0.6.3 each left one). Cosmetic only — the preconditions ask origin
    # directly and never consult these refs — so it must not be able to fail a release
    # that has already merged, hence check=False. Deliberately no refspec: `fetch
    # --prune origin main` would prune only within `main` and miss the release branch.
    run("git", "fetch", "--prune", "origin", check=False)
    run("git", "pull", "--ff-only", "origin", "main")
    landed = read_current_version()
    if landed != version:
        raise ReleaseError(
            f"main reports version {landed} after merging PR #{pr}, expected {version}. "
            f"Not tagging — inspect main before retrying."
        )
    print(f"  merged PR #{pr}; main is now {landed}")


def publish(tag: str, message: str, notes_file: str | None) -> None:
    args = ["gh", "release", "create", tag, "--title", message, "--latest"]
    args += ["--notes-file", notes_file] if notes_file else ["--generate-notes"]
    print(f"  published {run(*args)}")


def abandon_hint(branch: str, pr: int | None) -> str:
    lines = ["\n  The release branch is pushed. To abandon it:",
             "    git checkout main"]
    if pr is not None:
        lines.append(f"    gh pr close {pr} --delete-branch")
    else:
        lines.append(f"    git push origin --delete {branch}")
    lines.append(f"    git branch -D {branch}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut a Cambrian release.")
    parser.add_argument("version", help="the new version, e.g. 0.6.2")
    parser.add_argument("-m", "--message",
                        help="tag/release title; defaults to 'vX.Y.Z — Cambrian'")
    parser.add_argument("--notes-file",
                        help="markdown file for the GitHub Release body "
                             "(default: --generate-notes)")
    parser.add_argument("--no-release", action="store_true",
                        help="merge and tag, but skip the GitHub Release")
    parser.add_argument("--ci-timeout", type=int, default=CI_TIMEOUT_MINUTES,
                        metavar="MINUTES",
                        help=f"how long to wait for {REQUIRED_CHECK} "
                             f"(default: {CI_TIMEOUT_MINUTES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="check preconditions and bump, then revert; no git writes")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    tag = f"v{version}"
    branch = f"release/{tag}"
    message = args.message or f"{tag} — Cambrian"
    if not message.startswith(tag):
        message = f"{tag} — {message}"

    pr: int | None = None
    pushed = False
    try:
        check_preconditions(version, tag, branch)
        print(f"Releasing {read_current_version()} -> {version} as {tag}\n")
        bump(version)

        if args.dry_run:
            try:
                verify()
            finally:
                revert_bump()   # a dry run leaves no residue, pass or fail
            print(f"\nDry run OK — bump reverted, nothing committed. "
                  f"Would open {branch} and tag {tag}.")
            return 0

        try:
            verify()
        except ReleaseError:
            revert_bump()
            print("\n! Verification failed — bump reverted, nothing committed.")
            raise

        run("git", "checkout", "-b", branch)
        run("git", "commit", "-am", f"chore(release): {version}")
        print(f"\n  committed {version} on {branch}")

        # Past this point the state is public; report rather than auto-revert.
        pushed = True
        pr = open_pr(version, tag, branch)
        wait_for_check(pr, args.ci_timeout)
        merge_pr(pr, version)

        run("git", "tag", "-a", tag, "-m", message)
        pushed = False   # the branch is gone; only the tag is outstanding now
        try:
            run("git", "push", "origin", tag)
            print(f"  tagged and pushed {tag}")
        except ReleaseError:
            print(f"\n! Tag push failed. The release commit is on main and {tag} exists "
                  f"locally. Retry with:\n    git push origin {tag}")
            raise

        if not args.no_release:
            publish(tag, message, args.notes_file)

        print(f"\n{tag} released.")
        return 0

    except ReleaseError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if pushed:
            print(abandon_hint(branch, pr), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
