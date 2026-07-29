"""The version is declared in three places and they must never disagree.

`plugin.json` is what Claude Code shows and what `claude plugin update` compares;
`pyproject.toml` is what pip records for the installed engine; `__version__` is what
the package reports at runtime. Nothing mechanically ties them together — each bump
edits all three by hand — so a partial bump would ship a plugin whose manifest,
installed distribution, and runtime each claim a different version.

This is a release-hygiene guard, not engine behavior: it caught nothing historically,
but the 0.6.0/0.6.1 releases both bumped these files inside an unrelated commit, which
is exactly the situation where one of the three gets missed.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import cambrian_engine

_ROOT = Path(__file__).resolve().parents[1]
_PLUGIN_JSON = _ROOT / ".claude-plugin/plugin.json"
_PYPROJECT = _ROOT / "skills/ideate/scripts/pyproject.toml"


def _plugin_json_version() -> str:
    return json.loads(_PLUGIN_JSON.read_text(encoding="utf-8"))["version"]


def _pyproject_version() -> str:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def test_all_three_declared_versions_agree():
    manifest, dist, runtime = (
        _plugin_json_version(),
        _pyproject_version(),
        cambrian_engine.__version__,
    )
    assert manifest == dist == runtime, (
        "version drift across the three declaration sites — "
        f"plugin.json={manifest!r}, pyproject.toml={dist!r}, "
        f"cambrian_engine.__version__={runtime!r}. "
        "Bump all three together (scripts/release.py does this)."
    )


def test_version_is_semver_like():
    # The plugin marketplace and `claude plugin update` order versions; a non-numeric
    # or short form ("0.6", "0.6.2-dev") would sort unpredictably against real tags.
    assert re.fullmatch(r"\d+\.\d+\.\d+", cambrian_engine.__version__)
