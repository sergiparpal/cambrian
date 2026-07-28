"""Regression pins for the burgess-backport correctness fixes (Stage 1).

Each test pins one defect that was reproduced against this repo before the fix:

  1.1  non-finite fitness duplicated a candidate in the DPP slate
  1.2  _niche_slug collapsed every non-Latin categorical value into ONE MAP-Elites niche
  1.3  an omitted continuous axis binned to the extreme (bin 0) instead of the middle
  1.4  ingest warm-loads the embedder OUTSIDE the project lock
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from cambrian_engine import embed, pipeline
from cambrian_engine.archive import _MISSING_BUCKET, _niche_slug, axis_bucket, continuous_bin
from cambrian_engine.config import Axis
from cambrian_engine.diversity import bounded_quality, select_diverse
from cambrian_engine.state import State


# --------------------------------------------------------------------------- 1.1
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_quality_yields_distinct_slate_indices(bad):
    """Pre-fix this returned e.g. [0, 1, 0, 1] — the same idea twice in one slate."""
    vecs = np.eye(6)
    quality = np.array([1.0, 2.0, bad, 3.0, 4.0, 5.0])
    sel = select_diverse(vecs, 4, quality)
    assert len(set(sel)) == len(sel), f"duplicate index with {bad}: {sel}"


def test_bounded_quality_sanitizes_non_finite():
    q = bounded_quality(np.array([1.0, np.inf, np.nan, 2.0]), weight=1.0)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.7) and np.all(q <= 1.3)


def test_bounded_quality_all_non_finite_is_uniform():
    q = bounded_quality(np.array([np.nan, np.inf, -np.inf]), weight=1.0)
    assert np.all(np.isfinite(q))
    assert len(set(np.round(q, 12))) == 1  # uniform -> pure diversity


def test_bounded_quality_clean_input_unchanged():
    """The sanitizer must be a no-op on ordinary fitness."""
    raw = np.array([0.1, 0.5, 0.9])
    assert np.allclose(bounded_quality(raw, weight=0.3),
                       bounded_quality(raw.copy(), weight=0.3))


# --------------------------------------------------------------------------- 1.2
def test_non_latin_values_get_distinct_buckets():
    values = ["芸術", "成長", "технология", "مرحبا"]
    buckets = [_niche_slug(v) for v in values]
    assert len(set(buckets)) == len(values), f"collapsed: {buckets}"
    assert _MISSING_BUCKET not in buckets


def test_ascii_buckets_are_unchanged_and_readable():
    assert _niche_slug("Young Adults") == "young-adults"
    assert _niche_slug("b2b") == "b2b"
    assert _niche_slug("none") == _MISSING_BUCKET


def test_missing_and_blank_share_the_missing_bucket():
    ax = Axis(name="segment", type="categorical")
    assert _niche_slug("") == _MISSING_BUCKET
    assert _niche_slug("   ") == _MISSING_BUCKET
    assert axis_bucket(ax, None) == _MISSING_BUCKET


def test_niche_slug_is_deterministic_across_processes():
    """hash() is PYTHONHASHSEED-salted; niche ids must be stable between CLI invocations."""
    code = "from cambrian_engine.archive import _niche_slug; print(_niche_slug('芸術'))"
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(2)
    }
    assert len(outs) == 1


# --------------------------------------------------------------------------- 1.3
def test_continuous_bin_missing_is_mid_bin():
    ax = Axis(name="feasibility", type="continuous", range=(0.0, 1.0), bins=5)
    assert continuous_bin(ax, None) == 2           # mid, not 0 (the "far-fetched" extreme)
    assert continuous_bin(ax, "") == 2             # un-coercible reads as missing
    assert continuous_bin(ax, 0.05) == 0           # a real low value still bins low
    assert continuous_bin(ax, float("nan")) == 0   # garbage stays clamped, not neutralized


def test_omitting_candidates_do_not_all_share_the_extreme_bin():
    ax = Axis(name="boldness", type="continuous", range=(0.0, 1.0), bins=5)
    assert continuous_bin(ax, None) != continuous_bin(ax, 0.0)


# --------------------------------------------------------------------------- 1.4
def test_ingest_warm_loads_embedder_before_taking_the_lock(home, monkeypatch):
    """The embedder must be resolved BEFORE project_lock() — a first-run model download
    inside a 60s-stale lock lets a concurrent session steal it mid-cycle."""
    order: list[str] = []

    real_lock = State.project_lock

    def traced_lock(self, *a, **kw):
        order.append("lock")
        return real_lock(self, *a, **kw)

    monkeypatch.setattr(State, "project_lock", traced_lock)

    # Session resolves the process-cached embedder, so wrap that instance's ``embed``
    # (patching ``embed.get_embedder`` would miss session.py's from-import binding).
    emb = embed.get_embedder()
    real_embed = emb.embed

    def traced_embed(texts):
        order.append("embed")
        return real_embed(texts)

    axes = {
        "domain": "t", "unit_of_generation": "idea", "slate_size": 3,
        "axes": [{"name": "mechanism", "type": "open", "primary_novelty": True}],
    }
    cands = [{"id": f"c{i}", "text": f"idea number {i} about weather",
              "descriptor": {"mechanism": f"m{i}"}} for i in range(3)]
    emb.embed = traced_embed  # type: ignore[method-assign]
    try:
        pipeline.ingest("lockorder", cands, axes, seed=0, home=home)
    finally:
        del emb.embed  # drop the instance attribute; the class method is live again

    assert order, "neither seam was observed"
    assert order[0] == "embed", f"lock was taken before the embedder warm-load: {order}"
