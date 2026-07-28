"""Pipeline robustness: engine-knob pinning, null-engine guard, empty-cycle schema."""

from __future__ import annotations

from cambrian_engine import config, pipeline, selftest
from cambrian_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def test_open_niching_knobs_pinned_to_init_snapshot(home):
    # The open-axis NICHING knobs (open_niches / open_niche_freeze_factor) are
    # pinned at init like the axes geometry: a later cycle passing a DIFFERENT
    # value must be ignored, so the CVT partition can't be refit with a different k
    # than the cells already in the archive. (Other engine knobs stay per-cycle
    # overridable — see test_state_pruning.)
    axes = _generic()
    axes["engine"] = {"open_niches": 10}
    pipeline.init_project("pin", axes, seed=0, home=home)
    target = int(State("pin", home=home).read_meta()["candidates_per_generation"])

    axes2 = _generic()
    axes2["engine"] = {"open_niches": 99}  # operator edits the file mid-session
    pipeline.ingest("pin", selftest.diverse_candidates(target), axes2, seed=0, home=home)

    assert State("pin", home=home).read_meta()["engine"]["open_niches"] == 10


def test_metrics_survives_null_engine_block(home):
    # An older/hand-edited meta may carry "engine": null; metrics must not crash.
    pipeline.init_project("m", _generic(), seed=0, home=home)
    st = State("m", home=home)
    meta = st.read_meta()
    meta["engine"] = None
    st.write_meta(meta)
    res = pipeline.metrics("m", home=home)  # would AttributeError on None.get(...)
    assert "open_axis" in res


def test_empty_cycle_returns_full_schema(home):
    # An empty-candidate generation must return the same response shape as a normal
    # one, so consumers never KeyError on the advisory keys. Compared STRUCTURALLY
    # against a real cycle rather than a hand-listed key set: the hand-listed version
    # silently missed monitor["variety_erosion"], which is exactly the drift this
    # guards. `under_generation_note` is conditional on BOTH paths (present only when
    # the guard fires), so it is excluded from the parity set.
    conditional = {"under_generation_note"}
    pipeline.init_project("e", _generic(), seed=0, home=home)
    empty = pipeline.ingest("e", [], _generic(), seed=0, home=home)

    pipeline.init_project("e2", _generic(), seed=0, home=home)
    target = int(State("e2", home=home).read_meta()["candidates_per_generation"])
    normal = pipeline.ingest(
        "e2", selftest.diverse_candidates(target), _generic(), seed=0, home=home
    )

    assert set(empty) == set(normal)
    assert set(empty["monitor"]) - conditional == set(normal["monitor"]) - conditional
    assert empty["ask_policy"]["phase"] in ("explore", "refine")
    assert empty["monitor"]["under_generation"] is False
    assert empty["monitor"]["variety_eroding"] is False
    assert empty["monitor"]["collapsing"] is False


def test_empty_cycle_reflects_persisted_erosion_streak(home):
    # An empty generation is a no-op for the erosion sensor, but it must report the
    # LAST persisted streak's flag rather than a hard False — an in-progress streak
    # shouldn't read as "not eroding" just because one generation arrived empty.
    pipeline.init_project("ee", _generic(), seed=0, home=home)
    st = State("ee", home=home)
    meta = st.read_meta()
    persist = int(meta["engine"]["erosion_persist"])
    meta["erosion_streak"] = persist  # already at the flag threshold
    st.write_meta(meta)
    res = pipeline.ingest("ee", [], _generic(), seed=0, home=home)
    assert res["monitor"]["variety_eroding"] is True


def test_metrics_caps_cosine_vectors_but_reports_full_n(home):
    # With a tiny novelty_ref_cap, metrics computes mean_cosine over only the capped
    # most-novel elites yet still reports the FULL elite count and runs cleanly.
    axes = _generic()
    axes["engine"] = {"novelty_ref_cap": 3}
    pipeline.init_project("cap", axes, seed=0, home=home)
    target = int(State("cap", home=home).read_meta()["candidates_per_generation"])
    pipeline.ingest("cap", selftest.diverse_candidates(target), axes, seed=0, home=home)
    arc_n = len(State("cap", home=home).read_archive()["niches"])
    assert arc_n > 3, "fixture should produce more elites than the cap"
    res = pipeline.metrics("cap", home=home)
    assert res["n"] == arc_n              # true count, not the cap
    assert isinstance(res["mean_cosine"], float)
    assert res["coverage"] >= 1
