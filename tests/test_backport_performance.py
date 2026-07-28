"""Regression pins for the burgess-backport performance fixes (Stage 2).

  2.1  lazy-row FPS matches the reference full-matrix walk exactly
  2.2  _cap_by_novelty: identity at/below cap, most-novel above it; metrics reports mean_cosine_n
  2.3  a surface-reusable open-axis batch does no redundant embedding work
  2.4  an unchanged mechanism store is not rewritten
  2.5  vectors persist as npz, round-trip exactly, and legacy JSON still reads
"""
from __future__ import annotations

import json
from typing import List, Optional, Sequence

import numpy as np

from cambrian_engine import pipeline
from cambrian_engine.diversity import farthest_point_sampling
from cambrian_engine.state import State


# --------------------------------------------------------------------------- 2.1
def _reference_fps(vecs, k, start=0, seeds: Optional[Sequence[int]] = None) -> List[int]:
    """The pre-fix full-matrix implementation, kept here as the oracle."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    dist = 1.0 - vecs @ vecs.T
    if seeds:
        selected = [int(s) for s in seeds]
        min_d = dist[:, selected].min(axis=1)
    else:
        selected = [int(start)]
        min_d = dist[start].copy()
    while len(selected) < k:
        min_d[selected] = -np.inf
        j = int(np.argmax(min_d))
        selected.append(j)
        min_d = np.minimum(min_d, dist[j])
    return selected


def test_lazy_row_fps_matches_reference():
    rng = np.random.default_rng(7)
    for n, d in ((12, 8), (40, 16), (73, 5)):
        v = rng.standard_normal((n, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        for k in (1, 3, 8, n):
            assert farthest_point_sampling(v, k) == _reference_fps(v, k)
        for seeds in ([0], [1, 4], [0, 2, 5]):
            assert (farthest_point_sampling(v, 9, seeds=seeds)
                    == _reference_fps(v, 9, seeds=seeds))


# --------------------------------------------------------------------------- 2.2
def test_cap_by_novelty_is_identity_at_or_below_cap():
    from cambrian_engine.archive import Archive
    from cambrian_engine.config import Axis, AxesSpec

    spec = AxesSpec(domain="t", unit_of_generation="idea",
                    axes=[Axis(name="mechanism", type="open", primary_novelty=True)])
    arc = Archive(spec)
    ids = [f"e{i}" for i in range(5)]
    for i, eid in enumerate(ids):
        arc.place(eid, f"n{i}", {"mechanism": f"cell{i}"}, fitness=0.5, novelty=float(i))
    assert pipeline._cap_by_novelty(arc, ids, 10) == ids          # identity, order untouched
    assert pipeline._cap_by_novelty(arc, ids, 5) == ids
    top2 = pipeline._cap_by_novelty(arc, ids, 2)
    assert set(top2) == {"e4", "e3"}                              # most-novel first


def test_metrics_reports_mean_cosine_sample_size(home):
    axes = {
        "domain": "t", "unit_of_generation": "idea", "slate_size": 3,
        "axes": [{"name": "mechanism", "type": "open", "primary_novelty": True}],
    }
    cands = [{"id": f"c{i}", "text": f"a distinct idea about topic number {i}",
              "descriptor": {"mechanism": f"mechanism {i}"}} for i in range(6)]
    pipeline.init_project("mcn", axes, seed=0, home=home)
    pipeline.ingest("mcn", cands, axes, seed=0, home=home)
    m = pipeline.metrics("mcn", home=home)
    assert "mean_cosine_n" in m
    assert m["mean_cosine_n"] <= m["n"]


# --------------------------------------------------------------------------- 2.3
def test_open_cells_reuse_surface_rows(home):
    """Every descriptor omits the open axis, so open_texts == texts: with surface_vecs handed
    over, the embedder must not be called at all."""
    from cambrian_engine import embed
    from cambrian_engine.config import Axis, AxesSpec

    spec = AxesSpec(domain="t", unit_of_generation="idea",
                    axes=[Axis(name="mechanism", type="open", primary_novelty=True)])
    emb = embed.get_embedder()
    texts = [f"idea number {i}" for i in range(5)]
    surface = emb.embed(texts)

    calls = {"n": 0}
    real = emb.embed

    def counting(t):
        calls["n"] += 1
        return real(t)

    emb.embed = counting  # type: ignore[method-assign]
    try:
        _, cells, vecs = pipeline.assign_open_cells(
            spec, [{} for _ in texts], texts, emb, seed=0, surface_vecs=surface)
    finally:
        del emb.embed  # drop the instance attribute; the class method is live again

    assert calls["n"] == 0, "re-embedded text the caller already embedded"
    assert np.allclose(vecs, surface)
    assert len(cells) == len(texts)


# --------------------------------------------------------------------------- 2.4
def test_unchanged_mech_store_is_not_rewritten(home):
    """No open axis -> nothing writes the mechanism store, so its mtime must not move."""
    axes = {
        "domain": "t", "unit_of_generation": "idea", "slate_size": 3,
        "axes": [{"name": "angle", "type": "categorical"}],
    }
    pipeline.init_project("nomech", axes, seed=0, home=home)

    def mk(r):
        return [{"id": f"{r}{i}", "text": f"round {r} idea {i} about ships",
                 "descriptor": {"angle": f"a{i}"}} for i in range(4)]

    pipeline.ingest("nomech", mk("x"), axes, seed=0, home=home)
    st = State("nomech", home=home)
    first = st.mech_embeddings_path.stat().st_mtime_ns
    pipeline.ingest("nomech", mk("y"), axes, seed=0, home=home)
    assert st.mech_embeddings_path.stat().st_mtime_ns == first


# --------------------------------------------------------------------------- 2.5
def test_vector_store_roundtrips_exactly(home):
    st = State("vec", home=home).ensure()
    data = {"a": [0.1, -0.25, 3.5], "b": [1e-9, 2.0, -0.0], "file": [1.0, 2.0, 3.0]}
    st.write_embeddings(data)
    assert st.embeddings_path.exists() and st.embeddings_path.suffix == ".npz"
    assert st.read_embeddings() == data          # exact float identity, incl. the "file" key


def test_legacy_json_vector_store_still_reads_then_migrates(home):
    st = State("legacy", home=home).ensure()
    data = {"a": [0.5, 0.25], "b": [-1.0, 0.125]}
    st._legacy_embeddings_json.write_text(json.dumps(data), encoding="utf-8")
    assert st.read_embeddings() == data          # tolerant fallback
    st.write_embeddings(data)                    # next write migrates
    assert st.embeddings_path.exists()
    assert not st._legacy_embeddings_json.exists()


def test_reset_geometry_clears_legacy_vector_files(home):
    st = State("resetleg", home=home).ensure()
    st._legacy_embeddings_json.write_text('{"a": [1.0]}', encoding="utf-8")
    st._legacy_mech_embeddings_json.write_text('{"a": [1.0]}', encoding="utf-8")
    st.reset_geometry()
    assert not st._legacy_embeddings_json.exists()
    assert not st._legacy_mech_embeddings_json.exists()
    assert st.read_embeddings() == {}
