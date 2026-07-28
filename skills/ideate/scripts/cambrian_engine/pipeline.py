"""High-level orchestration for the CLI commands.

Each public function returns a JSON-serializable dict (or raises). The CLI in
``__main__`` is a thin wrapper that parses args, calls these, and prints JSON.

The ``ingest`` flow is the heart of one cycle:
embed → dedup → place (MAP-Elites over the resolved axes) → geometric novelty →
DPP diverse slate → anti-collapse monitor. The judge is never called here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:  # import for type hints only; avoids a runtime import cycle
    from .state import State

from . import (
    __version__,
    archive as archive_mod,
    config,
    diversity,
    gap,
    memory,
    monitor,
    novelty,
)
from .config import AxesSpec, Candidate
from .embed import dedupe, default_dedup_tau
from .session import Session

# Default tuning knobs. These are the **fallback defaults** for direct/library callers and the
# self-test's placement helper; the real ``ingest`` path resolves every knob from
# :class:`config.EngineConfig` (per-domain overridable). DERIVED from a default ``EngineConfig`` —
# the config dataclass is the single home for every value (these used to be hand-mirrored literals).
#
# The near-duplicate cosine threshold is per-embedder (see ``embed.default_dedup_tau``).
_DEFAULTS = config.EngineConfig()
KNN_K = _DEFAULTS.knn_k
# Open-axis (mechanism) niching. The partition is data-adaptive: cold-start fixed
# centroids until ``OPEN_NICHE_FREEZE_FACTOR * OPEN_NICHES`` mechanism embeddings
# have accumulated, then a one-time k-means fit freezes the cells (see
# ``_accumulate_and_maybe_freeze``).
OPEN_NICHES = _DEFAULTS.open_niches
# freeze after freeze_factor * open_niches survivor mechanisms accumulate. At 2 this
# is 48 (~4-5 generations of 12) so the data-adaptive partition actually activates in
# a realistic session, while keeping >=2 samples/centroid for a meaningful k-means
# fit. Most short sessions still never reach it and run on the (validated) cold-start
# partition; `ingest`/`metrics` expose accumulation progress so this is observable.
OPEN_NICHE_FREEZE_FACTOR = _DEFAULTS.open_niche_freeze_factor
MAX_DPP_POOL = _DEFAULTS.max_dpp_pool
# Cap on the dedup/novelty reference set (the most-novel elites). Dedup and k-NN
# novelty run against the archive elites every cycle (O(n·m)); without a cap this
# grows unbounded. Below the cap behavior is identical to using every elite.
NOVELTY_REF_CAP = _DEFAULTS.novelty_ref_cap
# How much the judge's (bounded) fitness is allowed to weight the DPP slate.
# 0 -> pure diversity; 1 -> full quality-diversity. Kept low so geometry owns
# the slate and the judge can only nudge ordering within an already-diverse pool.
QUALITY_WEIGHT = _DEFAULTS.quality_weight
# Bound on the persisted advisory gap series (entries in meta["gap_log"]); not an EngineConfig knob
# — it caps storage, not behavior.
GAP_LOG_CAP = 50


# --------------------------------------------------------------------------- #
# init-project
# --------------------------------------------------------------------------- #
def init_project(
    project: str,
    axes_source,
    seed: int = 0,
    home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create state dirs and snapshot the resolved axes for the session.

    The axes geometry goes to ``axes.json``; the agent-/session-level settings
    that ride alongside it (candidates-per-generation, judge rubric) are
    recorded in ``meta.json`` — kept out of the engine's :class:`AxesSpec`.

    Re-initing an EXISTING project whose axes geometry changed would otherwise leave
    stale niche keys mixed with new ones (the archive is keyed by the old axes). When
    the incoming axes differ from the persisted snapshot and geometric state exists,
    that state (archive / candidates / embeddings / mechanism embeddings / frozen
    open-nicher) is reset and the geometry-coupled meta series dropped; preference memory
    is preserved. Identical
    axes -> idempotent, nothing reset. Runs under the project lock (see ``ingest``).
    """
    spec, settings, econfig = config.load_all(axes_source)
    sess = Session(project, home=home, seed=seed).ensure()
    state = sess.state
    with state.project_lock():
        # Geometry-incompatible re-init: compare ONLY the axes list (what niche keys
        # are built from) — not slate_size/domain/unit, which don't affect placement.
        prev_axes = state.read_axes()
        reset = False
        if isinstance(prev_axes, dict) and prev_axes:
            geometry_changed = (
                prev_axes.get("axes") != [a.to_dict() for a in spec.axes]
            )
            if geometry_changed and state.read_archive().get("niches"):
                state.reset_geometry()
                reset = True
        sess.adopt_spec(spec)
        meta = state.read_meta()
        if reset:
            # The monitor/erosion/gap series and the cycle counter are tied to the
            # old geometry; drop them so calibration restarts under the new axes.
            for key in ("cycles", "cos_window", "novelty_window",
                        "erosion_streak", "gap_log"):
                meta.pop(key, None)
        meta.update(
            {
                "project": project,
                "domain": spec.domain,
                "unit_of_generation": spec.unit_of_generation,
                "candidates_per_generation": settings.candidates_per_generation,
                "judge_rubric": settings.judge_rubric,
                "engine": econfig.to_dict(),
                "seed": int(seed),
                "version": __version__,
            }
        )
        state.write_meta(meta)
    return {"ok": True, "domain": spec.domain, "reset": reset, "paths": state.paths()}


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def paths(project: str, home: Optional[Path] = None) -> Dict[str, Any]:
    """Ensure the project's state dir (incl. its ``tmp/`` scratch dir) and return
    the resolved paths. The skill calls this **before** writing its hand-off files
    so it can drop ``axes.json`` / ``candidates.json`` / ``event.json`` under
    ``tmp`` (inside the state home) instead of the user's cwd — keeping home and
    project-slug resolution entirely in the engine.
    """
    from .state import State

    return State(project, home=home).ensure().paths()


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall(project: str, k: int = 10, home: Optional[Path] = None) -> Dict[str, Any]:
    """Return memory for in-context injection: recent choices, pins, win tallies."""
    sess = Session(project, home=home)
    return memory.recall(sess.state, sess.domain, k=k)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def _parse_candidates(candidates) -> List[Candidate]:
    if isinstance(candidates, dict):
        candidates = candidates.get("candidates", [])
    if not isinstance(candidates, list):
        raise config.ConfigError("candidates must be a list (or {candidates: [...]})")
    return [Candidate.from_dict(c) for c in candidates]


def _survivor_novelty(
    surv_vecs: np.ndarray, existing_vecs: np.ndarray, k: int
) -> np.ndarray:
    """Mean k-NN distance of each survivor to (existing ∪ other survivors).

    novelty = mean k-NN distance to this session's own elites + batch; a variety
    proxy, NOT originality vs. prior art (no external referent is consulted).
    """
    n = surv_vecs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if existing_vecs.shape[0] > 0:
        ref = np.vstack([existing_vecs, surv_vecs])
        offset = existing_vecs.shape[0]
    else:
        ref = surv_vecs
        offset = 0
    dist = novelty.cosine_distance_matrix(surv_vecs, ref)
    # Mask each survivor's own row in the combined reference so it isn't its own
    # neighbour, then reuse the shared mean-k-NN kernel.
    dist[np.arange(n), offset + np.arange(n)] = np.inf
    return novelty.mean_knn_distance(dist, k, n_neighbors=ref.shape[0] - 1)


def _survivor_mechanism_novelty(
    open_vecs: np.ndarray,
    open_axis: Optional[Any],
    existing_ids: List[str],
    stored_mech_emb: Dict[str, List[float]],
    k: int,
    n_survivors: int,
) -> Optional[np.ndarray]:
    """Mechanism-space novelty for survivors: mean k-NN distance of each survivor's
    mechanism embedding to (archive mechanisms ∪ other survivors). Returns None when
    there is no open axis (nothing to measure). Same kernel as surface novelty."""
    if open_axis is None or open_vecs.shape[0] != n_survivors or n_survivors == 0:
        return None
    dim = int(open_vecs.shape[1])
    ref_ids = [i for i in existing_ids
               if i in stored_mech_emb and len(stored_mech_emb[i]) == dim]
    existing_mech = _stack_embeddings(ref_ids, stored_mech_emb, dim)
    return _survivor_novelty(open_vecs, existing_mech, k)


def _slate_item(record: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one elite record into a slate item for the agent/human.

    The ``novelty`` field carried here is a variety proxy — mean k-NN distance to
    this session's own elites + batch — NOT originality vs. prior art. The key is
    deliberately left named ``novelty``: consumers (the skill, the stubbed human)
    read it, so it is documented rather than renamed.
    """
    return {
        "id": record["id"],
        "text": record.get("text", ""),
        "descriptor": record.get("descriptor", {}),
        "genealogy": record.get("genealogy", {}),
        "niche_id": record.get("niche_id"),
        "coords": record.get("coords", {}),
        "novelty": round(float(record.get("novelty", 0.0)), 4),
        # Advisory mechanism-space novelty (S4): variety of this idea's mechanism vs
        # the session's own mechanisms. None for elites placed before this feature or
        # specs with no open axis. Measurement only — never steers selection.
        "mechanism_novelty": (
            round(float(record["mechanism_novelty"]), 4)
            if record.get("mechanism_novelty") is not None else None
        ),
        "fitness": round(float(record.get("fitness", 1.0)), 4),
        # The agent looks an embedding up by candidate id, so the ref IS the id;
        # kept as a distinct field so the contract survives if that ever changes.
        "embedding_ref": record["id"],
    }


def _open_axis_texts(
    open_axis: Any, descriptors: List[Dict[str, Any]], texts: List[str]
) -> List[str]:
    """The text to embed for the open axis: its descriptor value, else the idea."""
    return [str(d.get(open_axis.name) or t) for d, t in zip(descriptors, texts)]


def assign_open_cells(
    spec: AxesSpec,
    descriptors: List[Dict[str, Any]],
    texts: List[str],
    embedder,
    seed: int,
    nicher: Optional["archive_mod.CVTNicher"] = None,
    open_niches: int = OPEN_NICHES,
    surface_vecs: Optional[np.ndarray] = None,
) -> Tuple[Optional[Any], List[Optional[int]], np.ndarray]:
    """Voronoi cell per item for the primary "open" axis.

    Returns ``(open_axis, cells, open_vecs)`` where ``cells[i]`` is the item's
    cell index (or ``None`` when there is no open axis) and ``open_vecs`` are the
    embedded open-axis texts (so callers can accumulate them without re-embedding).
    When ``nicher`` is given (a frozen, data-fitted partition) it is used as-is;
    otherwise a deterministic cold-start partition is built. Shared by
    :func:`ingest` and the self-test so both place candidates identically.

    ``surface_vecs`` (row-aligned with ``texts``) lets the caller hand over the surface
    embeddings it already computed: an item whose open-axis text FALLS BACK to the idea text
    (no descriptor value) reuses its surface row instead of re-embedding the identical string.
    Exact by the embedder's per-string determinism — the same cross-call contract dedup already
    relies on when it compares this batch against stored elite vectors.
    """
    open_axis = spec.primary_axis
    n = len(texts)
    if open_axis is None or n == 0:
        return open_axis, [None] * n, np.zeros((0, 1), dtype=np.float32)
    open_texts = _open_axis_texts(open_axis, descriptors, texts)
    if surface_vecs is not None and surface_vecs.shape[0] == n:
        rows: List[Optional[np.ndarray]] = [
            surface_vecs[i] if open_texts[i] == texts[i] else None for i in range(n)
        ]
        miss = [i for i, r in enumerate(rows) if r is None]
        if miss:
            fresh = embedder.embed([open_texts[i] for i in miss])
            for j, i in enumerate(miss):
                rows[i] = fresh[j]
        open_vecs = np.asarray(rows, dtype=np.float32)
    else:
        open_vecs = embedder.embed(open_texts)
    if nicher is None:
        nicher = archive_mod.CVTNicher(dim=open_vecs.shape[1], k=open_niches, seed=seed)
    return open_axis, nicher.cells(open_vecs), open_vecs


def _frozen_open_nicher(
    on_state: Optional[Dict[str, Any]], open_axis: Optional[Any]
) -> Optional["archive_mod.CVTNicher"]:
    """The persisted frozen nicher, or ``None`` (cold start / no open axis)."""
    if open_axis is None or not on_state:
        return None
    if on_state.get("frozen") and on_state.get("centroids"):
        return archive_mod.CVTNicher.from_dict(on_state)
    return None


def _elite_open_cells(
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    open_axis: Any,
    embedder,
    nicher: "archive_mod.CVTNicher",
) -> Dict[str, int]:
    """Frozen-cell index for each niche, from its elite's open-axis embedding."""
    nids: List[str] = []
    texts: List[str] = []
    for nid, niche in arc.niches.items():
        rec = cand_store.get(niche.elite_id, {})
        # Falls back descriptor-value -> idea text -> "". An empty mechanism embeds
        # to a zero vector and lands in cell 0; such niches merge there on freeze.
        # Harmless in practice (candidates always carry text), just deterministic.
        mech = str(rec.get("descriptor", {}).get(open_axis.name) or rec.get("text") or "")
        nids.append(nid)
        texts.append(mech)
    if not texts:
        return {}
    cells = nicher.cells(embedder.embed(texts))
    return {nid: cell for nid, cell in zip(nids, cells)}


def _accumulate_and_maybe_freeze(
    state: State,
    on_state: Optional[Dict[str, Any]],
    open_axis: Optional[Any],
    open_vecs: np.ndarray,
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    spec: AxesSpec,
    embedder,
    seed: int,
    open_niches: int = OPEN_NICHES,
    freeze_factor: int = OPEN_NICHE_FREEZE_FACTOR,
) -> None:
    """Grow the mechanism-embedding buffer; freeze the partition once it's full.

    Until ``freeze_factor * open_niches`` open-axis embeddings have accumulated we
    stay on the cold-start partition (only the buffer grows). On the cycle that
    crosses the threshold we fit k-means **once**, persist the frozen centroids,
    and re-key the archive onto them (:meth:`Archive.rekey_open_axis`) so existing
    niche ids migrate without being scrambled. After freezing we never refit.
    """
    if open_axis is None:
        return
    on_state = dict(on_state or {})
    if on_state.get("frozen"):
        return  # already frozen — niche ids are fixed

    accum: List[List[float]] = list(on_state.get("accum", []))
    if open_vecs.shape[0]:
        accum.extend([[float(x) for x in v] for v in open_vecs])

    threshold = freeze_factor * open_niches
    if len(accum) < threshold:
        state.write_open_nicher({"frozen": False, "accum": accum})
        return

    # Threshold reached: fit once, freeze, and re-bucket the archive.
    nicher = archive_mod.CVTNicher.fit(
        np.asarray(accum, dtype=np.float32), k=open_niches, seed=seed
    )
    cell_by_nid = _elite_open_cells(arc, cand_store, open_axis, embedder, nicher)
    arc.rekey_open_axis(spec, open_axis.name, cell_by_nid)
    # keep cand_store display fields consistent with the re-keyed archive: slate
    # items are built from these elite records (`_slate_item`), so an elite placed
    # before the freeze would otherwise display a stale niche_id/coords. Elites are
    # the only candidates ever shown, so non-elite history is left untouched.
    for niche in arc.niches.values():
        rec = cand_store.get(niche.elite_id)
        if rec is not None:
            rec["niche_id"] = niche.id
            rec["coords"] = dict(niche.coords)

    frozen = nicher.to_dict()
    frozen["frozen"] = True
    state.write_open_nicher(frozen)


def _open_axis_status(
    state: State, spec: AxesSpec, open_niches: int, freeze_factor: int
) -> Dict[str, Any]:
    """Progress of the data-adaptive open-axis partition toward its one-time freeze.

    Surfaced by ``ingest`` and ``metrics`` so the fit-once-then-freeze feature is
    observable instead of silent: most short sessions never reach the threshold and
    run entirely on the deterministic cold-start partition (validated good on its
    own), but whether/when a session crosses into the frozen partition should be
    visible. Returns ``{"present": False}`` when the spec has no open axis.
    """
    if spec.primary_axis is None:
        return {"present": False}
    threshold = freeze_factor * open_niches
    on = state.read_open_nicher() or {}
    if on.get("frozen"):
        return {
            "present": True,
            "frozen": True,
            "partition": "frozen",
            "accumulated": threshold,
            "freeze_threshold": threshold,
            "progress": 1.0,
        }
    accumulated = len(on.get("accum", []))
    return {
        "present": True,
        "frozen": False,
        "partition": "cold_start",
        "accumulated": accumulated,
        "freeze_threshold": threshold,
        "progress": round(min(accumulated / threshold, 1.0), 3) if threshold else 1.0,
    }


def _maybe_prune_state(
    cand_store: Dict[str, Any],
    stored_emb: Dict[str, List[float]],
    keep_ids: set,
    threshold: int,
    stored_mech_emb: Optional[Dict[str, List[float]]] = None,
) -> int:
    """Drop candidate records + embeddings that are never read again, in place.

    Only runs once the store exceeds ``threshold`` (``0`` disables it). The keep set
    must be everything still referenced after the cycle: archive **elites** (dedup /
    novelty / slate), **pins** (parents), and the ids in preference **comparisons**
    (recall's learned ``preferred_values``). Everything else is dead weight — kept
    only as display history — so pruning it bounds the O(n) whole-file rewrite cost
    of long sessions without changing any engine output. Returns the count pruned.
    """
    if threshold <= 0 or len(cand_store) <= threshold:
        return 0
    drop = [cid for cid in cand_store if cid not in keep_ids]
    for cid in drop:
        cand_store.pop(cid, None)
        stored_emb.pop(cid, None)
        if stored_mech_emb is not None:
            stored_mech_emb.pop(cid, None)
    # stored_emb ids are a subset of cand_store ids in practice, but sweep any
    # orphaned (non-kept) embeddings too so the two stores stay aligned.
    for cid in [c for c in stored_emb if c not in keep_ids and c not in cand_store]:
        stored_emb.pop(cid, None)
    # Mirror the orphan sweep for the parallel mechanism store (advisory; S4).
    if stored_mech_emb is not None:
        for cid in [c for c in stored_mech_emb
                    if c not in keep_ids and c not in cand_store]:
            stored_mech_emb.pop(cid, None)
    return len(drop)


def _empty_cycle(
    state: "State",
    arc: "archive_mod.Archive",
    spec: AxesSpec,
    econfig: "config.EngineConfig",
) -> Dict[str, Any]:
    """Result dict for a generation with no candidates to ingest.

    Mirrors the normal-cycle response schema (the advisory keys present with
    neutral defaults) so a JSON consumer never KeyErrors on an empty generation.
    Nothing is persisted — an empty generation is a no-op for archive/monitor/
    sensor state, so the cycle and window counters are read but never advanced.
    """
    meta = state.read_meta()
    gen_index = int(meta.get("cycles", 0))
    mon = monitor.evaluate(
        np.zeros((0, 1)), arc.niche_counts(),
        baseline=list(meta.get("cos_window", [])),
        cos_threshold=econfig.monitor_cos_threshold,
        entropy_threshold=econfig.monitor_entropy_threshold,
        margin=econfig.monitor_margin,
        cos_ceiling=econfig.monitor_cos_ceiling,
        min_baseline=econfig.monitor_min_baseline,
    )
    mon["submitted"] = 0
    mon["target_candidates"] = int(meta.get("candidates_per_generation", 0) or 0)
    mon["under_generation"] = False
    # An empty generation is a no-op for the erosion sensor (no survivor novelty to
    # feed it), so report the LAST persisted streak's flag rather than a hard False —
    # an in-progress collapse streak shouldn't read as "not eroding" just because one
    # generation arrived empty. Nothing is persisted here, so the streak is unchanged.
    mon["variety_eroding"] = (
        int(meta.get("erosion_streak", 0)) >= econfig.erosion_persist
    )
    in_explore = (
        econfig.explore_until_generation > 0
        and gen_index < econfig.explore_until_generation
    )
    return {
        "slate": [],
        "ask_pairs": [],
        "ask_policy": {
            "generation": gen_index,
            "phase": "explore" if in_explore else "refine",
            "ask_sim_weight_effective": econfig.ask_sim_weight_for_generation(gen_index),
        },
        "monitor": mon,
        "parents": [],
        "slate_mechanism_novelty": None,
        "open_axis": _open_axis_status(
            state, spec, econfig.open_niches, econfig.open_niche_freeze_factor
        ),
    }


def _guard_embedding_dim(
    stored_emb: Dict[str, List[float]], vecs: np.ndarray, embedder, project: str
) -> None:
    """Fail loudly if a prior embedder wrote incompatible-dimension vectors.

    Mixing dimensions within a project would give dedup/novelty ragged arrays.
    """
    if not stored_emb:
        return
    existing_dim = len(next(iter(stored_emb.values())))
    if existing_dim != vecs.shape[1]:
        raise config.ConfigError(
            f"project {project!r} has {existing_dim}-dim embeddings but the "
            f"current embedder ({embedder.name!r}) produces {vecs.shape[1]}-dim "
            f"vectors; reuse the original embedder ($CAMBRIAN_EMBEDDER) or "
            f"start a fresh project."
        )


def _cap_by_novelty(arc: "archive_mod.Archive", ids: List[str], cap: int) -> List[str]:
    """Keep the ``cap`` most-novel of ``ids`` (all of them, order untouched, when already within
    cap). The ONE bounding rule for every quadratic pass over elite vectors — the dedup/novelty
    reference and the metrics mechanism-spread snapshot."""
    if len(ids) <= cap:
        return ids
    novelty_by_elite = {n.elite_id: n.novelty for n in arc.niches.values()}
    return sorted(ids, key=lambda eid: novelty_by_elite.get(eid, 0.0), reverse=True)[:cap]


def _novelty_reference_ids(
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cap: Optional[int] = None,
) -> List[str]:
    """Elite ids used as the dedup/novelty reference, capped to the most-novel.

    At or below ``cap`` this is exactly the embedded elites in archive order, so
    small-project behavior is unchanged. Above ``cap`` it keeps the ``cap``
    most-novel elites, bounding the O(n·m) dedup and k-NN novelty passes. ``cap``
    defaults to the module-level :data:`NOVELTY_REF_CAP`, read at call time so it
    stays overridable.
    """
    if cap is None:
        cap = NOVELTY_REF_CAP
    return _cap_by_novelty(arc, [eid for eid in arc.elite_ids() if eid in stored_emb], cap)


def _stack_embeddings(
    ids: List[str], stored_emb: Dict[str, List[float]], dim: int
) -> np.ndarray:
    """``(len(ids), dim)`` float32 matrix of the given ids' vectors (empty if none)."""
    if ids:
        return np.asarray([stored_emb[i] for i in ids], dtype=np.float32)
    return np.zeros((0, dim), dtype=np.float32)


def _place_survivors(
    survivors: List[Candidate],
    surv_vecs: np.ndarray,
    cells: List[Optional[int]],
    novelties: np.ndarray,
    open_axis: Optional[Any],
    spec: AxesSpec,
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    stored_emb: Dict[str, List[float]],
    open_vecs: Optional[np.ndarray] = None,
    mech_novelties: Optional[np.ndarray] = None,
    stored_mech_emb: Optional[Dict[str, List[float]]] = None,
) -> bool:
    """Insert each survivor into its niche; record its candidate + embedding.

    When the mechanism-space args are supplied (``open_vecs`` / ``mech_novelties`` /
    ``stored_mech_emb``) each survivor also gets an advisory ``mechanism_novelty`` on
    its record and its mechanism embedding persisted to the parallel store. They are
    defaulted so the function stays safe to call without them (measurement only — see
    CLAUDE.md / S4); they never touch placement, the elite rule, or the surface store.

    Returns ``True`` when this call actually changed ``stored_mech_emb`` — ``_persist_cycle``
    uses it to skip a byte-identical rewrite of the advisory S4 store in the common
    no-open-axis case.
    """
    mech_dirty = False
    for idx, c in enumerate(survivors):
        ocell = {}
        if open_axis is not None and cells[idx] is not None:
            ocell = {open_axis.name: cells[idx]}
        nid, coords = archive_mod.compute_niche(c.descriptor, spec, ocell)
        nov = float(novelties[idx])
        arc.place(c.id, nid, coords, fitness=c.fitness, novelty=nov)
        record = {**c.to_dict(), "niche_id": nid, "coords": coords, "novelty": nov}
        if mech_novelties is not None:
            record["mechanism_novelty"] = round(float(mech_novelties[idx]), 4)
        cand_store[c.id] = record
        stored_emb[c.id] = [float(x) for x in surv_vecs[idx]]
        if (stored_mech_emb is not None and open_vecs is not None
                and open_axis is not None and idx < open_vecs.shape[0]):
            stored_mech_emb[c.id] = [float(x) for x in open_vecs[idx]]
            mech_dirty = True
    return mech_dirty


def _select_slate(
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cand_store: Dict[str, Any],
    spec: AxesSpec,
    seed: int,
    max_dpp_pool: int = MAX_DPP_POOL,
    quality_weight: float = QUALITY_WEIGHT,
    discards: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """DPP diverse slate over the current niche elites. Returns ``(slate, ids)``.

    ``discards`` are the user's vetoed ids: an elite the user discarded is skipped so
    it stops re-appearing on the slate. This is a human veto applied to the presented
    pool only — the archive, niching, novelty, and the monitor are untouched — so when
    ``discards`` is empty the slate is identical to before.
    """
    discarded = discards or set()
    elites: List[Tuple[str, float, str]] = [
        (niche.elite_id, niche.fitness, nid)
        for nid, niche in arc.niches.items()
        if niche.elite_id and niche.elite_id in stored_emb
        and niche.elite_id not in discarded
    ]
    # cap the pool by novelty for latency
    if len(elites) > max_dpp_pool:
        elites.sort(
            key=lambda e: cand_store.get(e[0], {}).get("novelty", 0.0), reverse=True
        )
        elites = elites[:max_dpp_pool]
    if not elites:
        return [], []

    elite_ids = [e[0] for e in elites]
    elite_vecs = np.asarray([stored_emb[i] for i in elite_ids], dtype=np.float32)
    quality = np.asarray([e[1] for e in elites], dtype=np.float64)
    sel = diversity.select_diverse(
        elite_vecs, k=spec.slate_size, quality=quality, seed=seed,
        quality_weight=quality_weight,
    )
    slate_ids = [elite_ids[i] for i in sel]
    slate = [_slate_item(cand_store[i]) for i in slate_ids]
    return slate, slate_ids


def _persist_cycle(
    state: State,
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cand_store: Dict[str, Any],
    vecs: np.ndarray,
    embedder,
    mon: Dict[str, Any],
    econfig: "config.EngineConfig",
    novelty_window: List[float],
    erosion_streak: int,
    gap_record=None,
    stored_mech_emb: Optional[Dict[str, List[float]]] = None,
    mech_dirty: bool = True,
) -> None:
    """Write archive/embeddings/candidates, bump cycle metadata, roll the window."""
    state.write_archive(arc.to_dict())
    state.write_embeddings(stored_emb)
    # Advisory parallel store (S4). Written when this cycle changed it OR the paths()-listed file
    # is not on disk yet (so it is present — even as an empty {} — after any cycle); an unchanged
    # store (the common no-open-axis case) skips a byte-identical rewrite.
    if mech_dirty or not state.mech_embeddings_path.exists():
        state.write_mech_embeddings(stored_mech_emb or {})
    state.write_candidates(cand_store)
    meta = state.read_meta()
    meta["cycles"] = int(meta.get("cycles", 0)) + 1
    meta["embedder"] = embedder.name
    meta["embedding_dim"] = int(vecs.shape[1])
    meta["engine"] = econfig.to_dict()  # keep the resolved knobs visible/auditable
    # Roll the monitor's calibration window with this generation's mean cosine.
    # The window must track the project's *normal* diversity scale, never the
    # collapse it exists to detect. So once a calibrated baseline exists we exclude
    # any generation the RELATIVE rule flags as too similar — otherwise a sustained
    # collapse trains the baseline up past itself and the flag goes quiet (a
    # "boiling-frog" blind spot). While still bootstrapping (no calibrated baseline
    # yet) we add every generation regardless, so the window can form even under an
    # embedder whose natural cosine scale trips the absolute fallback.
    suppress = mon.get("calibrated") and mon.get("too_similar")
    if int(mon.get("n", 0)) >= 2 and not suppress:
        cos_window = list(meta.get("cos_window", []))
        cos_window.append(float(mon["mean_cosine"]))
        meta["cos_window"] = cos_window[-econfig.monitor_window:]
    # S2 — persist the variety-erosion sensor's OWN state, independent of the
    # cos_window calibration roll above. Already rolled/truncated by the assessor.
    meta["novelty_window"] = list(novelty_window)
    meta["erosion_streak"] = int(erosion_streak)
    # Advisory: accumulate per-cycle gap records so real sessions build before/after
    # evidence. Bounded; never read back by the engine. Independent of cos_window /
    # novelty_window — a separate series.
    if gap_record is not None and "gap" in gap_record:
        gap_log = list(meta.get("gap_log", []))
        gap_log.append({
            "cycle": int(meta["cycles"]),
            **{k: gap_record[k] for k in ("surface_spread", "mechanism_spread", "gap", "corr", "n")},
        })
        meta["gap_log"] = gap_log[-GAP_LOG_CAP:]  # bounded
    state.write_meta(meta)


def ingest(
    project: str,
    candidates,
    axes_source,
    seed: int = 0,
    home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Embed → dedup → place → novelty → archive → DPP → monitor for one cycle.

    The whole-project read-modify-write (archive / candidates / embeddings / meta)
    runs under a best-effort cross-process lock (``State.project_lock``) so two
    concurrent cycles on the same project can't clobber each other's generation (a
    lost update). On lock timeout it proceeds unlocked rather than deadlock.
    """
    spec, econfig = config.load_axes_and_engine(axes_source)
    sess = Session(project, home=home, seed=seed).ensure()
    cand_list = _parse_candidates(candidates)
    if cand_list:
        # Force the embedder's lazy model load BEFORE taking the project lock: the first-ever load
        # can DOWNLOAD the static model (~120 MB), and the lock's staleness window (_LOCK_STALE,
        # 60s) has no heartbeat — a slow download inside the lock would let a concurrent session
        # judge the lock abandoned, steal it, and interleave the read-modify-write this lock exists
        # to serialize. embed([]) resolves the model and returns an empty (0, dim) array; once
        # loaded this is a no-op on every later cycle.
        sess.embedder.embed([])
    with sess.state.project_lock():
        return _ingest_locked(sess, spec, econfig, project, cand_list, seed)


def _ingest_locked(
    sess: Session,
    spec: AxesSpec,
    econfig: "config.EngineConfig",
    project: str,
    cand_list: List[Candidate],
    seed: int,
) -> Dict[str, Any]:
    """One ingest cycle, run while holding the project lock (see :func:`ingest`).

    ``cand_list`` arrives already parsed — :func:`ingest` parses it (and warm-loads the embedder)
    OUTSIDE the lock so no slow, lock-irrelevant work runs inside the serialized section.
    """
    state = sess.state
    # The axes passed in are authoritative for this cycle; snapshot them only on
    # a fresh project so an existing project keeps its original resolved axes.
    prev_axes = state.read_axes()
    fresh = prev_axes is None
    if fresh:
        sess.adopt_spec(spec)
    else:
        # Geometry guard: niche keys are built from the axes list, so ingesting under
        # axes that differ from the project's snapshot would silently mix incompatible
        # niche keys into one archive (and metrics/parents read the snapshot, not these
        # axes). init-project resets geometry on an axes change; ingest must not be the
        # back door around that. Compare ONLY the axes list (what placement uses), not
        # slate_size/domain/unit. Fail loudly so the operator re-inits to change axes.
        if (isinstance(prev_axes, dict)
                and prev_axes.get("axes") != [a.to_dict() for a in spec.axes]):
            raise config.ConfigError(
                f"axes geometry passed to ingest differs from project {project!r}'s "
                f"snapshot; re-run init-project to change axes (it resets the geometry) "
                f"or pass the project's original axes"
            )
        # Engine config stays per-cycle overridable (state_prune_threshold, monitor
        # thresholds, ask weights …), but the open-axis NICHING knobs are pinned to
        # the init snapshot: open_niches / open_niche_freeze_factor set the CVT
        # partition's cell count and freeze point, so changing them mid-session
        # (before the partition freezes) would refit k-means with a different k than
        # the cells already placed in the archive. Pin just those from meta["engine"].
        snap = state.read_meta().get("engine")
        if isinstance(snap, dict):
            econfig = replace(
                econfig,
                open_niches=int(snap.get("open_niches", econfig.open_niches)),
                open_niche_freeze_factor=int(
                    snap.get("open_niche_freeze_factor",
                             econfig.open_niche_freeze_factor)
                ),
            )

    arc = archive_mod.Archive.from_dict(spec, state.read_archive())
    if not cand_list:
        return _empty_cycle(state, arc, spec, econfig)

    stored_emb: Dict[str, List[float]] = state.read_embeddings()
    stored_mech_emb: Dict[str, List[float]] = state.read_mech_embeddings()
    cand_store: Dict[str, Any] = state.read_candidates()

    embedder = sess.embedder
    vecs = embedder.embed([c.text for c in cand_list])
    _guard_embedding_dim(stored_emb, vecs, embedder, project)

    # Existing archive elites seed both dedup and the novelty reference, capped to
    # the most-novel novelty_ref_cap so the per-cycle cost stays bounded.
    existing_ids = _novelty_reference_ids(arc, stored_emb, cap=econfig.novelty_ref_cap)
    existing_vecs = _stack_embeddings(existing_ids, stored_emb, vecs.shape[1])

    tau = econfig.dedup_tau if econfig.dedup_tau is not None else default_dedup_tau(
        embedder.name
    )
    keep, _drop = dedupe(
        vecs, tau=tau, existing=existing_vecs if existing_vecs.shape[0] else None
    )
    survivors = [cand_list[i] for i in keep]
    surv_vecs = vecs[keep] if keep else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    # Open-axis niching is data-adaptive: use the frozen partition if one has been
    # fitted, else the deterministic cold-start partition.
    open_axis = spec.primary_axis
    on_state = state.read_open_nicher()
    frozen_nicher = _frozen_open_nicher(on_state, open_axis)
    open_axis, cells, open_vecs = assign_open_cells(
        spec, [c.descriptor for c in survivors], [c.text for c in survivors],
        embedder, seed, nicher=frozen_nicher, open_niches=econfig.open_niches,
        surface_vecs=surv_vecs,  # fallback items reuse their surface row
    )
    novelties = _survivor_novelty(surv_vecs, existing_vecs, econfig.knn_k)
    # S2 — per-generation survivor mean novelty, fed to the variety-erosion sensor
    # below (a separate, post-dedup series; the monitor still runs on RAW vectors).
    surv_mean_novelty = float(np.mean(novelties)) if novelties.size else None
    # S4 — mechanism-space novelty (advisory). Same k-NN kernel as surface novelty,
    # computed on the mechanism (open-axis) embeddings; never enters selection.
    mech_novelties = _survivor_mechanism_novelty(
        open_vecs, open_axis, existing_ids, stored_mech_emb,
        econfig.knn_k, len(survivors),
    )
    mech_dirty = _place_survivors(
        survivors, surv_vecs, cells, novelties, open_axis,
        spec, arc, cand_store, stored_emb,
        open_vecs=open_vecs, mech_novelties=mech_novelties,
        stored_mech_emb=stored_mech_emb,
    )

    # Accumulate the mechanism embeddings; once enough exist, fit + freeze the
    # partition once and re-key the archive onto the frozen cells.
    _accumulate_and_maybe_freeze(
        state, on_state, open_axis, open_vecs, arc, cand_store, spec,
        embedder, seed, open_niches=econfig.open_niches,
        freeze_factor=econfig.open_niche_freeze_factor,
    )

    # User vetoes (discards) are namespaced by the session domain, like pins. Drop
    # them from the presented slate so a discarded idea stops re-appearing.
    discards = set(sess.state.read_discards(sess.domain))
    slate, slate_ids = _select_slate(
        arc, stored_emb, cand_store, spec, seed,
        max_dpp_pool=econfig.max_dpp_pool, quality_weight=econfig.quality_weight,
        discards=discards,
    )

    # Read meta once: nothing rewrites it until _persist_cycle at the end, so the
    # rolling baseline and the per-generation target both come from one snapshot.
    meta_now = state.read_meta()

    # Monitor the RAW generation (pre-dedup) so a near-duplicate batch still
    # registers as collapsing — dedup would otherwise hide it behind survivors.
    # The baseline is the rolling window of prior generations' mean cosine, so the
    # similarity flag is calibrated to this project rather than a fixed constant.
    baseline = list(meta_now.get("cos_window", []))
    mon = monitor.evaluate(
        vecs, arc.niche_counts(), baseline=baseline,
        cos_threshold=econfig.monitor_cos_threshold,
        entropy_threshold=econfig.monitor_entropy_threshold,
        margin=econfig.monitor_margin,
        cos_ceiling=econfig.monitor_cos_ceiling,
        min_baseline=econfig.monitor_min_baseline,
    )

    # Prefilter guard (soft). The monitor above runs on the SUBMITTED generation,
    # so an agent that generates samey ideas is caught by `too_similar`. The blind
    # spot is the *prefilter* stage the engine never sees: the agent dropping
    # candidates as "off-brief" and cutting variety under cover of validity. We
    # can't see what was dropped, but we can see how many reached ingest — if that
    # is well below the per-generation target, flag it so the skill generates more
    # / prefilters less next round. Sensor is at the submitted-vs-target boundary,
    # NOT post-dedup survivors (dedup is the engine's own job, not the agent's).
    # This NEVER affects `collapsing` or the calibration window — purely advisory.
    target = int(meta_now.get("candidates_per_generation", 0) or 0)
    submitted = len(cand_list)
    mon["submitted"] = submitted
    mon["target_candidates"] = target
    if target > 0 and submitted < econfig.under_generation_ratio * target:
        mon["under_generation"] = True
        mon["under_generation_note"] = (
            f"only {submitted} candidates reached ingest vs target {target} "
            f"(< {econfig.under_generation_ratio:.0%}); possible over-prefiltering "
            f"— generate more / prefilter less next round"
        )
    else:
        mon["under_generation"] = False

    # S2 — variety-erosion sensor (advisory). Feeds the post-dedup survivor mean
    # novelty through the acceleration-of-decay assessor and attaches an advisory
    # flag to `mon`. It NEVER sets or influences `collapsing`, never touches the
    # monitor's calibration window (`cos_window`), and keeps its OWN series
    # (`novelty_window`) and streak counter (`erosion_streak`). It only reports.
    erosion = monitor.assess_variety_erosion(
        meta_now.get("novelty_window", []),
        int(meta_now.get("erosion_streak", 0)),
        surv_mean_novelty,
        submitted_healthy=not mon["under_generation"],
        window=econfig.erosion_window,
        accel_ratio=econfig.erosion_accel_ratio,
        persist=econfig.erosion_persist,
    )
    mon["variety_eroding"] = erosion["variety_eroding"]
    mon["variety_erosion"] = {  # advisory detail; never gates anything
        "streak": erosion["erosion_streak"],
        "slope_earlier": erosion["slope_earlier"],
        "slope_recent": erosion["slope_recent"],
        "note": "advisory; acceleration of survivor-novelty decay with healthy submits; "
                "never affects collapsing or the calibration window",
    }

    # Namespace preference memory by the session domain so ingest is consistent
    # with remember/recall/parents (all share Session's snapshot resolution).
    domain = sess.domain
    comparisons = state.read_comparisons(domain)
    # S3 — generation-aware ask-policy. gen_index is meta_now["cycles"], read before
    # _persist_cycle increments it, so the first ingest is generation 0. With the
    # schedule off (explore_until_generation == 0) this is exactly ask_sim_weight.
    gen_index = int(meta_now.get("cycles", 0))
    eff_sim = econfig.ask_sim_weight_for_generation(gen_index)
    ask_pairs = memory.select_ask_pairs(
        slate, stored_emb, comparisons, max_pairs=2,
        weights=(
            eff_sim,
            econfig.ask_uncertainty_weight,
            econfig.ask_novelty_weight,
        ),
    )
    in_explore = (
        econfig.explore_until_generation > 0
        and gen_index < econfig.explore_until_generation
    )
    ask_policy = {
        "generation": gen_index,
        "phase": "explore" if in_explore else "refine",
        "ask_sim_weight_effective": eff_sim,
    }

    # State hygiene (long sessions): drop candidate records/embeddings nothing reads
    # again. Keep set = archive elites + pins + comparison ids — exactly what
    # dedup/novelty/slate, parents, and recall consume — so output is unchanged.
    # Discards need no entry: their id list lives in discards.json (never pruned), a
    # discarded id that is still an elite is already kept via elite_ids(), and a
    # non-elite discarded id needs only its string, not its candidate record.
    keep_ids = set(arc.elite_ids())
    keep_ids.update(state.read_pins(domain))
    for ev in comparisons:
        if ev.get("type") == "comparison":
            keep_ids.update(i for i in (ev.get("winner"), ev.get("loser")) if i)
    mech_before = len(stored_mech_emb)
    _maybe_prune_state(
        cand_store, stored_emb, keep_ids, econfig.state_prune_threshold,
        stored_mech_emb=stored_mech_emb,
    )
    if len(stored_mech_emb) != mech_before:
        mech_dirty = True  # prune dropped mech entries — the store must be rewritten

    # Advisory surface/mechanism gap (measurement only; off by default). Never touches
    # selection, the monitor, the calibration window, or any gate — it only reads embeddings
    # the cycle already produced. See gap.py / CLAUDE.md.
    gap_record = None
    if econfig.gap_probe:
        try:
            # open_axis is already resolved above (== spec.primary_axis).
            slate_cands = [cand_store[i] for i in slate_ids if i in cand_store]
            slate_surf = _stack_embeddings(
                [c["id"] for c in slate_cands if c["id"] in stored_emb],
                stored_emb, vecs.shape[1],
            )
            if open_axis is not None and len(slate_cands) >= 2:
                descs = [c.get("descriptor", {}) for c in slate_cands]
                texts = [c.get("text", "") for c in slate_cands]
                mech_texts = _open_axis_texts(open_axis, descs, texts)
                # Reuse the persisted mechanism vectors — _place_survivors stored exactly these for
                # every elite placed under an open axis — and embed only the misses (e.g. an elite
                # placed before the S4 store existed). Exact by per-string embedder determinism;
                # this probe used to re-embed the whole slate every cycle.
                rows = [stored_mech_emb.get(c["id"]) for c in slate_cands]
                miss = [i for i, r in enumerate(rows) if r is None]
                if miss:
                    fresh = embedder.embed([mech_texts[i] for i in miss])
                    for j, i in enumerate(miss):
                        rows[i] = fresh[j]
                mech_vecs = np.asarray(rows, dtype=np.float32)
                gap_record = gap.surface_mechanism_gap(slate_surf, mech_vecs)
            else:
                gap_record = {"n": len(slate_cands), "surface_spread": 0.0,
                              "mechanism_spread": 0.0, "gap": 0.0, "corr": None,
                              "skipped": "no open axis or slate < 2"}
        except Exception as exc:  # never let an advisory probe break a cycle
            gap_record = {"skipped": f"gap probe failed ({exc})"}

    _persist_cycle(
        state, arc, stored_emb, cand_store, vecs, embedder, mon, econfig,
        erosion["novelty_window"], erosion["erosion_streak"], gap_record=gap_record,
        stored_mech_emb=stored_mech_emb, mech_dirty=mech_dirty,
    )
    # Advisory slate-level mean mechanism novelty (S4); None when no item carries it.
    _mvals = [s["mechanism_novelty"] for s in slate
              if s.get("mechanism_novelty") is not None]
    result = {
        "slate": slate,
        "ask_pairs": ask_pairs,
        "ask_policy": ask_policy,
        "monitor": mon,
        "parents": slate_ids,
        "slate_mechanism_novelty": (
            round(sum(_mvals) / len(_mvals), 4) if _mvals else None
        ),
        "open_axis": _open_axis_status(
            state, spec, econfig.open_niches, econfig.open_niche_freeze_factor
        ),
    }
    if gap_record is not None:
        result["surface_mechanism_gap"] = gap_record
    return result


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(project: str, home: Optional[Path] = None) -> Dict[str, Any]:
    """Current archive health: entropy, mean cosine, coverage, n."""
    sess = Session(project, home=home)
    arc = archive_mod.Archive.from_dict(sess.spec, sess.state.read_archive())
    stored_emb = sess.state.read_embeddings()
    elite_ids = [i for i in arc.elite_ids() if i in stored_emb]
    # Engine knobs from the persisted meta (fall back to the module defaults for older
    # projects whose meta predates the engine block).
    meta = sess.state.read_meta()
    eng = meta.get("engine") or {}
    open_niches = int(eng.get("open_niches", OPEN_NICHES))
    freeze_factor = int(eng.get("open_niche_freeze_factor", OPEN_NICHE_FREEZE_FACTOR))
    # The mean-pairwise-cosine snapshot is O(E²·d); cap the vectors it runs on to the
    # most-novel ``novelty_ref_cap`` elites so a large archive stays cheap. Entropy and
    # coverage still use the FULL niche occupancy (cheap counts) and ``n`` still reports
    # the true elite count. At/below the cap this is identical to a full computation; ABOVE it,
    # ``mean_cosine`` is over the most-novel subsample (which skews it toward "more diverse"), so
    # ``mean_cosine_n`` below reports the actual sample size — measurement-only, it never gates
    # selection, the monitor, or a verdict.
    cap = int(eng.get("novelty_ref_cap", NOVELTY_REF_CAP))
    cos_ids = _novelty_reference_ids(arc, stored_emb, cap=cap)
    # dim only matters in the empty case (the resulting (0, dim) array is never used in
    # arithmetic); take it from a stored vector when one exists.
    dim = len(next(iter(stored_emb.values()))) if stored_emb else 1
    cos_vecs = _stack_embeddings(cos_ids, stored_emb, dim=dim)
    mon = monitor.evaluate(cos_vecs, arc.niche_counts())
    # Advisory archive-level mechanism spread (S4): mean pairwise distance over the
    # elites' mechanism (open-axis) embeddings. This is ARCHIVE-scoped and distinct
    # from the slate-scoped `mechanism_spread` inside `surface_mechanism_gap` (gap.py).
    # Measurement only — never feeds selection, the monitor, or any gate.
    stored_mech_emb = sess.state.read_mech_embeddings()
    # Same cap discipline as the surface snapshot above: this pass ran the O(N²·d) pairwise matrix
    # over ALL elites' mechanism vectors while the surface pass was already capped, with the comment
    # explaining why. At/below the cap the id set — and so the advisory number — is unchanged; above
    # it, the cap most-novel. mech_spread is measurement-only (never feeds selection, the monitor,
    # or any gate), so the cap cannot perturb behavior.
    mech_ids = _cap_by_novelty(arc, [i for i in elite_ids if i in stored_mech_emb], cap)
    if len(mech_ids) >= 2:
        mdim = len(stored_mech_emb[mech_ids[0]])
        mvecs = np.asarray(
            [stored_mech_emb[i] for i in mech_ids if len(stored_mech_emb[i]) == mdim],
            dtype=np.float32,
        )
        mech_spread = round(float(diversity.mean_pairwise_distance(mvecs)), 4)
    else:
        mech_spread = None
    result = {
        "entropy": mon["entropy"],
        "mean_cosine": mon["mean_cosine"],
        "mean_cosine_n": len(cos_ids),   # elites mean_cosine ran on (< n above the cap)
        "coverage": mon["coverage"],
        "n": len(elite_ids),
        "mechanism_spread": mech_spread,
        "mechanism_n": len(mech_ids),
        "open_axis": _open_axis_status(
            sess.state, sess.spec, open_niches, freeze_factor
        ),
    }
    # Advisory passthrough of the gap probe's per-cycle log (see gap.py / ingest). Present
    # only after gap_probe has run; omitted otherwise so the off-path output is unchanged.
    # Never read back by the engine; never affects archive health, the monitor, or selection.
    gap_log = meta.get("gap_log")
    if gap_log:
        result["gap_log"] = gap_log
    return result


# --------------------------------------------------------------------------- #
# remember / parents
# --------------------------------------------------------------------------- #
def remember(project: str, event: Dict[str, Any],
             home: Optional[Path] = None) -> Dict[str, Any]:
    """Append a comparison/pin to this domain's preference memory."""
    sess = Session(project, home=home).ensure()
    return memory.remember(sess.state, sess.domain, event)


def parents(project: str, k: int = 4, seed: int = 0,
            home: Optional[Path] = None) -> Dict[str, Any]:
    """Diverse parents for the next generation; pinned stepping stones kept.

    Each parent's ``novelty`` is the same variety proxy as on the slate (mean k-NN
    distance to this session's own elites + batch), NOT originality vs. prior art.
    """
    sess = Session(project, home=home, seed=seed)
    arc = archive_mod.Archive.from_dict(sess.spec, sess.state.read_archive())
    stored_emb = sess.state.read_embeddings()
    cand_store = sess.state.read_candidates()
    # Session.domain is the shared snapshot-resolved namespace, so pins are read
    # from the namespace recall/ingest/remember wrote them to.
    pins = sess.state.read_pins(sess.domain)
    discards = sess.state.read_discards(sess.domain)
    elite_ids = [
        i for i in arc.elite_ids() if i in stored_emb and i not in set(discards)
    ]
    chosen = memory.select_parents(elite_ids, stored_emb, pins, k, discards=discards)
    records = []
    for cid in chosen:
        rec = cand_store.get(cid, {})
        records.append(
            {
                "id": cid,
                "text": rec.get("text", ""),
                "coords": rec.get("coords", {}),
                "niche_id": rec.get("niche_id"),
                "novelty": round(float(rec.get("novelty", 0.0)), 4),
                "pinned": cid in pins,
            }
        )
    return {"parents": records}
