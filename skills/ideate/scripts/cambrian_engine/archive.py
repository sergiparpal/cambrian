"""MAP-Elites archive over the resolved axes.

Each candidate is placed into a **niche** — a discrete cell of descriptor space —
and the archive keeps at most **one elite per niche**. Niche keys combine a
bucket per axis:

* ``categorical`` → the value itself (slugified),
* ``continuous``  → the bin index over its ``range``,
* ``open``        → a CVT (Voronoi) cell over the embedding of the axis value.

The within-niche elite is the higher-``fitness`` candidate (fitness comes from
the judge, upstream); ties break toward higher geometric novelty. The judge is
never called here — geometry owns diversity, and quality only ranks *within* an
already-diverse niche.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Axis, AxesSpec, Niche

# Niche-key slug: lowercase alnum-only, for stable categorical bucket labels.
_NICHE_SLUG_RE = re.compile(r"[^a-z0-9]+")
# The bucket for a genuinely absent categorical value (see `axis_bucket`).
_MISSING_BUCKET = "none"


def _niche_slug(value: Any) -> str:
    """Stable categorical bucket label for one descriptor value.

    A value whose every character is outside ``[a-z0-9]`` slugs to the empty string. Returning a
    shared literal there (as this used to) collapsed EVERY such value into one MAP-Elites niche —
    and since a niche holds one elite, ``Niche``'s elite rule then evicts all but the fittest. On a
    non-Latin brief (the case the default multilingual ``potion-multilingual-128M`` embedder exists
    to serve) that silently voided the categorical coverage axes: ``芸術`` and ``成長`` both bucketed
    as ``none``, so two genuinely different ideas fought over one cell. Distinct raw values must get
    distinct buckets, so hash the raw value when the slug carries no information.

    The sibling ``state._path_slug`` already guards this exact class for filesystem paths; this is
    the same rule, narrowed: only an EMPTY slug is disambiguated, so every ordinary ASCII label
    (``"Young Adults"`` -> ``"young-adults"``) keeps its readable, backward-compatible bucket. An
    empty/whitespace value is genuinely missing and shares ``_MISSING_BUCKET`` with an omitted one —
    as does the literal string ``"none"``, which means the same thing.
    """
    raw = str(value).strip()
    s = _NICHE_SLUG_RE.sub("-", raw.lower()).strip("-")
    if s:
        return s
    if not raw:
        return _MISSING_BUCKET
    # sha1 over utf-8 bytes: deterministic ACROSS PROCESSES, unlike hash(), which is
    # PYTHONHASHSEED-salted — niche ids must be stable between the CLI's separate invocations.
    return f"x-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row (zero rows pass through unchanged)."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class CVTNicher:
    """A **frozen** Voronoi partition of the open-axis embedding space.

    Despite the historical name this is *not* a running centroidal Voronoi
    tessellation: the centroids are fixed for the life of the nicher and a point
    is assigned to the centroid of maximum cosine similarity, so it always lands
    in the same cell. There are two ways to obtain the centroids:

    * **cold start** (the default constructor) — deterministic unit directions
      seeded by ``seed``. The boundaries are arbitrary w.r.t. the data, but they
      let the very first cycles assign deterministically before enough mechanism
      embeddings exist to fit a data-adaptive partition.
    * **fitted-and-frozen** (:meth:`fit`) — k-means centroids learned *once* over
      accumulated mechanism embeddings, L2-normalized, then frozen. Boundaries
      now follow where the data actually lies. The fit happens a single time and
      the centroids are persisted, so niche ids stay stable across later cycles.

    Either way the centroids never change after construction.
    """

    def __init__(
        self,
        dim: int,
        k: int = 16,
        seed: int = 0,
        centroids: Optional[np.ndarray] = None,
    ):
        self.seed = int(seed)
        if centroids is not None:
            self.centroids = _l2_rows(centroids)
            self.k = int(self.centroids.shape[0])
            self.dim = int(self.centroids.shape[1])
            return
        self.dim = int(dim)
        self.k = int(k)
        rng = np.random.default_rng(seed)
        self.centroids = _l2_rows(rng.standard_normal((self.k, self.dim)))

    @classmethod
    def fit(cls, vecs: np.ndarray, k: int, seed: int = 0) -> "CVTNicher":
        """Fit k-means **once** over accumulated embeddings; return a frozen nicher.

        Uses ``sklearn.cluster.KMeans(random_state=seed)`` (deterministic) and
        keeps the L2-normalized cluster centers as the frozen centroids. ``k`` is
        clamped to the number of available points.
        """
        import warnings

        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning

        vecs = np.asarray(vecs, dtype=np.float64)
        n = int(vecs.shape[0])
        k_eff = max(1, min(int(k), n))
        km = KMeans(n_clusters=k_eff, random_state=int(seed), n_init=10)
        # Accumulated idea embeddings are often clustered into fewer than k natural
        # groups (near-duplicate mechanisms, a low-diversity session). KMeans then
        # finds fewer distinct centers than requested and warns; that's expected and
        # harmless here — the surplus centroids sit in sparse regions and assignment
        # still works — so silence just that warning rather than spam the operator.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            km.fit(vecs)
        return cls(dim=int(vecs.shape[1]), k=k_eff, seed=seed,
                   centroids=km.cluster_centers_)

    def cell(self, vec: np.ndarray) -> int:
        vec = np.asarray(vec, dtype=np.float32)
        return int(np.argmax(self.centroids @ vec))

    def cells(self, vecs: np.ndarray) -> List[int]:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape[0] == 0:
            return []
        sims = vecs @ self.centroids.T  # (n, k)
        return [int(i) for i in np.argmax(sims, axis=1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroids": [[float(x) for x in row] for row in self.centroids],
            "k": self.k,
            "dim": self.dim,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVTNicher":
        return cls(
            dim=int(d.get("dim", 0)),
            k=int(d.get("k", 0)),
            seed=int(d.get("seed", 0)),
            centroids=np.asarray(d["centroids"], dtype=np.float32),
        )


def continuous_bin(axis: Axis, value: Any) -> int:
    lo, hi = axis.range  # type: ignore[misc]
    # A MISSING or un-coercible value maps to the NEUTRAL mid-bin, never bin 0. Bin 0 is a real
    # extreme (for `feasibility` it is "far-fetched"), so defaulting a candidate that simply OMITS
    # the axis there would (a) silently bias every such descriptor toward that extreme — the
    # opposite of the coverage axis's intent — and (b) collapse all omitting candidates into ONE
    # bin, dissolving the "one elite per bin" coverage guarantee. A middle default keeps the archive
    # honest when the agent forgets the axis; the guarantee is still only fully realized when the
    # descriptors DO carry it. (A finite-but-out-of-range GARBAGE value still clamps below; a
    # non-finite NaN/inf — an agent emitting "nan" — stays bin 0: that is a bad value, not an
    # omitted one.)
    mid = max(0, axis.bins // 2)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return mid
    # A non-finite value (NaN/inf, e.g. an agent emitting "nan") would make
    # ``int(frac * bins)`` raise; treat it as an out-of-range value -> bin 0.
    if not math.isfinite(v):
        return 0
    if hi <= lo:
        return 0
    frac = (v - lo) / (hi - lo)
    idx = int(frac * axis.bins)
    return max(0, min(axis.bins - 1, idx))


def axis_bucket(axis: Axis, value: Any, open_cell: Optional[int] = None) -> str:
    """Bucket label for one axis (used both for the niche key and display)."""
    if axis.type == "categorical":
        return _MISSING_BUCKET if value is None else _niche_slug(value)
    if axis.type == "continuous":
        return f"b{continuous_bin(axis, value)}"
    # open
    if open_cell is None:
        return "cell?"
    return f"cell{open_cell}"


def compute_niche(
    descriptor: Dict[str, Any],
    spec: AxesSpec,
    open_cells: Optional[Dict[str, int]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Return ``(niche_id, coords)`` for a candidate's descriptor."""
    open_cells = open_cells or {}
    coords: Dict[str, str] = {}
    parts: List[str] = []
    for axis in spec.axes:
        val = descriptor.get(axis.name)
        cell = open_cells.get(axis.name) if axis.type == "open" else None
        bucket = axis_bucket(axis, val, open_cell=cell)
        coords[axis.name] = bucket
        parts.append(f"{axis.name}={bucket}")
    return "|".join(parts), coords


class Archive:
    """One elite per niche, persisted as a plain dict."""

    def __init__(self, spec: AxesSpec):
        self.spec = spec
        self.niches: Dict[str, Niche] = {}
        self.counts: Dict[str, int] = {}

    @classmethod
    def from_dict(cls, spec: AxesSpec, data: Dict[str, Any]) -> "Archive":
        arc = cls(spec)
        for nid, rec in (data.get("niches", {}) or {}).items():
            arc.niches[nid] = Niche.from_dict(rec)
        arc.counts = dict(data.get("counts", {}) or {})
        return arc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "niches": {nid: n.to_dict() for nid, n in self.niches.items()},
            "counts": dict(self.counts),
        }

    def place(
        self,
        candidate_id: str,
        niche_id: str,
        coords: Dict[str, str],
        fitness: float,
        novelty: float,
    ) -> bool:
        """Insert a candidate; return True if it (newly) became the niche elite.

        Elite selection: higher fitness wins; ties break toward higher novelty.
        """
        self.counts[niche_id] = self.counts.get(niche_id, 0) + 1
        cur = self.niches.get(niche_id)
        if cur is None:
            self.niches[niche_id] = Niche(
                id=niche_id, coords=coords, elite_id=candidate_id,
                fitness=fitness, novelty=novelty,
            )
            return True
        better = (fitness > cur.fitness) or (
            fitness == cur.fitness and novelty > cur.novelty
        )
        if better:
            cur.elite_id = candidate_id
            cur.fitness = fitness
            cur.novelty = novelty
            cur.coords = coords
            return True
        return False

    def rekey_open_axis(
        self,
        spec: AxesSpec,
        open_axis_name: str,
        new_cell_by_nid: Dict[str, int],
    ) -> Dict[str, str]:
        """Re-assign every niche's open-axis bucket to a (frozen) cell and merge.

        ``new_cell_by_nid`` maps each current ``niche_id`` to its new open-axis
        cell index. Each niche's id is rebuilt with that cell in place of the old
        open bucket; niches whose rebuilt id now collides are **merged by the
        elite rule** (higher ``fitness`` wins; ties break toward higher
        ``novelty``), and their occupancy counts are summed. This is the one-time
        re-keying done when the open-axis partition freezes, so the archive is
        re-bucketed onto the frozen centroids without being scrambled.

        Returns ``{old_niche_id: new_niche_id}``.
        """
        new_niches: Dict[str, Niche] = {}
        new_counts: Dict[str, int] = {}
        remap: Dict[str, str] = {}
        for old_nid, niche in self.niches.items():
            coords = dict(niche.coords)
            cell = new_cell_by_nid.get(old_nid)
            if cell is not None:
                coords[open_axis_name] = f"cell{cell}"
            new_nid = "|".join(
                f"{a.name}={coords.get(a.name, 'none')}" for a in spec.axes
            )
            remap[old_nid] = new_nid
            cur = new_niches.get(new_nid)
            if cur is None:
                new_niches[new_nid] = Niche(
                    id=new_nid, coords=coords, elite_id=niche.elite_id,
                    fitness=niche.fitness, novelty=niche.novelty,
                )
            else:
                better = (niche.fitness > cur.fitness) or (
                    niche.fitness == cur.fitness and niche.novelty > cur.novelty
                )
                if better:
                    cur.elite_id = niche.elite_id
                    cur.fitness = niche.fitness
                    cur.novelty = niche.novelty
                    cur.coords = coords
            new_counts[new_nid] = new_counts.get(new_nid, 0) + self.counts.get(
                old_nid, 0
            )
        self.niches = new_niches
        self.counts = new_counts
        return remap

    def elite_ids(self) -> List[str]:
        return [n.elite_id for n in self.niches.values() if n.elite_id]

    def niche_counts(self) -> List[int]:
        return [self.counts.get(nid, 0) for nid in self.niches]

    def __len__(self) -> int:
        return len(self.niches)
