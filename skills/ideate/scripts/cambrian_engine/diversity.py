"""Diverse selection via DPP, plus diversity metrics.

A Determinantal Point Process favours subsets whose items are mutually
dissimilar: the probability of a set is proportional to the determinant of the
kernel submatrix, which is the squared volume they span. We use the standard
**fast greedy MAP** inference (Chen et al., 2018) to pick a diverse slate, with a
farthest-point fallback if anything degenerates.

Metrics here (mean pairwise distance, Vendi score) are used by the variety gate to
prove the diverse slate beats a non-diverse baseline.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .config import debug_enabled


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def build_kernel(
    vecs: np.ndarray,
    quality: Optional[np.ndarray] = None,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Build a PSD DPP kernel ``L = diag(q) (X Xᵀ) diag(q) + jitter·I``.

    ``X Xᵀ`` is a Gram matrix (always PSD). ``quality`` weights items so a DPP
    trades off diversity against per-item quality; default is uniform. The jitter
    makes it strictly PD so greedy log-det inference is well-conditioned.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    sim = vecs @ vecs.T
    if quality is not None:
        q = np.asarray(quality, dtype=np.float64).reshape(-1)
        sim = (q[:, None] * q[None, :]) * sim
    sim = sim + jitter * np.eye(n)
    return sim


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def greedy_map_dpp(kernel: np.ndarray, k: int, epsilon: float = 1e-10) -> List[int]:
    """Fast greedy MAP inference for a DPP (Chen et al., 2018).

    Returns up to ``k`` item indices, greedily maximizing the marginal gain in
    log-determinant.

    Variable names follow the paper's notation (kept terse so the algorithm
    lines up with it):

    * ``di2s``   — d_i², the squared marginal gain of adding each item next.
    * ``cis``    — the incremental Cholesky factors built row by row.
    * ``di_opt`` / ``ci_opt`` — the chosen item's d and its Cholesky column.
    * ``eis``    — the new Cholesky row contributed by the chosen item.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    n = kernel.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    cis = np.zeros((k, n))
    di2s = np.diag(kernel).astype(np.float64).copy()
    selected: List[int] = []
    j = int(np.argmax(di2s))
    selected.append(j)
    while len(selected) < k:
        t = len(selected) - 1
        ci_opt = cis[:t, j]
        di_opt = float(np.sqrt(max(di2s[j], 1e-300)))
        row = kernel[j, :]
        eis = (row - ci_opt @ cis[:t, :]) / di_opt
        cis[t, :] = eis
        di2s = di2s - np.square(eis)
        di2s[j] = -np.inf
        j = int(np.argmax(di2s))
        if di2s[j] < epsilon:
            break
        selected.append(j)
    return selected


def farthest_point_sampling(
    vecs: np.ndarray,
    k: int,
    start: int = 0,
    seeds: Optional[Sequence[int]] = None,
) -> List[int]:
    """Max-min cosine-distance greedy selection (also the DPP fallback).

    Returns indices into ``vecs`` that are mutually far apart, each chosen to
    maximize its minimum cosine distance to everything picked so far.

    ``seeds`` pre-selects indices (always kept, order preserved) and seeds the
    frontier — use it to extend an existing selection. When ``seeds`` is None the
    walk starts from ``start``. The result begins with the seeds and grows to
    ``k`` total; if more than ``k`` seeds are given they are all kept (so the
    length is ``max(k, len(seeds))``), since seeds are never dropped.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    dist = 1.0 - vecs @ vecs.T
    if seeds:
        selected = [int(s) for s in seeds]
        # min cosine distance from each row to its nearest already-selected seed
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


def bounded_quality(
    quality: np.ndarray,
    weight: float,
    lo: float = 0.7,
    hi: float = 1.3,
) -> np.ndarray:
    """Map raw judge fitness into a **bounded** multiplicative quality factor.

    The DPP kernel uses quality multiplicatively, so an unbounded fitness from the
    agent could swamp the diversity term. We first affine-rescale the *observed*
    fitness range to ``[lo, hi]`` (uniform fitness → all ones → pure diversity),
    then damp toward uniform by ``weight`` (a quality-diversity knob): ``weight=0``
    gives pure diversity, ``weight=1`` the full ``[lo, hi]`` spread. The result is
    clipped to ``[lo, hi]`` so quality can never dominate the kernel.
    """
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    # Sanitize non-finite fitness (NaN/±inf) in the single home. Left in, they poison qmin/qmax
    # and the affine rescale below into NaN quality, which propagates through build_kernel and
    # makes greedy DPP re-select an already-picked index — the SAME idea twice in one slate.
    # Map any non-finite into the observed FINITE range; if nothing is finite, treat quality as
    # uniform, which is exactly what the equal-fitness branch below already does (pure diversity).
    if not np.all(np.isfinite(q)):
        finite = q[np.isfinite(q)]
        if finite.size:
            fmin, fmax = float(finite.min()), float(finite.max())
            q = np.nan_to_num(q, nan=fmin, posinf=fmax, neginf=fmin)
        else:
            q = np.ones_like(q)
    qmin, qmax = float(q.min()), float(q.max())
    if qmax - qmin < 1e-12:
        rescaled = np.ones_like(q)
    else:
        rescaled = lo + (hi - lo) * (q - qmin) / (qmax - qmin)
    w = float(np.clip(weight, 0.0, 1.0))
    blended = (1.0 - w) * np.ones_like(q) + w * rescaled
    return np.clip(blended, lo, hi)


def select_diverse(
    vecs: np.ndarray,
    k: int,
    quality: Optional[np.ndarray] = None,
    seed: int = 0,
    quality_weight: float = 1.0,
) -> List[int]:
    """Pick ``k`` diverse item indices. Greedy DPP, farthest-point fallback.

    When ``quality`` is given it is bounded by :func:`bounded_quality` (using
    ``quality_weight``) before entering the kernel, so the slate is
    quality-*weighted* diversity: geometry still owns spread and the judge's
    fitness can only nudge the ordering, never collapse diversity.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    if quality is not None:
        quality = bounded_quality(quality, quality_weight)
    try:
        # Tie the early-stop threshold to the kernel jitter. Marginal gains floor at
        # ~jitter for a rank-deficient pool, so an absolute epsilon below the jitter
        # could never fire and the rank-deficiency top-up below would be dead code.
        # 10x the jitter flags "no real diversity left" without tripping on genuinely
        # diverse pools (whose gains are O(1)).
        jitter = 1e-6
        kernel = build_kernel(vecs, quality=quality, jitter=jitter)
        sel = greedy_map_dpp(kernel, k, epsilon=10.0 * jitter)
        if len(sel) >= min(k, n):
            return sel
        # Greedy stopped early (rank-deficient pool): top up with farthest-point,
        # seeded by the current selection so the fill EXTENDS that frontier rather
        # than restarting independently (which could append items near existing picks).
        fallback = farthest_point_sampling(vecs, k, seeds=sel)
        for idx in fallback:
            if idx not in sel:
                sel.append(idx)
            if len(sel) >= k:
                break
        return sel[:k]
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # Expected ways the kernel can degenerate (singular/ill-conditioned,
        # ragged input). Fall back to farthest-point, but re-raise under
        # CAMBRIAN_DEBUG so a genuine bug isn't masked by the fallback.
        if debug_enabled():
            raise
        return farthest_point_sampling(vecs, k)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pairwise_cosine_sims(vecs: np.ndarray) -> np.ndarray:
    """Flat array of cosine similarities over all unordered pairs ``(i < j)``.

    Empty when there are fewer than two rows. Assumes L2-normalized rows, so a
    dot product is the cosine. Shared by ``mean_pairwise_distance`` here and the
    monitor's ``mean_pairwise_cosine`` so the two can never drift apart.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float64)
    sims = vecs @ vecs.T
    return sims[np.triu_indices(n, k=1)]


def mean_pairwise_distance(vecs: np.ndarray) -> float:
    """Average cosine distance over all unordered pairs (higher == more diverse)."""
    pairs = pairwise_cosine_sims(vecs)
    if pairs.size == 0:
        return 0.0
    return float(np.mean(1.0 - pairs))


def vendi_score(vecs: np.ndarray) -> float:
    """Vendi score: effective number of distinct items (exp of von-Neumann
    entropy of the normalized similarity matrix). 1 == all identical, up to n."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    K = (vecs @ vecs.T) / n  # eigenvalues >= 0, sum to 1
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    w = w / w.sum()
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))
