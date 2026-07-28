"""Advisory surface/mechanism gap measurement.

The engine embeds two views of each idea: the full idea text (SURFACE) and the open /
``primary_novelty`` axis value, typically ``mechanism`` (MECHANISM). A slate can be spread
in surface space (varied wording) yet clustered in mechanism space (the same approach
reworded) — surface diversity then OVERSTATES approach diversity. This module measures that
gap. Pure measurement: advisory only, never a gate, and never fed into selection geometry
(the DPP ``q``).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from . import diversity


def surface_mechanism_gap(
    surface_vecs: Sequence[Sequence[float]],
    mechanism_vecs: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Spread in each space, their gap, and the pairwise-distance correlation.

    ``gap = surface_spread - mechanism_spread``: positive means the slate is more spread in
    surface space than in mechanism space (surface diversity overstates approach diversity).
    ``corr`` is the Pearson correlation between the pairwise surface distances and the
    pairwise mechanism distances; high ``corr`` means surface-distant pairs are also
    mechanism-distant (little gap). Returns a zero/None result gracefully when there are
    fewer than 2 aligned points.
    """
    surf = np.asarray(surface_vecs, dtype=np.float64)
    mech = np.asarray(mechanism_vecs, dtype=np.float64)
    ns = int(surf.shape[0]) if surf.ndim == 2 else 0
    nm = int(mech.shape[0]) if mech.ndim == 2 else 0
    n = min(ns, nm)
    if n < 2 or ns != nm:
        return {
            "n": n, "surface_spread": 0.0, "mechanism_spread": 0.0,
            "gap": 0.0, "corr": None,
        }
    s_spread = float(diversity.mean_pairwise_distance(surf))
    m_spread = float(diversity.mean_pairwise_distance(mech))
    # upper-triangular cosine distances via the shared pairwise primitive (this module used to
    # re-implement exactly the helper diversity.pairwise_cosine_sims exists to single-home)
    s_pd = 1.0 - diversity.pairwise_cosine_sims(surf)
    m_pd = 1.0 - diversity.pairwise_cosine_sims(mech)
    corr = None
    if s_pd.shape[0] >= 2 and float(np.std(s_pd)) > 1e-9 and float(np.std(m_pd)) > 1e-9:
        corr = float(np.corrcoef(s_pd, m_pd)[0, 1])
    return {
        "n": n,
        "surface_spread": round(s_spread, 4),
        "mechanism_spread": round(m_spread, 4),
        "gap": round(s_spread - m_spread, 4),
        "corr": round(corr, 4) if corr is not None else None,
    }
