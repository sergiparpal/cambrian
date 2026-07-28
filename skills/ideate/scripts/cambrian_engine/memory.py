"""Preference memory + heuristic active learning.

Three jobs:

* **remember** — append the user's A-vs-B comparisons and pins to local memory,
  namespaced per domain so switching domains keeps preferences separate.
* **recall** — summarize that memory for in-context injection next session.
* **active learning** — pick the most *informative* A-vs-B pairs to ask, and
  sample **diverse parents** for the next generation while never dropping a
  pinned stepping stone.

None of this judges novelty; it only reflects the human's revealed preferences
and keeps the search both informed and diverse.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .state import State

# Default weights for pair informativeness (similar + novel + undecided ==
# informative). These are the **fallback defaults** for direct/library callers;
# the ``ingest`` path resolves them from ``EngineConfig`` (per-domain overridable).
# A POSITIVE ``W_SIM`` asks the user to compare *similar* ideas — the boundary where
# the judge is most undecided, best for **learning the preference function**. Set it
# to 0 or negative (via ``engine.ask_sim_weight``) to instead surface *region-
# separating* pairs — better for **exploration**. The two readings are a deliberate
# policy choice the value-gate does not adjudicate, so it is left tunable.
W_SIM = 0.5
W_UNCERTAIN = 0.3
W_NOVELTY = 0.2


# --------------------------------------------------------------------------- #
# remember
# --------------------------------------------------------------------------- #
def remember(state: State, domain: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Append a comparison or pin to this domain's memory."""
    if not isinstance(event, dict) or "type" not in event:
        raise ValueError("event must be an object with a 'type'")
    etype = event["type"]
    if etype == "comparison":
        winner, loser = event.get("winner"), event.get("loser")
        if not winner or not loser:
            raise ValueError("comparison event needs 'winner' and 'loser'")
        state.append_comparison(
            domain,
            {
                "type": "comparison",
                "winner": winner,
                "loser": loser,
                "context": event.get("context", ""),
            },
        )
        return {"ok": True, "type": "comparison", "winner": winner, "loser": loser}
    if etype == "pin":
        cid = event.get("id")
        if not cid:
            raise ValueError("pin event needs an 'id'")
        pins = state.add_pin(domain, cid)
        return {"ok": True, "type": "pin", "id": cid, "pins": pins}
    if etype == "discard":
        cid = event.get("id")
        if not cid:
            raise ValueError("discard event needs an 'id'")
        # A discard is the user's negative lever — the symmetric opposite of a pin.
        # It removes the idea from future slates and parents (and un-pins it).
        discards = state.add_discard(domain, cid)
        return {"ok": True, "type": "discard", "id": cid, "discards": discards}
    raise ValueError(
        f"unknown event type {etype!r} (expected comparison|pin|discard)"
    )


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall(state: State, domain: str, k: int = 10) -> Dict[str, Any]:
    """Summarize memory for injection: recent comparisons, pins, discards, tallies."""
    comparisons = state.read_comparisons(domain)
    pins = state.read_pins(domain)
    discards = state.read_discards(domain)

    cand_store = state.read_candidates()
    wins: Counter = Counter()
    losses: Counter = Counter()
    value_wins: Counter = Counter()  # preferred descriptor values, learned from winners
    # ONE pass under ONE type guard: a second, guard-less loop used to tally value_wins, so a
    # schema-drifted non-comparison event carrying a `winner` key counted toward preferred values
    # while being — correctly — excluded from win_counts.
    for ev in comparisons:
        if ev.get("type") != "comparison":
            continue
        # Guard against schema-drifted/partial records so a missing winner or loser can't be
        # tallied as a ``None`` candidate in the summary.
        winner, loser = ev.get("winner"), ev.get("loser")
        if loser:
            losses[loser] += 1
        if not winner:
            continue
        wins[winner] += 1
        rec = cand_store.get(winner)
        if rec:
            for axis, val in (rec.get("descriptor") or {}).items():
                if isinstance(val, (str, int, bool)):
                    value_wins[f"{axis}={val}"] += 1

    return {
        "domain": domain,
        "preferences": comparisons[-k:],
        "pins": pins,
        "discards": discards,
        "summary": {
            "n_comparisons": len(comparisons),
            "win_counts": dict(wins.most_common(k)),
            "preferred_values": dict(value_wins.most_common(k)),
        },
    }


# --------------------------------------------------------------------------- #
# active learning: which pairs to ask
# --------------------------------------------------------------------------- #
def _compared_set(comparisons: Sequence[Dict[str, Any]]) -> Set[frozenset]:
    out: Set[frozenset] = set()
    for ev in comparisons:
        if ev.get("type") == "comparison":
            winner, loser = ev.get("winner"), ev.get("loser")
            if winner and loser:
                out.add(frozenset({winner, loser}))
    return out


def select_ask_pairs(
    slate: List[Dict[str, Any]],
    emb_by_id: Dict[str, Sequence[float]],
    comparisons: Optional[Sequence[Dict[str, Any]]] = None,
    max_pairs: int = 2,
    weights: Optional[Tuple[float, float, float]] = None,
) -> List[List[Any]]:
    """Pick the most-informative A-vs-B pairs from the slate.

    Informativeness is a weighted sum of (a) embedding **similarity** (a fine
    distinction the model is unsure about → max judge-disagreement), (b) **uncertain**
    relative quality (close fitness), and (c) the **novel** frontier — skipping pairs
    the user already decided. ``weights`` is ``(w_sim, w_uncertainty, w_novelty)``;
    it defaults to the module constants. A non-positive ``w_sim`` flips the policy
    from "compare similar" (learn preferences) to "compare region-separating"
    (explore).
    """
    w_sim, w_unc, w_nov = weights if weights is not None else (
        W_SIM, W_UNCERTAIN, W_NOVELTY
    )
    comparisons = comparisons or []
    decided = _compared_set(comparisons)
    n = len(slate)
    if n < 2:
        return []

    scored: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = slate[i], slate[j]
            ida, idb = a["id"], b["id"]
            if frozenset({ida, idb}) in decided:
                continue  # already resolved by the user
            va = emb_by_id.get(ida)
            vb = emb_by_id.get(idb)
            if va is None or vb is None or len(va) != len(vb):
                # Missing, or mismatched width across an embedder switch — treat as
                # uninformative rather than letting np.dot raise and abort selection.
                sim = 0.0
            else:
                sim = float(np.dot(np.asarray(va), np.asarray(vb)))
            uncertainty = 1.0 - abs(
                float(a.get("fitness", 1.0)) - float(b.get("fitness", 1.0))
            )
            mean_nov = 0.5 * (float(a.get("novelty", 0.0)) + float(b.get("novelty", 0.0)))
            score = w_sim * sim + w_unc * uncertainty + w_nov * mean_nov
            scored.append((score, i, j))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    framing = "region-separating" if w_sim <= 0 else "similar"
    out: List[List[Any]] = []
    for score, i, j in scored[:max_pairs]:
        out.append(
            [
                slate[i]["id"],
                slate[j]["id"],
                f"informative pair (score {score:.2f}): {framing}, undecided, "
                f"on the novel frontier",
            ]
        )
    return out


# --------------------------------------------------------------------------- #
# diverse parents (honoring pins)
# --------------------------------------------------------------------------- #
def select_parents(
    elite_ids: List[str],
    emb_by_id: Dict[str, Sequence[float]],
    pins: List[str],
    k: int,
    discards: Optional[Sequence[str]] = None,
) -> List[str]:
    """Diverse parent ids for the next generation; pins are ALWAYS included.

    Starts from the pinned stepping stones and greedily adds the elites farthest
    (in cosine distance) from everything chosen so far, until ``k`` is reached.
    Pins are never dropped, even if there are more pins than ``k``.

    ``discards`` are the user's vetoed ids — dropped from the elite pool so they are
    never bred from. They cannot collide with pins: ``add_pin``/``add_discard`` keep
    the two sets mutually exclusive (latest action wins), so a discard never strips a
    pin here.
    """
    from .diversity import farthest_point_sampling

    discarded = set(discards or ())

    # pins first, de-duplicated, order preserved — never dropped
    selected: List[str] = []
    seen: set = set()  # membership twin of `selected` (order lives in the list, lookups here)
    for p in pins:
        if p not in seen:
            selected.append(p)
            seen.add(p)

    pool = [
        e for e in elite_ids
        if e not in seen and e in emb_by_id and e not in discarded
    ]
    remaining = max(0, k - len(selected))
    if not pool or remaining == 0:
        return selected

    pool_vecs = np.asarray([emb_by_id[e] for e in pool], dtype=np.float64)
    seed_vecs = [
        np.asarray(emb_by_id[s], dtype=np.float64) for s in selected if s in emb_by_id
    ]

    if not seed_vecs:
        # no usable seed (e.g. pins lack embeddings): plain farthest-point fill
        idx = farthest_point_sampling(pool_vecs, remaining)
        return selected + [pool[i] for i in idx]

    # Seed the farthest-point frontier with the already-chosen vectors, then pick
    # the `remaining` pool items farthest from everything selected so far. We
    # stack [seeds; pool] so the seeds are plain indices into one matrix; picks
    # come back seeds-first, so we keep only the pool half (index >= n_seeds).
    n_seeds = len(seed_vecs)
    combined = np.vstack([np.asarray(seed_vecs, dtype=np.float64), pool_vecs])
    picks = farthest_point_sampling(
        combined, k=n_seeds + remaining, seeds=range(n_seeds)
    )
    selected += [pool[i - n_seeds] for i in picks if i >= n_seeds]
    return selected
