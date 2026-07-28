"""End-to-end self-test with a stubbed LLM and a stubbed human.

No interactive input, no live model. It exercises the whole loop on a
domain-neutral brief + generic axes and then proves two things:

* **Variety gate** — the engine's diverse slate beats a single-shot baseline on
  mean pairwise distance, Vendi score, and niche entropy; DPP selection beats
  naive first-N on the *same* pool, **shuffled** (so first-N isn't trivially the
  near-clones) and **averaged over several seeds** (so the win isn't a fluke);
  and a **null check** confirms DPP does not regress below a random subset on an
  already-uniform pool. This gate proves **variety** geometry *only* — not
  originality vs. the world, nor value; what it leaves unvalidated is spelled out
  in ``variety_gate["coverage_gaps"]``.
* **Induced-collapse reversal** — a samey generation trips the monitor, and once
  diversity pressure is raised the next generation recovers.
* **Live semantic check** (``--live`` only) — a paraphrase is more similar than an
  unrelated sentence under the real default embedder (the torch-free multilingual
  ``static`` model); skipped cleanly when that embedder can't be built/downloaded.
* **Originality probe** (advisory) — distance from the diverse slate to a held-out
  "obvious-set". Reported for visibility only; it is **not** part of the variety
  gate and **never** included in ``ok`` (originality is measured, never selected on).

The stubbed LLM is a canned candidate generator; the stubbed human auto-picks the
highest-novelty idea in each slate.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config, diversity, embed, gap, monitor, originality, pipeline
from .archive import compute_niche
from .state import State

# Margins the variety gate must clear on the seeded fixture.
MARGIN_MPD = 0.10
MARGIN_VENDI = 0.5
# Small explicit margin for the DPP-vs-first-N comparison so it matches the
# discipline of the other gate checks (no marginless float comparison that could go
# flaky if the averages converge). The observed gap is ~0.09, well above this.
MARGIN_DPP = 0.01

# Generations the diverse loop runs before its slate is measured.
SELFTEST_CYCLES = 2

_ANGLES = ["growth", "art", "community", "tech", "ritual", "play", "learning", "ecology"]
_SCOPES = ["personal", "local", "regional", "global"]
_FORMS = ["event", "tool", "story", "game", "service", "installation", "campaign", "wager"]
_MECHANISMS = [
    "referral incentives", "immersive spectacle", "interactive branching",
    "shared private audio", "time-limited scarcity", "peer teaching",
    "data sonification", "ambient signage", "gamified streaks",
    "cooperative puzzle", "reverse auction", "mycelial relay",
]
_NOUNS = [
    "lantern", "tide", "compass", "ember", "orchard", "signal",
    "mosaic", "beacon", "ledger", "kite", "prism", "anvil",
]
_OPERATORS = ["mutation", "analogy", "reframe", "scamper", "biomimicry", "inversion"]


# --------------------------------------------------------------------------- #
# Canned generators (the "stubbed LLM")
# --------------------------------------------------------------------------- #
def diverse_candidates(n: int, gen: int = 0, prefix: str = "d") -> List[Dict[str, Any]]:
    """Vary every axis widely -> well-spread embeddings and niches."""
    out = []
    for i in range(n):
        a = _ANGLES[i % len(_ANGLES)]
        s = _SCOPES[(i * 3 + 1) % len(_SCOPES)]
        f = _FORMS[(i * 5 + 2) % len(_FORMS)]
        m = _MECHANISMS[(i * 7 + gen) % len(_MECHANISMS)]
        noun = _NOUNS[(i * 11 + 3) % len(_NOUNS)]
        bold = round((i % 5) / 4.0, 2)
        out.append(
            {
                "id": f"{prefix}{gen}-{i}",
                "text": f"A {f} for {s} {a}: {m} using a {noun}",
                "descriptor": {
                    "angle": a, "scope": s, "form": f,
                    "boldness": bold, "mechanism": m,
                },
                "genealogy": {"operator_id": _OPERATORS[i % len(_OPERATORS)]},
            }
        )
    return out


def single_shot_candidates(n: int, prefix: str = "s") -> List[Dict[str, Any]]:
    """A clichéd single shot: one angle/form/mechanism, minor wording -> clustered."""
    verbs = ["boost", "increase", "drive", "grow", "raise", "lift", "improve", "scale"]
    out = []
    for i in range(n):
        verb = verbs[i % len(verbs)]
        noun = _NOUNS[i % 2]  # only two nouns -> very samey
        out.append(
            {
                "id": f"{prefix}-{i}",
                "text": f"A campaign to {verb} growth with referral incentives and a {noun}",
                "descriptor": {
                    "angle": "growth", "scope": "global", "form": "campaign",
                    "boldness": 0.3, "mechanism": "referral incentives",
                },
                "genealogy": {"operator_id": "single_shot"},
            }
        )
    return out


def collapsing_candidates(n: int, prefix: str = "x") -> List[Dict[str, Any]]:
    """Samey but not exact-duplicate: comfortably survives dedup (cos < 0.92) yet
    trips the monitor (cos > 0.55). Two words vary to sit mid-band."""
    nouns = ["birds", "trees", "rivers", "gardens", "markets", "bridges", "murals", "cafes"]
    adjs = ["vibrant", "quiet", "historic", "hidden", "bustling", "sleepy", "colorful", "windswept"]
    out = []
    for i in range(n):
        noun = nouns[i % len(nouns)]
        adj = adjs[i % len(adjs)]
        out.append(
            {
                "id": f"{prefix}-{i}",
                "text": f"A neighborhood mural festival celebrating {adj} local {noun} every weekend",
                "descriptor": {
                    "angle": "community", "scope": "local", "form": "event",
                    "boldness": 0.4, "mechanism": "immersive spectacle",
                },
                "genealogy": {"operator_id": "mutation"},
            }
        )
    return out


def dpp_isolation_candidates(n: int, slate_size: int) -> List[Dict[str, Any]]:
    """A pool whose first ``slate_size`` items are near-clones (so naive first-N
    is low-diversity) followed by diverse items (so DPP can find spread). Isolates
    the engine's selection value from the generator's."""
    clones = collapsing_candidates(slate_size, prefix="clone")
    rest = diverse_candidates(max(0, n - slate_size), gen=2, prefix="rest")
    return clones + rest


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def _slate_diversity(vecs: np.ndarray, niche_ids: List[str]) -> Dict[str, float]:
    counts = Counter(niche_ids)
    return {
        "mean_pairwise_distance": round(diversity.mean_pairwise_distance(vecs), 4),
        "vendi": round(diversity.vendi_score(vecs), 4),
        "niche_entropy": round(monitor.shannon_entropy(list(counts.values())), 4),
        "coverage": len(counts),
        "n": int(vecs.shape[0]),
    }


def _place(candidates, spec, embedder, seed):
    """Embed + niche a raw candidate list (no DPP, no dedup).

    Reuses the production placement (`pipeline.assign_open_cells`) so the
    baseline is niched exactly like the engine niches its own candidates.
    """
    texts = [c["text"] for c in candidates]
    descriptors = [c["descriptor"] for c in candidates]
    vecs = embedder.embed(texts)
    open_axis, cells, _open_vecs = pipeline.assign_open_cells(
        spec, descriptors, texts, embedder, seed
    )
    niche_ids = []
    for c, cell in zip(candidates, cells):
        ocell = {open_axis.name: cell} if (open_axis and cell is not None) else {}
        nid, _ = compute_niche(c["descriptor"], spec, ocell)
        niche_ids.append(nid)
    return vecs, niche_ids


def _mechanism_vecs(candidates, spec, embedder, seed):
    """Open-axis (mechanism) embeddings for raw candidates, via the production
    placement path so they match how the engine embeds mechanisms."""
    descriptors = [c["descriptor"] for c in candidates]
    texts = [c["text"] for c in candidates]
    _open_axis, _cells, open_vecs = pipeline.assign_open_cells(
        spec, descriptors, texts, embedder, seed
    )
    return open_vecs


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _diverse_engine_metrics(spec, settings, axes, seed, home, project):
    """Run the full engine loop with a stubbed human; return ``(metrics, state)``.

    ``state`` is the diverse project's handle, reused later for the file checks.
    """
    dproj = f"{project}-diverse"
    pipeline.init_project(dproj, axes, seed=seed, home=home)
    state = State(dproj, home=home)

    last = None
    for gen in range(SELFTEST_CYCLES):
        cands = diverse_candidates(settings.candidates_per_generation, gen=gen)
        last = pipeline.ingest(dproj, cands, axes, seed=seed, home=home)
        _stub_human(pipeline, dproj, last, home)

    slate = last["slate"] if last else []
    emb_store = state.read_embeddings()
    ids = [s["id"] for s in slate]
    vecs = np.asarray([emb_store[i] for i in ids if i in emb_store], dtype=np.float32)
    return _slate_diversity(vecs, [s["niche_id"] for s in slate]), state


def _single_shot_metrics(spec, settings, embedder, seed):
    """Clichéd single-shot baseline: a clustered, low-diversity slate."""
    cands = single_shot_candidates(settings.candidates_per_generation)
    vecs, niches = _place(cands, spec, embedder, seed)
    return _slate_diversity(vecs[: spec.slate_size], niches[: spec.slate_size])


def _dpp_isolation_metrics(spec, settings, embedder, seed, n_seeds=3):
    """DPP vs first-N over a SHUFFLED shared pool, averaged across seeds.

    Shuffling (fixed per-seed) means first-N is a *random* slice of the pool, not
    trivially the leading near-clones, so beating it is a real signal rather than
    an artifact of pool order. We average the mean-pairwise-distance over several
    seeds so a single lucky/unlucky draw doesn't decide the gate. The first seed's
    full metrics are kept as the reported example.
    """
    dpp_mpds, firstn_mpds = [], []
    dpp_example, firstn_example = None, None
    # A pool large enough that a random first-N reliably draws some near-clones,
    # so the geometric gap is unmistakable under any embedder (not just hash).
    pool_n = max(4 * spec.slate_size, settings.candidates_per_generation)
    for s in range(seed, seed + n_seeds):
        cands = dpp_isolation_candidates(pool_n, spec.slate_size)
        vecs, niches = _place(cands, spec, embedder, s)
        order = np.random.default_rng(s).permutation(len(cands))
        vecs = vecs[order]
        niches = [niches[i] for i in order]
        sel = diversity.select_diverse(vecs, k=spec.slate_size, seed=s)
        dpp = _slate_diversity(vecs[sel], [niches[i] for i in sel])
        first_n = _slate_diversity(
            vecs[: spec.slate_size], niches[: spec.slate_size]
        )
        dpp_mpds.append(dpp["mean_pairwise_distance"])
        firstn_mpds.append(first_n["mean_pairwise_distance"])
        if dpp_example is None:
            dpp_example, firstn_example = dpp, first_n
    dpp_example["mean_pairwise_distance_avg"] = round(float(np.mean(dpp_mpds)), 4)
    firstn_example["mean_pairwise_distance_avg"] = round(float(np.mean(firstn_mpds)), 4)
    return dpp_example, firstn_example


def _null_check(spec, settings, embedder, seed, trials=50, eps=0.02):
    """On a uniformly-diverse pool DPP must not regress below a random k-subset.

    There is nothing for selection to "gain" when every item is already spread
    out, so DPP's mean pairwise distance should be at least the random-subset mean
    (minus a small epsilon). This guards against the variety gate rewarding DPP for
    a degenerate pool rather than for genuine selection skill.
    """
    cands = diverse_candidates(max(2 * spec.slate_size, 16), gen=9, prefix="null")
    vecs, _ = _place(cands, spec, embedder, seed)
    sel = diversity.select_diverse(vecs, k=spec.slate_size, seed=seed)
    dpp_mpd = diversity.mean_pairwise_distance(vecs[sel])
    rng = np.random.default_rng(seed)
    rand = [
        diversity.mean_pairwise_distance(
            vecs[rng.choice(len(cands), size=spec.slate_size, replace=False)]
        )
        for _ in range(trials)
    ]
    rand_mean = float(np.mean(rand))
    return {
        "dpp_mpd": round(dpp_mpd, 4),
        "random_mean_mpd": round(rand_mean, 4),
        "passed": bool(dpp_mpd >= rand_mean - eps),
    }


def _value_elite_wins_niche(axes, seed, home, project) -> Dict[str, Any]:
    """The variety gate's first VALUE assertion: higher fitness wins its niche.

    Two candidates share an identical descriptor (so they map to the **same**
    niche) but differ only in ``fitness`` (0.9 vs 0.6 — both inside the judge's
    contractual ``[0, 1]`` range, see ``judge_rubric.md``) and in ``text`` (so
    neither is deduped away). NB: ``[0.7, 1.3]`` is the range ``bounded_quality``
    affine-rescales fitness *into* for the DPP kernel — it is not a valid range for
    fitness itself, and the elite rule below is scale-free regardless. The engine
    must make the higher-fitness candidate the niche elite and surface it on the
    slate. The **sanity swap** — flipping the two fitness values on a fresh project
    — must flip which id is the elite, proving the check exercises the real fitness
    path (``archive.place``'s elite rule), not candidate order or id. This asserts
    only the EXISTING low-weight within-niche fitness behavior; it adds no new
    value power and never touches selection geometry.
    """
    descriptor = {
        "angle": "growth", "scope": "global", "form": "campaign",
        "boldness": 0.3, "mechanism": "referral incentives",
    }

    def _elite_on_slate(fit_hi: float, fit_lo: float, proj: str):
        # Same descriptor -> same niche; distinct text -> both survive dedup.
        cands = [
            {"id": "hi", "text": "A storefront loyalty rebate paid in store credit",
             "descriptor": dict(descriptor), "fitness": fit_hi},
            {"id": "lo", "text": "A weekend pop-up market with neighborhood vendors",
             "descriptor": dict(descriptor), "fitness": fit_lo},
        ]
        pipeline.init_project(proj, axes, seed=seed, home=home)
        res = pipeline.ingest(proj, cands, axes, seed=seed, home=home)
        slate_ids = [s["id"] for s in res["slate"]]
        elite_id = slate_ids[0] if slate_ids else None  # one niche -> one elite
        return elite_id, slate_ids

    norm_elite, norm_slate = _elite_on_slate(0.9, 0.6, f"{project}-value")
    swap_elite, _ = _elite_on_slate(0.6, 0.9, f"{project}-value-swap")
    passed = bool(
        norm_elite == "hi" and "hi" in norm_slate and swap_elite == "lo"
    )
    return {
        "normal_elite": norm_elite,
        "swapped_elite": swap_elite,
        "high_fitness_on_slate": "hi" in norm_slate,
        "passed": passed,
    }


def _live_semantic_check(live: bool) -> Dict[str, Any]:
    """Under the real embedder, a paraphrase must beat an unrelated sentence.

    Only runs on a ``--live`` invocation, and **skips cleanly** (without failing
    the self-test) when the live embedder can't be built — e.g. its package is
    missing or the model weights can't be downloaded offline.
    """
    if not live:
        return {"ran": False, "skipped": True, "reason": "not a --live run"}
    try:
        emb = embed.get_embedder()  # the live default (static), set by run()
        a = emb.embed(["a quiet library for focused study"])[0]
    except Exception as exc:  # pragma: no cover - depends on the environment
        return {
            "ran": False,
            "skipped": True,
            "reason": f"live embedder unavailable ({exc})",
        }
    para = emb.embed(["a calm reading room for concentrated work"])[0]
    unrel = emb.embed(["an explosive monster-truck demolition derby"])[0]
    sim_para = float(np.dot(a, para))
    sim_unrel = float(np.dot(a, unrel))
    return {
        "ran": True,
        "skipped": False,
        "sim_paraphrase": round(sim_para, 4),
        "sim_unrelated": round(sim_unrel, 4),
        "passed": bool(sim_para > sim_unrel),
    }


def _originality_probe(spec, settings, embedder, seed) -> Dict[str, Any]:
    """Advisory: how far the engine's diverse slate sits from an "obvious-set".

    **Advisory only** — never a pass/fail gate, never part of ``ok``, and never
    fed into the DPP ``q`` / selection geometry. It supplies the external-ish
    referent the engine otherwise lacks, as a pure measurement: build a small
    fixture of clichéd one-liners for the generic brief, split it **disjointly**
    into ``O_train`` (first half) and a held-out ``O_test`` (second half), and
    score the slate's distance-to-obvious against **O_test only**. Holding out
    ``O_test`` keeps the measure non-circular — scoring against the half a
    generator was told to avoid would be a Goodhart trap. Skips cleanly (like
    :func:`_live_semantic_check`) if the embedder can't be built/used.
    """
    # A clichéd obvious-set for the generic, growth/marketing-ish selftest brief,
    # mirroring single_shot_candidates' platitude style. NOTE: this is the
    # self-test's *fixture*; the skill builds its own obvious-set live per brief
    # (a construction recipe, not this shared object) — see SKILL.md / Item 4.
    obvious = [
        "Launch a referral program to boost growth",
        "Run a viral social media giveaway campaign",
        "Offer a limited-time discount to drive signups",
        "Partner with influencers to increase reach",
        "Start a points-based loyalty rewards program",
        "Send a retargeting email blast to lapsed users",
    ]
    half = len(obvious) // 2
    # Disjoint split: obvious[:half] is O_train (the repellent a live generator would
    # avoid; the canned diverse_candidates here ignores it). Only the held-out O_test
    # is scored, which keeps the measure non-circular.
    o_test = obvious[half:]
    try:
        o_test_vecs = embedder.embed(o_test)
        # The engine's diverse slate: place diverse candidates, let DPP pick.
        diverse = diverse_candidates(settings.candidates_per_generation)
        dvecs, _ = _place(diverse, spec, embedder, seed)
        sel = diversity.select_diverse(dvecs, k=spec.slate_size, seed=seed)
        slate_vecs = dvecs[sel]
        # A clearly clichéd slate, for the sanity comparison below.
        cliche = single_shot_candidates(settings.candidates_per_generation)
        cvecs, _ = _place(cliche, spec, embedder, seed)
        cliche_vecs = cvecs[: spec.slate_size]
    except Exception as exc:  # pragma: no cover - depends on the environment
        return {"ran": False, "skipped": True, "reason": f"embedder unavailable ({exc})"}
    slate = originality.originality_scores(slate_vecs, o_test_vecs)
    cliche_score = originality.originality_scores(cliche_vecs, o_test_vecs)
    return {
        "ran": True,
        "skipped": False,
        "slate_originality_vs_heldout": slate["slate_mean"],
        # Printed sanity only — the diverse slate should sit further from the
        # obvious set than a clichéd one. NOT part of `ok`; never gates anything.
        "cliche_baseline_vs_heldout": cliche_score["slate_mean"],
        "sanity_diverse_more_original": bool(
            slate["slate_mean"] > cliche_score["slate_mean"]
        ),
        "note": "advisory; not a gate; held-out half so it isn't circular",
    }


def _surface_mechanism_gap_probe(spec, settings, embedder, seed) -> Dict[str, Any]:
    """Advisory: the diverse slate's surface vs mechanism spread, plus the gap that a
    surface-diverse / mechanism-monotone slate shows by contrast.

    **Advisory only** — never a gate, never part of ``ok``, never fed into selection.
    Sanity: holding the SAME (varied) surface texts but collapsing every mechanism to one
    string must produce a LARGER gap (lower mechanism_spread) than the genuinely-varied
    slate. Skips cleanly (like :func:`_live_semantic_check`) if the embedder can't be used.
    """
    try:
        diverse = diverse_candidates(settings.candidates_per_generation)
        surf, _ = _place(diverse, spec, embedder, seed)
        sel = diversity.select_diverse(surf, k=spec.slate_size, seed=seed)
        slate_surf = surf[sel]
        slate_cands = [diverse[i] for i in sel]
        mech_varied = _mechanism_vecs(slate_cands, spec, embedder, seed)
        # Same surface, mechanism collapsed to a single string -> the gap by construction.
        mono_cands = [
            dict(c, descriptor={**c["descriptor"], "mechanism": "referral incentives"})
            for c in slate_cands
        ]
        mech_mono = _mechanism_vecs(mono_cands, spec, embedder, seed)
    except Exception as exc:  # pragma: no cover - depends on the environment
        return {"ran": False, "skipped": True, "reason": f"embedder unavailable ({exc})"}
    diverse_slate = gap.surface_mechanism_gap(slate_surf, mech_varied)
    monotone = gap.surface_mechanism_gap(slate_surf, mech_mono)
    return {
        "ran": True,
        "skipped": False,
        "diverse_slate": diverse_slate,
        "mechanism_monotone_fixture": monotone,
        # Printed sanity only — NOT part of `ok`, never gates anything.
        "sanity_monotone_gap_larger": bool(monotone["gap"] > diverse_slate["gap"]),
        "note": "advisory; surface vs mechanism spread; never a gate, never in selection",
    }


def _collapse_reversal(spec, settings, axes, seed, home, project):
    """A samey generation must trip the monitor; the next diverse one recovers.

    First warm up the monitor's rolling baseline with a couple of diverse
    generations, so the similarity flag is calibrated to *this embedder's*
    natural cosine scale before collapse/recovery are judged. Without the warm-up
    the absolute fallback (tuned for the hash fixture) misfires under a sentence
    embedder, where even diverse short ideas sit well above 0.55 — exactly the
    misfire the calibrated monitor exists to prevent. This exercises the relative
    path end-to-end under both embedders.
    """
    cproj = f"{project}-collapse"
    pipeline.init_project(cproj, axes, seed=seed, home=home)
    for g in range(2):
        pipeline.ingest(
            cproj,
            diverse_candidates(settings.candidates_per_generation, gen=g, prefix="warm"),
            axes, seed=seed, home=home,
        )
    collapsed = pipeline.ingest(
        cproj, collapsing_candidates(8), axes, seed=seed, home=home
    )
    recovered = pipeline.ingest(
        cproj,
        diverse_candidates(settings.candidates_per_generation, gen=5, prefix="rec"),
        axes, seed=seed, home=home,
    )
    col_mon, rec_mon = collapsed["monitor"], recovered["monitor"]
    checks = {
        "collapse_detected": bool(col_mon["collapsing"]),
        "recovered_quiet": not rec_mon["collapsing"],
        "diversity_recovered": rec_mon["mean_cosine"] < col_mon["mean_cosine"],
    }
    return {
        "collapsed_monitor": col_mon,
        "recovered_monitor": rec_mon,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(project: str = "selftest", live: bool = False, seed: int = 0,
        home: Optional[Path] = None) -> Dict[str, Any]:
    # The self-test must be hermetic. With no explicit home (the CLI path) it
    # would otherwise write to the persistent default home under fixed project
    # names; init-project never resets a project, so MAP-Elites occupancy would
    # accumulate across runs and eventually drive normalized niche entropy below
    # the collapse threshold — falsely failing the collapse-reversal check. Run
    # in a throwaway temp home instead. Callers that pass a home (the test suite)
    # already isolate themselves and skip this.
    if home is None:
        with tempfile.TemporaryDirectory(prefix="cambrian-selftest-") as tmp:
            return run(project=project, live=live, seed=seed, home=Path(tmp))

    # Deterministic embedder unless a live run is explicitly requested. Save and
    # restore $CAMBRIAN_EMBEDDER so running the self-test never mutates the
    # caller's global env (the test suite calls run() in-process); the cache is
    # reset on both ends so neither the self-test nor the caller inherits a stale
    # embedder built under the other's setting.
    prev_embedder = os.environ.get(embed.ENV_VAR)
    # A live run exercises the real default embedder (the torch-free multilingual
    # 'static' model); non-live stays on the deterministic, download-free 'hash'.
    os.environ[embed.ENV_VAR] = embed.DEFAULT_PROVIDER if live else "hash"
    embed.reset_cache()
    try:
        spec = config.load_generic_axes()
        settings = config.load_session_settings(config.generic_axes_path())
        axes = spec.to_dict()
        embedder = embed.get_embedder()

        engine_metrics, state = _diverse_engine_metrics(
            spec, settings, axes, seed, home, project
        )
        base_metrics = _single_shot_metrics(spec, settings, embedder, seed)
        dpp_metrics, firstn_metrics = _dpp_isolation_metrics(
            spec, settings, embedder, seed
        )

        null_check = _null_check(spec, settings, embedder, seed)
        value_check = _value_elite_wins_niche(axes, seed, home, project)

        variety_gate = {
            "engine": engine_metrics,
            "single_shot": base_metrics,
            "dpp_on_pool": dpp_metrics,
            "first_n_on_pool": firstn_metrics,
            "null_check": null_check,
            "value_check": value_check,
            # This gate validates VARIETY geometry plus one narrow VALUE assertion
            # (higher within-niche fitness wins its niche). State plainly what is
            # still uncovered so the name can't be mistaken for a full value gate.
            "coverage_gaps": [
                "originality: no external referent is checked (advisory probe only)",
                "value: partial — higher within-niche fitness is asserted to win "
                "its niche; cross-niche value (which niches matter) is unguarded",
            ],
            "checks": {
                "mpd_beats_single_shot": engine_metrics["mean_pairwise_distance"]
                > base_metrics["mean_pairwise_distance"] + MARGIN_MPD,
                "vendi_beats_single_shot": engine_metrics["vendi"]
                > base_metrics["vendi"] + MARGIN_VENDI,
                "entropy_beats_single_shot": engine_metrics["niche_entropy"]
                > base_metrics["niche_entropy"],
                # averaged over shuffled seeds, so it isn't an artifact of pool order
                "dpp_beats_first_n": dpp_metrics["mean_pairwise_distance_avg"]
                > firstn_metrics["mean_pairwise_distance_avg"] + MARGIN_DPP,
                # DPP doesn't regress below random when there's nothing to gain
                "null_no_regression": null_check["passed"],
                # first VALUE check: higher within-niche fitness wins its niche
                # (and the swap sanity flips the elite). See _value_elite_wins_niche.
                "value_elite_wins_niche": value_check["passed"],
            },
        }
        variety_gate["passed"] = all(variety_gate["checks"].values())

        reversal = _collapse_reversal(spec, settings, axes, seed, home, project)
        semantic = _live_semantic_check(live)
        # Advisory only: measures distance-to-obvious against a held-out half. It is
        # deliberately NOT added to variety_gate["checks"] and NOT part of `ok`.
        # Never feed originality into the DPP `q` / selection geometry.
        originality_probe = _originality_probe(spec, settings, embedder, seed)
        # Advisory only: surface vs mechanism spread of the slate. Deliberately NOT a
        # variety_gate check and NOT part of `ok`; never fed into selection geometry.
        gap_probe = _surface_mechanism_gap_probe(spec, settings, embedder, seed)

        written = {
            name: Path(p).exists()
            for name, p in state.paths().items()
            if name != "root"
        }
        # open_nicher.json is written only for specs WITH an open axis; it's reported
        # above for visibility but kept out of the gate (the generic axes here do have
        # one, but the gate stays honest about what is actually mandatory).
        files_ok = all(v for k, v in written.items() if k != "open_nicher")
        # A skipped semantic check doesn't fail the gate; a ran-and-failed one does.
        semantic_ok = (not semantic.get("ran")) or bool(semantic.get("passed"))

        ok = bool(
            variety_gate["passed"] and reversal["passed"] and files_ok and semantic_ok
        )
        return {
            "ok": ok,
            "variety_gate": variety_gate,
            "collapse_reversal": reversal,
            "live_semantic": semantic,
            "originality_probe": originality_probe,
            "gap_probe": gap_probe,
            "state_files_written": written,
            "cycles": SELFTEST_CYCLES,
            "embedder": embed.DEFAULT_PROVIDER if live else "hash",
        }
    finally:
        if prev_embedder is None:
            os.environ.pop(embed.ENV_VAR, None)
        else:
            os.environ[embed.ENV_VAR] = prev_embedder
        embed.reset_cache()


def _stub_human(pipeline, project, result, home) -> None:
    """Pick the best idea by a blend of judge ``fitness`` and intra-session ``novelty``.

    Ranks each slate item by ``0.5 * fitness + 0.5 * novelty`` (previously novelty
    alone), so the stubbed human models *value* as well as variety — letting the
    gate catch a value regression rather than only a diversity one. Records a
    comparison (winner vs loser) and pins the winner.
    """
    slate = result.get("slate", [])
    if len(slate) < 2:
        return
    ranked = sorted(
        slate,
        key=lambda s: 0.5 * s.get("fitness", 1.0) + 0.5 * s.get("novelty", 0.0),
    )
    winner, loser = ranked[-1]["id"], ranked[0]["id"]
    pipeline.remember(
        project, {"type": "comparison", "winner": winner, "loser": loser}, home=home
    )
    pipeline.remember(project, {"type": "pin", "id": winner}, home=home)
