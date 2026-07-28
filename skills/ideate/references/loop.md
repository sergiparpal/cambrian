# The full ideate loop

This is the authoritative procedure. SKILL.md is the summary; when in doubt,
follow this file. Everything is domain-agnostic: the only domain knowledge is the
**axes** you resolve in step 1, per session.

Set once:

```
ENGINE = "<PYBIN>" -m cambrian_engine    # <PYBIN> = contents of the engine-python.txt
                                           # pointer (see step 0 for its location)
PROJECT = <short slug for this session, e.g. "campaign-jun14">
```

All engine commands read/write JSON and take `--project PROJECT`. Randomized
steps take `--seed <int>` — reuse the same seed within a session for
reproducibility.

**Scratch files go in the project's `tmp/`, never your cwd.** Once the engine is
ready (step 0), resolve the per-project scratch dir and reuse it for every hand-off
file (`axes.json`, `candidates.json`, `event.json`):

```
ENGINE paths --project PROJECT      # -> {"root","meta","axes",...,"tmp"}; sets up the dir
TMP = <the "tmp" field of that output>   # e.g. ~/.cambrian/<project>/tmp
```

Writing scratch files under `$TMP` (inside the state home, `~/.cambrian`
or `$CAMBRIAN_HOME`) keeps the user's working directory clean and avoids
collisions across concurrent sessions. Never write these files to the cwd.

---

## 0. Ensure the engine

The engine venv **auto-provisions in the background** when the plugin loads (a
`SessionStart` hook runs `bootstrap.py`), so it is usually ready before you need it.
Find the interpreter pointer at the **first** path that exists:

- `${CLAUDE_PLUGIN_DATA}/venv/engine-python.txt`  — marketplace install (the venv
  lives in the plugin's persistent data dir, so it survives plugin updates);
- `${CLAUDE_SKILL_DIR}/.venv/engine-python.txt`    — dev install (`--plugin-dir .`
  or `bash scripts/setup.sh`).

Read that file; its contents are `<PYBIN>`.

**If neither pointer exists yet**, the one-time setup is still running, hasn't
started, **or failed in the background**. The detached worker logs to
`provision.log` next to the venv — check it **first** so you don't re-run blind:

```
# the log sits at <venv-parent>/provision.log, i.e. the FIRST that exists:
tail -n 40 "${CLAUDE_PLUGIN_DATA}/provision.log"   # marketplace install
tail -n 40 "${CLAUDE_SKILL_DIR}/provision.log"     # dev install
```

If the tail shows a real failure (e.g. a wheel won't compile, no network, Python
too old), relay it to the user — that is the actual diagnosis. Then tell the user
once that you're setting up the engine (a one-time, multi-minute download of ML
libraries + a small embedding model) and finish/await it in the foreground
(idempotent; waits for any in-progress background provision):

```
"<PY>" "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py" --venv "${CLAUDE_PLUGIN_DATA}/venv"
# <PY> = python3 (macOS/Linux/WSL) or py / python (Windows)
```

`bootstrap.py` creates the venv (using `uv` if it is on PATH, else `python -m venv`)
and installs deps — the **static** multilingual CPU embedder
(`minishlab/potion-multilingual-128M`, no API key, ~120 MB, torch-free) by default.
Re-read the pointer afterwards. **If it still fails**, show the fresh tail of
`provision.log` again alongside the foreground error and stop — the combination is
the actionable diagnosis (e.g. Python 3.11+ missing, or a wheel that won't build on
this OS). For the higher-fidelity English-only embedder, the user installs
`requirements-local.txt` and sets `CAMBRIAN_EMBEDDER=local`; a hosted embedder is
`CAMBRIAN_EMBEDDER=api` plus provider env vars (a stub until wired up).

---

## 1. Resolve axes (per session)

Diversity is only meaningful relative to descriptor axes. Resolve them with this
cascade, then snapshot them.

**(a) Named domain.** If the user names a domain that has a config in
`${CLAUDE_SKILL_DIR}/config/domains/examples/` (e.g. they say "marketing
campaign"), read that YAML and use it.

**(b) Inferred + confirmed.** Otherwise infer 4–6 axes from the brief following
`axis_inference.md`. Mark exactly one `open` axis as `primary_novelty: true` —
this is the main novelty carrier. Then ask the user ONE short confirmation
question, e.g.:

> I'll spread ideas across these axes: **audience, register, format, edginess
> (0–1), and mechanism** (the main novelty axis). Running at *balanced* reach —
> say *wild* for further-out or *conservative* for safer. Want to change any?

Accept a quick tweak or an "ok". Do not block for long.

The **same one-line confirm states the reach** (see `SKILL.md` → *Reach*): pick it from
the user's phrasing if they gave one, else default to `balanced`. Reach is a
language-layer directive only — it changes the descriptors you write and the operators
you pick, never anything the engine computes. The user can change it any round.

**(c) Generic fallback.** If you cannot infer well-separated axes, load
`${CLAUDE_SKILL_DIR}/config/domains/generic.yaml` and tell the user you used
neutral axes.

Write the resolved axes to `$TMP/axes.json` (the scratch dir from the top of this
file) in this shape:

```json
{
  "domain": "ad-hoc:campaign-ideas",
  "unit_of_generation": "concept",
  "axes": [
    {"name": "audience", "type": "categorical"},
    {"name": "register", "type": "categorical"},
    {"name": "format", "type": "categorical"},
    {"name": "edginess", "type": "continuous", "range": [0, 1]},
    {"name": "mechanism", "type": "open", "primary_novelty": true}
  ],
  "slate_size": 6,
  "candidates_per_generation": 12
}
```

Then:

```
ENGINE init-project --project PROJECT --axes $TMP/axes.json --seed 7
ENGINE recall --project PROJECT          # returns {domain, preferences, pins, discards}
```

Inject the `recall` output into your context — it carries the user's prior
A-vs-B choices, pinned stepping stones, and discarded ideas for this domain (don't
re-surface a discarded id; the engine already excludes them from slates and parents).

---

## 2. Generate (you)

Use `operators.md`. Produce `candidates_per_generation` ideas by applying
**several different operators** (don't lean on one). Each candidate is an object:

```json
{
  "id": "c-0007",
  "text": "Idea: <the concept, concretely>",
  "descriptor": {"audience": "...", "register": "...", "format": "...",
                 "edginess": 0.7, "mechanism": "<the core how, in a few words>"},
  "genealogy": {"parent_ids": ["c-0001"], "operator_id": "analogy"}
}
```

Rules:
- Generation is **mechanism-first**: pick a distinct `mechanism` (the open/primary
  axis "how") for each idea, then realize each as a surface `text` — surface variety
  follows mechanism variety, never the reverse. This composes with the
  descriptor-discipline rule below (no two candidates share a `mechanism` string).
  **Reach never narrows this — mechanism variety is always maximal.** Reach caps the
  *degree* of divergence (`boldness`), never its *kind*.
- `descriptor` keys must be exactly the resolved axis names. Continuous axes get a
  number in range; the `open`/primary axis gets a short free-text "how".
  Spread continuous values to the **extremes of the reach band**, not the middle
  (`balanced` → `boldness` in `[0, 0.75]`; see `operators.md` → "Descriptor
  discipline"). When the axes carry `feasibility`, use it as the coverage axis it is
  — `balanced` spreads across `[0.2, 1.0]` — and do **not** also fold plausibility
  into `fitness`, or you double-count it.
- Make each new idea differ from the ones already shown. If you have `parents`
  from a previous cycle, treat them as stepping stones to vary, not repeat.
- Concrete beats abstract. One vivid sentence is better than a vague category.

---

## 3. Prefilter (you, as the skeptical judge)

Apply `judge_rubric.md`. Drop only candidates that are **invalid, off-brief, or
incoherent**. Do NOT cut ideas for being weird, risky, or unlike the others —
that is exactly the variety the engine needs. But weird and *incoherent* are
different properties: see `judge_rubric.md` → "Coherence is not weirdness — tell
them apart", whose one-line test ("can you say why the mechanism actually works?")
is what separates a bold idea from an absurd one. Killing a broken idea does not
count against your over-filtering budget. Optionally attach `fitness` (0–1)
for within-niche ranking; never use it to reduce diversity. Pairwise comparison
is allowed where it sharpens a validity call.

Write survivors to `$TMP/candidates.json` (a JSON list, or `{"candidates": [...]}`).

---

## 4. Ingest (engine owns diversity)

```
ENGINE ingest --project PROJECT --candidates $TMP/candidates.json --axes $TMP/axes.json --seed 7
```

Returns:

```json
{
  "slate": [ {"id","text","descriptor","genealogy","coords","niche_id","novelty","mechanism_novelty","fitness","embedding_ref"}, ... ],
  "ask_pairs": [ ["idA","idB","why this pair is worth asking"], ... ],
  "ask_policy": {"generation": 0, "phase": "refine", "ask_sim_weight_effective": 0.5},
  "monitor": {"collapsing": false, "too_similar": false, "calibrated": false,
              "mean_cosine": 0.18, "cos_limit": 0.55, "baseline_n": 0, "entropy": 2.1,
              "normalized_entropy": 0.88, "coverage": 9, "n": 12, "reasons": [],
              "submitted": 12, "target_candidates": 12, "under_generation": false,
              "variety_eroding": false, "variety_erosion": {"streak": 0, "...": "..."}},
  "parents": ["id", "..."],
  "slate_mechanism_novelty": 0.57,
  "open_axis": {"present": true, "frozen": false, "partition": "cold_start",
                "accumulated": 12, "freeze_threshold": 48, "progress": 0.25},
  "surface_mechanism_gap": {"n": 6, "surface_spread": 0.62, "mechanism_spread": 0.41,
                            "gap": 0.21, "corr": 0.34}
}
```

`surface_mechanism_gap` is **present only when `engine.gap_probe: true`** is set in the
resolved axes (default off ⇒ absent). It is advisory measurement — see §8.

`ask_policy` (the explore/refine schedule behind `ask_pairs`) and `slate_mechanism_novelty`
(the slate's mean mechanism-space novelty, paired with each item's `mechanism_novelty`) are
**advisory/observability only** — no action required. `monitor.variety_eroding` *is*
actionable; see §7.

`submitted` / `target_candidates` / `under_generation` are the **prefilter guard**:
the engine sees only the candidates you submitted, so if you prefiltered away so many
that `under_generation` is `true`, you may be cutting variety under cover of validity
(see step 7).

`open_axis` is **observability only** (no action needed): it reports the data-adaptive
mechanism partition's progress toward its one-time freeze. Most sessions stay on the
`cold_start` partition (good on its own); it flips to `frozen` once `accumulated`
reaches `freeze_threshold`.

The engine embedded survivors with a **different model family** from you,
deduped near-duplicates, placed each into a MAP-Elites niche over the resolved
axes, scored geometric novelty, kept one elite per niche, and ran DPP to choose
the diverse `slate`. You did not pick the slate — geometry did.

---

## 5. Present + ask (human in the loop)

Show the `slate`. For each idea, show its `coords` (the niche buckets) so the user
can see *why* it is considered distinct — e.g. "mechanism=cell7, edginess=b3".
Surfacing coordinates is important: embedding-diversity is not always
human-perceived distinctness, so let the user judge.

**De-obscure each slate item.** Add a plain "*how it works in practice*" clause and one
concrete instantiation ("e.g., …") so a bold idea reads as clear, not cryptic. If you
cannot write that clause for an idea, it was incoherent and should have been cut at the
prefilter — drop it now.

Then ask ONLY the `ask_pairs`, as short A-vs-B questions:

> Which sharpens the direction better — **A** (…) or **B** (…)? (or "neither")

Keep the *comparison* questions to the returned pairs. But then **explicitly invite
the user to pin any idea(s) they want to keep exploring from — including ideas that
were not in the asked pairs**, e.g.:

> Want me to pin any of these as a stepping stone? Any of them — not just A/B.

Pinning is deliberately *not* limited to the asked pairs, and it is the user's main
lever. The `ask_pairs` only feed a **low-weight** preference signal — recorded in
memory and recalled into the next round, never able to prune variety or pick the
slate; a **pin** is
the strong, durable signal — it is always kept as a parent for the next generation
(step 6, `select_parents` never drops a pin) and recalled across sessions. So when
the user likes a slate idea you didn't ask about, prompt them to pin it instead of
letting it pass unrecorded. Pins **and** any free-form comparison the user volunteers
(beyond the two suggested pairs) are both recorded in step 6.

Then offer the **negative lever**: the user may **discard** any idea(s) they don't
want carried forward, e.g.:

> Want me to drop any of these so we stop building on them? (any of them)

A **discard** is the symmetric opposite of a pin. The engine drops a discarded idea
from **future slates** (it stops re-appearing) and never breeds from it as a parent,
and the discard **persists across sessions** like a pin. Pin and discard of the same
id are **mutually exclusive, latest action wins** — re-pinning a discarded idea
un-discards it, and discarding a pinned idea un-pins it. Discarding is the user's
call; never discard on your own to enforce a taste or trim variety — that authority
belongs to the geometry + the human, never the judge.

---

## 6. Remember + parents + loop

For each user answer, record it:

```
# a pairwise preference
echo '{"type":"comparison","winner":"idA","loser":"idB","context":"..."}' > $TMP/event.json
ENGINE remember --project PROJECT --event $TMP/event.json

# a pin
echo '{"type":"pin","id":"idA"}' > $TMP/event.json
ENGINE remember --project PROJECT --event $TMP/event.json

# a discard (negative of a pin: dropped from future slates + parents, persists;
# mutually exclusive with a pin of the same id — latest action wins)
echo '{"type":"discard","id":"idB"}' > $TMP/event.json
ENGINE remember --project PROJECT --event $TMP/event.json
```

Then fetch diverse parents for the next generation (pins are always honored, discards
are always excluded):

```
ENGINE parents --project PROJECT --k 4 --seed 7
```

Loop back to step 2 with those parents, or stop when the user is satisfied. At any
time, `ENGINE metrics --project PROJECT` reports `{entropy, mean_cosine,
coverage, n, open_axis}` for the archive (`open_axis` = mechanism-partition freeze
progress).

---

## 7. React to the monitor

**Collapse.** If `monitor.collapsing` is true (or `reasons` is non-empty), the search
is converging. Next round, raise diversity pressure:
- switch to operators you haven't used (analogy/biomimicry/constraint-randomizer);
- explicitly forbid the crowded niches ("no more X-mechanism ideas");
- demand each new idea be far from the recent set;
- widen an axis (e.g. push edginess to its extremes);
- consider **widening reach** for a round (`balanced` → `wild`, see `SKILL.md` →
  *Reach*) so the operator mix and `boldness` band open up — mention it, since the
  user can always override or pull it back.

**Under-generation (prefilter guard).** If `monitor.under_generation` is true, far
fewer candidates reached `ingest` than the per-generation target — you likely
**over-prefiltered**. The engine deduplicates and judges novelty itself, so your only
job in the prefilter is to drop the *invalid / off-brief / incoherent*. Next round:
- generate the full `candidates_per_generation` and prefilter **less** — keep weird,
  risky, or off-beat ideas; they are the variety the geometry needs;
- only the genuinely invalid should be cut, never the merely unusual.

**Variety erosion (advisory, S2).** If `monitor.variety_eroding` is true, survivor
novelty is decaying *faster over time* (the decay is accelerating) while your submitted
count is healthy — the signature of a generator quietly **regressing to the mode** even
though nothing has tripped `collapsing` yet. It is advisory (it never gates anything and
`monitor.variety_erosion` carries the detail: `streak`, `slope_earlier`, `slope_recent`),
but treat it as an early-warning version of collapse: next round, reach for unused
operators and push *mechanism* variety (the open axis) harder before it becomes a full
collapse. A single quiet generation is noise; a raised flag means it has persisted.

Never remove, bypass, or "trust the judge instead of" the monitor. The
anti-convergence machinery is the point of the tool.

---

## 8. Session-end gap summary (advisory, only when enabled)

This section applies **only when `engine.gap_probe: true`** is set in the resolved axes.
With it off (the default), `ingest` never returns a `surface_mechanism_gap` block — skip
this section entirely; the loop's behavior and output are unchanged.

When it is on, each `ingest` return carries that advisory block:
- `surface_spread` — how spread the slate is in **wording** (the idea-text the engine
  already uses for novelty/DPP).
- `mechanism_spread` — how spread it is in **approach** (the open / `mechanism` axis text).
- `gap = surface_spread − mechanism_spread` — positive ⇒ the slate reads more varied than
  its underlying approaches are.
- `corr` — do wording-distant pairs also tend to be approach-distant? High ⇒ yes (little
  gap); low ⇒ ideas can read very differently yet lean on the same approach.

**Never report these per cycle** — one ~6-idea slate gives only ~15 pairwise distances, so a
single cycle is too noisy to read. Instead, at **session end** (the user stops, or asks to
wrap up), read the durable per-cycle series with `ENGINE metrics --project PROJECT` (its
`gap_log` field, present once the probe has run) and summarize the **trend** across the
session, in **plain language, no embedding jargon**. Translate, e.g.:
- small gap + high `corr` ⇒ "Your ideas varied in *how they work*, not just how they're
  worded — the variety looks real all the way down."
- persistent positive gap + low `corr` ⇒ "Heads-up: these read quite differently, but
  several lean on the same underlying approach (e.g. *<name the repeated mechanism>*) — the
  variety is more in the wording than in the mechanism."

Keep these honesty caveats in the summary:
- It compares your **idea text** against your **mechanism descriptor** (both embedded by the
  engine's model), so it reflects how distinctly *you* described each mechanism, not a
  ground-truth taxonomy of approaches — a measured proxy, like `novelty` (variety, not
  world-novelty) and the obvious-set (your notion of cliché).
- It is **advisory only**: it never picked or reordered any slate and never fed the engine's
  selection. You are reporting a measurement, not a verdict.

**Optional nudge (only if the user wants it).** When the gap is persistently high you MAY
offer to push *approach* variety next round via the "Zoom on mechanism" steering tactic
(generate several genuinely distinct mechanisms). That is a **generation**-side choice you
already own — the same kind of reaction as to `monitor.collapsing` — not a change to
selection. Never wire the gap into the engine's selection, parents, or fitness, and never
act on it unless the user asks.

The series is persisted to `meta["gap_log"]` (last 50 records) and surfaced by `ENGINE
metrics` (so the summary doesn't depend on what's still in context); see CLAUDE.md for
offline before/after analysis.

---

## Steering tactics (optional, on user request)

- **Navigate the archive:** ask for ideas in a specific niche, or "more like the
  pinned one but bolder".
- **Constrain an axis:** lock `register=deadpan` and vary the rest.
- **Zoom on mechanism:** request several distinct mechanisms for the same surface
  idea (varies only the primary-novelty axis).
- **Stop conditions:** user says stop, or coverage/diversity plateaus across two
  cycles with no new niches.
