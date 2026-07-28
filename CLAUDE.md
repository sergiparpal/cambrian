# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code **plugin** named `cambrian`. It ships **one model-invoked skill**
(`skills/ideate/`, invoked as `/cambrian:ideate <brief>`) plus a bundled, server-less
**Python "Cambrian engine"** (`skills/ideate/scripts/cambrian_engine/`, a CLI) that owns the
anti-convergence math.

The work is split across two halves that meet at a JSON contract:

- **The LLM half** — `skills/ideate/SKILL.md` + `skills/ideate/references/*.md`. These are
  *instructions to a future Claude instance*, which acts as the runtime "agent": it generates idea
  variations, prefilters for validity/on-brief, and runs the human selection dialogue. There is no
  extra chat-LLM API key — Claude itself is the generator and judge.
- **The engine half** — the deterministic Python CLI. Pure, JSON-in/JSON-out, no LLM calls.

When editing here you may be doing either job: maintaining the Python engine, or refining the
prose contract in `SKILL.md`/`references/` that future Claude instances execute.

## The one invariant: diversity is decoupled from the judge

This principle is load-bearing across `SKILL.md`, `pipeline.py`, `archive.py`, and `monitor.py`,
and any change must preserve it:

- **Geometry (the engine) owns *novelty / diversity*** — embeddings, MAP-Elites niching, k-NN
  novelty, DPP selection.
- **The LLM (agent) owns only *validity* and *within-niche ranking***. The judge filters off-brief
  candidates and may attach a `fitness` (0–1) that ranks *within* a niche. That fitness is also
  allowed a **bounded, low-weight** say in the DPP slate (quality-weighted diversity): it is
  affine-rescaled and **clipped to a [0.7, 1.3] multiplier** and blended at `QUALITY_WEIGHT=0.3`, so
  it can nudge ordering but can **never** prune variety, collapse the slate's diversity, or pick the
  final slate. (Set the weight to 0 for pure diversity.)
- **The user is the real selector.** The engine proposes a diverse slate + the most-informative
  A-vs-B pairs; the human chooses and pins "stepping stones". The human may also **discard**
  ideas — the negative of a pin. A discard is a **human veto**, applied only as a filter on the
  *presented* slate pool and the *parent* pool (so a discarded idea stops re-appearing and is
  never bred from); it is **never** wired into novelty, the DPP `q`/kernel, niching, fitness, or
  the monitor. Because it is the user pruning (not a quality/cliché heuristic), it preserves the
  invariant while honoring "the user is the real selector". Pins and discards are **mutually
  exclusive, latest action wins** (`add_pin`/`add_discard` keep one set free of the other).
- The embedder is deliberately a **different model family** from Claude (default
  `minishlab/potion-multilingual-128M`; opt-in high-fidelity `BAAI/bge-small-en-v1.5`) so
  "what's novel" isn't judged by the lineage that generated the ideas.
- **Originality (distance-to-cliché) is measured, never selected on.** Unlike geometric *novelty*
  (which has no external referent), `originality.py` scores how far an idea sits from an "obvious-set"
  of clichés. It is **measurement only**: never wired into the DPP `q`/kernel, never part of the
  self-test's `ok`, never steers the engine. The agent uses the obvious-set at *generation* time (the
  `anti_cliche` operator — repel clichés); the engine only ever *measures* the result. Keeping it out
  of selection is precisely what stops a cliché signal from pruning variety.
- The **anti-collapse monitor is never bypassed** — when it flags convergence the skill raises
  diversity pressure next round; it is never removed or worked around.
- **Reach is a language-layer directive, never an engine input.** The `conservative` / `balanced`
  (default) / `wild` dial in `SKILL.md` caps divergence in **degree** (the `boldness` band) and
  biases the operator mix and `feasibility` lean; it **never** caps divergence in **kind** — the
  `mechanism` (open) axis always spreads maximally. It exists purely in the prose contract: the
  agent applies it when writing descriptors and picking operators, and there is **zero** engine
  code behind it. Do not mistake it for a quality or selection signal — it is not read by the DPP
  `q`/kernel, novelty, niching, or the monitor, all of which still only measure dispersion over
  whatever the agent submits. The companion `feasibility` axis is an ordinary **coverage** axis
  (one elite per bin), not a fitness term.

**What "the engine owns novelty" precisely means (and what it doesn't).** Niche *placement* on
**every** axis is driven by the agent's `descriptor` text by design — a categorical mislabel or a
differently-worded `mechanism` lands an idea in a different niche, changing the elite competition and
thus the DPP pool. The open axis is **not special** here: its Voronoi cell is the embedding of the
agent's mechanism descriptor (`_open_axis_texts`), so the agent's *word choice* influences placement
on it just as on the categorical axes. The purity guarantee the design actually makes — and keeps —
is narrower and load-bearing: **k-NN novelty, dedup, and the DPP kernel are all computed on the
idea-text embedding** produced by the **independent embedder** (a different model family from Claude).
So even where the agent's descriptors steer placement, they run through that independent geometry —
what "leaks" is the agent's word choice, never Claude's own similarity judgment. (For stricter purity
one *could* niche the open axis on the full idea-text embedding instead of the descriptor; we keep the
descriptor deliberately, because it is the focused "core how" that gives the mechanism axis its
intended semantics.)

## Install & provisioning

Two install paths, one provisioner (`skills/ideate/scripts/bootstrap.py`):

- **End users (marketplace):** `/plugin marketplace add sergiparpal/Cambrian`
  then `/plugin install cambrian@sergiparpal`. A `SessionStart` hook
  (`hooks/hooks.json`, `async: true`) runs a thin Node dispatcher (`hooks/provision.mjs`)
  that hands off to `bootstrap.py` in a detached background process right after load —
  non-blocking, idempotent, concurrency-safe. The venv is
  built in **`${CLAUDE_PLUGIN_DATA}/venv`** (the plugin's persistent data dir, so it
  survives plugin updates) and the engine is installed **non-editable** there. Default
  install is the **torch-free multilingual stack** (the `static` model2vec embedder,
  `potion-multilingual-128M`, ~120 MB); the heavier `local` (bge / sentence-transformers)
  embedder is opt-in via `requirements-local.txt`.
- **Developers:** `bash skills/ideate/scripts/setup.sh` (or `python3
  skills/ideate/scripts/bootstrap.py`) builds `skills/ideate/.venv` with the engine
  installed **editable**, then `claude --plugin-dir .`.

`bootstrap.py` uses `uv` when it is on PATH (faster) and falls back to `python -m venv`
+ `pip` otherwise; it never auto-installs `uv` or any other tool. It writes the resolved
interpreter path to `<venv>/engine-python.txt` (the skill reads this to locate the
interpreter) and a content hash to `<venv>/install.stamp` (rebuild trigger on a plugin
update that changes deps or — in non-editable mode — engine sources). The three
`hooks/provision.*` files are thin: `provision.mjs` is a cross-platform Node dispatcher
(the single command `hooks.json` registers) that detects the OS and invokes **only** the
matching launcher — `provision.sh` (POSIX) or `provision.ps1` (native Windows) — so the
wrong-OS launcher never fails noisily in the hook log; each launcher just finds a Python
≥ 3.11 and hands off to `bootstrap.py --background`. If the engine isn't ready when
`/ideate` runs, the skill
shows a one-time "setting up…" message and finishes the build in the foreground.

## Commands

Run Python through the engine's own venv (`skills/ideate/.venv/bin/python` in dev), not
system Python.

```bash
# One-time setup: create skills/ideate/.venv and install deps (idempotent)
bash skills/ideate/scripts/setup.sh

# Full test suite (hermetic: hash embedder + temp state home, no model download)
skills/ideate/.venv/bin/python -m pytest -q

# A single test file / test
skills/ideate/.venv/bin/python -m pytest tests/test_diversity_monitor.py -q
skills/ideate/.venv/bin/python -m pytest -k "dpp" -q

# Offline end-to-end self-test (the correctness contract; exits 0 on pass)
skills/ideate/.venv/bin/python -m cambrian_engine selftest
# ...with the live embedder instead of the hash one:
skills/ideate/.venv/bin/python -m cambrian_engine selftest --live

# Load the plugin in Claude Code without installing it, then validate it
claude --plugin-dir .
claude plugin validate .            # or: claude plugin validate --strict .
```

Set `CAMBRIAN_DEBUG=1` to get a full traceback from the CLI instead of the clean one-line error.

## Engine CLI

`python -m cambrian_engine <command> --project <id> [--axes axes.json] [--seed N]` — every command
reads/writes JSON, prints JSON to stdout, errors to stderr with a non-zero exit.

| Command | Does |
| :-- | :-- |
| `init-project` | create state dirs, snapshot the resolved axes + session settings |
| `paths` | ensure the project state dir (incl. its `tmp/` scratch dir) + return resolved paths |
| `recall` | return preference memory for in-context injection |
| `ingest` | embed → dedup → place → novelty → archive → DPP → monitor (one cycle) |
| `remember` | append a comparison/pin/discard to preference memory |
| `parents` | diverse parents for the next generation (pins always kept) |
| `metrics` | archive health (entropy, mean cosine, coverage, n) + open-axis freeze progress |
| `selftest` | full loop with stubbed LLM + human; variety gate + collapse reversal + advisory originality probe |

## Architecture

**Module layering** (`skills/ideate/scripts/cambrian_engine/`), lowest to highest:

- `config.py` — foundation. Dataclasses (`Axis`, `AxesSpec`, `Candidate`, `Niche`,
  `SessionSettings`, `EngineConfig`) and axes loading/validation. The engine **never assumes a
  domain**; every command receives resolved axes. **Tuning knobs live in `EngineConfig`** (dedup τ,
  KNN k, open-niche count + freeze factor, DPP pool, novelty-ref cap, quality weight, monitor
  thresholds/margins/window), overridable per domain via an optional `engine:` block; defaults
  reproduce the original behavior and `ingest` resolves them with `load_engine_config`. An
  **unknown key** in that block is a loud `ConfigError` (the valid set is exactly what
  `EngineConfig.to_dict` emits), so a typo'd knob can never silently run on defaults. The
  pipeline/monitor module constants remain only as fallback defaults for direct callers and the
  self-test. The **pipeline** ones are now *derived* from a default `EngineConfig` (`_DEFAULTS`)
  rather than hand-mirrored literals, so `test_engine_config` is a **structural** guard for those
  (the names must exist and stay aligned) and still a genuine value-drift guard for the
  monitor/memory constants. `config.require_sklearn` gives the two sklearn-dependent call sites
  (`embed.HashingEmbedder`, `archive.CVTNicher.fit`) one actionable `ConfigError` instead of a raw
  `ModuleNotFoundError`. `Candidate.from_dict`
  also **rejects empty/whitespace `text`** (it would embed to a zero vector that never dedups and
  always scores maximally novel — poisoning dedup and novelty), mirroring the empty-id guard.
- `embed.py`, `novelty.py`, `diversity.py`, `monitor.py`, `archive.py` — the math (see below).
- `originality.py` — advisory distance-to-cliché *measurement*, deliberately isolated from the math
  above: it is never imported by the selection geometry and only ever consumed as a reported number
  (see the invariant section). `novelty.py` is variety vs. this session; `originality.py` is distance
  from an external obvious-set.
- `state.py`, `memory.py` — file-based persistence and preference memory. The active learner
  (`select_ask_pairs`) scores candidate A-vs-B questions by a weighted sum of embedding
  **similarity**, fitness **uncertainty**, and **novelty**. The default weights favor *similar*
  pairs — the judge-disagreement boundary, best for **learning the preference function** — but that
  is a deliberate policy, not a proven optimum (the variety gate does not test it). The two competing
  readings (learn-preference vs. **explore** by comparing region-separating pairs) are reconciled by
  making the weights tunable: `engine.ask_sim_weight ≤ 0` flips the policy to explore. Defaults
  reproduce the original behavior. `engine.explore_until_generation` (S3) **schedules** that policy
  by generation index: the first that-many 0-indexed generations ask region-separating (explore)
  pairs (`-abs(ask_sim_weight)`), then switch to the configured `ask_sim_weight` (refine). It
  defaults to `0` (off — flat `ask_sim_weight` every generation, no change to the global default)
  and is opt-in; the explore magnitude tracks the refine magnitude by design (one knob). It only
  changes *which* pairs are asked — never selection geometry, never who decides.
- `session.py` — per-invocation context (`Session`): bundles the `State` handle, the resolved
  preference **domain** (memory namespace), the axes **spec**, and the **embedder**, all resolved
  lazily. It centralizes the rule that the memory namespace *is* the domain of the persisted axes
  snapshot, so `ingest`/`recall`/`remember`/`parents` can never drift on which namespace a
  project's memory lives in.
- `pipeline.py` — orchestration; one public function per CLI command, built on `Session` and each
  returning a JSON-serializable dict. **This is where to start reading.**
- `__main__.py` — thin argparse wrapper over `pipeline`.

**The `ingest` cycle** (`pipeline.ingest`, the heart of the loop):
`embed → dedup (cosine > per-embedder τ) → place into MAP-Elites niches → geometric k-NN novelty →
keep one elite per niche → DPP diverse slate → anti-collapse monitor`.
Subtlety worth preserving: the **monitor runs on the RAW pre-dedup generation vectors**, so a
near-duplicate batch still registers as collapsing instead of being hidden behind survivors.
`ingest` also **warm-loads the embedder before taking the project lock** (`sess.embedder.embed([])`):
the first-ever load can download ~120 MB, and the lock's staleness window (`_LOCK_STALE`, 60s) has no
heartbeat — a slow download *inside* the lock would let a concurrent session judge it abandoned,
steal it, and interleave the very read-modify-write the lock serializes.
Two thresholds that used to be fixed constants are now **calibrated**: the dedup τ is per-embedder
(`embed.default_dedup_tau`), and the monitor's similarity flag is relative to a rolling window of
recent generations' mean cosine (`baseline + margin`, with an absolute safety ceiling), falling back
to the absolute threshold until the window has enough samples. Crucially, once calibrated the window
trains only on **healthy** generations — `_persist_cycle` excludes any generation the relative rule
flags as too similar — so a sustained collapse can't drag the baseline up past itself and blind the
monitor (a "boiling-frog" failure). While still bootstrapping (no calibrated baseline yet) every
generation is admitted, so the window can form even under an embedder whose natural cosine scale
trips the absolute fallback.

**Prefilter guard** (the load-bearing invariant, mechanically sensed): the monitor covers the
*generation* stage (it runs on raw submitted vectors), but the agent's **prefilter** — dropping
candidates as "off-brief" — is the one stage the engine never sees, so an agent could cut variety
under cover of validity. `ingest` therefore compares `len(cand_list)` **submitted** (pre-dedup, *not*
post-dedup survivors — dedup is the engine's own job) against `candidates_per_generation`, and emits a
**soft** `monitor.under_generation` flag (with `submitted`/`target_candidates`) when it falls below
`engine.under_generation_ratio` (default 0.6). The flag is advisory: it never touches `collapsing` or
the calibration window; `SKILL.md`/`loop.md` tell the agent to generate more / prefilter less.

**Variety-erosion sensor** (S2, `monitor.assess_variety_erosion`, also advisory): `variety_eroding`
is a monitor flag raised when the per-generation **survivor mean novelty** decays *faster over time*
(the decay **accelerates**) while the submitted count is healthy — the signature of a generator
quietly regressing to the mode. By design it thresholds the **change in the decay rate**, never the
novelty *level* and never the rate *absolutely*: a long-but-healthy session sits low and decays
slowly (decelerating), so it can never false-positive. The flag fires only after the acceleration
holds `engine.erosion_persist` (K, default 2) consecutive generations, over a window of
`engine.erosion_window` (W, default 5) with margin `engine.erosion_accel_ratio` (ρ, default 0.5 ⇒
recent ≥ 1.5× earlier). It is **strictly advisory** — it never sets or influences `collapsing`,
never touches the monitor's calibration `cos_window`, and keeps its **own** persisted series
(`novelty_window`) and streak counter (`erosion_streak`).

**Surface/mechanism gap probe** (`gap.py`, `surface_mechanism_gap`, advisory measurement only):
quantifies whether a slate's *wording* diversity (the surface idea-text embedding the engine
already uses for novelty/DPP) **overstates** its *approach* diversity (the open/`mechanism`-axis
embedding). It reports `surface_spread`, `mechanism_spread`, `gap = surface − mechanism`, and the
pairwise-distance correlation `corr`. Like `novelty` (variety, not world-novelty) and
`originality.py` (distance-to-cliché), it is **measured, never selected on**: never in
`selftest`'s `ok`, never wired into the DPP `q`/kernel, parents, fitness, or the monitor.
`engine.gap_probe` (default **off** ⇒ zero cost, output byte-for-byte unchanged) turns on per-cycle
emission: `ingest` attaches `surface_mechanism_gap` to its result and appends a bounded record to
`meta["gap_log"]` (last 50), which `metrics` then surfaces. **Before/after recipe:** set
`engine.gap_probe: true` in the domain config, run the **same briefs** before and after a
generation-side change, then compare the distributions of `gap`, `mechanism_spread`, and `corr`
from each session's `gap_log` (`ENGINE metrics --project PROJECT`, or `meta.json`). Reading: a small
`gap` with high `corr` and `mechanism_spread ≈ surface_spread` means surface diversity already
tracks approach diversity; a persistent positive `gap` with low `corr` means it does not. When
enabled, the skill also gives the user a plain-language, session-end summary of the gap
(`SKILL.md` step 10 / `loop.md` §8) — still advisory: it reports the measurement and never
steers selection.

**Mechanism-first generation + mechanism-space novelty** (G4 + S4): generation is
**mechanism-first** (G4, prose) — the agent commits to a distinct `mechanism` (the open /
`primary_novelty` axis value) *before* writing each surface idea, since that is the axis the
engine niches on (`SKILL.md` step 4 / `operators.md` → `principle_first`). To make that
observable, the engine also reports **novelty in mechanism space** (S4): `mechanism_novelty`
per slate item (k-NN distance of an idea's mechanism embedding to the session's accumulated
mechanisms — the *same* k-NN kernel as the surface `novelty`, run on the open-axis embedding),
`slate_mechanism_novelty` on the `ingest` result, and `mechanism_spread` / `mechanism_n` in
`metrics`, all backed by a parallel `mech_embeddings.npz` store (filled from `open_vecs`, the
same L2-normalized embedder). S4 is **advisory measurement only** — like `novelty`,
`originality.py`, and the gap probe it is **measured, never selected on**: never in dedup, the
DPP `q`/kernel, parents, the fitness clip, the monitor, or `selftest`'s `ok`; the slate is still
selected on surface geometry. Note the deliberate name reuse: the archive-scoped
`metrics["mechanism_spread"]` (mean pairwise distance over elite mechanism vectors) is **distinct
from** the slate-scoped `mechanism_spread` inside the optional `surface_mechanism_gap` block
(`gap.py`) — different dict, different scope. Reading: if `mechanism_spread` tracks the surface
spread (and `mechanism_novelty` stays healthy across a session), mechanism diversity is keeping
up; a persistent low `mechanism_spread` with high surface diversity is the residual gap.

**Niching** (`archive.py`): a niche key combines one bucket per axis — `categorical` → the value,
`continuous` → bin index over its range, `open` → a **frozen Voronoi cell** over the *embedding* of
the axis value (`CVTNicher`). Two bucketing rules are load-bearing and easy to get wrong: a
categorical value whose slug comes out **empty** (any non-Latin label — the case the multilingual
default embedder exists to serve) is bucketed by a stable sha1 of the raw value rather than a shared
literal, so `芸術` and `成長` no longer collapse into one niche and void the coverage axes; and a
**missing** continuous value bins to the neutral **mid**-bin, never bin 0, since bin 0 is a real
extreme and defaulting there both biases every omitting descriptor toward it and collapses them all
into one bin. (A non-finite value is a *bad* value, not an omitted one, and still clamps to bin 0.) The open-axis partition is **data-adaptive (fit-once-then-freeze)**:
early cycles use deterministic cold-start centroids seeded by `--seed`; once `OPEN_NICHE_FREEZE_FACTOR
* OPEN_NICHES` (now **2 × 24 = 48**, ~4–5 generations — lowered from 4× so the data-adaptive partition
actually activates in a realistic session) mechanism embeddings have accumulated, k-means is fit
**once** (`KMeans(random_state=seed)`, with the benign "fewer distinct clusters than k" warning
silenced — clustered idea embeddings legitimately under-fill the partition), the centroids are
persisted (`open_nicher.json`), and the archive is **re-keyed** onto them (`Archive.rekey_open_axis`,
merging collapsed niches by the elite rule). It never refits after freezing, so niche ids stay stable.
Exactly one axis may be the `primary_novelty` "open" axis (the novelty carrier). Within a niche, higher
`fitness` wins; ties break toward higher novelty. Most short sessions never reach the threshold and run
entirely on the cold-start partition (validated good under the real embedder), so `ingest`/`metrics`
surface an **`open_axis`** progress block (`accumulated` / `freeze_threshold` / `progress` / `frozen`)
to make the otherwise-silent freeze observable.

**Embedders** (`embed.py`), selected by `CAMBRIAN_EMBEDDER` (`static` default):
- `static` — model2vec `minishlab/potion-multilingual-128M`, **256-dim, 101 languages**, CPU,
  **numpy-only inference (no torch)**, ~120 MB, lazily downloaded. The default for real runs.
- `local` — sentence-transformers `BAAI/bge-small-en-v1.5`, **384-dim, English-only**, CPU, pulls
  the ~2 GB torch stack (opt-in via `requirements-local.txt`); the high-fidelity escape hatch.
- `hash` — deterministic char-n-gram `HashingVectorizer`, no downloads (tests + offline selftest).
- `api` — a reserved name with no wired backend; selecting it raises a clean `ConfigError`.

Switching the default from `local` (384-dim) to `static` (256-dim) is **breaking for projects
persisted under the old default**: `_guard_embedding_dim` refuses to mix widths, so an old project
must be re-embedded or pinned to `CAMBRIAN_EMBEDDER=local`. Each family has its own dedup τ in
`DEDUP_TAU_BY_EMBEDDER` (`static: 0.93`, calibrated on an EN+ES near-dup/distinct sample).

All embedders return **L2-normalized rows**, so cosine similarity is a plain dot product — this
assumption is relied on throughout the math modules. `ingest` guards against mixing embedding
dimensions within a project and fails loudly. It also **guards the axes geometry**: niche keys are
built from the axes list, so `ingest` refuses an `--axes` whose axes differ from the project's
persisted snapshot (it compares the axes list only — not `slate_size`/domain/unit) with a clean
`ConfigError`, rather than silently mixing incompatible niche keys into one archive. To change axes
you re-run `init-project`, which resets the geometry-coupled state (archive / candidates /
embeddings / **mechanism embeddings** / frozen open-nicher) while preserving preference memory;
`ingest` is deliberately not a back door around that guard.

**Axes resolution** (per session, done by the agent in `SKILL.md`, validated by `config.py`): named
domain config → inferred-and-confirmed axes → `config/domains/generic.yaml` fallback. Domain
templates live in `config/domains/examples/*.yaml`; `_schema.md` documents the format. Nothing
about a domain is baked into the plugin or engine.

**State** (`state.py`) is written **outside the plugin** so reinstalls don't wipe it:
`~/.cambrian/<project>/` (override the base with `CAMBRIAN_HOME`). Writes
are atomic (temp file + `os.replace`). Per-project files are `meta.json` (project/session
settings), `axes.json` (the resolved axes geometry — kept separate from settings so the engine's
`AxesSpec` stays pure), `archive.json`, `candidates.json`, `embeddings.npz`, `mech_embeddings.npz`
(the parallel mechanism/open-axis embedding store backing S4 — see Mechanism-space novelty), and
`open_nicher.json`
(the frozen CVT centroids, written once the open-axis partition freezes — see Niching). Preference memory
(`comparisons.jsonl`, `pins.json`, `discards.json`) lives in a per-domain sub-directory,
**namespaced per domain** so switching domains keeps preferences separate. `discards.json` is the
user's veto list (excluded from slates + parents, mutually exclusive with `pins.json`). A per-project `tmp/` scratch dir (created by
`State.ensure`, surfaced by the `paths` command) holds the skill's hand-off files (`axes.json`,
`candidates.json`, `event.json`) inside the state home so they never clutter the user's cwd or
collide across concurrent sessions. The two **vector** stores are binary `.npz` (float64, so the
round-trip is exact); a project written by an older engine still loads via a tolerant fallback read
of the legacy `embeddings.json` / `mech_embeddings.json`, which the next write migrates and removes
(`reset_geometry` unlinks both shapes so leftovers can't shadow a fresh geometry). The machine-only
JSON stores (`candidates.json`, `open_nicher.json`) are written **compact** — same values and key
order, fewer bytes. `candidates.json`/`embeddings.npz` are rewritten whole each
cycle (the advisory `mech_embeddings.npz` only when the cycle actually changed it);
for long sessions `ingest` **prunes** records nothing reads again once the store exceeds
`engine.state_prune_threshold` (default 2000, 0 disables) — keeping exactly archive elites, pins, and
comparison ids, so the pruning is output-neutral.

**The self-test is the correctness contract** (`selftest.py`). It enforces a **variety gate**
(renamed from "value gate": it proves *variety geometry* plus one narrow within-niche value
assertion, not value in general) — the engine's diverse slate must beat a single-shot baseline on
mean pairwise distance, Vendi score, and niche entropy, and DPP must beat naive first-N on the
*same* pool — plus an **induced-collapse
reversal** (a samey generation trips the monitor; the next generation recovers once diversity
pressure rises). It also runs an **advisory originality probe** (`_originality_probe`): it scores the
diverse slate's distance to a **held-out** half (`O_test`) of a clichéd obvious-set — held out so the
measure stays non-circular — and reports it as a printed-only sanity number that is **never part of
`ok`**. Treat a `selftest` failure as a real regression in the diversity guarantees.

## Conventions & gotchas

- **`requirements*.txt` are the source of truth for dependencies** (`pyproject.toml` keeps
  `dependencies = []` on purpose). `requirements.txt` is **runtime only** and what end-user
  background installs build from; `requirements-dev.txt` (`-r requirements.txt` + `pytest`) adds the
  test runner and is what `bootstrap.py` installs in editable/dev mode and what CI installs.
  `requirements-local.txt` is the opt-in torch `local` embedder. Deps are **compatible-release
  bounded** (`numpy>=1.26,<3`, …) — lower bound = minimum tested, upper bound = the next *untested*
  major, so an auto-provisioned install never silently adopts a breaking major; bump caps
  deliberately. `setup.sh` just execs `bootstrap.py`, which resolves runtime-vs-dev and runs
  `pip install -r <reqs>` then `pip install -e . --no-deps`.
- Tests are hermetic via `tests/conftest.py` (forces `CAMBRIAN_EMBEDDER=hash` and an isolated
  `CAMBRIAN_HOME`). Keep new tests offline — never trigger a model download.
- Determinism matters: niching/DPP/CVT take a `--seed`; reuse it across a session's cycles.
- `config.ConfigError` messages are user-facing (printed by the CLI) — write them for the operator.
- `docs/PAPER.md` is the reference-architecture paper (rationale and positioning), not the
  implementation spec — `SKILL.md` + this file are the spec.
- **Git workflow:** commit work directly on `main` — do **not** create a new branch unless
  the user explicitly asks for one. (Overrides the default "branch before committing on the
  default branch" behavior.)
