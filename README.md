# Cambrian

![cambrian-explosion.jpg](images/cambrian-explosion.jpg)

A **domain-agnostic** Claude Code plugin that turns a creative brief in *any*
subject into a **diverse, non-cliché slate** of ideas, using a blind-variation →
diverse-archive → human-selection loop with you (the user) as the selector.

It ships as **one model-invoked skill** (`/cambrian:ideate`) plus a
bundled, server-less **Python engine** that owns the anti-convergence math. The
LLM parts — generating variations and the skeptical judge prefilter — are done by
the agent (Claude) itself, so **no extra chat-LLM API key is needed**.

## Why it's different

- **Diversity is decoupled from the judge.** Geometry owns *novelty* (embeddings,
  MAP-Elites niching, k-NN novelty, DPP selection); the judge only filters
  *validity / on-brief* and ranks *within* a niche. Its fitness has just a
  **bounded, low-weight** say in the DPP slate — quality-weighted diversity
  (weight 0.3, fitness clipped to a [0.7, 1.3] multiplier) — so it can never
  prune variety or collapse the slate's diversity, and **you** remain the
  final selector.
- **A different embedding family from the agent.** The default embedder is
  `model2vec` (`minishlab/potion-multilingual-128M`, CPU) — a different model
  family from Claude, **multilingual (101 languages)**, and torch-free — so
  "what's novel" isn't judged by the same lineage that generated the ideas, in
  any language. A higher-fidelity English-only `BAAI/bge-small-en-v1.5` embedder
  is available as an opt-in.
- **Anti-cliché generation, measured honestly.** Before generating, Claude maps
  the **~6 most obvious answers** to *your* brief and deliberately generates
  *away* from them (a recipe re-derived per brief, not a fixed list). Originality
  is then reported advisorily as an idea's distance to a **held-out** half of that
  obvious set — so you're never scored against the very clichés you steered around.
  It **hedges** cliché rather than guaranteeing world-novelty, and it is
  **measurement only**: the cliché signal never feeds the selection geometry, so
  it can't prune variety.
- **Mechanism-first generation, novelty measured where it counts.** Because the
  engine niches ideas on their *mechanism* (the open "how it works" axis), Claude
  commits to a **distinct mechanism before writing each surface idea** — so variety
  is pursued in approach-space, not just wording. To keep that honest, the engine
  also reports **novelty in mechanism space** (the same k-NN novelty, computed on
  each idea's mechanism embedding against the session's accumulated mechanisms)
  alongside the surface novelty — and an opt-in **surface/mechanism gap** probe that
  flags when wording diversity *overstates* approach diversity. Both are **advisory
  measurement only**, never wired into selection.
- **An anti-collapse monitor that's never bypassed.** Shannon entropy over niche
  occupancy + mean pairwise cosine flag convergence; the similarity signal is
  **calibrated to a rolling baseline** (and the dedup threshold is per-embedder),
  so it doesn't misfire when the embedder or domain changes. When it fires, the
  skill raises diversity pressure next round. Two **advisory** sensors back it up
  without ever touching its verdict or the selection geometry: a **prefilter
  guard** that flags when the agent submits too few candidates (possible
  over-prefiltering), and a **variety-erosion** sensor that flags when survivor
  novelty starts *decaying faster over time* — the signature of a generator
  quietly regressing to the mode. Both only nudge the skill to widen the search.
- **Axes resolved per session.** "Domain-agnostic" doesn't remove the need for
  descriptor axes — it resolves them per session (named domain → inferred &
  confirmed → generic fallback). Nothing about a domain is baked into the plugin.

## Install (one command)

Requirements: Claude Code (latest), Python 3.11+ on your PATH. Works the same in the
**CLI** and the **desktop app**, on **Windows (incl. WSL), macOS, and Linux**.

In Claude Code, add this repo as a plugin marketplace and install the plugin:

```
/plugin marketplace add sergiparpal/Cambrian
/plugin install cambrian@sergiparpal
```

**Updating:** grab the latest version anytime with:

```
claude plugin update cambrian@sergiparpal
```

That's it — no clone, no `setup.sh`, no `--plugin-dir`. On the next session start, the
plugin **provisions its own Python engine in the background**: it builds a virtualenv
and installs the default stack (numpy, scikit-learn, and the multilingual
`model2vec` embedder, `minishlab/potion-multilingual-128M`). The embedder weights are
**~120 MB** and need **only numpy at inference — no PyTorch**, so the first-run download
is small and fast, and the whole build runs **non-blocking** — Claude Code stays usable throughout.
The venv is stored in the plugin's persistent data directory, so it **survives plugin
updates**.

If `uv` is on your PATH it's used for a faster install; otherwise the bundled
`python -m venv` + `pip` path is used. Nothing is auto-installed onto your system
beyond that venv.

Then invoke the skill with a brief in ANY subject:

```
/cambrian:ideate names for a privacy-first calendar app
/cambrian:ideate research hypotheses for why week-2 retention dropped
```

If you run `/cambrian:ideate` before that one-time setup finishes, the skill says it's
**setting up the engine** and continues on its own once it's ready — you never run a
setup step yourself. (If Python 3.11+ isn't found, it says so, with a fix.)

## What happens when you run it

You don't need any of the math below to use the plugin — a session is just a
back-and-forth chat, and **you are the one choosing the ideas**. Here is what each
step looks like and what you do.

1. **You give a brief.** Run `/cambrian:ideate <your brief>` for whatever
   you want ideas about — product names, campaign angles, plot twists, research
   hypotheses, anything. That's the only command you have to remember.
2. **Claude confirms the "angles" — one quick question.** Before generating, Claude
   proposes a handful of *axes*: the dimensions it will deliberately spread ideas
   across (for names, say: tone, imagery, length, and how the name is built). It asks
   **one** short question to confirm them. Just reply "ok", or tweak in plain words —
   "make it edgier", "drop the length one". *(If you named a known domain it skips
   straight ahead.)* That same line states the **reach** — how far out the ideas go.
   It runs at *balanced* by default; say **"wild"** for further-out ideas or
   **"conservative"** for safer ones, at any point in the session. Reach changes how
   *bold* the ideas are, never how *varied* they are — every slate is still built from
   genuinely different approaches.
3. **Claude shows you a varied slate.** Claude drafts a batch of ideas, quietly drops
   any that are off-brief or don't make sense, and the engine picks a few that are as
   *different from one another* as possible — not just the "best" ones. Each idea comes
   with a short note on why it counts as distinct, so you can see the spread. **You
   read them over.**
4. **You steer — pin your favorites, and answer a couple of quick comparisons.**
   Claude asks only the few most useful **A-vs-B** questions — "which points in a
   better direction, A or B?" (you can also say "neither"). But the main lever is
   **pinning**: tell Claude to pin *any* idea you like as a "stepping stone" to keep
   exploring from — **including ones it didn't ask you about**. If a slate idea you
   love isn't one of the two being compared, just say "pin that one" — a pin is the
   strong, lasting signal (the comparisons only fine-tune the ranking), so your
   favorites are never lost just because they weren't in a question. The opposite lever
   is there too: say "drop that one" to **discard** ideas you don't want — they
   disappear from later slates and are never built on again (re-pin to undo). Pins and
   discards both persist for next time.
5. **Claude runs another round, building on your picks.** Using what you preferred and
   pinned, Claude generates a fresh batch — pushing for ideas that are genuinely *new*,
   not variations on the same theme. If things start looking samey, a built-in monitor
   notices and forces more variety the next round.
6. **Repeat until you're happy, then stop.** Loop as many rounds as you like; say
   "stop" (or "that's enough") when you have what you need. Your preferences are
   remembered for the next time you ideate in the same kind of domain.

In short: **you give a brief, confirm the angles once, then react to slates and pin
favorites while Claude keeps widening the search** until you're satisfied. The rest of
this README is for people who want to develop on it or understand the internals.

## Local development (fallback)

To hack on the plugin from a checkout instead of installing it:

```bash
# 1. Build the engine venv (Windows / macOS / Linux)
python3 skills/ideate/scripts/bootstrap.py     # Windows: python ... or py ...
#    or, equivalently:  bash skills/ideate/scripts/setup.sh

# 2. Load the plugin in Claude Code without installing it
claude --plugin-dir .

# 3. Invoke the skill
/cambrian:ideate names for a privacy-first calendar app
```

In this mode the venv lives at `skills/ideate/.venv/` and the engine is installed
editable, so source edits take effect without a rebuild. Validate the plugin at any
time:

```bash
claude plugin validate .          # or: claude plugin validate --strict .
```

### The one configuration choice (embedding provider)

By default the engine uses the **static** `model2vec` embedder
(`minishlab/potion-multilingual-128M`) — no API key, CPU-only, **multilingual**,
torch-free (~120 MB), downloaded once on first use. Select a different provider with
the `CAMBRIAN_EMBEDDER` environment variable before launching Claude Code:

```bash
export CAMBRIAN_EMBEDDER=local   # static | local | hash
```

- **`static`** (default) — `potion-multilingual-128M`, 256-dim, 101 languages,
  numpy-only inference.
- **`local`** — higher-fidelity **English-only** `BAAI/bge-small-en-v1.5` (384-dim).
  Pulls the ~2 GB PyTorch stack, so it's **opt-in**: install it with
  `pip install -r skills/ideate/scripts/requirements-local.txt`.
- **`hash`** — deterministic, dependency-light char-n-gram embedder used by the tests
  and the offline self-test (no model download).

(`api` is a **reserved name** for a future hosted provider. There is no backend behind it,
so selecting it fails immediately with a clear error rather than at first use.)

> **Switching the default is breaking for existing projects.** `static` is 256-dim and
> `local` is 384-dim; the engine refuses to mix embedding widths within one project. A
> project created/persisted under one embedder can't be re-ingested under another —
> start a **new** project, or pin the original embedder (e.g.
> `export CAMBRIAN_EMBEDDER=local`) and re-embed if you must switch.

## How a session works (under the hood)

The plain-language walkthrough is in [**What happens when you run it**](#what-happens-when-you-run-it)
above; this is the same loop with the internals. The skill follows
`skills/ideate/references/loop.md`. One cycle:

1. **Resolve axes.** If you name a domain with a config in
   `skills/ideate/config/domains/examples/`, it's loaded. Otherwise the agent
   **infers 4–6 axes** from your brief (marking one `open` axis as the primary
   novelty carrier) and confirms them with **one short question**. If it can't,
   it falls back to `config/domains/generic.yaml`.
2. **Generate (agent).** Claude first maps the brief's ~6 most obvious answers,
   then applies several variation operators (`references/operators.md`) to draft
   candidates that deliberately steer *away* from those clichés. Generation is
   **mechanism-first**: Claude commits to a distinct `mechanism` (the open novelty
   axis the engine niches on) *before* writing each surface idea. Every candidate
   carries a descriptor on the resolved axes and genealogy.
3. **Prefilter (agent).** Claude applies `references/judge_rubric.md` to drop only
   invalid / off-brief candidates — never to cut variety.
4. **Ingest (engine).** Survivors are embedded, deduped, placed into MAP-Elites
   niches over the resolved axes (the open "mechanism" axis uses a **data-adaptive
   partition** — deterministic cold-start cells that fit once via k-means and then
   freeze, so niche ids stay stable), scored for novelty, kept one-elite-per-niche,
   and a **DPP** picks a quality-weighted diverse slate (geometry dominates; the
   judge's bounded fitness only nudges ordering). The **anti-collapse monitor** runs,
   plus the two **advisory** sensors (prefilter guard + variety erosion) that ask the
   skill to widen the search without ever influencing selection. Two more **advisory
   measurements** ride along: **mechanism-space novelty** (the surface k-NN novelty,
   recomputed on the mechanism axis) and an opt-in **surface/mechanism gap** probe
   (`engine.gap_probe`, default off) that quantifies whether wording diversity
   overstates approach diversity — both reported, never wired into selection.
5. **Select (you).** Claude shows the slate with each idea's niche coordinates,
   asks only the most-informative A-vs-B pairs, and **explicitly invites you to pin
   any idea — not just the ones it asked about**. A pin is the strong, durable
   preference signal (always kept as a parent for the next generation, recalled
   across sessions); the A-vs-B answers only refine a bounded, low-weight `fitness`
   that can never prune variety or pick the slate. It also invites you to **discard**
   ideas — the negative of a pin: a human veto that drops the idea from future slates
   and parents and persists across sessions (mutually exclusive with a pin of the same
   id, latest action wins). Discarding is filter-only — it never feeds novelty, DPP, or
   the monitor. *(The pair policy is tunable: by
   default it favors **similar**, boundary-clarifying pairs to learn your preference;
   an opt-in `explore_until_generation` schedule asks **region-separating** pairs in
   the first few generations, then switches to refine.)*
6. **Remember & loop.** Choices/pins/discards go to local preference memory (namespaced
   per domain); diverse parents seed the next generation (pins kept, discards excluded).

## Adding a domain template

Domain templates are optional conveniences — the skill works without them. To add
one, drop a YAML file in `skills/ideate/config/domains/examples/` following
`skills/ideate/config/domains/_schema.md`:

```yaml
domain: naming
unit_of_generation: name
axes:
  - {name: tone, type: categorical}
  - {name: imagery, type: categorical}
  - {name: length, type: continuous, range: [1, 4]}
  - {name: feasibility, type: continuous, range: [0, 1]}   # far-fetched <-> usable (coverage axis)
  - {name: construction, type: open, primary_novelty: true}
judge_rubric: references/judge_rubric.md
slate_size: 6
candidates_per_generation: 12
```

Exactly one axis carries `primary_novelty: true` — the `open` axis the engine niches
mechanisms on. A `feasibility` dial is worth carrying on most domains: it's an ordinary
**coverage** axis (one elite per bin, so a far-fetched idea can't evict a buildable one
and vice versa), *not* a quality score — omit it only when another axis already does that
job, as `testability` does in `research_hypotheses.yaml`.

Then a user who says "name ideas (naming)" gets these axes; everyone else gets
inferred-or-generic axes. See `references/axis_inference.md` for how inference
works. A template may also carry an optional `engine:` block to override tuning
knobs (open-niche count, dedup τ, quality weight, monitor thresholds,
variety-erosion window/persistence, ask-pair policy & `explore_until_generation`
schedule, the opt-in `gap_probe`, …) per domain — defaults reproduce the standard
behavior; see `_schema.md` for the keys.

## The engine CLI (for the curious / for tests)

```
python -m cambrian_engine <command> --project <id> [--axes axes.json] [--seed N]
```

| Command | Does |
| :-- | :-- |
| `init-project` | create state dirs, snapshot the resolved axes + session settings |
| `paths` | ensure the project state dir (incl. its `tmp/` scratch dir) + return resolved paths |
| `recall` | return preference memory for in-context injection |
| `ingest` | embed → dedup → place → novelty → archive → DPP → monitor |
| `remember` | append a comparison/pin/discard to preference memory |
| `parents` | diverse parents for the next generation (pins always kept, discards excluded; pins whose idea an axes change cleared come back under `stale_pins`) |
| `metrics` | archive health (entropy, mean cosine, coverage, n) + mechanism spread + open-axis freeze progress |
| `selftest` | full loop with a stubbed LLM + human; variety gate + collapse reversal + advisory originality probe |

Runtime state is written **outside** the plugin (so reinstalls don't wipe it):
`~/.cambrian/<project>/...`, preferences namespaced per domain.
Override the base directory with `CAMBRIAN_HOME`.

## Running the self-test & the suite

```bash
# offline end-to-end check (stubbed LLM + human, no model download); exits 0
skills/ideate/.venv/bin/python -m cambrian_engine selftest

# unit / property / e2e tests
skills/ideate/.venv/bin/python -m pytest -q
```

The self-test enforces a **variety gate** — the engine's diverse slate must beat a
single-shot baseline on mean pairwise distance, Vendi score, and niche entropy,
and DPP must beat naive first-N selection on the same pool (**shuffled** so
first-N isn't trivially the near-clones, and **averaged over several seeds**),
with a **null check** that DPP doesn't regress below a random subset on an
already-uniform pool — plus an **induced-collapse reversal** (a samey generation
trips the monitor; the next generation recovers once diversity pressure rises).
A `--live` run adds a semantic sanity check (a paraphrase beats an unrelated
sentence under the real default `static` embedder), skipped cleanly when that
embedder can't be built or downloaded.

## Layout

```
Cambrian/                  # plugin root
├── .claude-plugin/
│   ├── plugin.json                    # manifest
│   └── marketplace.json               # marketplace entry (for /plugin marketplace add)
├── hooks/
│   ├── hooks.json                     # SessionStart hook: auto-provision the venv
│   ├── provision.mjs                  # Node dispatcher: picks the matching OS launcher
│   ├── provision.sh                   # POSIX launcher (sh / Git Bash / WSL)
│   └── provision.ps1                  # Windows-PowerShell launcher
├── skills/ideate/
│   ├── SKILL.md                       # model-invoked orchestration (concise)
│   ├── references/                    # loop, operators, judge rubric, axis inference
│   ├── config/domains/                # _schema.md, generic.yaml, examples/*.yaml
│   └── scripts/
│       ├── bootstrap.py               # cross-platform self-provisioning installer
│       ├── setup.sh, pyproject.toml   # dev setup wrapper; package metadata (deps in requirements*.txt)
│       ├── requirements.txt           # runtime deps (version-bounded)
│       ├── requirements-dev.txt       # + pytest (dev/CI); requirements-local.txt = opt-in torch embedder
│       └── cambrian_engine/         # the Python engine (CLI)
├── tests/                             # pytest (dev-only)
├── docs/PAPER.md                      # reference-architecture paper (rationale)
└── README.md
```

## Background

The design rationale — why diversity is owned by geometry while the judge's say in the slate stays
bounded, and how this maps onto blind-variation/selective-retention, Quality-Diversity (MAP-Elites,
novelty search), and DPP selection — is written up in [`docs/PAPER.md`](docs/PAPER.md).

## License

MIT — see `LICENSE`.
