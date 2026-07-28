# Variation operators (domain-agnostic)

These are **blind-variation** operators: ways to generate candidates that differ
structurally, so the diversity engine has real spread to work with. None assumes a
subject — they take the brief and current parents/stepping-stones and produce a
new idea plus its descriptor and genealogy.

**How to use them**
- Each generation, apply **several different operators**, not one. Mixing
  operators is what creates spread; reusing one collapses it.
- For every candidate, record `genealogy.operator_id` (the operator's key below)
  and `genealogy.parent_ids` (the ideas it varied, if any).
- Aim each operator at a *different* region: vary the `open`/primary-novelty axis
  (the core "how") aggressively; vary categorical/continuous axes to cover the
  grid.
- Prefer concrete, single-sentence ideas over vague categories.

When the monitor reports collapse, switch to operators you have not used this
session, especially `analogy`, `biomimicry`, `random_stimulus`, and `inversion`.

**Reach controls the operator *mix*** (see `SKILL.md` → *Reach*). The operators split by
how far they reach from the brief:
- *recombinative* (shorter reach) — `mutation`, `combination`, `morphological`,
  `scamper`, `constraint`, `principle_first`: recombine and re-angle the brief's
  own material.
- *far-reaching* (longer reach) — `analogy` to distant domains, `transformation`
  (10×/self-destruct), `random_stimulus`, `inversion`, `biomimicry`: jump to a
  foreign source.

`conservative` leans on the recombinative set; `balanced` mixes (mostly
recombinative, one or two far-reaching per round); `wild` reaches freely. In every
mode still apply **several different** operators per round — reach shrinks how far
each jumps, never how many distinct angles you take.

---

## Core operators

### `mutation`
Take one parent and change a single dimension hard: swap the audience, flip the
register, push a continuous axis to an extreme, or replace the mechanism while
keeping the surface. Smallest-step explorer; good for filling nearby niches.

### `combination`
Fuse two unrelated parents (or the brief + an unrelated concept) into one idea
that needs both to make sense. Forces a mechanism neither parent had.

### `analogy`
Map the brief's structure onto a distant source domain ("what is the X of Y?")
and import that domain's mechanism. The strongest novelty driver — reach for far
sources (other industries, nature, games, rituals, physics).

### `transformation`
Apply a transform to an existing idea: reverse it, exaggerate it 10×, shrink it to
a single gesture, make it continuous/ongoing, make it collaborative, make it
self-destruct. Changes the *form* axis while preserving intent.

### `reframing`
Restate the underlying goal/problem, then solve the reframed version. Changing the
question ("we assumed it's about A — what if it's about B?") opens niches no
amount of mutation reaches.

### `inversion`
Do the opposite of the obvious solution, or solve the anti-goal, then invert the
result back. Surfaces ideas that cliché-following never produces.

### `constraint`
Impose an arbitrary, severe constraint (no budget, one word, must work offline,
must be free, must take 5 seconds) and design to it. Constraints force fresh
mechanisms.

### `random_stimulus`
Draw a random unrelated noun/image/verb and force a connection to the brief. A
deliberate jolt out of the current cluster; use when ideas feel samey.

### `biomimicry`
Borrow a mechanism from a living system (swarming, symbiosis, camouflage,
metamorphosis, mycelial networks) and adapt it. A reliable source of mechanisms
for the primary-novelty axis.

### `anti_cliche`
Before generating, enumerate the **~6 most obvious / cliché answers** to the brief
and split them into **O_train** (first half) and a held-out **O_test** (second
half). Treat **O_train** as a *forbidden zone*: deliberately generate ideas that
sit far from it — different mechanism, different framing, different form — so the
slate starts away from the crowd the engine would otherwise have to disperse. Do
**not** look at or optimize toward **O_test**; it is reserved for honest,
after-the-fact originality measurement (validate against O_test, **never**
O_train — see SKILL.md). Build the obvious-set fresh per brief (a construction
recipe, not a fixed list). Pairs naturally with `inversion` and `reframing`, which
already move away from the obvious.

---

## Structured idea generators (use to systematically cover the grid)

### `scamper`
Run the brief through SCAMPER prompts — **S**ubstitute, **C**ombine, **A**dapt,
**M**odify/magnify, **P**ut to other use, **E**liminate, **R**everse. Each prompt
is effectively a different operator; tag the candidate with the sub-letter in the
text if useful.

### `morphological`
List the key parameters of the solution (often your categorical axes), enumerate
several options per parameter, then assemble unusual *combinations* of options.
Directly targets unoccupied MAP-Elites niches.

### `triz`
When the brief contains a trade-off ("we want more A but that costs B"), apply a
TRIZ inventive principle (segmentation, asymmetry, "the other way round",
nesting, prior counteraction, self-service) to dissolve the contradiction rather
than compromise it.

### `principle_first`
Pick the **mechanism before the surface**. Enumerate candidate mechanisms that sit
far from `O_train` and from each other; assign **one distinct mechanism per idea**;
then realize each as a surface concept. Surface variety follows mechanism variety,
never the reverse. Pairs naturally with `analogy` and `biomimicry` (strong mechanism
sources) and with `anti_cliche` (steer mechanisms away from the obvious-set).

---

## Descriptor discipline

Niche placement — **including the open axis** — is driven entirely by *your*
descriptor word choices, **before** the engine runs. Lazy or duplicate descriptors
collapse niches up front, so the engine never sees the variety you generated.
Protect this input contract. For every candidate, fill the resolved axes honestly:
- **categorical** → pick the value that truly describes the idea (don't default
  everything to the same value — that collapses niches).
- **continuous** → a number in range that reflects the idea. **Prefer the
  extremes of the allowed band** over clustering everything in the middle — but
  the band itself is set by the **reach dial** (see `SKILL.md` → *Reach*). Under
  `balanced` reach `boldness` lives in `[0, 0.75]`, under `conservative` in
  `[0, 0.5]`; spread to the ends *of that band* (so niche resolution is
  preserved), don't blow past it. Reach caps the *degree* of divergence
  (`boldness`); it never caps divergence in *kind* (the `mechanism` axis, which
  always spreads maximally).
- **`feasibility`** (when the axes carry it) → how buildable-vs-far-fetched the
  idea is, `0`–`1`. This is a **coverage axis**: because MAP-Elites keeps one
  elite per feasibility bin, a bold idea can't evict a buildable one from its
  niche, so the archive always *retains* practical options for the slate to draw
  and for you to build from next round (a coverage guarantee, not a promise the
  final six always contain one). Reach sets the lean: `conservative` clusters ideas
  high (`0.6`–`1.0`, mostly buildable), `balanced` spreads across `[0.2, 1.0]`,
  `wild` uses the full range. `feasibility` is a descriptor, **not** `fitness` —
  don't fold plausibility into the fitness score too, or you double-count it.
- **open / primary-novelty** → a few words naming the *core mechanism* — the
  thing that makes this idea work. The `mechanism` is **chosen first** (Layer 1 of
  generation: see `principle_first`), and the surface idea is written to express it —
  so mechanism distinctness is decided up front, not retrofitted. This is the
  **primary-novelty niche** the engine carries, so **no two candidates may share an
  identical `mechanism` string unless they genuinely share a mechanism.** Distinct
  ideas must name distinct mechanisms.
- Make every axis value **meaningfully distinct**, not a cosmetic re-wording of
  another candidate's. Two descriptors that differ only superficially land in the
  same niche and compete as one — variety you never get back.
