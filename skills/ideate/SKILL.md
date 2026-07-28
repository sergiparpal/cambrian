---
name: ideate
description: >
  Generate a diverse, non-cliché slate of ideas/concepts from a brief in ANY
  domain, using a blind-variation + diverse-archive loop with the user selecting
  in chat. Use when the user asks to brainstorm, ideate, explore an idea space,
  find fresh/original angles, generate many distinct options, or escape clichéd
  or samey concepts — regardless of subject (marketing, product, research,
  naming, design, fiction, strategy, etc.). A "reach" dial (conservative |
  balanced | wild) tunes how far out the ideas go without narrowing their
  variety. The deterministic diversity engine (embeddings, MAP-Elites, novelty,
  DPP, anti-collapse monitor) runs locally; you do the generating and judging.
allowed-tools: Bash, Read, Write
---

# Cambrian — ideate

Brief: $ARGUMENTS

You amplify creativity by pairing **your** generation + judgment with a local
**diversity engine** that owns the anti-convergence math. Diversity is decoupled
from quality: geometry (the engine) decides what is *new*; you only filter what
is *valid/on-brief* and rank *within* a niche. Never let the judge pick the final
slate. The user is the real selector.

## Reach — how far out (the divergence dial)

Maximum divergence is not always what the user wants: it produces extravagant, and
sometimes absurd or cryptic, ideas. **Reach** is the tempered middle band between
greatest divergence and collapse-to-the-mean. It caps divergence in *degree*
(`boldness`) and biases the operator mix and `feasibility` lean — it **never** caps
divergence in *kind* (the `mechanism` axis always spreads maximally, so every slate
is still made of genuinely different approaches).

| reach | `boldness` band | `feasibility` lean | operators | feel |
| --- | --- | --- | --- | --- |
| `conservative` | `[0, 0.5]` | high (`0.6`–`1.0`) | recombinative | spirited but buildable |
| **`balanced`** (default) | `[0, 0.75]` | spread `[0.2, 1.0]` | mixed | novel, still coherent |
| `wild` | `[0, 1]` | full `[0, 1]` | far-reaching welcome | maximal divergence |

Pick the reach from the user's phrasing if they gave one (in `$ARGUMENTS` or the brief's
tone); otherwise **default to `balanced`** and mention it in the same one-line confirm as
the axes (step 3) — e.g. "Running at *balanced* reach; say *wild* for further-out or
*conservative* for safer." The user can change it any round.

Reach is a **language-layer directive** you apply when assigning descriptors and choosing
operators. The engine is unchanged: it still only measures dispersion over whatever you
submit. Coherence (the judge's validity gate) is enforced at **every** reach — `wild` means
further-out, never incoherent.

Follow `${CLAUDE_SKILL_DIR}/references/loop.md` exactly. Summary of one session:

1. **Locate the engine interpreter.** The venv auto-provisions in the background when
   the plugin loads, so it is usually ready already. Read the interpreter pointer from
   the **first** of these that exists, and set `ENGINE = "<PYBIN>" -m cambrian_engine`
   (quote `<PYBIN>` — Windows paths may contain spaces):
   - `${CLAUDE_PLUGIN_DATA}/venv/engine-python.txt`  (marketplace install)
   - `${CLAUDE_SKILL_DIR}/.venv/engine-python.txt`   (dev `--plugin-dir` / `setup.sh`)
2. **If neither pointer exists yet**, the one-time setup is still running, hasn't
   started, or **failed in the background**. First check the background log so you
   don't re-run blind — it sits next to the venv:
   `tail -n 40 "${CLAUDE_PLUGIN_DATA}/provision.log"` (marketplace) or
   `tail -n 40 "${CLAUDE_SKILL_DIR}/provision.log"` (dev); relay any real failure it
   shows. Then tell the user **once**: "⏳ Setting up the Cambrian engine — a one-time
   download of ML libraries and a small embedding model. This can take a few minutes;
   I'll continue automatically when it's ready." Then run the bootstrap in the
   foreground (idempotent; it waits for any in-progress background provision):
   `"<PY>" "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py" --venv "${CLAUDE_PLUGIN_DATA}/venv"`
   where `<PY>` is `python3` (macOS/Linux/WSL) or `py`/`python` (Windows). When it
   finishes, re-read the pointer. If the venv still can't be built, show the fresh
   `provision.log` tail alongside the bootstrap's error and stop — together they are
   the actionable diagnosis (e.g. Python 3.11+ missing). Then choose a short `PROJECT`
   id for this session.
3. **Resolve axes for this session** (diversity is only meaningful relative to a
   set of descriptor axes). Cascade:
   - if the user named a domain that has a config in
     `${CLAUDE_SKILL_DIR}/config/domains/examples/`, load it; else
   - **infer** 4–6 descriptor axes from the brief using
     `${CLAUDE_SKILL_DIR}/references/axis_inference.md` (mark exactly one `open`
     axis as the primary novelty carrier) and **confirm them with ONE short
     question** the user can accept or tweak — the same one-line confirm also
     states the **reach** (see *Reach* above); else
   - load `${CLAUDE_SKILL_DIR}/config/domains/generic.yaml`.
   Resolve the per-project scratch dir with `ENGINE paths --project PROJECT` and
   use its `tmp` field as `$TMP` (inside the state home, never your cwd) for every
   hand-off file. Write the resolved axes to `$TMP/axes.json`, then run
   `ENGINE init-project --project PROJECT --axes $TMP/axes.json` and
   `ENGINE recall --project PROJECT`.
4. **Generate** candidates yourself using
   `${CLAUDE_SKILL_DIR}/references/operators.md`. Apply several different
   operators; for each candidate report its `descriptor` on the resolved axes and
   its `genealogy` (parent ids + operator id). Push for variety — each new
   approach must differ from the ones already shown.
   - **Give every candidate an `id` that is unique for the life of the PROJECT** —
     across generations *and* across sessions resuming the same project, not just
     within one batch. The id keys the archive, the embeddings, and preference
     memory, so a reused one would rewrite the archived idea it names. Prefix ids
     with the generation (`g0-1`, `g1-1`, …). The engine rejects a collision with a
     clear error; renumber and re-ingest if you hit one.
   - **Before generating, map the clichés (anti-cliché directive).** Enumerate the
     **~6 most obvious / cliché answers** to *this* brief — the responses anyone
     would reach for first. Split them in half: **O_train** (first ~3 — keep these
     in view and deliberately generate *away* from them; see the `anti_cliche`
     operator in `operators.md`) and **O_test** (last ~3 — set aside; do **not**
     look at them while generating, and never optimize toward them). Build the
     obvious-set fresh per brief — it is a construction recipe, not a fixed list.
   - **Honest measurement, never Goodhart.** After the slate is presented (step 7)
     you *may* report each idea's originality as its distance to the **held-out
     O_test** (advisory only — it never picks or ranks the slate). **Never**
     validate against **O_train**: scoring against the very clichés you steered
     away from just confirms you followed instructions (a Goodhart trap); the
     held-out O_test is the only honest referent.
   - **Caveat.** "Obvious" is still *your* (Claude's) notion of cliché, not the
     world's. This **hedges** cliché; it does **not** guarantee novelty against
     prior art or the wider world.
   - **Generate mechanism-first (two layers).**
     - *Layer 1 — mechanism first.* Before writing a candidate's `text`, decide its
       **mechanism** (the open / `primary_novelty` axis value): the core "how it
       works" in a few words. Choose it to sit **far from** (a) the obvious-set
       mechanisms (`O_train`) and (b) the mechanisms already chosen for other
       candidates this round and already in the archive.
     - *Layer 2 — surface second.* Only then write the idea `text` that **expresses
       that mechanism**. The surface is a realization of the mechanism, not the other
       way round.
     - **Same mechanism = same idea.** Two ideas with the same mechanism are the same
       idea — vary the mechanism, then the surface follows. This is what the engine
       niches on (the open axis), so mechanism variety is what produces a genuinely
       diverse slate.
     - **Caveat.** "Mechanism" here is *your* notion of approach; distinct mechanism
       strings hedge mechanism-monotony, they do not guarantee functionally
       independent mechanisms.
   - **Descriptor discipline (protect niche placement).** Niche placement — incl.
     the open axis — runs on *your* descriptor words **before** the engine sees
     anything, so lazy or duplicate descriptors collapse niches up front. Give no
     two candidates an identical `mechanism` string unless they genuinely share a
     mechanism (the open-axis `mechanism` text carries the primary-novelty niche);
     prefer the **extremes of the reach band** (see *Reach* above) on continuous
     axes (e.g. `boldness`) over the middle;
     and make each axis value **meaningfully distinct**, not a cosmetic variation.
     See `operators.md` → "Descriptor discipline".
5. **Prefilter** yourself using `${CLAUDE_SKILL_DIR}/references/judge_rubric.md`
   to drop only invalid / off-brief candidates. NEVER judge novelty here. You may
   attach a within-niche `fitness` (0–1); you may NOT use it to cut variety.
6. **Ingest.** Write survivors to `$TMP/candidates.json` and run
   `ENGINE ingest --project PROJECT --candidates $TMP/candidates.json --axes $TMP/axes.json`.
7. **Present** the returned `slate` (show each idea with its niche `coords` so the
   user can judge distinctness). **De-obscure each item**: add a plain "how it works
   in practice" clause and one concrete instantiation, so a bold idea reads as clear
   rather than cryptic. Ask only the returned `ask_pairs` as short
   A-vs-B questions, then **explicitly invite the user to pin any idea they like as
   a "stepping stone" — including ideas that were NOT in the asked pairs** (e.g.
   "Want me to pin any of these to keep exploring from? — any of them, not just A/B").
   Pinning is the user's strongest, most durable preference signal: pins are always
   kept as parents for the next generation and recalled across sessions, whereas the
   A-vs-B answers only feed a **low-weight** preference signal (recorded in memory,
   recalled into the next round) that can never prune variety or pick the slate. So
   if the user's favorite is an idea you didn't ask about, prompt them to
   pin it rather than let it pass unrecorded. Then also **offer the negative lever —
   the user may discard any ideas they don't want carried forward** (e.g. "Want me to
   drop any of these so we stop building on them?"). A discard is the symmetric
   opposite of a pin: the engine removes that idea from future slates and never breeds
   from it, and it persists across sessions (re-pinning an id un-discards it — pin and
   discard of the same id are mutually exclusive, latest action wins). Discarding is
   the user's call, never yours — never discard to enforce a taste or cut variety.
   Note on the `novelty`
   field: it is **mean k-NN distance to this session's own ideas (elites + this
   batch)** — a *variety* proxy, NOT originality vs. prior art / the world. Read a
   high `novelty` as "unlike the other ideas in this run", and don't present it to
   the user as proof an idea is novel to the world. Each slate item also carries
   `mechanism_novelty` — the variety of its *mechanism* vs the session's own
   mechanisms (a mechanism-space analog of `novelty`, same k-NN kernel on the open
   axis). Advisory like `novelty`/originality: report it, never select on it.
8. **Record & continue.** For each answer/pin/discard run `ENGINE remember`
   (`{"type":"pin","id":…}`, `{"type":"discard","id":…}`, or
   `{"type":"comparison","winner":…,"loser":…}`); then `ENGINE parents` to get diverse
   parents (pins always kept, discards always excluded) and loop from step 4, or stop
   on the user's command. Use the `parents` **command** for this — `ingest`'s
   `slate_ids` is only the slate's id list and honors neither pins nor discards. Ids
   returned under `stale_pins` are pins whose idea was cleared by an axes change:
   never breed from them.
9. **React to the monitor.** If `monitor.collapsing` is true, raise diversity
   directives next round (new operators, forbid the crowded niches, demand
   distance from recent ideas) — never remove or bypass the monitor. If
   `monitor.under_generation` is true, you over-prefiltered: next round generate the
   full target and cut **only** invalid/off-brief ideas, never the merely unusual.
   If `monitor.variety_eroding` is true (advisory early warning: survivor novelty is
   decaying *faster over time* while submits stay healthy — a generator regressing to
   the mode before `collapsing` trips), push unused operators and *mechanism* variety
   next round. See `references/loop.md` §7.
10. **Session-end gap summary (advisory; only when `gap_probe` is enabled).** If — and only
    if — this session's `ingest` output has carried a `surface_mechanism_gap` block (it
    appears only when `engine.gap_probe: true` is set in the resolved axes; by default it is
    absent and this step is skipped entirely), then when the session ends give a short
    **plain-language** read of whether the slates were varied in *approach* or only in
    *wording*. Fetch the durable series with `ENGINE metrics` (its `gap_log` field) and
    summarize across the session's cycles — **never per cycle** (one ~6-idea slate is too
    small to read). This is **measurement only, exactly like `novelty` / originality**: it
    never changed the slates and must never steer selection; you only report it. Full guide
    in `references/loop.md` → §8.

Read `${CLAUDE_SKILL_DIR}/references/loop.md` for exact JSON shapes, engine
contracts, and steering tactics. Never hard-code a domain — always use the axes
resolved in step 3.
