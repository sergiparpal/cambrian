# Inferring descriptor axes from a brief

When the user does not name a shipped domain, you infer the axes for this session.
Good axes are what make "diverse" meaningful, so spend a little care here — then
confirm with ONE short question.

## Goal

Produce **4–6 axes** that:
- are **independent** (vary one without forcing another),
- together **span** the interesting space of the brief,
- are **legible** to the user (they can picture what each means),
- include exactly **one `open` axis** marked `primary_novelty: true` — the main
  carrier of novelty (usually the *mechanism / approach / core "how"*).

## Axis types

- `categorical` — a discrete choice with a handful of natural values (audience,
  tone, format, channel, genre, material…). You don't have to enumerate values;
  the engine buckets by the value you assign each idea.
- `continuous` — a dial with a numeric `range` (boldness, complexity, intimacy,
  time-horizon, cost). Always give `range: [lo, hi]`.
- `open` — a free-text dimension that can't be pre-enumerated (the mechanism,
  the underlying metaphor, the approach). Mark the most novelty-bearing one
  `primary_novelty: true`. The engine niches it geometrically over embeddings.

## A recipe

1. Identify the **unit of generation** (idea | concept | hypothesis | name |
   feature | plot | strategy …). That sets `unit_of_generation`.
2. Find the **mechanism axis**: "what makes two solutions *fundamentally*
   different here?" Make that the `open`, `primary_novelty` axis.
3. Add 1–2 **audience/context** categoricals: who it's for, where/when it lives.
4. Add 1–2 **form/style** categoricals: shape, format, register, genre.
5. Add a **`feasibility` continuous dial** (`range: [0.0, 1.0]`, far-fetched↔buildable)
   whenever the unit can be more or less practical — ideas, concepts, features,
   strategies, hypotheses. It is a **coverage axis**: MAP-Elites keeps buildable
   ideas in their own niches (a bold idea can't evict a practical one), which is what
   lets the **reach** dial keep practical options alive rather than drowning in
   extravagant ones. Skip it only when "buildable" isn't meaningful for the unit
   (names, taglines). Then add **at most one other** continuous dial the user clearly
   cares about (`boldness` safe↔daring, simple↔complex, cheap↔lavish, near↔far term).
6. Drop redundant axes (if two move together, keep one). Aim for 4–6 total.

## Worked examples (illustrative — infer fresh each time)

**Brief: "names for a new productivity app."**
```yaml
unit_of_generation: name
axes:
  - {name: tone, type: categorical}                 # playful / serious / cryptic
  - {name: imagery, type: categorical}              # nature / motion / tools / abstract
  - {name: length, type: continuous, range: [1, 4]} # syllables
  - {name: construction, type: open, primary_novelty: true}  # pun / coinage / metaphor / compound
```

**Brief: "research hypotheses for why retention dropped."**
```yaml
unit_of_generation: hypothesis
axes:
  - {name: locus, type: categorical}                # product / pricing / market / ops
  - {name: user_segment, type: categorical}         # new / power / dormant
  - {name: time_horizon, type: continuous, range: [0, 1]}  # acute <-> structural
  - {name: feasibility, type: continuous, range: [0, 1]}   # far-fetched <-> testable (coverage axis)
  - {name: causal_mechanism, type: open, primary_novelty: true}  # the proposed "why"
```

**Brief: "concepts for a public art installation."**
```yaml
unit_of_generation: concept
axes:
  - {name: site, type: categorical}                 # plaza / transit / water / rooftop
  - {name: sense, type: categorical}                # visual / sonic / tactile / olfactory
  - {name: scale, type: continuous, range: [0, 1]}  # intimate <-> monumental
  - {name: feasibility, type: continuous, range: [0, 1]}  # far-fetched <-> buildable (coverage axis)
  - {name: interaction, type: categorical}          # passive / participatory / generative
  - {name: device, type: open, primary_novelty: true}  # the core artistic mechanism
```

## Confirm, then proceed

Show the inferred axes in one line and ask the user to accept or tweak:

> Axes for this session: **tone, imagery, length (1–4 syllables), construction
> (the main novelty axis)**. Good, or change any?

Take a quick edit or an "ok". If you cannot find well-separated, meaningful axes,
**fall back to `config/domains/generic.yaml`** and say so — don't ship vague axes.
