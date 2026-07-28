"""Core types and per-session axes resolution.

This module is the lowest-level foundation: it defines the shared dataclasses
(``Axis``, ``AxesSpec``, ``Candidate``, ``Niche``) and loads/validates the
resolved axes for a session. Axes can arrive three ways, all producing an
identical ``AxesSpec``:

* a **named** domain config (``config/domains/*.yaml``),
* an **inferred** ``axes.json`` written by the agent, or
* the **generic** fallback (``config/domains/generic.yaml``).

The engine never assumes a domain — every command receives the resolved axes.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # pyyaml is a hard dependency, but keep the import error legible.
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "pyyaml is required (pip install -r requirements.txt)"
    ) from exc


AXIS_TYPES = ("categorical", "continuous", "open")


class ConfigError(ValueError):
    """Raised when an axes spec is malformed. Message is user-facing."""


def require_sklearn(what: str) -> None:
    """Raise an actionable ``ConfigError`` when scikit-learn — the peer hard-dep behind the ``hash``
    embedder's ``HashingVectorizer`` and the open-axis nicher's ``KMeans`` — is absent, instead of
    leaking a raw ``ModuleNotFoundError``. ONE home for the message so the two call sites
    (``embed.HashingEmbedder``, ``archive.CVTNicher.fit``) can't drift. The caller imports the
    specific submodule after this returns. Deliberately does NOT suggest ``CAMBRIAN_EMBEDDER=hash``:
    the hash embedder itself needs sklearn, so re-provisioning is the only real remedy."""
    try:
        import sklearn  # noqa: F401 — presence probe; the caller imports the submodule next
    except ImportError as exc:
        raise ConfigError(
            f"Cambrian {what} unavailable: the 'scikit-learn' package is not installed in the "
            f"engine venv. Re-run provisioning (the SessionStart hook, or "
            f"`python skills/ideate/scripts/bootstrap.py`)."
        ) from exc


def debug_enabled() -> bool:
    """Whether ``CAMBRIAN_DEBUG`` requests full tracebacks instead of clean errors.

    Treats unset/empty and the explicit off-values ``0``/``false``/``no``/``off``
    as disabled, so ``CAMBRIAN_DEBUG=0`` doesn't accidentally *enable* debugging
    (a bare ``os.environ.get`` would, since ``"0"`` is truthy).
    """
    return os.environ.get("CAMBRIAN_DEBUG", "").strip().lower() not in (
        "", "0", "false", "no", "off",
    )


# --------------------------------------------------------------------------- #
# Core dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Axis:
    """One descriptor axis of the idea space.

    ``type`` is one of ``categorical`` | ``continuous`` | ``open``. ``range`` is
    required for continuous axes. ``primary_novelty`` marks the single "open"
    axis that carries most of the novelty (e.g. a mechanism/approach axis); it is
    niched via CVT over embeddings.
    """

    name: str
    type: str
    range: Optional[Tuple[float, float]] = None
    primary_novelty: bool = False
    bins: int = 5  # discretization granularity for continuous axes

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type}
        if self.range is not None:
            out["range"] = list(self.range)
        if self.primary_novelty:
            out["primary_novelty"] = True
        if self.type == "continuous" and self.bins != 5:
            out["bins"] = self.bins
        return out


@dataclass
class AxesSpec:
    """The fully-resolved descriptor space for a session.

    This is purely the engine's *geometry*: the descriptor axes plus
    ``slate_size`` (the only runtime knob the engine itself consumes, when
    sizing the DPP slate). Agent-/session-level settings that merely ride
    alongside the axes in the same config file (how many candidates the agent
    drafts, which rubric it prefilters with) live in :class:`SessionSettings`
    so they don't leak into the engine's core type.
    """

    domain: str
    unit_of_generation: str
    axes: List[Axis]
    slate_size: int = 6

    @property
    def axis_names(self) -> List[str]:
        return [a.name for a in self.axes]

    def axis(self, name: str) -> Axis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(name)

    @property
    def primary_axis(self) -> Optional[Axis]:
        for a in self.axes:
            if a.primary_novelty:
                return a
        # Fall back to the first open axis if none flagged.
        for a in self.axes:
            if a.type == "open":
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "unit_of_generation": self.unit_of_generation,
            "axes": [a.to_dict() for a in self.axes],
            "slate_size": self.slate_size,
        }

    def __eq__(self, other: object) -> bool:  # value equality for "load identically"
        if not isinstance(other, AxesSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        # Defining __eq__ otherwise makes the class unhashable; keep hashing
        # consistent with equality (same resolved spec -> same hash).
        return hash(json.dumps(self.to_dict(), sort_keys=True))


@dataclass
class SessionSettings:
    """Agent-/session-level settings that live in the axes config but are NOT
    engine geometry.

    The engine never acts on these; the **agent** reads ``judge_rubric`` (which
    rubric to prefilter with) and ``candidates_per_generation`` (how many ideas
    to draft per cycle), and the self-test reads the latter. Keeping them out of
    :class:`AxesSpec` keeps the engine's core type purely about the descriptor
    space, sharpening the engine/agent contract boundary.
    """

    candidates_per_generation: int = 12
    judge_rubric: str = "references/judge_rubric.md"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_per_generation": self.candidates_per_generation,
            "judge_rubric": self.judge_rubric,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionSettings":
        if not isinstance(d, dict):
            raise ConfigError(
                f"settings must be an object, got {type(d).__name__}"
            )
        try:
            per_gen = int(d.get("candidates_per_generation", 12))
        except (TypeError, ValueError):
            raise ConfigError("'candidates_per_generation' must be an integer")
        if per_gen < 1:
            raise ConfigError("'candidates_per_generation' must be >= 1")
        return cls(
            candidates_per_generation=per_gen,
            judge_rubric=str(d.get("judge_rubric", "references/judge_rubric.md")),
        )


@dataclass
class EngineConfig:
    """Tuning knobs the engine reads, with per-domain overrides.

    These used to be scattered module constants (``DEDUP_TAU``, ``KNN_K``,
    ``OPEN_NICHES`` …), tuned for one embedder. They live here so a domain config
    can override any of them via an optional ``engine:`` block while the defaults
    reproduce the original behavior exactly. Like :class:`SessionSettings`, this
    is *not* engine geometry — it rides alongside the axes in the same file and is
    parsed out separately, so :class:`AxesSpec` stays pure.

    ``dedup_tau`` defaults to ``None``, meaning "use the per-embedder default"
    (see ``embed.default_dedup_tau``); set it to pin a fixed threshold.
    """

    # niching
    open_niches: int = 24
    # freeze the open-axis partition once freeze_factor * open_niches survivor
    # mechanisms accumulate; 2 => 48 (~4-5 generations) so it activates in a real
    # session while keeping the k-means fit meaningful. See pipeline.py for the
    # rationale; cold-start runs until then.
    open_niche_freeze_factor: int = 2
    # novelty / dedup
    knn_k: int = 5
    dedup_tau: Optional[float] = None
    novelty_ref_cap: int = 500
    # DPP slate
    max_dpp_pool: int = 200
    quality_weight: float = 0.3
    # anti-collapse monitor
    monitor_cos_threshold: float = 0.55
    monitor_entropy_threshold: float = 0.50
    monitor_margin: float = 0.15
    monitor_cos_ceiling: float = 0.80
    monitor_window: int = 5
    monitor_min_baseline: int = 2
    # variety-erosion sensor (S2); see monitor.assess_variety_erosion
    erosion_window: int = 5          # W
    erosion_accel_ratio: float = 0.5  # rho
    erosion_persist: int = 2         # K
    # prefilter guard: flag a soft "under_generation" signal when the agent submits
    # fewer than this fraction of ``candidates_per_generation`` to ingest (the agent
    # may be over-prefiltering and cutting variety under cover of "off-brief").
    under_generation_ratio: float = 0.6
    # state hygiene: once the candidate store exceeds this many records, drop the
    # ones never read again (everything but archive elites, pins, and comparison
    # ids) to bound the per-cycle whole-file rewrite cost. 0 disables pruning.
    state_prune_threshold: int = 2000
    # active-learning pair policy (memory.select_ask_pairs). Weights of the
    # informativeness score: embedding similarity, fitness uncertainty, novelty.
    # Defaults reproduce the original behavior (compare *similar* ideas → learn the
    # preference function). Set ask_sim_weight <= 0 to compare *region-separating*
    # pairs instead (explore). Allowed range [-1, 1].
    ask_sim_weight: float = 0.5
    ask_uncertainty_weight: float = 0.3
    ask_novelty_weight: float = 0.2
    # active-learning pair policy schedule (S3). Number of initial 0-indexed
    # generations that use a region-separating (explore) weight before switching to
    # the configured ask_sim_weight (refine). 0 DISABLES the schedule -> flat
    # ask_sim_weight for every generation (current behavior, no silent flip).
    # Recommended value to enable: 1 (explore on generation 0, refine from 1).
    explore_until_generation: int = 0
    # advisory surface/mechanism gap probe (measurement only). When True, ingest emits a
    # `surface_mechanism_gap` record on the slate and appends it to a bounded meta log. Off
    # by default: zero cost, output unchanged. Never affects selection or any gate.
    gap_probe: bool = False

    def ask_sim_weight_for_generation(self, generation: int) -> float:
        """Effective ask-pair similarity weight for a 0-indexed generation.

        Default (explore_until_generation == 0): the configured flat
        ``ask_sim_weight`` for every generation. When explore_until_generation > 0,
        the first that-many generations use a region-separating (explore) weight,
        ``-abs(ask_sim_weight)``; later generations use ``ask_sim_weight`` (refine).
        One knob: the explore magnitude tracks the refine magnitude by design.
        """
        if self.explore_until_generation > 0 and generation < self.explore_until_generation:
            return -abs(self.ask_sim_weight)
        return self.ask_sim_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open_niches": self.open_niches,
            "open_niche_freeze_factor": self.open_niche_freeze_factor,
            "knn_k": self.knn_k,
            "dedup_tau": self.dedup_tau,
            "novelty_ref_cap": self.novelty_ref_cap,
            "max_dpp_pool": self.max_dpp_pool,
            "quality_weight": self.quality_weight,
            "monitor_cos_threshold": self.monitor_cos_threshold,
            "monitor_entropy_threshold": self.monitor_entropy_threshold,
            "monitor_margin": self.monitor_margin,
            "monitor_cos_ceiling": self.monitor_cos_ceiling,
            "monitor_window": self.monitor_window,
            "monitor_min_baseline": self.monitor_min_baseline,
            "erosion_window": self.erosion_window,
            "erosion_accel_ratio": self.erosion_accel_ratio,
            "erosion_persist": self.erosion_persist,
            "under_generation_ratio": self.under_generation_ratio,
            "state_prune_threshold": self.state_prune_threshold,
            "ask_sim_weight": self.ask_sim_weight,
            "ask_uncertainty_weight": self.ask_uncertainty_weight,
            "ask_novelty_weight": self.ask_novelty_weight,
            "explore_until_generation": self.explore_until_generation,
            "gap_probe": self.gap_probe,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        """Build from a full config dict, reading its optional ``engine:`` block."""
        if not isinstance(d, dict):
            raise ConfigError(f"config must be an object, got {type(d).__name__}")
        eng = d.get("engine", {})
        if eng is None:  # `engine:` present but empty
            eng = {}
        if not isinstance(eng, dict):
            raise ConfigError(f"'engine' must be an object, got {type(eng).__name__}")
        base = cls()

        # Reject unknown keys rather than silently ignoring them: a typo'd knob
        # (e.g. 'qualtiy_weight') would otherwise be a no-op and the domain would
        # quietly run on defaults. The valid set is exactly what ``to_dict`` emits.
        known = set(base.to_dict())
        unknown = sorted(k for k in eng if k not in known)
        if unknown:
            raise ConfigError(
                f"unknown engine config key(s): {', '.join(unknown)}; "
                f"valid keys are {', '.join(sorted(known))}"
            )

        def pos_int(key: str, default: int) -> int:
            v = eng.get(key, default)
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ConfigError(f"engine.{key} must be an integer")
            if v < 1:
                raise ConfigError(f"engine.{key} must be >= 1")
            return v

        def non_neg_int(key: str, default: int) -> int:
            v = eng.get(key, default)
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ConfigError(f"engine.{key} must be an integer")
            if v < 0:
                raise ConfigError(f"engine.{key} must be >= 0")
            return v

        def int_min(key: str, default: int, minimum: int) -> int:
            v = eng.get(key, default)
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ConfigError(f"engine.{key} must be an integer")
            if v < minimum:
                raise ConfigError(f"engine.{key} must be >= {minimum}")
            return v

        def unit_float(key: str, default: float, lo: float = 0.0,
                       hi: float = 1.0) -> float:
            v = eng.get(key, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ConfigError(f"engine.{key} must be a number")
            if not (lo <= v <= hi):
                raise ConfigError(f"engine.{key} must be in [{lo}, {hi}]")
            return v

        def flag(key: str, default: bool) -> bool:
            v = eng.get(key, default)
            if not isinstance(v, bool):
                raise ConfigError(f"engine.{key} must be a boolean")
            return v

        dedup_tau = eng.get("dedup_tau", None)
        if dedup_tau is not None:
            try:
                dedup_tau = float(dedup_tau)
            except (TypeError, ValueError):
                raise ConfigError("engine.dedup_tau must be a number or null")
            if not (0.0 < dedup_tau <= 1.0):
                raise ConfigError("engine.dedup_tau must be in (0, 1]")

        return cls(
            open_niches=pos_int("open_niches", base.open_niches),
            open_niche_freeze_factor=pos_int(
                "open_niche_freeze_factor", base.open_niche_freeze_factor
            ),
            knn_k=pos_int("knn_k", base.knn_k),
            dedup_tau=dedup_tau,
            novelty_ref_cap=pos_int("novelty_ref_cap", base.novelty_ref_cap),
            max_dpp_pool=pos_int("max_dpp_pool", base.max_dpp_pool),
            quality_weight=unit_float("quality_weight", base.quality_weight),
            monitor_cos_threshold=unit_float(
                "monitor_cos_threshold", base.monitor_cos_threshold
            ),
            monitor_entropy_threshold=unit_float(
                "monitor_entropy_threshold", base.monitor_entropy_threshold
            ),
            monitor_margin=unit_float("monitor_margin", base.monitor_margin),
            monitor_cos_ceiling=unit_float(
                "monitor_cos_ceiling", base.monitor_cos_ceiling
            ),
            monitor_window=pos_int("monitor_window", base.monitor_window),
            monitor_min_baseline=pos_int(
                "monitor_min_baseline", base.monitor_min_baseline
            ),
            erosion_window=int_min("erosion_window", base.erosion_window, 3),
            erosion_accel_ratio=unit_float(
                "erosion_accel_ratio", base.erosion_accel_ratio, lo=0.0, hi=5.0
            ),
            erosion_persist=pos_int("erosion_persist", base.erosion_persist),
            under_generation_ratio=unit_float(
                "under_generation_ratio", base.under_generation_ratio
            ),
            state_prune_threshold=non_neg_int(
                "state_prune_threshold", base.state_prune_threshold
            ),
            ask_sim_weight=unit_float(
                "ask_sim_weight", base.ask_sim_weight, lo=-1.0, hi=1.0
            ),
            ask_uncertainty_weight=unit_float(
                "ask_uncertainty_weight", base.ask_uncertainty_weight, lo=-1.0, hi=1.0
            ),
            ask_novelty_weight=unit_float(
                "ask_novelty_weight", base.ask_novelty_weight, lo=-1.0, hi=1.0
            ),
            explore_until_generation=non_neg_int(
                "explore_until_generation", base.explore_until_generation
            ),
            gap_probe=flag("gap_probe", base.gap_probe),
        )


@dataclass
class Candidate:
    """An idea produced by the agent and consumed by ``ingest``."""

    id: str
    text: str
    descriptor: Dict[str, Any] = field(default_factory=dict)
    genealogy: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 1.0  # within-niche quality from the judge; NOT novelty

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        if not isinstance(d, dict):
            raise ConfigError(f"candidate must be an object, got {type(d).__name__}")
        cid = d.get("id")
        if not cid or not isinstance(cid, str):
            raise ConfigError("candidate missing a non-empty string 'id'")
        text = d.get("text", "")
        if not isinstance(text, str):
            raise ConfigError(f"candidate {cid!r} 'text' must be a string")
        if not text.strip():
            # Empty/whitespace text embeds to a zero vector, which never dedups
            # (cosine 0 < tau) and always scores maximally novel — silently
            # poisoning dedup and the novelty signal. Reject it like an empty id.
            raise ConfigError(f"candidate {cid!r} 'text' must be a non-empty string")
        descriptor = d.get("descriptor", {}) or {}
        if not isinstance(descriptor, dict):
            raise ConfigError(f"candidate {cid!r} 'descriptor' must be an object")
        genealogy = d.get("genealogy", {}) or {}
        if not isinstance(genealogy, dict):
            raise ConfigError(f"candidate {cid!r} 'genealogy' must be an object")
        fitness = d.get("fitness", 1.0)
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            raise ConfigError(f"candidate {cid!r} 'fitness' must be a number")
        if not math.isfinite(fitness):
            # A non-finite fitness (NaN/inf) silently poisons elite selection
            # (NaN comparisons are always False) and the DPP quality kernel.
            raise ConfigError(f"candidate {cid!r} 'fitness' must be finite")
        return cls(id=cid, text=text, descriptor=descriptor,
                   genealogy=genealogy, fitness=fitness)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "descriptor": dict(self.descriptor),
            "genealogy": dict(self.genealogy),
            "fitness": self.fitness,
        }


@dataclass
class Niche:
    """One MAP-Elites cell: at most one elite candidate per niche."""

    id: str
    coords: Dict[str, Any] = field(default_factory=dict)
    elite_id: Optional[str] = None
    fitness: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "coords": dict(self.coords),
            "elite_id": self.elite_id,
            "fitness": self.fitness,
            "novelty": self.novelty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Niche":
        return cls(
            id=d["id"],
            coords=dict(d.get("coords", {})),
            elite_id=d.get("elite_id"),
            fitness=float(d.get("fitness", 0.0)),
            novelty=float(d.get("novelty", 0.0)),
        )


# --------------------------------------------------------------------------- #
# Axes loading & validation
# --------------------------------------------------------------------------- #
def _coerce_range(value: Any, axis_name: str) -> Tuple[float, float]:
    # A ``str`` already fails the (list, tuple) check, so no separate str branch is needed.
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(
            f"axis {axis_name!r}: 'range' must be a [lo, hi] pair"
        )
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise ConfigError(f"axis {axis_name!r}: 'range' bounds must be numbers")
    if not hi > lo:
        raise ConfigError(
            f"axis {axis_name!r}: 'range' hi ({hi}) must be greater than lo ({lo})"
        )
    return (lo, hi)


def _axis_from_dict(d: Any) -> Axis:
    if not isinstance(d, dict):
        raise ConfigError(f"each axis must be an object, got {type(d).__name__}")
    name = d.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("axis missing a non-empty string 'name'")
    atype = d.get("type")
    if atype not in AXIS_TYPES:
        raise ConfigError(
            f"axis {name!r}: 'type' must be one of {AXIS_TYPES}, got {atype!r}"
        )
    arange: Optional[Tuple[float, float]] = None
    if atype == "continuous":
        if "range" not in d:
            raise ConfigError(f"axis {name!r}: continuous axis requires a 'range'")
        arange = _coerce_range(d["range"], name)
    elif "range" in d and d["range"] is not None:
        arange = _coerce_range(d["range"], name)
    primary = bool(d.get("primary_novelty", False))
    try:
        bins = int(d.get("bins", 5))
    except (TypeError, ValueError):
        raise ConfigError(f"axis {name!r}: 'bins' must be an integer")
    if bins < 1:
        raise ConfigError(f"axis {name!r}: 'bins' must be >= 1")
    return Axis(name=name, type=atype, range=arange,
                primary_novelty=primary, bins=bins)


def axes_spec_from_dict(d: Dict[str, Any]) -> AxesSpec:
    """Build and validate an :class:`AxesSpec` from a plain dict (json or yaml)."""
    if not isinstance(d, dict):
        raise ConfigError(f"axes spec must be an object, got {type(d).__name__}")
    raw_axes = d.get("axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ConfigError("axes spec must contain a non-empty 'axes' list")
    axes = [_axis_from_dict(a) for a in raw_axes]

    names = [a.name for a in axes]
    if len(names) != len(set(names)):
        raise ConfigError(f"axis names must be unique, got {names}")

    n_primary = sum(1 for a in axes if a.primary_novelty)
    if n_primary > 1:
        raise ConfigError(
            "at most one axis may set 'primary_novelty: true'; "
            f"{n_primary} did"
        )

    try:
        slate = int(d.get("slate_size", 6))
    except (TypeError, ValueError):
        raise ConfigError("'slate_size' must be an integer")
    if slate < 1:
        raise ConfigError("'slate_size' must be >= 1")

    return AxesSpec(
        domain=str(d.get("domain", "ad-hoc")),
        unit_of_generation=str(d.get("unit_of_generation", "idea")),
        axes=axes,
        slate_size=slate,
    )


def _load_config_dict(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Read a raw config dict from a dict, a ``.json`` file, or a ``.yaml`` file.

    Shared by :func:`load_axes` and :func:`load_session_settings` so axes and
    settings parse identically out of the same file.
    """
    if isinstance(source, dict):
        return source
    path = Path(source)
    if not path.exists():
        raise ConfigError(f"config source not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            # Be forgiving: try yaml (a superset of json) for unknown extensions.
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse config file {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"config file {path} is empty")
    return data


def load_axes(source: Union[str, Path, Dict[str, Any]]) -> AxesSpec:
    """Load axes from a dict, a ``.json`` file, or a ``.yaml``/``.yml`` file.

    All three paths produce an identical :class:`AxesSpec`.
    """
    return axes_spec_from_dict(_load_config_dict(source))


def load_session_settings(
    source: Union[str, Path, Dict[str, Any]]
) -> SessionSettings:
    """Load the agent-/session-level :class:`SessionSettings` from the same
    dict/file the axes come from (missing keys fall back to defaults)."""
    return SessionSettings.from_dict(_load_config_dict(source))


def load_engine_config(source: Union[str, Path, Dict[str, Any]]) -> EngineConfig:
    """Load the engine tuning :class:`EngineConfig` (its optional ``engine:``
    block) from the same dict/file the axes come from; defaults reproduce the
    original behavior."""
    return EngineConfig.from_dict(_load_config_dict(source))


def load_all(
    source: Union[str, Path, Dict[str, Any]]
) -> Tuple[AxesSpec, SessionSettings, EngineConfig]:
    """Load axes + session settings + engine config in a SINGLE file read.

    The three single-purpose loaders each re-read and re-parse the source; a caller
    that needs more than one (``init_project``) should use this to touch the file
    once. Equivalent to calling the three loaders, just without the extra reads.
    """
    d = _load_config_dict(source)
    return (
        axes_spec_from_dict(d),
        SessionSettings.from_dict(d),
        EngineConfig.from_dict(d),
    )


def load_axes_and_engine(
    source: Union[str, Path, Dict[str, Any]]
) -> Tuple[AxesSpec, EngineConfig]:
    """Load axes + engine config in one file read (the pair ``ingest`` consumes).

    Deliberately does NOT parse :class:`SessionSettings`: ``ingest`` never reads them,
    and an axes file with malformed settings must not fail a cycle that doesn't touch
    them (settings are validated once, at ``init`` time).
    """
    d = _load_config_dict(source)
    return axes_spec_from_dict(d), EngineConfig.from_dict(d)


def generic_axes_path() -> Path:
    """Absolute path to the bundled neutral fallback ``generic.yaml``."""
    # cambrian_engine/config.py -> .../skills/ideate/scripts/cambrian_engine
    # generic.yaml -> .../skills/ideate/config/domains/generic.yaml
    skill_dir = Path(__file__).resolve().parents[2]
    return skill_dir / "config" / "domains" / "generic.yaml"


def load_generic_axes() -> AxesSpec:
    """Load the bundled generic fallback axes."""
    return load_axes(generic_axes_path())
