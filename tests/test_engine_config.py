"""Engine tuning lifted to config: defaults reproduce behavior; per-domain overrides.

The knobs (dedup tau, KNN k, open-niche count, DPP pool, monitor thresholds …)
used to be scattered module constants. They now live in ``EngineConfig`` with an
optional per-domain ``engine:`` block; the defaults must reproduce the original
behavior and modules must read from the resolved config.
"""

from __future__ import annotations

import numpy as np
import pytest

from cambrian_engine import config, embed, memory, monitor, pipeline
from cambrian_engine.config import EngineConfig
from cambrian_engine.state import State


def test_defaults_match_module_constants():
    # Drift guard: EngineConfig defaults mirror the module-level fallback constants.
    # The pipeline constants are now DERIVED from a default EngineConfig, so for those this is a
    # STRUCTURAL check (the names must exist and stay aligned) rather than a value-drift check;
    # for the monitor/memory constants it remains a genuine value-drift guard.
    c = EngineConfig()
    assert c.open_niches == pipeline.OPEN_NICHES
    assert c.open_niche_freeze_factor == pipeline.OPEN_NICHE_FREEZE_FACTOR
    assert c.knn_k == pipeline.KNN_K
    assert c.novelty_ref_cap == pipeline.NOVELTY_REF_CAP
    assert c.max_dpp_pool == pipeline.MAX_DPP_POOL
    assert c.quality_weight == pipeline.QUALITY_WEIGHT
    assert c.monitor_cos_threshold == monitor.DEFAULT_COS_THRESHOLD
    assert c.monitor_entropy_threshold == monitor.DEFAULT_ENTROPY_THRESHOLD
    assert c.monitor_margin == monitor.DEFAULT_MARGIN
    assert c.monitor_cos_ceiling == monitor.DEFAULT_COS_CEILING
    assert c.monitor_window == 5
    assert c.monitor_min_baseline == monitor.DEFAULT_MIN_BASELINE
    assert c.erosion_window == monitor.DEFAULT_EROSION_WINDOW
    assert c.erosion_accel_ratio == monitor.DEFAULT_EROSION_ACCEL_RATIO
    assert c.erosion_persist == monitor.DEFAULT_EROSION_PERSIST
    # dedup tau defaults to per-embedder (None), which for hash is the global default
    assert c.dedup_tau is None
    assert embed.default_dedup_tau("hash") == embed.DEFAULT_DEDUP_TAU
    # ask-pair weights mirror the memory module fallback constants
    assert c.ask_sim_weight == memory.W_SIM
    assert c.ask_uncertainty_weight == memory.W_UNCERTAIN
    assert c.ask_novelty_weight == memory.W_NOVELTY


def test_load_defaults_and_overrides():
    assert config.load_engine_config({}) == EngineConfig()
    cfg = config.load_engine_config(
        {"engine": {"open_niches": 8, "quality_weight": 0.0, "dedup_tau": 0.8}}
    )
    assert cfg.open_niches == 8
    assert cfg.quality_weight == 0.0
    assert cfg.dedup_tau == 0.8
    # unspecified keys keep their defaults
    assert cfg.knn_k == EngineConfig().knn_k
    # prefilter-guard ratio: default and override
    assert EngineConfig().under_generation_ratio == 0.6
    assert config.load_engine_config(
        {"engine": {"under_generation_ratio": 0.25}}
    ).under_generation_ratio == 0.25
    # state-prune threshold: default, override, and 0 (disabled) is valid
    assert EngineConfig().state_prune_threshold == 2000
    assert config.load_engine_config(
        {"engine": {"state_prune_threshold": 50}}
    ).state_prune_threshold == 50
    assert config.load_engine_config(
        {"engine": {"state_prune_threshold": 0}}
    ).state_prune_threshold == 0
    # ask-pair weights: override, including a negative sim weight (explore mode)
    cfg_explore = config.load_engine_config({"engine": {"ask_sim_weight": -0.5}})
    assert cfg_explore.ask_sim_weight == -0.5
    assert cfg_explore.ask_novelty_weight == EngineConfig().ask_novelty_weight
    # ask-policy schedule: default off (flat), opt-in via explore_until_generation
    assert EngineConfig().explore_until_generation == 0
    assert config.load_engine_config(
        {"engine": {"explore_until_generation": 2}}
    ).explore_until_generation == 2
    # variety-erosion sensor knobs: override all three
    cfg_er = config.load_engine_config(
        {"engine": {"erosion_window": 7, "erosion_accel_ratio": 1.0, "erosion_persist": 3}}
    )
    assert (cfg_er.erosion_window, cfg_er.erosion_accel_ratio, cfg_er.erosion_persist) == (7, 1.0, 3)
    # advisory gap probe flag: default off, opt-in boolean
    assert EngineConfig().gap_probe is False
    assert config.load_engine_config({"engine": {"gap_probe": True}}).gap_probe is True


def test_example_domain_engine_overrides():
    path = config.generic_axes_path().parent / "examples" / "research_hypotheses.yaml"
    cfg = config.load_engine_config(path)
    assert cfg.open_niches == 32
    assert cfg.quality_weight == 0.4
    # the axes themselves still parse independently of the engine block
    assert config.load_axes(path).primary_axis.name == "causal_mechanism"


@pytest.mark.parametrize(
    "bad, needle",
    [
        ({"engine": {"open_niches": 0}}, "open_niches"),
        ({"engine": {"quality_weight": 2}}, "quality_weight"),
        ({"engine": {"dedup_tau": 1.5}}, "dedup_tau"),
        ({"engine": {"knn_k": "x"}}, "knn_k"),
        ({"engine": {"under_generation_ratio": 1.5}}, "under_generation_ratio"),
        ({"engine": {"state_prune_threshold": -1}}, "state_prune_threshold"),
        ({"engine": {"ask_sim_weight": 2}}, "ask_sim_weight"),
        ({"engine": {"ask_novelty_weight": -2}}, "ask_novelty_weight"),
        ({"engine": {"explore_until_generation": -1}}, "explore_until_generation"),
        ({"engine": {"erosion_window": 2}}, "erosion_window"),
        ({"engine": {"erosion_accel_ratio": 6}}, "erosion_accel_ratio"),
        ({"engine": {"erosion_persist": 0}}, "erosion_persist"),
        ({"engine": {"gap_probe": "yes"}}, "gap_probe"),
        ({"engine": []}, "engine"),
        # A misspelled knob is rejected, not silently ignored (would run defaults).
        ({"engine": {"qualtiy_weight": 0.9}}, "unknown engine config key"),
        ({"engine": {"open_niches": 8, "totally_bogus": 1}}, "totally_bogus"),
    ],
)
def test_malformed_engine_raises(bad, needle):
    with pytest.raises(config.ConfigError) as exc:
        EngineConfig.from_dict(bad)
    assert needle in str(exc.value)


def test_ingest_respects_open_niches_override(home):
    # An engine override must reach ingest: shrink open_niches + freeze factor so
    # the partition freezes at the overridden threshold with the overridden count.
    axes = {
        "domain": "ov",
        "axes": [{"name": "m", "type": "open", "primary_novelty": True}],
        "engine": {"open_niches": 4, "open_niche_freeze_factor": 2},  # threshold 8
    }
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    cands = [
        {"id": f"c{i}", "text": f"mechanism {w} {i}", "descriptor": {"m": f"mechanism {w} {i}"}}
        for i, w in enumerate(words)
    ]
    pipeline.ingest("ov", cands, axes, seed=0, home=home)
    on = State("ov", home=home).read_open_nicher()
    assert on["frozen"] is True
    assert np.asarray(on["centroids"]).shape[0] == 4  # the overridden open_niches


def test_init_project_persists_engine_config(home):
    axes = {
        "domain": "p",
        "axes": [{"name": "m", "type": "open", "primary_novelty": True}],
        "engine": {"open_niches": 10, "dedup_tau": 0.9},
    }
    pipeline.init_project("p", axes, home=home)
    meta = State("p", home=home).read_meta()
    assert meta["engine"]["open_niches"] == 10
    assert meta["engine"]["dedup_tau"] == 0.9
