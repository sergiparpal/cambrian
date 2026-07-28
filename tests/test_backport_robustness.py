"""Regression pins for the burgess-backport robustness fixes (Stage 3)."""
from __future__ import annotations

import pytest

from cambrian_engine import diversity, embed, memory, pipeline
from cambrian_engine.config import ConfigError, EngineConfig
from cambrian_engine.state import State


def test_api_provider_fails_at_selection_with_a_clear_message():
    embed.reset_cache()
    with pytest.raises(ConfigError, match="no wired backend"):
        embed.get_embedder("api")
    embed.reset_cache()


def test_unknown_provider_lists_the_real_providers():
    embed.reset_cache()
    with pytest.raises(ValueError, match="static\\|hash\\|local"):
        embed.get_embedder("nope")
    embed.reset_cache()


def test_dedupe_default_tau_tracks_the_constant():
    import inspect
    sig = inspect.signature(embed.dedupe)
    assert sig.parameters["tau"].default == embed.DEFAULT_DEDUP_TAU


def test_pipeline_constants_are_derived_from_engine_config():
    c = EngineConfig()
    assert (pipeline.KNN_K, pipeline.OPEN_NICHES, pipeline.MAX_DPP_POOL,
            pipeline.NOVELTY_REF_CAP, pipeline.QUALITY_WEIGHT) == (
        c.knn_k, c.open_niches, c.max_dpp_pool, c.novelty_ref_cap, c.quality_weight)


def test_kernel_jitter_has_one_home():
    import inspect
    assert (inspect.signature(diversity.build_kernel).parameters["jitter"].default
            == diversity.KERNEL_JITTER)


def test_gap_module_has_no_private_pairwise_copy():
    from cambrian_engine import gap
    assert not hasattr(gap, "_pairwise_cos_distances")


def test_require_sklearn_raises_an_actionable_config_error(monkeypatch):
    """With sklearn absent the operator gets a remedy, not a ModuleNotFoundError."""
    import builtins

    from cambrian_engine import config as config_mod

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("no sklearn")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ConfigError, match="scikit-learn"):
        config_mod.require_sklearn("hash embedder")


def test_value_wins_ignores_non_comparison_events(home):
    """A schema-drifted record carrying a `winner` key must not train preferred values."""
    st = State("prefguard", home=home).ensure()
    st.write_candidates({"c1": {"id": "c1", "descriptor": {"angle": "wild"}}})
    st.append_comparison("d", {"type": "note", "winner": "c1"})   # NOT a comparison
    out = memory.recall(st, "d")
    assert out["summary"]["n_comparisons"] == 1
    assert not out["summary"].get("preferred_values")


def test_coerce_range_rejects_a_string():
    from cambrian_engine.config import axes_spec_from_dict
    with pytest.raises(ConfigError, match="range"):
        axes_spec_from_dict({
            "domain": "t", "unit_of_generation": "idea",
            "axes": [{"name": "x", "type": "continuous", "range": "01"}],
        })


def test_parents_dedup_preserves_pin_order():
    picked = memory.select_parents(
        elite_ids=[], emb_by_id={}, pins=["p2", "p1", "p2", "p3"], k=5)
    assert picked == ["p2", "p1", "p3"]
