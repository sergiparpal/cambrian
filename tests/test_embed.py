"""Phase 2: embeddings are well-shaped & deterministic; dedup works; the
provider switch loads without import errors."""

from __future__ import annotations

import numpy as np
import pytest

from cambrian_engine import embed
from cambrian_engine.embed import (
    DEFAULT_DEDUP_TAU,
    HashingEmbedder,
    dedupe,
    default_dedup_tau,
    get_embedder,
    l2_normalize,
    reset_cache,
)

TEXTS = [
    "Launch a referral program rewarding loyal customers.",
    "Host an underwater sculpture exhibit at midnight.",
    "Turn the onboarding flow into a choose-your-own-adventure.",
]


def test_shape_and_dtype():
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed(TEXTS)
    assert vecs.shape == (3, 256)
    assert vecs.dtype == np.float32


def test_rows_are_normalized():
    vecs = HashingEmbedder().embed(TEXTS)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic():
    a = HashingEmbedder().embed(TEXTS)
    b = HashingEmbedder().embed(TEXTS)
    assert np.array_equal(a, b)


def test_empty_input():
    vecs = HashingEmbedder(dim=64).embed([])
    assert vecs.shape == (0, 64)


def test_near_duplicate_is_more_similar_than_distinct():
    emb = HashingEmbedder()
    base = "Launch a referral program rewarding loyal customers."
    near = "Launch a referral program rewarding loyal customers!"
    far = "Host an underwater sculpture exhibit at midnight."
    v = emb.embed([base, near, far])
    sim_near = float(np.dot(v[0], v[1]))
    sim_far = float(np.dot(v[0], v[2]))
    assert sim_near > 0.9
    assert sim_near > sim_far + 0.3


def test_dedupe_removes_near_duplicate_keeps_distinct():
    emb = HashingEmbedder()
    base = "Launch a referral program rewarding loyal customers."
    near = "Launch a referral program rewarding loyal customers!"
    far = "Host an underwater sculpture exhibit at midnight."
    v = emb.embed([base, near, far])
    keep, drop = dedupe(v, tau=0.92)
    assert keep == [0, 2]
    assert drop == [1]


def test_dedupe_against_existing():
    emb = HashingEmbedder()
    existing = emb.embed(["Launch a referral program rewarding loyal customers."])
    new = emb.embed(
        [
            "Launch a referral program rewarding loyal customers!",  # dup of existing
            "Host an underwater sculpture exhibit at midnight.",  # distinct
        ]
    )
    keep, drop = dedupe(new, tau=0.92, existing=existing)
    assert keep == [1]
    assert drop == [0]


def test_dedupe_empty():
    assert dedupe(np.zeros((0, 8), dtype=np.float32)) == ([], [])


def test_l2_normalize_handles_zero_row():
    out = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert out.shape == (1, 4)
    assert np.all(np.isfinite(out))


def test_provider_switch_loads_without_import_errors():
    reset_cache()
    # All providers construct cheaply; static/local defer the model download to
    # first use, so constructing them here must not import model2vec/torch.
    assert get_embedder("static").name == "static"
    assert get_embedder("hash").name == "hash"
    assert get_embedder("local").name == "local"
    reset_cache()


def test_default_provider_is_static(monkeypatch):
    # The torch-free multilingual embedder is the default for real runs: with no
    # CAMBRIAN_EMBEDDER set, get_embedder() resolves to 'static'.
    assert embed.DEFAULT_PROVIDER == "static"
    reset_cache()
    monkeypatch.delenv(embed.ENV_VAR, raising=False)
    assert get_embedder().name == "static"
    reset_cache()


def test_unknown_provider_raises():
    reset_cache()
    with pytest.raises(ValueError):
        get_embedder("nope")
    reset_cache()


def test_env_var_selects_provider(monkeypatch):
    reset_cache()
    monkeypatch.setenv(embed.ENV_VAR, "hash")
    assert get_embedder().name == "hash"
    reset_cache()


def test_template_method_normalizes_any_provider():
    # A provider that returns wildly unnormalized rows still comes out unit-norm,
    # because Embedder.embed() centralizes l2_normalize — a new provider can't
    # forget to normalize.
    class RawProvider(embed.Embedder):
        name = "raw"
        dim = 3

        def _embed_raw(self, texts):
            return np.array([[3.0, 4.0, 0.0]] * len(texts), dtype=np.float32)

    out = RawProvider().embed(["a", "b"])
    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # the empty case still short-circuits to (0, dim) without calling _embed_raw
    assert RawProvider().embed([]).shape == (0, 3)


def test_empty_embed_resolves_lazy_dim():
    # An embedder whose dim is resolved lazily (model loaded on first real embed)
    # must still return a correctly-WIDE empty array, so an accidental embed([])
    # before the first real call can't yield a 0-width array that would later trip
    # pipeline._guard_embedding_dim.
    class LazyDim(embed.Embedder):
        name = "lazy"

        def __init__(self):
            self.dim = 0  # unknown until "loaded"

        def _ensure(self):
            self.dim = 7

        def _dim_for_empty(self):
            self._ensure()
            return self.dim

        def _embed_raw(self, texts):
            self._ensure()
            return np.ones((len(texts), self.dim), dtype=np.float32)

    assert LazyDim().embed([]).shape == (0, 7)


def test_dedup_tau_is_per_embedder():
    # Each family gets its own near-duplicate threshold; the scales differ.
    assert default_dedup_tau("hash") == 0.92
    assert default_dedup_tau("static") == 0.93
    assert default_dedup_tau("local") == 0.94
    assert default_dedup_tau("local") != default_dedup_tau("hash")
    assert default_dedup_tau("static") != default_dedup_tau("local")
    # unknown families fall back to the global default
    assert default_dedup_tau("mystery-provider") == DEFAULT_DEDUP_TAU
