"""Candidate ids are the primary key of every store — collisions must fail loudly.

Pre-fix, a duplicate id inside one batch (or an id reused across generations with new
text) silently corrupted the archive: two niches ended up naming the SAME elite id while
only the last record survived in ``candidates.json``, so one niche pointed at another
idea's text/coords/embedding and the slate rendered the identical item twice. Dedup does
not catch it — it compares text, not ids.
"""

from __future__ import annotations

import pytest

from cambrian_engine import config, pipeline
from cambrian_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def _cand(cid: str, text: str, angle: str = "art", mechanism: str = "immersive spectacle"):
    return {
        "id": cid,
        "text": text,
        "descriptor": {"angle": angle, "scope": "local", "form": "event",
                       "boldness": 0.5, "mechanism": mechanism},
    }


def test_duplicate_id_within_a_batch_is_rejected(home):
    pipeline.init_project("dup", _generic(), seed=0, home=home)
    cands = [
        _cand("same", "A festival of floating lanterns on the river at dusk"),
        _cand("same", "A quarterly reverse auction for surplus industrial parts",
              angle="tech", mechanism="reverse auction"),
    ]
    with pytest.raises(config.ConfigError, match="duplicate candidate id"):
        pipeline.ingest("dup", cands, _generic(), seed=0, home=home)


def test_duplicate_id_names_the_offender(home):
    pipeline.init_project("dup2", _generic(), seed=0, home=home)
    cands = [
        _cand("ok-1", "A rooftop observatory that opens only on cloudy nights"),
        _cand("clash", "A lending library for unfinished manuscripts"),
        _cand("clash", "A monthly swap meet for broken electronics", angle="tech"),
    ]
    with pytest.raises(config.ConfigError, match="clash"):
        pipeline.ingest("dup2", cands, _generic(), seed=0, home=home)


def test_id_reuse_across_generations_with_new_text_is_rejected(home):
    pipeline.init_project("reuse", _generic(), seed=0, home=home)
    pipeline.ingest(
        "reuse", [_cand("x1", "A festival of floating lanterns on the river at dusk")],
        _generic(), seed=0, home=home,
    )
    with pytest.raises(config.ConfigError, match="already used"):
        pipeline.ingest(
            "reuse",
            [_cand("x1", "A quarterly reverse auction for surplus industrial parts",
                   angle="tech", mechanism="reverse auction")],
            _generic(), seed=0, home=home,
        )


def test_verbatim_resubmission_is_allowed(home):
    # Re-submitting an UNCHANGED candidate is a harmless no-op (dedup drops it at
    # cosine 1.0), so only a text CHANGE under an existing id is an error.
    pipeline.init_project("same", _generic(), seed=0, home=home)
    c = _cand("y1", "A rooftop observatory that opens only on cloudy nights")
    pipeline.ingest("same", [c], _generic(), seed=0, home=home)
    res = pipeline.ingest("same", [c], _generic(), seed=0, home=home)  # must not raise
    assert res["slate"], "the archived elite should still be selectable"


def test_archive_never_names_one_elite_from_two_niches(home):
    # The invariant the guards protect: an elite id identifies exactly one niche.
    pipeline.init_project("inv", _generic(), seed=0, home=home)
    cands = [
        _cand("a", "A festival of floating lanterns on the river at dusk"),
        _cand("b", "A quarterly reverse auction for surplus industrial parts",
              angle="tech", mechanism="reverse auction"),
        _cand("c", "A lending library for unfinished manuscripts",
              angle="learning", mechanism="peer teaching"),
    ]
    res = pipeline.ingest("inv", cands, _generic(), seed=0, home=home)
    niches = State("inv", home=home).read_archive()["niches"]
    elite_ids = [n["elite_id"] for n in niches.values()]
    assert len(elite_ids) == len(set(elite_ids))
    slate_ids = [s["id"] for s in res["slate"]]
    assert len(slate_ids) == len(set(slate_ids)), "the same idea must not appear twice"
