"""A pin can outlive the idea it names; it must not become a contentless parent.

``init-project`` resets the geometry (archive / candidates / embeddings) when the axes
change but deliberately PRESERVES preference memory. Pre-fix, ``parents`` then emitted the
surviving pin as a parent with ``text: ""`` and no coords — and ``loop.md`` §6 tells the
agent to breed from whatever ``parents`` returns.
"""

from __future__ import annotations

import json

from cambrian_engine import config, pipeline, selftest
from cambrian_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def _pin_first_slate_item(project, home) -> str:
    res = pipeline.ingest(
        project, selftest.diverse_candidates(8), _generic(), seed=0, home=home
    )
    cid = res["slate"][0]["id"]
    pipeline.remember(project, {"type": "pin", "id": cid}, home=home)
    return cid


def test_parents_omit_pins_whose_record_was_reset(home):
    pipeline.init_project("g", _generic(), seed=0, home=home)
    pinned = _pin_first_slate_item("g", home)
    assert any(p["id"] == pinned and p["text"] for p in
               pipeline.parents("g", home=home)["parents"])

    axes2 = json.loads(json.dumps(_generic()))
    axes2["axes"].append({"name": "tone", "type": "categorical"})
    assert pipeline.init_project("g", axes2, seed=0, home=home)["reset"] is True

    out = pipeline.parents("g", home=home)
    assert all(p["text"] for p in out["parents"]), "no contentless parent may be returned"
    assert pinned not in [p["id"] for p in out["parents"]]
    assert out["stale_pins"] == [pinned]
    assert "stale_pins_note" in out


def test_pin_is_preserved_in_memory_after_a_reset(home):
    # The pin itself is NOT dropped — preference memory outliving the geometry is the
    # documented behavior; only its use as a breeding parent is suppressed.
    pipeline.init_project("g2", _generic(), seed=0, home=home)
    pinned = _pin_first_slate_item("g2", home)
    axes2 = json.loads(json.dumps(_generic()))
    axes2["axes"].append({"name": "tone", "type": "categorical"})
    pipeline.init_project("g2", axes2, seed=0, home=home)

    st = State("g2", home=home)
    assert pinned in st.read_pins("generic")
    assert pinned in pipeline.recall("g2", home=home)["pins"]


def test_no_stale_key_when_every_pin_still_resolves(home):
    # The off-path output is unchanged: no stale_pins key in the ordinary case.
    pipeline.init_project("g3", _generic(), seed=0, home=home)
    _pin_first_slate_item("g3", home)
    out = pipeline.parents("g3", home=home)
    assert "stale_pins" not in out
    assert "stale_pins_note" not in out
