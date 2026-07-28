"""Pins and discards share ONE lock, because each write touches both files.

``add_pin`` un-discards and ``add_discard`` un-pins (mutually exclusive, latest action
wins). Pre-fix each guarded only "its own" file and wrote the other unlocked, so two
concurrent ``remember`` calls took DIFFERENT locks and nothing serialized them — an id
could land in both lists, or a pin could be lost.
"""

from __future__ import annotations

import threading

from cambrian_engine.state import State


def test_pin_and_discard_take_the_same_lock(home):
    st = State("p", home=home).ensure()
    held = st.pins_path("d").parent / (st.pins_path("d").name + ".lock")
    held.parent.mkdir(parents=True, exist_ok=True)
    held.mkdir()
    try:
        # With the shared lock held, an add_discard must WAIT rather than sail past on
        # a different (discards) lock. It still completes — the lock is best-effort and
        # proceeds after the timeout — so we only assert it was actually delayed.
        done = threading.Event()
        threading.Thread(
            target=lambda: (st.add_discard("d", "x"), done.set()), daemon=True
        ).start()
        assert not done.wait(0.5), "add_discard did not contend for the pins lock"
    finally:
        held.rmdir()
    done.wait(15)


def test_pin_then_discard_leaves_exactly_one_membership(home):
    st = State("p2", home=home).ensure()
    st.add_pin("d", "a")
    assert st.read_pins("d") == ["a"] and st.read_discards("d") == []
    st.add_discard("d", "a")
    assert st.read_pins("d") == [] and st.read_discards("d") == ["a"]
    st.add_pin("d", "a")
    assert st.read_pins("d") == ["a"] and st.read_discards("d") == []


def test_concurrent_pins_and_discards_stay_mutually_exclusive(home):
    st = State("p3", home=home).ensure()
    ids = [f"c{i}" for i in range(12)]

    def pin_all():
        for cid in ids:
            st.add_pin("d", cid)

    def discard_alternating():
        for cid in ids[::2]:
            st.add_discard("d", cid)

    threads = [threading.Thread(target=pin_all),
               threading.Thread(target=discard_alternating)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    pins, discards = set(st.read_pins("d")), set(st.read_discards("d"))
    assert not (pins & discards), f"id in BOTH lists: {sorted(pins & discards)}"
    # Every id ends up recorded somewhere — neither writer's work is lost wholesale.
    assert pins | discards == set(ids)
