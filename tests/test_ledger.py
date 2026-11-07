import copy

import pytest

from rvn import records
from rvn.crypto import Identity, sign_object
from rvn.ledger import Ledger, MissingPrev, check_entry, entry_hash


def _identity_body(ident):
    return {"node_id": ident.node_id, "name": "N", "orcid": "0000-0002-1825-0097", "affiliation": "U",
            "created_at": records.now_iso()}


def _dispute_body(ident, text="claim"):
    return {"target_entry_hash": "c" * 64, "claim": text, "author_node_id": ident.node_id,
            "created_at": records.now_iso()}


def _chain(ident, n):
    """identity entry followed by n-1 dispute entries, properly chained."""
    entries = [sign_object(ident, {"kind": "identity", "author": ident.node_id, "seq": 0, "prev": None,
                                   "body": _identity_body(ident)})]
    for i in range(1, n):
        entries.append(sign_object(ident, {"kind": "dispute", "author": ident.node_id, "seq": i,
                                           "prev": entry_hash(entries[-1]), "body": _dispute_body(ident, f"c{i}")}))
    return entries


def test_append_and_verify():
    ident = Identity.generate()
    led = Ledger(":memory:")
    for e in _chain(ident, 4):
        assert led.add(e) == "added"
    assert led.count() == 4
    assert led.verify_all() == []
    assert led.add(_chain(ident, 1)[0]) in ("added", "duplicate")


def test_tampered_body_is_rejected():
    ident = Identity.generate()
    e = _chain(ident, 1)[0]
    bad = copy.deepcopy(e)
    bad["body"]["name"] = "Someone Else"
    assert any("signature" in p for p in check_entry(bad))
    with pytest.raises(ValueError):
        Ledger(":memory:").add(bad)


def test_cannot_sign_for_someone_else():
    alice, mallory = Identity.generate(), Identity.generate()
    body = _identity_body(alice)  # names alice
    e = sign_object(mallory, {"kind": "identity", "author": mallory.node_id, "seq": 0, "prev": None, "body": body})
    assert any("must equal the entry author" in p for p in check_entry(e))


def test_rewriting_history_is_detected():
    ident = Identity.generate()
    chain = _chain(ident, 3)
    led = Ledger(":memory:")
    for e in chain:
        led.add(e)
    # An entry that points at the wrong previous entry (seq 2 linking to seq 0).
    bad = sign_object(ident, {"kind": "dispute", "author": ident.node_id, "seq": 3, "prev": entry_hash(chain[0]),
                              "body": _dispute_body(ident)})
    with pytest.raises(ValueError):
        led.add(bad)
    # An entry that points at something we never saw.
    orphan = sign_object(ident, {"kind": "dispute", "author": ident.node_id, "seq": 5, "prev": "d" * 64,
                                 "body": _dispute_body(ident)})
    with pytest.raises(MissingPrev):
        led.add(orphan)


def test_fork_is_kept_and_visible():
    ident = Identity.generate()
    chain = _chain(ident, 2)
    fork = sign_object(ident, {"kind": "dispute", "author": ident.node_id, "seq": 1, "prev": entry_hash(chain[0]),
                               "body": _dispute_body(ident, "a different history")})
    led = Ledger(":memory:")
    for e in chain + [fork]:
        led.add(e)
    conflicts = led.conflicts()
    assert len(conflicts) == 1 and conflicts[0]["seq"] == 1 and len(conflicts[0]["hashes"]) == 2
    assert led.verify_all() == []  # both sides are valid; nothing picks a winner


def test_sync_sends_only_what_is_missing():
    a, b = Identity.generate(), Identity.generate()
    full = Ledger(":memory:")
    for e in _chain(a, 5) + _chain(b, 2):
        full.add(e)
    partial = Ledger(":memory:")
    for e in full.entries(author=a.node_id)[:3]:
        partial.add(e)
    missing = full.entries_since(partial.summary())
    assert len(missing) == 2 + 2  # a's seq 3-4 and all of b
    res = partial.ingest(missing)
    assert res["added"] == 4 and partial.count() == full.count()
    assert full.entries_since(partial.summary()) == []


def test_sync_spreads_forks():
    ident = Identity.generate()
    chain = _chain(ident, 3)
    fork = sign_object(ident, {"kind": "dispute", "author": ident.node_id, "seq": 1, "prev": entry_hash(chain[0]),
                               "body": _dispute_body(ident, "other")})
    knows_fork = Ledger(":memory:")
    for e in chain + [fork]:
        knows_fork.add(e)
    honest = Ledger(":memory:")
    for e in chain:
        honest.add(e)
    honest.ingest(knows_fork.entries_since(honest.summary()))
    assert len(honest.conflicts()) == 1


def test_ingest_out_of_order():
    ident = Identity.generate()
    chain = _chain(ident, 4)
    led = Ledger(":memory:")
    res = led.ingest(list(reversed(chain)))
    assert res["added"] == 4
