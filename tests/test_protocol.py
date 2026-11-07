"""The protocol from spec section 5, over real HTTP on localhost."""

import pytest

from conftest import extracted
from rvn import matching, records
from rvn.crypto import sign_object
from rvn.ledger import entry_hash
from rvn.net import NetError, call

SERVING = ["llm serving", "kv cache", "batching"]


def _publish(node, pid, date, topics, text, draft=None):
    return node.publish_paper(node.prepare_paper(extracted(pid, f"Paper about {text}", date, topics, text)),
                              draft=draft)


def test_two_nodes_on_one_machine_sync(tmp_path):
    """Build plan step 2: two nodes, no relay, one machine."""
    from conftest import ORCID
    from rvn.ai import HeuristicAI
    from rvn.net import Server
    from rvn.node import Node

    a = Node.create(tmp_path / "a", name="A", orcid=ORCID, affiliation="U", topics=SERVING, ai=HeuristicAI())
    b = Node.create(tmp_path / "b", name="B", orcid=ORCID, affiliation="U", topics=SERVING, ai=HeuristicAI())
    _publish(a, "2601.00001", "2026-01", SERVING, "kv cache paging")
    server = Server(a).start()
    try:
        res = b.sync_from(server.url)  # pull what A has, push what B has
        assert res["added"] == 4 and res["pushed"] == 2
        assert a.ledger.summary() == b.ledger.summary() and a.ledger.count() == 6
        again = b.sync_from(server.url)
        assert again["added"] == 0 and again["pushed"] == 0
    finally:
        server.stop()


def test_full_flow_with_relay_and_offline_node(net):
    net.relay("relay1")
    net.relay("relay2")
    alice = net.node("alice", SERVING + ["speculative decoding"],
                     ["Paged memory for the kv cache in llm serving", "Batching requests for llm serving"])
    bob = net.node("bob", ["kv cache", "quantization", "llm serving"],
                   ["kv cache quantization for llm serving memory"])
    dan = net.node("dan", ["batching", "request scheduling", "llm serving"],
                   ["Request scheduling and batching for llm serving"], online=False)
    carol = net.node("carol", ["compilers for ml", "autotuning", "tensor programs"],
                     ["Autotuning tensor programs with learned cost models"])
    for n in (alice, bob, dan, carol):
        n.tick()

    _publish(bob, "2501.00001", "2025-01", ["kv cache", "quantization", "llm serving"],
             "kv cache quantization llm serving memory")
    _publish(dan, "2502.00001", "2025-02", ["batching", "request scheduling", "llm serving"],
             "request scheduling batching llm serving")
    for n in (alice, bob, dan, carol):
        n.tick()

    res = _publish(alice, "2601.00001", "2026-01", SERVING, "paged kv cache llm serving batching")
    matched = dict(res["matches"])
    assert bob.node_id in matched and dan.node_id in matched
    assert carol.node_id not in matched  # no shared topic
    assert res["delivery"][bob.node_id] == "direct"
    assert res["delivery"][dan.node_id] == "relay"  # dan is offline

    # Bob is online: his AI proposes, he confirms.
    bob.process_inbox()
    note = [n for n in bob.inbox() if n["paper"]["paper_id"] == "arXiv:2601.00001"][0]
    assert note["proposals"], "heuristic AI should see the overlap"
    p = note["proposals"][0]
    assert p["from_paper"] == "arXiv:2501.00001" and p["to_paper"] == "arXiv:2601.00001"
    out = bob.confirm(p["id"], explanation="Both manage KV cache memory; mine compresses it, hers pages it.")
    assert not out["held"]

    # Alice is told (spec 5.7).
    notices = alice.notices()
    assert notices and notices[-1]["connection"]["confirmed_by"] == bob.node_id

    # Dan comes back and picks up his mail from the relay.
    net.serve(dan)
    dan.tick()
    assert any(n["paper"]["paper_id"] == "arXiv:2601.00001" for n in dan.inbox())

    # Everyone converges, and every node computes the same reputation on its own.
    for _ in range(2):
        for n in (alice, bob, dan, carol):
            n.tick()
    counts = {n.ledger.count() for n in (alice, bob, dan, carol)}
    assert len(counts) == 1
    for n in (alice, bob, dan, carol):
        assert matching.reputation(n.ledger)[alice.node_id]["connections"] == 1
        assert n.ledger.verify_all() == [] and n.ledger.conflicts() == []


def test_reject_publishes_nothing_and_irrelevant_feeds_fingerprint(net):
    alice = net.node("alice", SERVING, ["kv cache paging"])
    bob = net.node("bob", SERVING, ["kv cache compression"])
    _publish(bob, "2501.00001", "2025-01", SERVING, "kv cache compression")
    net.settle(alice, bob)
    _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache paging")
    _publish(alice, "2601.00002", "2026-01", SERVING, "kv cache eviction")
    bob.process_inbox()
    inbox = bob.inbox()
    assert len(inbox) == 2
    before = bob.ledger.count()
    bob.reject(inbox[0]["proposals"][0]["id"])
    assert bob.ledger.count() == before
    with pytest.raises(ValueError):
        bob.reject(inbox[0]["proposals"][0]["id"])

    old = bob.ledger.latest("fingerprint", bob.node_id)["body"]["recent_vector"]
    bob.mark_irrelevant(inbox[1]["id"])
    new = bob.refresh_fingerprint()["body"]["recent_vector"]
    from rvn.embed import cosine
    v = inbox[1]["paper"]["paper_vector"]
    assert cosine(new, v) < cosine(old, v)
    assert [n["id"] for n in bob.inbox()] == [inbox[0]["id"]]


def test_threshold_ignores_low_reputation_senders(net):
    alice = net.node("alice", SERVING, ["kv cache paging"])
    bob = net.node("bob", SERVING, ["kv cache compression"])
    bob.config["threshold"] = 1
    _publish(bob, "2501.00001", "2025-01", SERVING, "kv cache compression")
    net.settle(alice, bob)
    _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache paging")
    assert bob.process_inbox()["ignored"] == 1


def test_draft_connection_is_held_until_the_draft_is_published(net):
    alice = net.node("alice", SERVING, ["kv cache paging"])
    bob = net.node("bob", SERVING, ["kv cache compression"])
    bob.config["allow_drafts"] = True
    (bob.home / "drafts" / "next-paper.md").write_text("kv cache paging llm serving batching extension draft")
    net.settle(alice, bob)
    _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache paging llm serving batching")
    bob.tick()
    props = bob.inbox()[0]["proposals"]
    draft_prop = [p for p in props if p["from_paper"] == "draft:next-paper"][0]
    before = bob.ledger.count()
    assert bob.confirm(draft_prop["id"], relation="inspired")["held"]
    assert bob.ledger.count() == before  # nothing public about the draft yet

    res = _publish(bob, "2607.00001", "2026-07", SERVING, "kv cache paging extension", draft="next-paper")
    assert len(res["released"]) == 1
    conn = bob.ledger.get(res["released"][0])["body"]
    assert conn["from_paper"] == "arXiv:2607.00001" and conn["to_paper"] == "arXiv:2601.00001"
    assert conn["relation"] == "inspired"
    alice.sync_all()
    assert matching.reputation(alice.ledger)[alice.node_id]["weighted"] == 2


def test_dispute_is_attached_and_spreads(net):
    alice = net.node("alice", SERVING, ["kv cache paging"])
    carol = net.node("carol", SERVING, ["kv cache"])
    _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache paging")
    net.settle(alice, carol)
    target = carol.ledger.entries(kind="paper")[0]
    d = carol.dispute(entry_hash(target)[:12], "The evidence field overstates the results.")
    alice.sync_all()
    found = alice.disputes_for(entry_hash(target))
    assert [entry_hash(x) for x in found] == [entry_hash(d)]


def test_messages_must_be_signed_by_the_sender(net):
    alice = net.node("alice", SERVING, ["kv cache"])
    bob = net.node("bob", SERVING, ["kv cache"])
    paper = _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache")["entry"]
    forged = sign_object(bob.identity, {"type": "notification", "from": alice.node_id, "to": bob.node_id,
                                        "sent_at": records.now_iso(), "payload": {"paper_entry": paper}})
    with pytest.raises(NetError, match="bad message signature"):
        call(bob.config["public_url"], "/inbox", {"message": forged})


def test_undeliverable_messages_wait_in_the_outbox(net):
    alice = net.node("alice", SERVING, ["kv cache paging"])
    bob = net.node("bob", SERVING, ["kv cache compression"], online=False)
    net.settle(alice, bob)
    alice.config["network"]["relays"] = ["http://127.0.0.1:9"]  # the relay is gone; nothing listens there
    res = _publish(alice, "2601.00001", "2026-01", SERVING, "kv cache paging")
    assert res["delivery"][bob.node_id] == "queued"
    net.serve(bob)
    alice.add_peer(bob.announcement())
    assert alice.retry_outbox() == 1
    assert len(bob.inbox()) == 1
