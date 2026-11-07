import pytest

from conftest import ORCID, extracted
from rvn import matching, synth
from rvn.ai import HeuristicAI
from rvn.embed import cosine, get_model
from rvn.node import LimitError, Node


def _node(tmp_path, name, topics=("llm serving", "kv cache", "batching")):
    return Node.create(tmp_path / name, name=name, orcid=ORCID, affiliation="U", topics=list(topics),
                       ai=HeuristicAI())


def _share(*nodes):
    """Give every node every entry, as if they had synced."""
    for _ in range(2):
        for a in nodes:
            for b in nodes:
                if a is not b:
                    b.ledger.ingest(a.ledger.entries())


def _paper(node, pid, date="2025-01", topics=("llm serving", "kv cache", "batching"), text="kv cache paging"):
    return node.publish_paper(node.prepare_paper(extracted(pid, f"Paper {pid}", date, list(topics), text)))


# ---- first-level matching on the fake network (build plan step 3) ----------

def test_synthetic_network_matching():
    rows = synth.evaluate(n_profiles=50, seed=0, noise=1.0)
    assert len(rows) == len(synth.AREAS)
    for r in rows:
        assert r["candidates"] <= matching.TOP_K
        assert r["recall_at_k"] == 1.0, r
        assert r["precision_at_members"] >= 0.8, r


def test_first_level_skips_people_without_shared_topics():
    ledger, profiles = synth.build(n_profiles=16, seed=3, noise=0.0)
    paper = synth.probe_paper("compilers")
    ranked = matching.first_level(ledger, paper, exclude=set())
    areas = {p["node_id"]: p["area"] for p in profiles}
    assert ranked and all(areas[nid] == "compilers" for nid, _ in ranked)


def test_first_level_keeps_top_k_and_excludes_author():
    ledger, profiles = synth.build(n_profiles=50, seed=1, noise=1.0)
    paper = synth.probe_paper("serving")
    paper["topics"] = [t for a in synth.AREAS.values() for t in a[0]]  # touch every area
    me = profiles[0]["node_id"]
    ranked = matching.first_level(ledger, paper, exclude={me}, k=30)
    assert len(ranked) == 30 and me not in [n for n, _ in ranked]
    assert [s for _, s in ranked] == sorted([s for _, s in ranked], reverse=True)


# ---- fingerprints -------------------------------------------------------------

def test_irrelevant_marks_push_recent_vector_away():
    emb = get_model("hash-bow-v1")
    works = [(emb.embed("kv cache paging for llm serving"), None), (emb.embed("gpu kernels for attention"), None)]
    bad = emb.embed("federated learning privacy on phones")
    _, before = matching.build_fingerprint_vectors(works, [])
    _, after = matching.build_fingerprint_vectors(works, [bad])
    assert cosine(after, bad) < cosine(before, bad)


def test_old_work_counts_less_in_recent_vector():
    from datetime import datetime, timezone
    emb = get_model("hash-bow-v1")
    old = emb.embed("compiler loop tiling autotuning")
    new = emb.embed("kv cache llm serving batching")
    long_term, recent = matching.build_fingerprint_vectors(
        [(old, datetime(2015, 1, 1, tzinfo=timezone.utc)), (new, datetime(2026, 1, 1, tzinfo=timezone.utc))], [])
    assert cosine(recent, new) > cosine(long_term, new)


def test_merge_topics():
    got = matching.merge_topics(["A", "b", "c"], [["B", "d"], ["d", "e"], ["d"]])
    assert got == ["A", "b", "c", "d", "e"]
    assert len(matching.merge_topics([str(i) for i in range(10)], [["x", "y"]])) == 10


# ---- reputation (spec 4 and 6) ---------------------------------------------------

def test_reputation_rules(tmp_path):
    alice, bob, mallory = _node(tmp_path, "alice"), _node(tmp_path, "bob"), _node(tmp_path, "mallory")
    _paper(alice, "2601.00001", "2026-01")
    _paper(alice, "2601.00002", "2026-01")
    _paper(bob, "2602.00001", "2026-02")
    _share(alice, bob, mallory)

    bob._publish_connection("arXiv:2602.00001", "arXiv:2601.00001", "builds_on", "Uses Alice's method.")
    bob._publish_connection("arXiv:2602.00001", "arXiv:2601.00001", "builds_on", "Same again.")  # counts once
    bob._publish_connection("arXiv:2602.00001", "arXiv:2601.00002", "overlaps", "Similar question.")
    # Mallory did not write 2602.00001, so her claim does not count.
    mallory._publish_connection("arXiv:2602.00001", "arXiv:2601.00001", "inspired", "Fake.")
    # Alice citing herself does not count.
    alice._publish_connection("arXiv:2601.00002", "arXiv:2601.00001", "builds_on", "My own earlier paper.")
    # to_paper is not in the ledger: nobody to credit.
    bob._publish_connection("arXiv:2602.00001", "arXiv:2501.99999", "cites", "Not in the ledger.")
    _share(alice, bob, mallory)

    for n in (alice, bob, mallory):  # every node computes the same thing on its own
        rep = matching.reputation(n.ledger)
        assert rep[alice.node_id] == {"connections": 2, "weighted": 3}
        assert bob.node_id not in rep and mallory.node_id not in rep


def test_new_node_paper_limit(tmp_path):
    alice, bob = _node(tmp_path, "alice"), _node(tmp_path, "bob")
    for i in range(3):
        _paper(alice, f"2601.0000{i}", "2026-01")
    with pytest.raises(LimitError):
        alice.check_paper_limit("arXiv:2601.00009")
    alice.check_paper_limit("arXiv:2601.00001")  # updating an existing paper is fine
    _paper(bob, "2602.00001", "2026-02")
    _share(alice, bob)
    bob._publish_connection("arXiv:2602.00001", "arXiv:2601.00001", "cites", "Cites it.")
    _share(alice, bob)
    alice.check_paper_limit("arXiv:2601.00009")  # earned a connection: limit lifted


def test_over_limit_papers_are_flagged_for_others(tmp_path):
    alice = _node(tmp_path, "alice")
    hashes = []
    for i in range(4):
        body = alice.prepare_paper(extracted(f"2601.0000{i}", "T", "2026-01", ["a", "b", "c"], "x"))
        from rvn.ledger import entry_hash
        hashes.append(entry_hash(alice._append("paper", body)))  # bypass the local check
    assert matching.over_limit_papers(alice.ledger) == {hashes[3]}
