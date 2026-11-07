"""A whole network on one laptop: two relays and four researchers.

Runs spec section 5 end to end over real HTTP on localhost, and prints
what happens at each step. One researcher is offline at first, to show
relays holding mail. Uses the heuristic stand-in instead of a real AI,
so the "proposals" are just text similarity.

    python scripts/demo.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rvn import matching  # noqa: E402
from rvn.ai import HeuristicAI  # noqa: E402
from rvn.ledger import entry_hash  # noqa: E402
from rvn.net import Server  # noqa: E402
from rvn.node import Node  # noqa: E402
from rvn.synth import make_orcid  # noqa: E402

import random  # noqa: E402

RNG = random.Random(7)


def say(text=""):
    print(text, flush=True)


def paper(pid, title, date, topics, problem, contribution, useful_for):
    return {
        "paper_id": pid, "title": title, "authors": ["(demo)"], "date": date, "problem": problem,
        "contribution": contribution, "method_summary": contribution, "evidence": "Moderate: demo data.",
        "builds_on": [], "useful_for": useful_for, "limitations": "Not stated by the authors.",
        "topics": topics, "extractor": "hand-written demo / paper-value-extraction v0.1",
    }


def main():
    home = Path(tempfile.mkdtemp(prefix="rvn-demo-"))
    servers = []

    def start(node, relay=False):
        node.relay = relay
        s = Server(node, relay=relay).start()
        node.config["public_url"] = s.url
        servers.append(s)
        return s

    def make(name, topics, past, relays):
        return Node.create(home / name.lower().replace(" ", "-"), name=name, orcid=make_orcid(RNG), affiliation="Demo U",
                           topics=topics, past_work=past, network={"relays": relays}, ai=HeuristicAI(0.3))

    try:
        say("1. Two universities start relays.")
        relays = []
        for uni in ("Relay North", "Relay South"):
            r = make(uni, ["relay", "infrastructure", "network"], [], [])
            relays.append(start(r, relay=True).url)
            say(f"   {uni} at {relays[-1]}")

        say("\n2. Four researchers create nodes. Dan's laptop is closed (offline).")
        alice = make("Alice Chen", ["llm serving", "kv cache", "batching"],
                     [{"text": "Continuous batching of requests for llm serving throughput", "date": "2024"},
                      {"text": "Memory management of the kv cache in llm inference", "date": "2025"}], relays)
        bob = make("Bob Okafor", ["kv cache", "quantization", "llm serving"],
                   [{"text": "Low-bit quantization of weights for llm inference", "date": "2024"}], relays)
        carol = make("Carol Diaz", ["compilers for ml", "autotuning", "tensor programs"],
                     [{"text": "Autotuning tensor programs with learned cost models", "date": "2023"}], relays)
        dan = make("Dan Weiss", ["request scheduling", "batching", "llm serving"],
                   [{"text": "Fair request scheduling for multi-tenant llm serving", "date": "2025"}], relays)
        for n in (alice, bob, carol):
            start(n)
        everyone = (alice, bob, carol, dan)
        for _ in range(2):
            for n in everyone:
                n.tick()
        say(f"   every ledger now holds {alice.ledger.count()} entries (identities + fingerprints)")

        say("\n3. Bob and Dan publish earlier papers.")
        bob.publish_paper(bob.prepare_paper(paper(
            "2503.00001", "Four-bit KV cache for long-context LLM serving", "2025-03",
            ["kv cache", "quantization", "llm serving"],
            "The kv cache limits batch size in llm serving.",
            "We quantize the kv cache to four bits, halving kv cache memory with small accuracy loss in llm serving.",
            ["A researcher working on llm serving could use this to fit larger batches.",
             "A researcher working on kv cache paging could use this to store more blocks."])))
        dan.publish_paper(dan.prepare_paper(paper(
            "2504.00001", "Deadline-aware batching for LLM requests", "2025-04",
            ["request scheduling", "batching", "llm serving"],
            "Batching llm serving requests hurts tail latency.",
            "A scheduler that forms batches by request deadlines in llm serving.",
            ["A researcher working on batching could use this to meet latency targets.",
             "A researcher working on llm serving could use this to set slos."])))
        for n in everyone:
            n.tick()

        say("\n4. Alice publishes a new paper (spec 5, steps 1-4).")
        res = alice.publish_paper(alice.prepare_paper(paper(
            "2601.00001", "Paged KV cache with continuous batching", "2026-01",
            ["llm serving", "kv cache", "batching"],
            "The kv cache wastes memory in llm serving, so batches stay small.",
            "We page the kv cache in fixed blocks and combine it with continuous batching of llm serving requests.",
            ["A researcher working on kv cache quantization could use this to combine paging and compression.",
             "A researcher working on request scheduling could use this to batch more requests."])))
        names = {n.node_id: n.config["name"] for n in everyone}
        say(f"   entry {res['hash'][:16]}; first-level matching picked {len(res['matches'])} people:")
        for nid, score in res["matches"]:
            say(f"     {names.get(nid, nid[:10]):12s} score {score:.2f}  delivered: {res['delivery'][nid]}")
        say("   Carol shares no topic with the paper, so she is not notified.")

        say("\n5. Bob's node runs second-level matching. Bob reads the proposal and confirms it.")
        bob.process_inbox()
        note = [n for n in bob.inbox() if n["paper"]["paper_id"] == "arXiv:2601.00001"][0]
        for p in note["proposals"]:
            say(f"   proposal [{p['id']}]: his {p['from_paper']} {p['relation']} {p['to_paper']}")
            say(f"     \"{p['explanation']}\"")
        prop = note["proposals"][0]
        out = bob.confirm(prop["id"], explanation="Both manage KV cache memory for serving: Alice pages it, "
                                                   "my work compresses it. The two could be combined.")
        say(f"   published connection {out['hash'][:16]}")
        say(f"   Alice's node got a notice: {alice.notices()[-1]['connection']['explanation']}")

        say("\n6. Dan opens his laptop. His node collects mail from the relays.")
        start(dan)
        dan.tick()
        for note in dan.inbox():
            say(f"   inbox: \"{note['paper']['title']}\" from {names[note['sender']]}, "
                f"{len(note['proposals'])} proposal(s)")
            for p in note["proposals"]:
                say(f"     \"{p['explanation']}\"")
                dan.reject(p["id"])
                say("     Dan thinks this one is too thin and rejects it. Nothing is published.")

        say("\n7. Carol disagrees with Bob's connection and files a dispute (it is shown, never resolved).")
        for _ in range(2):
            for n in everyone:
                n.tick()
        conn = [e for e in carol.ledger.entries(kind="connection") if e["author"] == bob.node_id][0]
        carol.dispute(entry_hash(conn)[:12], "Bob's paper predates Alice's; this is overlap at most, not reuse.")

        say("\n8. Everyone syncs. Each node computes reputation from its own copy of the ledger.")
        for _ in range(2):
            for n in everyone:
                n.tick()
        for n in everyone:
            rep = matching.reputation(n.ledger).get(alice.node_id, {"connections": 0, "weighted": 0})
            say(f"   {n.config['name']:12s} sees {n.ledger.count()} entries, Alice's reputation "
                f"{rep['connections']} connections ({rep['weighted']} weighted), "
                f"{len(n.disputes_for(entry_hash(conn)))} dispute(s) on Bob's connection, "
                f"problems: {len(n.ledger.verify_all())}")
        say(f"\nDone. Node folders are in {home} (deleted on exit).")
    finally:
        for s in servers:
            s.stop()
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
