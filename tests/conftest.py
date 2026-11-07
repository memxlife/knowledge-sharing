import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rvn.ai import HeuristicAI  # noqa: E402
from rvn.net import Server  # noqa: E402
from rvn.node import Node  # noqa: E402

ORCID = "0000-0002-1825-0097"  # ORCID's own published example


def extracted(paper_id, title, date, topics, text, builds_on=()):
    """A minimal valid extraction-skill output."""
    return {
        "paper_id": paper_id, "title": title, "authors": ["A. Person"], "date": date,
        "problem": f"{text} is hard.", "contribution": f"We make {text} better. It works.",
        "method_summary": f"A method for {text}.", "evidence": "Moderate: one benchmark.",
        "builds_on": list(builds_on),
        "useful_for": [f"A researcher working on {text} could use this to go faster.",
                       f"A researcher working on {topics[0]} could use this to compare."],
        "limitations": "Not stated by the authors.", "topics": topics, "extractor": "test",
    }


@pytest.fixture
def example_record():
    return json.loads((ROOT / "skill" / "examples" / "2309.06180.json").read_text())


class Net:
    """A little network on localhost: relays, online nodes, offline nodes."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.servers = []
        self.relay_urls = []

    def relay(self, name="Relay"):
        n = Node.create(self.tmp / name, name=name, orcid=ORCID, affiliation="Uni",
                        topics=["relay", "infrastructure", "network"], ai=HeuristicAI())
        n.relay = True
        s = Server(n, relay=True).start()
        n.config["public_url"] = s.url
        self.servers.append(s)
        self.relay_urls.append(s.url)
        return n

    def node(self, name, topics, past, online=True, threshold=0.3):
        n = Node.create(self.tmp / name, name=name, orcid=ORCID, affiliation="Uni", topics=topics,
                        past_work=[{"text": t, "date": "2025"} for t in past],
                        network={"relays": self.relay_urls}, ai=HeuristicAI(threshold))
        if online:
            self.serve(n)
        n.announce()
        n.sync_all()  # push our first entries to the relays
        return n

    def serve(self, n):
        s = Server(n).start()
        n.config["public_url"] = s.url
        self.servers.append(s)
        return s

    def settle(self, *nodes, rounds=2):
        """Let everyone find each other and sync, like a few minutes of `rvn serve`."""
        for _ in range(rounds):
            for n in nodes:
                n.tick()

    def close(self):
        for s in self.servers:
            s.stop()


@pytest.fixture
def net(tmp_path):
    n = Net(tmp_path)
    n.relay("relay0")
    yield n
    n.close()
