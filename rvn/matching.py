"""Interest fingerprints, the topic index, and reputation.

Everything here is computed locally from the ledger. Two nodes with the
same ledger get the same answers, so there is nothing to tamper with.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import records
from .embed import cosine, normalize, weighted_mean
from .ledger import entry_hash

TOP_K = 30
RECENT_YEARS = 2
OLD_WORK_WEIGHT = 0.2      # weight of work older than RECENT_YEARS in recent_vector
IRRELEVANT_PUSH = 0.3      # how hard "irrelevant" marks push recent_vector away
NEW_NODE_PAPER_LIMIT = 3

# Spec section 6: builds_on and inspired count more than cites or overlaps.
RELATION_WEIGHT = {"builds_on": 2, "inspired": 2, "cites": 1, "overlaps": 1, "contradicts": 1}


# ---- fingerprints ----------------------------------------------------------

def build_fingerprint_vectors(
    works: list[tuple[list[float], datetime | None]],
    irrelevant: list[list[float]],
    now: datetime | None = None,
) -> tuple[list[float], list[float]]:
    """long_term_vector: mean of all past work.
    recent_vector: the same, but work older than two years counts at 0.2,
    and papers the person marked irrelevant are pushed away from.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365 * RECENT_YEARS)
    vecs = [v for v, _ in works]
    long_term = normalize(weighted_mean(vecs))
    weights = [1.0 if (d is None or d >= cutoff) else OLD_WORK_WEIGHT for _, d in works]
    recent = weighted_mean(vecs, weights)
    total = sum(weights) or 1.0
    recent = [x / total for x in recent]
    if irrelevant and recent:
        away = weighted_mean(irrelevant)
        n = len(irrelevant)
        recent = [r - IRRELEVANT_PUSH * a / n for r, a in zip(recent, away)]
    return long_term, normalize(recent)


def merge_topics(declared: list[str], paper_topic_lists: list[list[str]], limit: int = 10) -> list[str]:
    """The person's own topics first, then the topics their papers use most."""
    out: list[str] = []
    seen: set[str] = set()

    def add(t: str):
        key = records.normalize_topic(t)
        if key and key not in seen and len(out) < limit:
            seen.add(key)
            out.append(t.strip())

    for t in declared:
        add(t)
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for topics in paper_topic_lists:
        for t in topics:
            k = records.normalize_topic(t)
            counts[k] = counts.get(k, 0) + 1
            labels.setdefault(k, t)
    for k in sorted(counts, key=lambda k: (-counts[k], k)):
        add(labels[k])
    return out


# ---- topic index and first-level matching ---------------------------------

def latest_fingerprints(ledger) -> dict[str, dict]:
    out = {}
    for author in ledger.authors():
        e = ledger.latest("fingerprint", author)
        if e:
            out[author] = e["body"]
    return out


def topic_index(fingerprints: dict[str, dict]) -> dict[str, set[str]]:
    """topic label -> the people interested in it."""
    index: dict[str, set[str]] = {}
    for node_id, fp in fingerprints.items():
        for t in fp.get("topics", []):
            index.setdefault(records.normalize_topic(t), set()).add(node_id)
    return index


def first_level(ledger, paper: dict, exclude: set[str], k: int = TOP_K) -> list[tuple[str, float]]:
    """Spec 5.3: look up the paper's topics, gather people, rank by cosine
    against each person's recent and long-term vectors, keep the top k."""
    fps = latest_fingerprints(ledger)
    index = topic_index(fps)
    candidates: set[str] = set()
    for t in paper.get("topics", []):
        candidates |= index.get(records.normalize_topic(t), set())
    candidates -= exclude
    pv = paper["paper_vector"]
    scored = []
    for node_id in candidates:
        fp = fps[node_id]
        if len(fp["recent_vector"]) != len(pv):
            continue  # made with a different embedding model; not comparable
        score = 0.5 * cosine(pv, fp["recent_vector"]) + 0.5 * cosine(pv, fp["long_term_vector"])
        scored.append((node_id, round(score, 4)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]


# ---- reputation ------------------------------------------------------------

def paper_authors(ledger) -> dict[str, set[str]]:
    """paper_id -> node ids that published a value record for it."""
    out: dict[str, set[str]] = {}
    for e in ledger.entries(kind="paper"):
        pid = records.normalize_paper_id(e["body"]["paper_id"])
        out.setdefault(pid, set()).add(e["author"])
    return out


def counted_connections(ledger) -> list[dict]:
    """Connection records that count (spec 4 and 6):
    - the confirmer published from_paper (they wrote the later work),
    - to_paper is in the ledger (so someone gets the credit),
    - it is not a self-connection (confirmer did not also publish to_paper),
    - each (confirmer, from, to, relation) counts once.
    """
    authors = paper_authors(ledger)
    seen = set()
    out = []
    for e in ledger.entries(kind="connection"):
        b = e["body"]
        frm = records.normalize_paper_id(b["from_paper"])
        to = records.normalize_paper_id(b["to_paper"])
        key = (b["confirmed_by"], frm, to, b["relation"])
        if key in seen:
            continue
        if b["confirmed_by"] not in authors.get(frm, set()):
            continue
        if to not in authors or b["confirmed_by"] in authors[to]:
            continue
        seen.add(key)
        out.append({**b, "from_paper": frm, "to_paper": to, "credited": sorted(authors[to])})
    return out


def reputation(ledger) -> dict[str, dict]:
    """node id -> {"connections": plain count, "weighted": weighted count}."""
    rep: dict[str, dict] = {}
    for c in counted_connections(ledger):
        for node_id in c["credited"]:
            r = rep.setdefault(node_id, {"connections": 0, "weighted": 0})
            r["connections"] += 1
            r["weighted"] += RELATION_WEIGHT[c["relation"]]
    return rep


def over_limit_papers(ledger) -> set[str]:
    """Hashes of paper entries past the new-node limit (spec 6).

    A node with no confirmed inbound connection may publish three papers.
    Later ones stay in the ledger but are not matched or notified until
    the author earns a first connection.
    """
    rep = reputation(ledger)
    out = set()
    by_author: dict[str, list[dict]] = {}
    for e in ledger.entries(kind="paper"):
        by_author.setdefault(e["author"], []).append(e)
    for author, entries in by_author.items():
        if rep.get(author, {}).get("connections", 0) > 0:
            continue
        seen_ids: list[str] = []
        for e in entries:  # already in seq order
            pid = records.normalize_paper_id(e["body"]["paper_id"])
            if pid not in seen_ids:
                seen_ids.append(pid)
            if seen_ids.index(pid) >= NEW_NODE_PAPER_LIMIT:
                out.add(entry_hash(e))
    return out
