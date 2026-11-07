"""Turning text into vectors.

Vectors from different models cannot be compared, so the whole network must
use one model. Which one is still an open decision (spec section 10). The
network file names it, and every node refuses to compare vectors of the
wrong size.

v0 ships one model, "hash-bow-v1": word and word-pair counts hashed into 512
slots. It is crude, but it needs no download, gives the same answer on every
machine, and is good enough to tell a compiler paper from a privacy paper.
To swap in a real embedding model, add a class with the same two methods and
register it in MODELS.
"""

from __future__ import annotations

import hashlib
import math
import re

_STOP = set(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having here how i if in into is it its itself just more most no nor not of off on once only or
    other our out over own same should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where which while who whom why will
    with would you your paper papers we our show shows shown use used using based new approach method
    methods results result work propose proposed present presents existing""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOP and not t.isdigit()]


def _slot(feature: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(d, "big")
    return n % dim, (1.0 if (n >> 63) & 1 else -1.0)


class HashBow:
    name = "hash-bow-v1"
    dim = 512

    def embed(self, text: str) -> list[float]:
        toks = tokens(text)
        feats = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        counts: dict[str, int] = {}
        for f in feats:
            counts[f] = counts.get(f, 0) + 1
        v = [0.0] * self.dim
        for f, c in counts.items():
            i, sign = _slot(f, self.dim)
            v[i] += sign * (1.0 + math.log(c))
        return normalize(v)


MODELS = {HashBow.name: HashBow}


def get_model(name: str):
    if name not in MODELS:
        raise ValueError(f"unknown embedding model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]()


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return [0.0] * len(v)
    return [round(x / n, 5) for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def weighted_mean(vectors: list[list[float]], weights: list[float] | None = None) -> list[float]:
    if not vectors:
        return []
    weights = weights or [1.0] * len(vectors)
    out = [0.0] * len(vectors[0])
    for v, w in zip(vectors, weights):
        for i, x in enumerate(v):
            out[i] += w * x
    return out


def paper_text(record: dict) -> str:
    """The text of a paper value record that gets embedded."""
    parts = [
        record.get("title", ""),
        record.get("problem", ""),
        record.get("contribution", ""),
        record.get("method_summary", ""),
        " ".join(record.get("useful_for", [])),
        " ".join(record.get("topics", [])),
    ]
    return "\n".join(p for p in parts if p)
