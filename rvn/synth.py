"""A synthetic network of fake profiles, for testing first-level matching
(build plan step 3). Nothing here touches the real network.

Each fake researcher belongs to one research area. Their topics and past
work are drawn mostly from that area, with some noise from others. Then we
publish one probe paper per area and check that matching finds that area's
people.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from . import matching, records
from .crypto import Identity, sign_object
from .embed import get_model, paper_text
from .ledger import Ledger, entry_hash

AREAS = {
    "serving": (
        ["llm serving", "kv cache", "batching", "request scheduling", "speculative decoding"],
        "serving latency throughput requests batch decode prefill tokens cache memory paged preemption "
        "queue slo goodput draft verify speculative streaming endpoint replicas autoscaling",
    ),
    "compilers": (
        ["compilers for ml", "tensor programs", "autotuning", "kernel fusion", "graph optimization"],
        "compiler tensor program loop tiling schedule autotuning search cost model fusion operator lowering "
        "codegen ir polyhedral layout vectorize template tvm triton mlir",
    ),
    "distributed": (
        ["distributed training", "pipeline parallelism", "tensor parallelism", "collective communication",
         "fully sharded data parallel"],
        "parallelism pipeline stage microbatch bubble allreduce collective sharding optimizer states gradient "
        "synchronization interconnect nvlink bandwidth partition placement zero megatron",
    ),
    "quantization": (
        ["quantization", "low-precision training", "sparsity", "pruning", "knowledge distillation"],
        "quantization bits int8 int4 fp8 outliers calibration rounding weights activations pruning sparse "
        "structured magnitude distillation student teacher accuracy compression",
    ),
    "cluster": (
        ["cluster scheduling", "gpu sharing", "resource allocation", "fault tolerance", "elastic training"],
        "cluster scheduler jobs gpus allocation fairness utilization preemption elastic failures checkpoint "
        "recovery multitenant sharing placement queueing makespan heterogeneity",
    ),
    "federated": (
        ["federated learning", "privacy-preserving ml", "on-device learning", "edge inference",
         "energy efficiency"],
        "federated clients devices privacy differential secure aggregation heterogeneity mobile edge battery "
        "energy communication rounds personalization stragglers noniid",
    ),
    "data": (
        ["data pipelines", "data loading", "dataset curation", "retrieval-augmented generation",
         "vector databases"],
        "data loading pipeline preprocessing shuffle storage io dataset curation deduplication filtering "
        "retrieval index vector embedding search nearest neighbor documents",
    ),
    "hardware": (
        ["hardware accelerators", "fpga", "in-memory computing", "performance modeling", "tpu"],
        "accelerator hardware systolic array dataflow fpga asic memory bandwidth sram dram roofline "
        "performance model simulation chip design energy",
    ),
}
GENERIC = "system evaluation design implementation scale efficient large model training inference workload".split()


def make_orcid(rng: random.Random) -> str:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(15))
    total = 0
    for ch in digits:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    d = digits + ("X" if check == 10 else str(check))
    return f"{d[0:4]}-{d[4:8]}-{d[8:12]}-{d[12:16]}"


def _text(rng: random.Random, area: str, n: int = 18) -> str:
    own = AREAS[area][1].split()
    other = AREAS[rng.choice([a for a in AREAS if a != area])][1].split()
    words = rng.sample(own, min(n, len(own))) + rng.sample(other, 3) + rng.sample(GENERIC, 4)
    rng.shuffle(words)
    return " ".join(words)


def build(n_profiles: int = 50, seed: int = 0, noise: float = 0.5, ledger: Ledger | None = None):
    """Return (ledger, profiles). profiles: list of {node_id, area, identity}."""
    rng = random.Random(seed)
    ledger = ledger or Ledger(":memory:")
    emb = get_model("hash-bow-v1")
    areas = list(AREAS)
    profiles = []
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    for i in range(n_profiles):
        area = areas[i % len(areas)]
        ident = Identity.generate()
        topics = rng.sample(AREAS[area][0], rng.randint(3, 5))
        for _ in range(2):  # side interests: these pull people into other areas' candidate lists
            if rng.random() < noise:
                topics.append(rng.choice(AREAS[rng.choice([a for a in areas if a != area])][0]))
        topics = list(dict.fromkeys(topics))
        works = []
        for _ in range(rng.randint(3, 5)):
            year = rng.randint(2016, 2026)
            works.append((emb.embed(_text(rng, area)), datetime(year, 6, 1, tzinfo=timezone.utc)))
        long_term, recent = matching.build_fingerprint_vectors(works, [], now=now)
        e0 = sign_object(ident, {"kind": "identity", "author": ident.node_id, "seq": 0, "prev": None, "body": {
            "node_id": ident.node_id, "name": f"Fake Researcher {i:02d} ({area})", "orcid": make_orcid(rng),
            "affiliation": "Synthetic University", "created_at": records.now_iso()}})
        ledger.add(e0)
        e1 = sign_object(ident, {"kind": "fingerprint", "author": ident.node_id, "seq": 1, "prev": entry_hash(e0),
                                 "body": {"node_id": ident.node_id, "version": 1, "topics": topics,
                                          "long_term_vector": long_term, "recent_vector": recent,
                                          "updated_at": records.now_iso()}})
        ledger.add(e1)
        profiles.append({"node_id": ident.node_id, "area": area, "identity": ident})
    return ledger, profiles


def probe_paper(area: str, seed: int = 1) -> dict:
    rng = random.Random(seed)
    emb = get_model("hash-bow-v1")
    topics = rng.sample(AREAS[area][0], 3)
    body = {
        "paper_id": f"2601.{10000 + list(AREAS).index(area)}",
        "title": f"Probe paper on {area}",
        "problem": _text(rng, area, 10),
        "contribution": _text(rng, area, 14),
        "method_summary": _text(rng, area, 16),
        "useful_for": [_text(rng, area, 8), _text(rng, area, 8)],
        "topics": topics,
    }
    body["paper_vector"] = emb.embed(paper_text(body))
    return body


def evaluate(n_profiles: int = 50, seed: int = 0, noise: float = 0.5, k: int = matching.TOP_K) -> list[dict]:
    """For each area, publish a probe and measure who first-level matching picks."""
    ledger, profiles = build(n_profiles, seed, noise)
    by_area: dict[str, set[str]] = {}
    for p in profiles:
        by_area.setdefault(p["area"], set()).add(p["node_id"])
    rows = []
    for i, area in enumerate(AREAS):
        paper = probe_paper(area, seed=seed * 100 + i)
        ranked = matching.first_level(ledger, paper, exclude=set(), k=k)
        ids = [nid for nid, _ in ranked]
        members = by_area[area]
        top_n = ids[: len(members)]
        rows.append({
            "area": area,
            "members": len(members),
            "candidates": len(ids),
            "precision_at_members": sum(1 for x in top_n if x in members) / len(members),
            "recall_at_k": sum(1 for x in ids if x in members) / len(members),
        })
    return rows
