"""Do two AI models extract the same record from the same paper?
(Build plan step 1: twenty papers, two models, measure agreement on
contribution and useful_for.)

Put model A's records in one folder and model B's in another, one JSON file
per paper. Files are paired by paper_id.

Scores, all from 0 to 1:
  contribution   cosine similarity of the two contribution texts
  useful_for     each statement is paired with its closest statement on the
                 other side; the average of those best matches, both ways
  topics         overlap of the two topic sets (Jaccard)
  evidence       1 if both start with the same strength label, else 0

The text similarity uses the network's embedding model. With hash-bow-v1
it mostly measures shared wording, so treat it as a screen: low scores
mean "a person should read both", not "one is wrong".
"""

from __future__ import annotations

import json
from pathlib import Path

from . import records
from .embed import cosine, get_model

FLAG_BELOW = 0.5


def _load(folder: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(folder).glob("*.json")):
        rec = json.loads(f.read_text())
        pid = records.normalize_paper_id(rec.get("paper_id", ""))
        if pid:
            out[pid] = rec
    return out


def _soft_overlap(a: list[str], b: list[str], embed) -> float:
    if not a or not b:
        return 0.0
    va, vb = [embed(x) for x in a], [embed(x) for x in b]
    ab = sum(max(cosine(x, y) for y in vb) for x in va) / len(va)
    ba = sum(max(cosine(y, x) for x in va) for y in vb) / len(vb)
    return (ab + ba) / 2


def _label(evidence: str) -> str:
    head = (evidence or "").split(":", 1)[0].strip().lower()
    return head if head in ("strong", "moderate", "preliminary") else ""


def compare(rec_a: dict, rec_b: dict, model: str = "hash-bow-v1") -> dict:
    embed = get_model(model).embed
    ta = {records.normalize_topic(t) for t in rec_a.get("topics", [])}
    tb = {records.normalize_topic(t) for t in rec_b.get("topics", [])}
    la, lb = _label(rec_a.get("evidence", "")), _label(rec_b.get("evidence", ""))
    return {
        "contribution": round(cosine(embed(rec_a.get("contribution", "")), embed(rec_b.get("contribution", ""))), 3),
        "useful_for": round(_soft_overlap(rec_a.get("useful_for", []), rec_b.get("useful_for", []), embed), 3),
        "topics": round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 0.0,
        "evidence": 1.0 if la and la == lb else 0.0,
    }


def compare_folders(folder_a: Path, folder_b: Path, model: str = "hash-bow-v1") -> dict:
    a, b = _load(folder_a), _load(folder_b)
    shared = sorted(set(a) & set(b))
    rows = [{"paper_id": pid, **compare(a[pid], b[pid], model)} for pid in shared]
    keys = ("contribution", "useful_for", "topics", "evidence")
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0.0 for k in keys}
    flagged = [r["paper_id"] for r in rows if r["contribution"] < FLAG_BELOW or r["useful_for"] < FLAG_BELOW]
    return {"pairs": rows, "mean": mean, "flagged": flagged,
            "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a))}
