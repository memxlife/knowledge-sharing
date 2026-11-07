"""Second-level matching: the recipient's own AI looks at an incoming paper.

The node does not care which AI you use. Two backends:

  command    Runs a shell command you choose, sends the prompt on stdin, reads
             JSON from stdout. Works with any CLI: `claude -p`,
             `llm -m gpt-4o`, `ollama run llama3`, and so on.

  heuristic  No AI at all. Proposes an "overlaps" link when two papers'
             vectors are close. For tests and the local demo only.

Direction of a proposal: from_paper is always the recipient's own work (a
paper of theirs, or a draft). to_paper is the incoming paper. The spec only
lets the author of from_paper confirm a connection, so the recipient can
never claim a link on someone else's behalf.
"""

from __future__ import annotations

import json
import shlex
import subprocess

from . import records
from .embed import cosine

FORWARD_RELATIONS = ("cites", "builds_on", "inspired")  # only possible from later work

PROMPT = """You help a researcher spot real connections between a newly published paper and their own work.

NEW PAPER (published by someone else):
{incoming}

THE RESEARCHER'S OWN WORK (each item has an id):
{mine}

Task: list the connections that are clearly real and specific. Most papers have none; an empty list is a good answer.

For each connection give:
- "from_paper": the id of the researcher's item (copy it exactly from the list above)
- "relation": one of
    "cites"        their item cites or should cite the new paper
    "builds_on"    their item uses the new paper's method, data or result
    "inspired"     the new paper gave them the idea for their item
    "overlaps"     both address the same question in a similar way
    "contradicts"  the two reach conflicting findings
  "cites", "builds_on" and "inspired" are only allowed when their item is dated after the new paper, or is a draft.
- "explanation": under 100 words, plain language, naming the specific shared idea, method or result.

Reply with only a JSON array, no other text. Example:
[{{"from_paper": "arXiv:2401.01234", "relation": "overlaps", "explanation": "Both ..."}}]
"""


_SHOWN = ("title", "date", "problem", "contribution", "useful_for", "limitations", "topics")


def build_prompt(incoming: dict, mine: list[dict]) -> str:
    def dump(obj):
        return json.dumps(obj, indent=2, ensure_ascii=False)

    items = []
    for m in mine:
        if m["paper_id"].startswith("draft:"):
            item = {"id": m["paper_id"], "kind": "unpublished draft", "text": m["text"][:3000]}
        else:
            item = {"id": m["paper_id"], **{k: m[k] for k in _SHOWN if k in m}}
        items.append(dump(item))
    new = {"id": incoming.get("paper_id"), **{k: incoming[k] for k in _SHOWN if k in incoming}}
    return PROMPT.format(incoming=dump(new), mine="\n\n".join(items) or "(none)")


def _extract_json(text: str):
    for open_, close in (("[", "]"), ("{", "}")):
        start, end = text.find(open_), text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def clean_proposals(raw, incoming: dict, mine: list[dict]) -> list[dict]:
    """Keep only proposals that follow the rules. Drop the rest quietly."""
    if isinstance(raw, dict):
        raw = raw.get("proposals") or raw.get("connections") or []
    if not isinstance(raw, list):
        return []
    by_id = {m["paper_id"]: m for m in mine}
    new_date = records.parse_date(incoming.get("date"))
    out, seen = [], set()
    for p in raw:
        if not isinstance(p, dict):
            continue
        frm, rel, why = p.get("from_paper"), p.get("relation"), p.get("explanation")
        if frm not in by_id or rel not in records.RELATIONS or not isinstance(why, str) or not why.strip():
            continue
        if rel in FORWARD_RELATIONS and not frm.startswith("draft:"):
            mine_date = records.parse_date(by_id[frm].get("date"))
            if new_date and mine_date and mine_date < new_date:
                continue  # an older paper cannot cite or build on a newer one
        words = why.split()
        if len(words) >= 100:
            why = " ".join(words[:99])
        if (frm, rel) in seen:
            continue
        seen.add((frm, rel))
        out.append({"from_paper": frm, "relation": rel, "explanation": why.strip()})
    return out


class CommandAI:
    name = "command"

    def __init__(self, command: str, timeout: int = 300):
        self.command = command
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        res = subprocess.run(
            shlex.split(self.command), input=prompt, capture_output=True, text=True, timeout=self.timeout
        )
        if res.returncode != 0:
            raise RuntimeError(f"AI command failed ({res.returncode}): {res.stderr.strip()[:500]}")
        return res.stdout

    def propose(self, incoming: dict, mine: list[dict]) -> list[dict]:
        if not mine:
            return []
        reply = self.complete(build_prompt(incoming, mine))
        return clean_proposals(_extract_json(reply), incoming, mine)


class HeuristicAI:
    """Stand-in for tests and the demo. Not a real judge of connections."""

    name = "heuristic"

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def propose(self, incoming: dict, mine: list[dict]) -> list[dict]:
        out = []
        for m in mine:
            sim = cosine(incoming["paper_vector"], m["vector"])
            if sim >= self.threshold:
                shared = sorted(
                    {records.normalize_topic(t) for t in incoming.get("topics", [])}
                    & {records.normalize_topic(t) for t in m.get("topics", [])}
                )
                topic_note = f" Shared topics: {', '.join(shared)}." if shared else ""
                out.append({
                    "from_paper": m["paper_id"],
                    "relation": "overlaps",
                    "explanation": f"Similar text (cosine {sim:.2f}) between this work and "
                                   f"\"{incoming.get('title', '')}\".{topic_note}",
                })
        return clean_proposals(out, incoming, mine)


def make_ai(config: dict):
    backend = (config or {}).get("backend", "heuristic")
    if backend == "command":
        if not config.get("command"):
            raise ValueError("ai.command is empty; set it to a CLI that reads a prompt on stdin")
        return CommandAI(config["command"], int(config.get("timeout", 300)))
    if backend == "heuristic":
        return HeuristicAI(float(config.get("threshold", 0.3)))
    raise ValueError(f"unknown ai backend {backend!r}")
