"""The five record types from spec section 4, and their checks.

Field names are fixed. Unknown fields are ignored: they are allowed in a
record (and are covered by its signature) but nothing reads them.

Every check returns a list of problems. An empty list means the record is fine.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .crypto import is_node_id

RELATIONS = ("cites", "builds_on", "inspired", "overlaps", "contradicts")

# Which body field names the author of each record kind.
AUTHOR_FIELD = {
    "identity": "node_id",
    "fingerprint": "node_id",
    "paper": "author_node_id",
    "connection": "confirmed_by",
    "dispute": "author_node_id",
}
KINDS = tuple(AUTHOR_FIELD)

# Fields of a paper value record that a person reviews and may edit.
PAPER_TEXT_FIELDS = (
    "title", "authors", "date", "problem", "contribution", "method_summary",
    "evidence", "builds_on", "useful_for", "limitations", "topics",
)
# What the extraction skill must produce. The node adds the rest
# (paper_vector, author_node_id, created_at, signature).
EXTRACTED_FIELDS = ("paper_id",) + PAPER_TEXT_FIELDS + ("extractor",)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_date(value) -> datetime | None:
    """Paper dates may be '2024', '2024-03' or '2024-03-15'."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    for fmt, length in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        if len(s) == length:
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return parse_time(s)


def word_count(text: str) -> int:
    return len(text.split())


# ---- paper ids -------------------------------------------------------------

_ARXIV_NEW = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"^([a-z\-]+(\.[A-Za-z]{2})?/\d{7})(v\d+)?$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


def normalize_paper_id(raw: str) -> str | None:
    """Return 'arXiv:2401.01234' or 'doi:10.1145/...', or None if not a paper id.

    Versions are dropped (v1 and v2 are the same paper). DOIs are lowercased
    because they are case-insensitive.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    low = s.lower()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/", "arxiv:"):
        if low.startswith(prefix):
            s = s[len(prefix):]
            m = _ARXIV_NEW.match(s) or _ARXIV_OLD.match(s)
            return f"arXiv:{m.group(1)}" if m else None
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if low.startswith(prefix):
            s = s[len(prefix):]
            break
    if _DOI.match(s):
        return f"doi:{s.lower()}"
    m = _ARXIV_NEW.match(s) or _ARXIV_OLD.match(s)
    return f"arXiv:{m.group(1)}" if m else None


# ---- ORCID -----------------------------------------------------------------

_ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def orcid_ok(orcid) -> bool:
    """Format and checksum check (ISO 7064 11-2). It does not prove ownership."""
    if not isinstance(orcid, str) or not _ORCID.match(orcid):
        return False
    digits = orcid.replace("-", "")
    total = 0
    for ch in digits[:-1]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


# ---- field helpers ---------------------------------------------------------

def _str(body, name, problems, required=True):
    v = body.get(name)
    if v is None:
        if required:
            problems.append(f"{name}: missing")
        return None
    if not isinstance(v, str) or (required and not v.strip()):
        problems.append(f"{name}: must be a non-empty string")
        return None
    return v


def _str_list(body, name, problems, lo=None, hi=None):
    v = body.get(name)
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        problems.append(f"{name}: must be a list of non-empty strings")
        return None
    if lo is not None and len(v) < lo or hi is not None and len(v) > hi:
        problems.append(f"{name}: needs {lo} to {hi} items, has {len(v)}")
    return v


def _vector(body, name, problems):
    v = body.get(name)
    if not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) for x in v):
        problems.append(f"{name}: must be a non-empty list of numbers")


def _time(body, name, problems):
    if parse_time(body.get(name)) is None:
        problems.append(f"{name}: must be an ISO 8601 time")


def _node(body, name, problems):
    if not is_node_id(body.get(name)):
        problems.append(f"{name}: must be a node id (64 hex characters)")


def _max_words(body, name, limit, problems):
    v = body.get(name)
    if isinstance(v, str) and word_count(v) > limit:
        problems.append(f"{name}: {word_count(v)} words, limit is {limit}")


# ---- per kind --------------------------------------------------------------

def check_identity(b: dict) -> list[str]:
    p: list[str] = []
    _node(b, "node_id", p)
    _str(b, "name", p)
    if not orcid_ok(b.get("orcid")):
        p.append("orcid: required, format 0000-0000-0000-0000 with a valid check digit")
    _str(b, "affiliation", p)
    _str(b, "scholar_url", p, required=False)
    _time(b, "created_at", p)
    return p


def check_fingerprint(b: dict) -> list[str]:
    p: list[str] = []
    _node(b, "node_id", p)
    if not isinstance(b.get("version"), int) or b["version"] < 1:
        p.append("version: must be an integer of 1 or more")
    _str_list(b, "topics", p, 3, 10)
    _vector(b, "long_term_vector", p)
    _vector(b, "recent_vector", p)
    _time(b, "updated_at", p)
    return p


def check_extracted(b: dict) -> list[str]:
    """Checks on the extraction skill's output, before the node adds its fields."""
    p: list[str] = []
    if normalize_paper_id(b.get("paper_id")) is None:
        p.append("paper_id: must be an arXiv id (e.g. 2401.01234) or a DOI (10.xxxx/...)")
    _str(b, "title", p)
    _str_list(b, "authors", p, 1, None)
    if parse_date(b.get("date")) is None:
        p.append("date: must look like 2024, 2024-03 or 2024-03-15")
    for name in ("problem", "contribution", "method_summary", "evidence", "limitations"):
        _str(b, name, p)
    _max_words(b, "method_summary", 150, p)
    builds_on = b.get("builds_on")
    if not isinstance(builds_on, list) or not all(isinstance(x, str) and x.strip() for x in builds_on):
        p.append("builds_on: must be a list of strings (it may be empty)")
    _str_list(b, "useful_for", p, 2, 5)
    _str_list(b, "topics", p, 3, 8)
    _str(b, "extractor", p)
    return p


def check_paper(b: dict) -> list[str]:
    p = check_extracted(b)
    _vector(b, "paper_vector", p)
    _node(b, "author_node_id", p)
    _time(b, "created_at", p)
    return p


def check_connection(b: dict) -> list[str]:
    p: list[str] = []
    for name in ("from_paper", "to_paper"):
        if normalize_paper_id(b.get(name)) is None:
            p.append(f"{name}: must be a paper id")
    if b.get("relation") not in RELATIONS:
        p.append(f"relation: must be one of {', '.join(RELATIONS)}")
    _str(b, "explanation", p)
    if isinstance(b.get("explanation"), str) and word_count(b["explanation"]) >= 100:
        p.append("explanation: must be under 100 words")
    _node(b, "confirmed_by", p)
    _time(b, "created_at", p)
    return p


def check_dispute(b: dict) -> list[str]:
    p: list[str] = []
    t = b.get("target_entry_hash")
    if not isinstance(t, str) or not re.fullmatch(r"[0-9a-f]{64}", t):
        p.append("target_entry_hash: must be a 64 character hex hash")
    _str(b, "claim", p)
    if isinstance(b.get("claim"), str) and word_count(b["claim"]) >= 200:
        p.append("claim: must be under 200 words")
    _node(b, "author_node_id", p)
    _time(b, "created_at", p)
    return p


CHECKS = {
    "identity": check_identity,
    "fingerprint": check_fingerprint,
    "paper": check_paper,
    "connection": check_connection,
    "dispute": check_dispute,
}


def check_body(kind: str, body) -> list[str]:
    if kind not in CHECKS:
        return [f"unknown kind {kind!r}"]
    if not isinstance(body, dict):
        return ["body must be a JSON object"]
    return CHECKS[kind](body)


def normalize_topic(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())
