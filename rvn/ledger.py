"""The append-only ledger.

Every entry is one signed record wrapped in a small envelope:

    {
      "kind":      "paper",            # identity | fingerprint | paper | connection | dispute
      "author":    "<node id>",        # who signed it
      "seq":       4,                  # 0, 1, 2, ... per author
      "prev":      "<hash of seq 3>",  # null for seq 0
      "body":      { ...record fields from spec section 4... },
      "signature": "<Ed25519 over everything above>"
    }

The spec's `signature` field lives on the envelope, so it covers the record
and the link to the author's previous entry. Because each entry names the
hash of the one before it, an author cannot quietly rewrite their history.

If two different entries claim the same (author, seq), that is a conflict.
Both are kept and both are shown. Nothing picks a winner.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from . import records
from .crypto import canonical, is_node_id, object_hash, verify


class MissingPrev(Exception):
    """The entry points at a previous entry we have not seen yet."""


def entry_hash(entry: dict) -> str:
    return object_hash(entry)


def check_entry(entry) -> list[str]:
    """Checks that need nothing but the entry itself."""
    if not isinstance(entry, dict):
        return ["entry must be a JSON object"]
    p: list[str] = []
    kind, author, seq, prev = entry.get("kind"), entry.get("author"), entry.get("seq"), entry.get("prev")
    if kind not in records.KINDS:
        p.append(f"kind: unknown {kind!r}")
    if not is_node_id(author):
        p.append("author: must be a node id")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        p.append("seq: must be a whole number, 0 or more")
    elif seq == 0 and prev is not None:
        p.append("prev: must be null for seq 0")
    elif seq > 0 and not (isinstance(prev, str) and len(prev) == 64):
        p.append("prev: must be the hash of the author's previous entry")
    if p:
        return p
    unsigned = {k: v for k, v in entry.items() if k != "signature"}
    if not verify(author, canonical(unsigned), entry.get("signature")):
        return ["signature: does not verify against the author's key"]
    body = entry.get("body")
    p += records.check_body(kind, body)
    if not p and body.get(records.AUTHOR_FIELD[kind]) != author:
        p.append(f"body.{records.AUTHOR_FIELD[kind]} must equal the entry author")
    return p


class Ledger:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS entries (
                hash TEXT PRIMARY KEY,
                author TEXT NOT NULL,
                seq INTEGER NOT NULL,
                prev TEXT,
                kind TEXT NOT NULL,
                entry TEXT NOT NULL,
                received_order INTEGER NOT NULL
            )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS by_author ON entries(author, seq)")
        self._db.execute("CREATE INDEX IF NOT EXISTS by_kind ON entries(kind)")
        self._db.commit()

    # ---- writing ---------------------------------------------------------

    def add(self, entry: dict) -> str:
        """Check and store one entry.

        Returns "added" or "duplicate". Raises MissingPrev if the previous
        entry is unknown, ValueError if the entry is invalid.
        """
        problems = check_entry(entry)
        if problems:
            raise ValueError("; ".join(problems))
        h = entry_hash(entry)
        with self._lock:
            if self._db.execute("SELECT 1 FROM entries WHERE hash=?", (h,)).fetchone():
                return "duplicate"
            if entry["seq"] > 0:
                row = self._db.execute("SELECT author, seq FROM entries WHERE hash=?", (entry["prev"],)).fetchone()
                if row is None:
                    raise MissingPrev(f"{entry['author'][:12]}… seq {entry['seq']} points at an unknown entry")
                if row != (entry["author"], entry["seq"] - 1):
                    raise ValueError("prev: must be the same author's entry with seq one lower")
            order = self._db.execute("SELECT COALESCE(MAX(received_order), 0) + 1 FROM entries").fetchone()[0]
            self._db.execute(
                "INSERT INTO entries VALUES (?,?,?,?,?,?,?)",
                (h, entry["author"], entry["seq"], entry.get("prev"), entry["kind"], json.dumps(entry), order),
            )
            self._db.commit()
            return "added"

    def ingest(self, entries: list[dict]) -> dict:
        """Add many entries from a peer. Sorted first so chains go in order."""
        result = {"added": 0, "duplicate": 0, "missing_prev": 0, "invalid": 0, "errors": []}
        good = [e for e in entries if isinstance(e, dict)]
        result["invalid"] += len(entries) - len(good)
        good.sort(key=lambda e: (str(e.get("author")), e.get("seq") if isinstance(e.get("seq"), int) else -1))
        for e in good:
            try:
                result[self.add(e)] += 1
            except MissingPrev:
                result["missing_prev"] += 1
            except ValueError as err:
                result["invalid"] += 1
                result["errors"].append(str(err))
        return result

    # ---- reading ---------------------------------------------------------

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [json.loads(r[0]) for r in self._db.execute(sql, args).fetchall()]

    def get(self, h: str) -> dict | None:
        rows = self._rows("SELECT entry FROM entries WHERE hash=?", (h,))
        return rows[0] if rows else None

    def find(self, prefix: str) -> list[dict]:
        """Entries whose hash starts with prefix (so people can type 10 characters)."""
        return self._rows("SELECT entry FROM entries WHERE hash LIKE ? ORDER BY hash", (prefix + "%",))

    def entries(self, kind: str | None = None, author: str | None = None) -> list[dict]:
        sql, args = "SELECT entry FROM entries WHERE 1=1", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if author:
            sql += " AND author=?"
            args.append(author)
        return self._rows(sql + " ORDER BY author, seq, hash", args)

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def authors(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT author FROM entries ORDER BY author")]

    def head(self, author: str) -> tuple[int, str] | None:
        """Highest (seq, hash) for an author. On a fork, the lowest hash wins the tie."""
        with self._lock:
            row = self._db.execute(
                "SELECT seq, hash FROM entries WHERE author=? ORDER BY seq DESC, hash LIMIT 1", (author,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def latest(self, kind: str, author: str) -> dict | None:
        rows = self._rows(
            "SELECT entry FROM entries WHERE kind=? AND author=? ORDER BY seq DESC, hash LIMIT 1", (kind, author)
        )
        return rows[0] if rows else None

    def conflicts(self) -> list[dict]:
        """Places where one author signed two different entries with the same seq."""
        with self._lock:
            rows = self._db.execute(
                """SELECT author, seq, GROUP_CONCAT(hash) FROM entries
                   GROUP BY author, seq HAVING COUNT(*) > 1 ORDER BY author, seq"""
            ).fetchall()
        return [{"author": a, "seq": s, "hashes": sorted(h.split(","))} for a, s, h in rows]

    # ---- sync ------------------------------------------------------------

    def summary(self) -> dict:
        """What we have: {author: [highest seq, its hash]}. Sent when pulling."""
        with self._lock:
            rows = self._db.execute(
                """SELECT author, seq, MIN(hash) FROM entries e
                   WHERE seq = (SELECT MAX(seq) FROM entries WHERE author = e.author)
                   GROUP BY author"""
            ).fetchall()
        return {a: [s, h] for a, s, h in rows}

    def entries_since(self, have: dict) -> list[dict]:
        """Answer a pull: every entry newer than what the asker has, per author.

        If the asker's latest entry for an author differs from ours at the
        same seq, or we already know that author has a fork, we send all of
        that author's entries so the asker sees both sides.
        """
        out: list[dict] = []
        forked = {c["author"] for c in self.conflicts()}
        for author in self.authors():
            mine = self.head(author)
            theirs = have.get(author) if isinstance(have, dict) else None
            if author in forked or not (
                isinstance(theirs, list) and len(theirs) == 2 and isinstance(theirs[0], int)
            ):
                out += self.entries(author=author)
                continue
            seq, h = theirs
            if mine[0] < seq:
                continue  # they are ahead of us on this author
            with self._lock:
                same = self._db.execute(
                    "SELECT 1 FROM entries WHERE author=? AND seq=? AND hash=?", (author, seq, h)
                ).fetchone()
            if same:
                out += self._rows(
                    "SELECT entry FROM entries WHERE author=? AND seq>? ORDER BY seq, hash", (author, seq)
                )
            else:
                out += self.entries(author=author)
        return out

    # ---- audit -----------------------------------------------------------

    def verify_all(self) -> list[str]:
        """Re-check every signature and every chain link."""
        problems = []
        known = {}
        for e in self.entries():
            known[entry_hash(e)] = e
        for h, e in known.items():
            for p in check_entry(e):
                problems.append(f"{h[:12]}: {p}")
            if e["seq"] > 0:
                prev = known.get(e["prev"])
                if prev is None or prev["author"] != e["author"] or prev["seq"] != e["seq"] - 1:
                    problems.append(f"{h[:12]}: broken link to previous entry")
        return problems
