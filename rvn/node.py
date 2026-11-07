"""The node: one per researcher, on their own machine.

It holds the identity key, the local ledger, private state (mailbox,
proposals, peers), and runs the protocol from spec section 5.

Files in the node's home folder:
    key.pem       private signing key (keep it secret; mode 600)
    config.json   name, relays, embedding model, AI command, threshold
    ledger.db     the shared ledger (public)
    private.db    inbox, proposals, peers, outbox; never leaves this machine
    drafts/       optional: drafts the AI may compare against (if allowed)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import matching, records
from .ai import make_ai
from .crypto import Identity, is_node_id, object_hash, sign_object, verify_object
from .embed import cosine, get_model, paper_text
from .ledger import Ledger, MissingPrev, check_entry, entry_hash
from .net import NetError, call

MAILBOX_DAYS = 30
SECOND_LEVEL_CANDIDATES = 8  # own papers shown to the AI per notification


class LimitError(Exception):
    pass


DEFAULT_NETWORK = {"relays": [], "embedding_model": "hash-bow-v1"}


class Node:
    def __init__(self, home: Path | str, ai=None):
        self.home = Path(home)
        self.config = json.loads((self.home / "config.json").read_text())
        self.identity = Identity.load(self.home / "key.pem")
        self.node_id = self.identity.node_id
        self.ledger = Ledger(self.home / "ledger.db")
        self.embedder = get_model(self.config["network"]["embedding_model"])
        self.ai = ai if ai is not None else make_ai(self.config.get("ai"))
        self.relay = False  # set True when serving as a relay
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.home / "private.db", check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS past_work (id INTEGER PRIMARY KEY, text TEXT, date TEXT);
            CREATE TABLE IF NOT EXISTS peers (node_id TEXT PRIMARY KEY, url TEXT, relay INTEGER,
                announced_at TEXT, announcement TEXT);
            CREATE TABLE IF NOT EXISTS seen_messages (hash TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY, sender TEXT, paper_hash TEXT,
                paper TEXT, score REAL, received_at TEXT, status TEXT, note TEXT);
            CREATE TABLE IF NOT EXISTS proposals (id INTEGER PRIMARY KEY, notification_id INTEGER,
                from_paper TEXT, to_paper TEXT, relation TEXT, explanation TEXT, status TEXT, entry_hash TEXT);
            CREATE TABLE IF NOT EXISTS irrelevant (paper_hash TEXT PRIMARY KEY, vector TEXT);
            CREATE TABLE IF NOT EXISTS notices (id INTEGER PRIMARY KEY, sender TEXT, connection TEXT,
                received_at TEXT);
            CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY, recipient TEXT, message TEXT);
            CREATE TABLE IF NOT EXISTS mailbox_cursor (relay TEXT PRIMARY KEY, last INTEGER);
            CREATE TABLE IF NOT EXISTS relay_mailbox (id INTEGER PRIMARY KEY, recipient TEXT, hash TEXT UNIQUE,
                message TEXT, stored_at TEXT);
            """
        )
        self._db.commit()

    # ---- setup -----------------------------------------------------------

    @classmethod
    def create(cls, home: Path | str, *, name: str, orcid: str, affiliation: str, topics: list[str],
               scholar_url: str | None = None, past_work: list[dict] | None = None,
               network: dict | None = None, ai_config: dict | None = None, public_url: str | None = None,
               ai=None) -> "Node":
        home = Path(home)
        if (home / "key.pem").exists():
            raise FileExistsError(f"{home} already has a node")
        if not records.orcid_ok(orcid):
            raise ValueError("ORCID is required and must look like 0000-0002-1825-0097 (valid check digit)")
        if not 3 <= len(topics) <= 10:
            raise ValueError("pick 3 to 10 topics")
        home.mkdir(parents=True, exist_ok=True)
        (home / "drafts").mkdir(exist_ok=True)
        Identity.generate().save(home / "key.pem")
        config = {
            "name": name,
            "public_url": public_url,
            "network": {**DEFAULT_NETWORK, **(network or {})},
            "declared_topics": topics,
            "threshold": 0,
            "allow_drafts": False,
            "ai": ai_config or {"backend": "heuristic"},
        }
        (home / "config.json").write_text(json.dumps(config, indent=2))
        node = cls(home, ai=ai)
        for w in past_work or []:
            node._db.execute("INSERT INTO past_work (text, date) VALUES (?, ?)", (w["text"], w.get("date")))
        node._db.commit()
        identity = {"node_id": node.node_id, "name": name, "orcid": orcid, "affiliation": affiliation,
                    "created_at": records.now_iso()}
        if scholar_url:
            identity["scholar_url"] = scholar_url
        node._append("identity", identity)
        node.refresh_fingerprint()
        return node

    def save_config(self) -> None:
        (self.home / "config.json").write_text(json.dumps(self.config, indent=2))

    @property
    def relays(self) -> list[str]:
        return list(self.config["network"].get("relays", []))

    # ---- own entries -----------------------------------------------------

    def _append(self, kind: str, body: dict) -> dict:
        with self._lock:
            head = self.ledger.head(self.node_id)
            entry = {
                "kind": kind,
                "author": self.node_id,
                "seq": 0 if head is None else head[0] + 1,
                "prev": None if head is None else head[1],
                "body": body,
            }
            entry = sign_object(self.identity, entry)
            self.ledger.add(entry)
            return entry

    def my_papers(self) -> dict[str, dict]:
        """paper_id -> my latest paper entry for it."""
        out = {}
        for e in self.ledger.entries(kind="paper", author=self.node_id):
            out[records.normalize_paper_id(e["body"]["paper_id"])] = e
        return out

    def refresh_fingerprint(self) -> dict:
        """Recompute and publish the interest fingerprint (spec 4)."""
        with self._lock:
            past = self._db.execute("SELECT text, date FROM past_work").fetchall()
            bad = [json.loads(v) for (v,) in self._db.execute("SELECT vector FROM irrelevant")]
        papers = [e["body"] for e in self.my_papers().values()]
        works = [(self.embedder.embed(t), records.parse_date(d)) for t, d in past]
        works += [(p["paper_vector"], records.parse_date(p["date"])) for p in papers]
        if not works:
            works = [(self.embedder.embed(" ".join(self.config["declared_topics"])), None)]
        bad = [v for v in bad if len(v) == self.embedder.dim]
        long_term, recent = matching.build_fingerprint_vectors(works, bad)
        prev = self.ledger.latest("fingerprint", self.node_id)
        body = {
            "node_id": self.node_id,
            "version": prev["body"]["version"] + 1 if prev else 1,
            "topics": matching.merge_topics(self.config["declared_topics"], [p["topics"] for p in papers]),
            "long_term_vector": long_term,
            "recent_vector": recent,
            "updated_at": records.now_iso(),
        }
        return self._append("fingerprint", body)

    # ---- publishing a paper (spec 5, steps 1-4) ---------------------------

    def prepare_paper(self, extracted: dict, extractor: str | None = None) -> dict:
        """Turn the extraction skill's JSON into a paper value record body.

        Call this after the person has reviewed and edited the text, so the
        vector matches what they approved.
        """
        extracted = dict(extracted)
        if extractor:
            extracted["extractor"] = extractor
        problems = records.check_extracted(extracted)
        if problems:
            raise ValueError("extraction output has problems:\n  " + "\n  ".join(problems))
        body = {k: extracted[k] for k in records.EXTRACTED_FIELDS}
        body["paper_id"] = records.normalize_paper_id(body["paper_id"])
        body["paper_vector"] = self.embedder.embed(paper_text(body))
        body["author_node_id"] = self.node_id
        body["created_at"] = records.now_iso()
        return body

    def check_paper_limit(self, paper_id: str) -> None:
        mine = self.my_papers()
        if paper_id in mine:
            return  # an update to a paper already published
        rep = matching.reputation(self.ledger).get(self.node_id, {}).get("connections", 0)
        if rep == 0 and len(mine) >= matching.NEW_NODE_PAPER_LIMIT:
            raise LimitError(
                f"new nodes may publish {matching.NEW_NODE_PAPER_LIMIT} papers until someone confirms a "
                "connection to one of them"
            )

    def publish_paper(self, body: dict, draft: str | None = None) -> dict:
        self.check_paper_limit(body["paper_id"])
        entry = self._append("paper", body)
        self.refresh_fingerprint()
        self.broadcast()
        matches = matching.first_level(self.ledger, body, exclude={self.node_id})
        delivery = {}
        for node_id, score in matches:
            delivery[node_id] = self.send(node_id, "notification", {"paper_entry": entry, "score": score})
        released = self._release_held(draft, body["paper_id"]) if draft else []
        return {"entry": entry, "hash": entry_hash(entry), "matches": matches, "delivery": delivery,
                "released": released}

    # ---- messages ----------------------------------------------------------

    def send(self, to: str, mtype: str, payload: dict) -> str:
        """Deliver directly if the peer is reachable, else through a relay.
        Returns "direct", "relay" or "queued" (will retry on the next tick)."""
        msg = sign_object(self.identity, {"type": mtype, "from": self.node_id, "to": to,
                                          "sent_at": records.now_iso(), "payload": payload})
        how = self._deliver(msg)
        if how == "queued":
            with self._lock:
                self._db.execute("INSERT INTO outbox (recipient, message) VALUES (?, ?)", (to, json.dumps(msg)))
                self._db.commit()
        return how

    def _deliver(self, msg: dict) -> str:
        url = self.peer_url(msg["to"])
        if url:
            try:
                call(url, "/inbox", {"message": msg}, timeout=3)
                return "direct"
            except NetError:
                pass
        for relay in self.relays:
            try:
                call(relay, "/mailbox", {"message": msg}, timeout=5)
                return "relay"
            except NetError:
                continue
        return "queued"

    def retry_outbox(self) -> int:
        with self._lock:
            rows = self._db.execute("SELECT id, message FROM outbox").fetchall()
        sent = 0
        for row_id, raw in rows:
            if self._deliver(json.loads(raw)) != "queued":
                with self._lock:
                    self._db.execute("DELETE FROM outbox WHERE id=?", (row_id,))
                    self._db.commit()
                sent += 1
        return sent

    def _check_message(self, msg) -> str | None:
        if not isinstance(msg, dict):
            return "message must be an object"
        if msg.get("type") not in ("notification", "connection_notice"):
            return "unknown message type"
        if not is_node_id(msg.get("from")) or not verify_object(msg["from"], msg):
            return "bad message signature"
        if not isinstance(msg.get("payload"), dict):
            return "payload must be an object"
        return None

    def receive_message(self, msg) -> dict:
        problem = self._check_message(msg)
        if problem:
            raise ValueError(problem)
        if msg.get("to") != self.node_id:
            raise ValueError("message is addressed to another node")
        h = object_hash(msg)
        with self._lock:
            if self._db.execute("SELECT 1 FROM seen_messages WHERE hash=?", (h,)).fetchone():
                return {"ok": True, "duplicate": True}
        want = {"notification": ("paper_entry", "paper"), "connection_notice": ("connection_entry", "connection")}
        key, kind = want[msg["type"]]
        entry = msg["payload"].get(key)
        problems = check_entry(entry)
        if problems or entry["kind"] != kind or entry["author"] != msg["from"]:
            raise ValueError("embedded entry is invalid: " + "; ".join(problems or ["wrong kind or author"]))
        try:
            self.ledger.add(entry)
        except MissingPrev:
            pass  # we are behind on this author; the next sync fills the gap
        now = records.now_iso()
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO seen_messages VALUES (?)", (h,))
            if kind == "paper":
                score = msg["payload"].get("score")
                self._db.execute(
                    "INSERT INTO notifications (sender, paper_hash, paper, score, received_at, status, note) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (msg["from"], entry_hash(entry), json.dumps(entry["body"]),
                     score if isinstance(score, (int, float)) else None, now, "new", ""),
                )
            else:
                self._db.execute("INSERT INTO notices (sender, connection, received_at) VALUES (?,?,?)",
                                 (msg["from"], json.dumps(entry["body"]), now))
            self._db.commit()
        return {"ok": True}

    # ---- second-level matching and the human step (spec 5, steps 5-7) -----

    def _my_items(self) -> list[dict]:
        items = []
        for pid, e in self.my_papers().items():
            b = e["body"]
            items.append({**{k: b[k] for k in ("title", "date", "problem", "contribution", "useful_for",
                                                "limitations", "topics")},
                          "paper_id": pid, "vector": b["paper_vector"]})
        if self.config.get("allow_drafts"):
            for f in sorted((self.home / "drafts").glob("*")):
                if f.suffix.lower() in (".md", ".txt", ".tex") and f.is_file():
                    text = f.read_text(errors="replace")
                    items.append({"paper_id": f"draft:{f.stem}", "title": f.stem, "text": text,
                                  "vector": self.embedder.embed(text), "topics": []})
        return items

    def process_inbox(self) -> dict:
        """Run second-level matching on every new notification."""
        with self._lock:
            rows = self._db.execute("SELECT id, sender, paper_hash, paper FROM notifications WHERE status='new'"
                                    ).fetchall()
        if not rows:
            return {"matched": 0, "ignored": 0, "errors": 0}
        rep = matching.reputation(self.ledger)
        over = matching.over_limit_papers(self.ledger)
        items = self._my_items()
        done = {"matched": 0, "ignored": 0, "errors": 0}
        for nid, sender, phash, raw in rows:
            paper = json.loads(raw)
            note = None
            if phash in over:
                note = "author is past the new-node paper limit"
            elif rep.get(sender, {}).get("weighted", 0) < self.config.get("threshold", 0):
                note = "sender is below your reputation threshold"
            elif len(paper["paper_vector"]) != self.embedder.dim:
                note = "paper uses a different embedding model"
            if note:
                self._set_status(nid, "ignored", note)
                done["ignored"] += 1
                continue
            mine = sorted(items, key=lambda m: -cosine(paper["paper_vector"], m["vector"]))
            mine = mine[:SECOND_LEVEL_CANDIDATES]
            try:
                proposals = self.ai.propose(paper, mine)
            except Exception as e:  # noqa: BLE001 - leave it "new" and retry later
                self._set_status(nid, "new", f"AI failed: {e}")
                done["errors"] += 1
                continue
            to_paper = records.normalize_paper_id(paper["paper_id"])
            with self._lock:
                for p in proposals:
                    self._db.execute(
                        "INSERT INTO proposals (notification_id, from_paper, to_paper, relation, explanation, "
                        "status) VALUES (?,?,?,?,?,?)",
                        (nid, p["from_paper"], to_paper, p["relation"], p["explanation"], "proposed"),
                    )
                self._db.commit()
            self._set_status(nid, "matched", f"{len(proposals)} proposal(s)")
            done["matched"] += 1
        return done

    def _set_status(self, nid: int, status: str, note: str) -> None:
        with self._lock:
            self._db.execute("UPDATE notifications SET status=?, note=? WHERE id=?", (status, note, nid))
            self._db.commit()

    def inbox(self, include_done: bool = False) -> list[dict]:
        sql = "SELECT id, sender, paper_hash, paper, score, received_at, status, note FROM notifications"
        if not include_done:
            sql += " WHERE status IN ('new', 'matched')"
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id").fetchall()
            out = []
            for nid, sender, phash, raw, score, at, status, note in rows:
                props = self._db.execute(
                    "SELECT id, from_paper, to_paper, relation, explanation, status FROM proposals "
                    "WHERE notification_id=? ORDER BY id", (nid,)).fetchall()
                out.append({
                    "id": nid, "sender": sender, "paper_hash": phash, "paper": json.loads(raw), "score": score,
                    "received_at": at, "status": status, "note": note,
                    "proposals": [dict(zip(("id", "from_paper", "to_paper", "relation", "explanation", "status"), p))
                                  for p in props],
                })
        return out

    def _proposal(self, pid: int) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT id, notification_id, from_paper, to_paper, relation, explanation, status FROM proposals "
                "WHERE id=?", (pid,)).fetchone()
        if not row:
            raise KeyError(f"no proposal {pid}")
        return dict(zip(("id", "notification_id", "from_paper", "to_paper", "relation", "explanation", "status"),
                        row))

    def confirm(self, proposal_id: int, explanation: str | None = None, relation: str | None = None) -> dict:
        """The human agrees a proposed connection is real. Publishes it."""
        p = self._proposal(proposal_id)
        if p["status"] not in ("proposed",):
            raise ValueError(f"proposal {proposal_id} is already {p['status']}")
        if explanation is not None:
            p["explanation"] = explanation.strip()
        if relation is not None:
            p["relation"] = relation
        if p["relation"] not in records.RELATIONS:
            raise ValueError(f"relation must be one of {', '.join(records.RELATIONS)}")
        if not p["explanation"] or records.word_count(p["explanation"]) >= 100:
            raise ValueError("explanation must be non-empty and under 100 words")
        with self._lock:
            self._db.execute("UPDATE proposals SET explanation=?, relation=? WHERE id=?",
                             (p["explanation"], p["relation"], proposal_id))
            self._db.commit()
        if p["from_paper"].startswith("draft:"):
            self._set_proposal(proposal_id, "held", None)
            return {"held": True, "draft": p["from_paper"][len("draft:"):]}
        entry = self._publish_connection(p["from_paper"], p["to_paper"], p["relation"], p["explanation"])
        self._set_proposal(proposal_id, "confirmed", entry_hash(entry))
        return {"held": False, "entry": entry, "hash": entry_hash(entry)}

    def _publish_connection(self, frm: str, to: str, relation: str, explanation: str) -> dict:
        body = {"from_paper": frm, "to_paper": to, "relation": relation, "explanation": explanation,
                "confirmed_by": self.node_id, "created_at": records.now_iso()}
        problems = records.check_connection(body)
        if problems:
            raise ValueError("; ".join(problems))
        entry = self._append("connection", body)
        self.broadcast()
        # Spec 5.7: tell the original author.
        for author in matching.paper_authors(self.ledger).get(to, set()) - {self.node_id}:
            self.send(author, "connection_notice", {"connection_entry": entry})
        return entry

    def _release_held(self, draft: str, paper_id: str) -> list[str]:
        """A draft became a published paper: publish the connections the
        person already confirmed for it."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, to_paper, relation, explanation FROM proposals WHERE status='held' AND from_paper=?",
                (f"draft:{draft}",)).fetchall()
        out = []
        for pid, to, rel, why in rows:
            entry = self._publish_connection(paper_id, to, rel, why)
            with self._lock:
                self._db.execute("UPDATE proposals SET from_paper=? WHERE id=?", (paper_id, pid))
                self._db.commit()
            self._set_proposal(pid, "confirmed", entry_hash(entry))
            out.append(entry_hash(entry))
        return out

    def _set_proposal(self, pid: int, status: str, h: str | None) -> None:
        with self._lock:
            self._db.execute("UPDATE proposals SET status=?, entry_hash=? WHERE id=?", (status, h, pid))
            self._db.commit()

    def reject(self, proposal_id: int) -> None:
        """The human says no. Nothing is published."""
        p = self._proposal(proposal_id)
        if p["status"] != "proposed":
            raise ValueError(f"proposal {proposal_id} is already {p['status']}")
        self._set_proposal(proposal_id, "rejected", None)

    def mark_irrelevant(self, notification_id: int) -> None:
        """The whole notification was not for me. Feeds the next fingerprint."""
        with self._lock:
            row = self._db.execute("SELECT paper_hash, paper FROM notifications WHERE id=?",
                                   (notification_id,)).fetchone()
            if not row:
                raise KeyError(f"no notification {notification_id}")
            self._db.execute("UPDATE notifications SET status='irrelevant' WHERE id=?", (notification_id,))
            self._db.execute("UPDATE proposals SET status='rejected' WHERE notification_id=? AND status='proposed'",
                             (notification_id,))
            self._db.execute("INSERT OR IGNORE INTO irrelevant VALUES (?, ?)",
                             (row[0], json.dumps(json.loads(row[1])["paper_vector"])))
            self._db.commit()

    def notices(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, sender, connection, received_at FROM notices ORDER BY id").fetchall()
        return [{"id": i, "sender": s, "connection": json.loads(c), "received_at": at} for i, s, c, at in rows]

    # ---- disputes ----------------------------------------------------------

    def resolve(self, prefix: str) -> dict:
        found = self.ledger.find(prefix.lower())
        if not found:
            raise KeyError(f"no entry starts with {prefix}")
        if len(found) > 1:
            raise KeyError(f"{prefix} matches {len(found)} entries; type more characters")
        return found[0]

    def dispute(self, target: str, claim: str) -> dict:
        entry = self.resolve(target)
        body = {"target_entry_hash": entry_hash(entry), "claim": claim.strip(), "author_node_id": self.node_id,
                "created_at": records.now_iso()}
        problems = records.check_dispute(body)
        if problems:
            raise ValueError("; ".join(problems))
        d = self._append("dispute", body)
        self.broadcast()
        return d

    def disputes_for(self, h: str) -> list[dict]:
        return [e for e in self.ledger.entries(kind="dispute") if e["body"]["target_entry_hash"] == h]

    # ---- peers and sync ---------------------------------------------------

    def info(self, relay: bool = False) -> dict:
        return {"node_id": self.node_id, "name": self.config["name"], "relay": relay or self.relay,
                "public_url": self.config.get("public_url")}

    def announcement(self) -> dict:
        return sign_object(self.identity, {"node_id": self.node_id, "url": self.config.get("public_url"),
                                           "relay": self.relay, "announced_at": records.now_iso()})

    def add_peer(self, ann) -> bool:
        if not isinstance(ann, dict) or not is_node_id(ann.get("node_id")):
            return False
        if not verify_object(ann["node_id"], ann) or records.parse_time(ann.get("announced_at")) is None:
            return False
        if ann["node_id"] == self.node_id:
            return False
        url = ann.get("url")
        if url is not None and not (isinstance(url, str) and url.startswith(("http://", "https://"))):
            return False
        with self._lock:
            row = self._db.execute("SELECT announced_at FROM peers WHERE node_id=?", (ann["node_id"],)).fetchone()
            if row and records.parse_time(row[0]) > records.parse_time(ann["announced_at"]):
                return False
            self._db.execute("INSERT OR REPLACE INTO peers VALUES (?,?,?,?,?)",
                             (ann["node_id"], url, int(bool(ann.get("relay"))), ann["announced_at"],
                              json.dumps(ann)))
            self._db.commit()
        return True

    def peer_announcements(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT announcement FROM peers").fetchall()
        return [json.loads(r[0]) for r in rows] + [self.announcement()]

    def peer_url(self, node_id: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT url FROM peers WHERE node_id=?", (node_id,)).fetchone()
        return row[0] if row else None

    def _peer_urls(self, limit: int = 8) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT url FROM peers WHERE url IS NOT NULL ORDER BY announced_at DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [r[0] for r in rows]

    def _targets(self) -> list[str]:
        mine = self.config.get("public_url")
        seen, out = set(), []
        for u in self.relays + self._peer_urls():
            if u and u != mine and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def announce(self) -> int:
        ok = 0
        ann = self.announcement()
        for relay in self.relays:
            try:
                call(relay, "/peers", {"announcement": ann})
                ok += 1
            except NetError:
                pass
        return ok

    def discover(self) -> int:
        """Ask relays and known peers who they know about."""
        added = 0
        for url in self._targets():
            try:
                res = call(url, "/peers")
            except NetError:
                continue
            for ann in (res or {}).get("peers", []):
                added += self.add_peer(ann)
        return added

    def sync_from(self, url: str) -> dict:
        """Pull what the peer has and we lack, then push what we have and it lacks.

        The pull is spec 7's sync. The push matters for nodes nobody can
        reach (offline or behind a home router): it is how their entries
        get to the relays at all.
        """
        res = call(url, "/sync", {"have": self.ledger.summary()}, timeout=30) or {}
        result = self.ledger.ingest(res.get("entries") or [])
        missing = self.ledger.entries_since(res.get("have") or {})
        if missing:
            call(url, "/entries", {"entries": missing}, timeout=30)
        result["pushed"] = len(missing)
        return result

    def sync_all(self) -> dict:
        total = {"added": 0, "pushed": 0, "reached": 0, "failed": 0}
        for url in self._targets():
            try:
                r = self.sync_from(url)
                total["added"] += r["added"]
                total["pushed"] += r["pushed"]
                total["reached"] += 1
            except NetError:
                total["failed"] += 1
        return total

    def broadcast(self) -> int:
        """Spread new entries now instead of waiting for the next tick."""
        return self.sync_all()["reached"]

    def fetch_mailbox(self) -> int:
        """Collect messages relays held for us while we were offline."""
        got = 0
        for relay in self.relays:
            with self._lock:
                row = self._db.execute("SELECT last FROM mailbox_cursor WHERE relay=?", (relay,)).fetchone()
            after = row[0] if row else 0
            try:
                res = call(relay, f"/mailbox?to={self.node_id}&after={after}")
            except NetError:
                continue
            for msg in res.get("messages", []):
                try:
                    r = self.receive_message(msg)
                    got += 0 if r.get("duplicate") else 1
                except ValueError:
                    pass
            with self._lock:
                self._db.execute("INSERT OR REPLACE INTO mailbox_cursor VALUES (?, ?)",
                                 (relay, max(after, int(res.get("last", after)))))
                self._db.commit()
        return got

    # ---- relay side ----------------------------------------------------------

    def relay_store(self, msg) -> dict:
        problem = self._check_message(msg)
        if problem:
            raise ValueError(problem)
        if not is_node_id(msg.get("to")):
            raise ValueError("message has no valid recipient")
        if msg["to"] == self.node_id:
            return self.receive_message(msg)
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO relay_mailbox (recipient, hash, message, stored_at) "
                             "VALUES (?,?,?,?)", (msg["to"], object_hash(msg), json.dumps(msg), records.now_iso()))
            self._db.commit()
        return {"ok": True}

    def relay_fetch(self, to: str, after: int) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT id, message FROM relay_mailbox WHERE recipient=? AND id>? ORDER BY id "
                                    "LIMIT 500", (to, after)).fetchall()
        return {"messages": [json.loads(m) for _, m in rows], "last": rows[-1][0] if rows else after}

    def prune_relay_mailbox(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=MAILBOX_DAYS)).replace(microsecond=0)
        cutoff = cutoff.isoformat().replace("+00:00", "Z")
        with self._lock:
            self._db.execute("DELETE FROM relay_mailbox WHERE stored_at < ?", (cutoff,))
            self._db.commit()

    # ---- the background loop -------------------------------------------------

    def tick(self) -> dict:
        """One round of housekeeping. `rvn serve` runs this every minute."""
        out = {"announced": self.announce(), "new_peers": self.discover()}
        out["sync"] = self.sync_all()
        out["mail"] = self.fetch_mailbox()
        out["outbox_sent"] = self.retry_outbox()
        out["inbox"] = self.process_inbox()
        if self.relay:
            self.prune_relay_mailbox()
        return out
