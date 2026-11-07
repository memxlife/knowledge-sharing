"""Command line: `rvn <command>`. Run `rvn -h` for the list."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import matching, records
from .ledger import entry_hash
from .net import Server

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOME = Path(os.environ.get("RVN_HOME", Path.home() / ".rvn"))


def _node(args):
    from .node import Node

    if not (args.home / "key.pem").exists():
        sys.exit(f"No node in {args.home}. Run `rvn init` first, or pass --home.")
    return Node(args.home)


def _names(node) -> dict[str, str]:
    return {e["author"]: e["body"]["name"] for e in node.ledger.entries(kind="identity")}


def _who(node, node_id: str, names=None) -> str:
    names = names if names is not None else _names(node)
    return f"{names.get(node_id, 'unknown')} ({node_id[:10]})"


def _read_past_work(path: Path) -> list[dict]:
    """Blank-line separated paragraphs. A paragraph may start with a year,
    like '2021: Title. Abstract...', which dates that piece of work."""
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^((?:19|20)\d{2})\s*[:|\-]\s*(.*)$", para, re.S)
        out.append({"text": m.group(2), "date": m.group(1)} if m else {"text": para, "date": None})
    return out


# ---- commands ----------------------------------------------------------------

def cmd_init(args):
    from .node import Node

    network = json.loads(Path(args.network).read_text()) if args.network else {}
    ai_config = {"backend": "command", "command": args.ai_command} if args.ai_command else {"backend": "heuristic"}
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    past = _read_past_work(Path(args.past_work)) if args.past_work else []
    node = Node.create(args.home, name=args.name, orcid=args.orcid, affiliation=args.affiliation,
                       scholar_url=args.scholar_url, topics=topics, past_work=past, network=network,
                       ai_config=ai_config, public_url=args.public_url)
    print(f"Created node {node.node_id}")
    print(f"  home: {args.home}")
    print(f"  relays: {', '.join(node.relays) or 'none (add a network file with --network)'}")
    print(f"  past work items embedded: {len(past)}")
    if not args.ai_command:
        print("  AI: heuristic stand-in. For real matching set one, e.g.\n"
              "      rvn config ai.backend command\n      rvn config ai.command 'claude -p'")
    print("Keep key.pem private. It is your identity on the network.")


def cmd_config(args):
    node = _node(args)
    if args.key is None:
        print(json.dumps(node.config, indent=2))
        return
    cur = node.config
    parts = args.key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    if args.value is None:
        print(json.dumps(cur.get(parts[-1]), indent=2))
        return
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    cur[parts[-1]] = value
    node.save_config()
    print(f"{args.key} = {json.dumps(value)}")


def cmd_skill(args):
    prompt = (ROOT / "skill" / "prompt.md").read_text()
    topics = (ROOT / "skill" / "topics.txt").read_text()
    labels = [line.strip() for line in topics.splitlines() if line.strip() and not line.startswith("#")]
    print(prompt.replace("{{TOPICS}}", "\n".join(f"- {t}" for t in labels)))


def _show_record(body: dict) -> None:
    print(f"  paper_id:  {body['paper_id']}")
    print(f"  title:     {body['title']}")
    print(f"  authors:   {', '.join(body['authors'])}   date: {body['date']}")
    for k in ("problem", "contribution", "method_summary", "evidence", "limitations"):
        print(f"  {k}:\n    {body[k]}")
    print("  builds_on:\n" + "\n".join(f"    - {x}" for x in body["builds_on"]) if body["builds_on"] else
          "  builds_on: (none)")
    print("  useful_for:\n" + "\n".join(f"    - {x}" for x in body["useful_for"]))
    print(f"  topics:    {', '.join(body['topics'])}")
    print(f"  extractor: {body['extractor']}")


def _edit(record: dict) -> dict:
    editable = {k: record[k] for k in records.PAPER_TEXT_FIELDS}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(editable, f, indent=2, ensure_ascii=False)
        path = f.name
    subprocess.call([os.environ.get("EDITOR", "nano"), path])
    edited = json.loads(Path(path).read_text())
    os.unlink(path)
    return {**record, **{k: edited[k] for k in records.PAPER_TEXT_FIELDS if k in edited}}


def cmd_publish(args):
    node = _node(args)
    extracted = json.loads(Path(args.file).read_text())
    if args.edit:
        extracted = _edit(extracted)
    problems = records.check_extracted({**extracted, **({"extractor": args.extractor} if args.extractor else {})})
    if problems:
        sys.exit("The record has problems:\n  " + "\n  ".join(problems))
    vocab = {records.normalize_topic(t) for t in (ROOT / "skill" / "topics.txt").read_text().splitlines()
             if t.strip() and not t.startswith("#")}
    print("Review the record (spec 5.1). Re-run with --edit to change any text field.\n")
    _show_record({**extracted, "extractor": args.extractor or extracted.get("extractor")})
    free = [t for t in extracted["topics"] if records.normalize_topic(t) not in vocab]
    if free:
        print(f"\n  note: not in the shared vocabulary: {', '.join(free)}")
    if not args.yes and input("\nPublish this record? [y/N] ").strip().lower() != "y":
        print("Not published.")
        return
    print("Syncing first so matching sees the latest fingerprints...")
    node.sync_all()
    body = node.prepare_paper(extracted, extractor=args.extractor)
    res = node.publish_paper(body, draft=args.draft)
    names = _names(node)
    print(f"\nPublished {body['paper_id']} as entry {res['hash'][:16]}")
    print(f"Sent to {len(res['matches'])} matched researcher(s) (at most {matching.TOP_K}):")
    for node_id, score in res["matches"]:
        print(f"  {score:.3f}  {_who(node, node_id, names):40s}  {res['delivery'][node_id]}")
    if res["released"]:
        print(f"Published {len(res['released'])} connection(s) you had confirmed for draft '{args.draft}'.")


def cmd_inbox(args):
    node = _node(args)
    node.fetch_mailbox()
    done = node.process_inbox()
    if done["errors"]:
        print(f"(AI failed on {done['errors']} notification(s); they will be retried)")
    names = _names(node)
    rep = matching.reputation(node.ledger)
    items = node.inbox(include_done=args.all)
    if not items:
        print("Inbox is empty.")
        return
    for n in items:
        p = n["paper"]
        r = rep.get(n["sender"], {}).get("weighted", 0)
        print(f"#{n['id']}  {p['title']}  [{p['paper_id']}]")
        print(f"     from {_who(node, n['sender'], names)}, reputation {r}, match score {n['score']}, "
              f"status {n['status']}{' - ' + n['note'] if n['note'] else ''}")
        print(f"     contribution: {p['contribution']}")
        for pr in n["proposals"]:
            print(f"     [{pr['id']}] {pr['status']}: your {pr['from_paper']} {pr['relation']} this paper")
            print(f"         {pr['explanation']}")
        print()
    print("Next: `rvn confirm ID`, `rvn reject ID`, or `rvn irrelevant #ID` for a whole notification.")


def cmd_confirm(args):
    node = _node(args)
    res = node.confirm(args.proposal, explanation=args.explanation, relation=args.relation)
    if res["held"]:
        print(f"Saved. This links your draft '{res['draft']}', so it will be published when you run "
              f"`rvn publish FILE --draft {res['draft']}`.")
    else:
        print(f"Published connection {res['hash'][:16]}. The paper's author has been notified.")


def cmd_reject(args):
    _node(args).reject(args.proposal)
    print("Rejected. Nothing was published.")


def cmd_irrelevant(args):
    node = _node(args)
    node.mark_irrelevant(int(str(args.notification).lstrip("#")))
    print("Marked irrelevant. Your next fingerprint update will lean away from it "
          "(`rvn fingerprint` to update now).")


def cmd_fingerprint(args):
    node = _node(args)
    e = node.refresh_fingerprint()
    node.broadcast()
    print(f"Published fingerprint v{e['body']['version']}: topics {', '.join(e['body']['topics'])}")


def cmd_dispute(args):
    node = _node(args)
    d = node.dispute(args.target, args.claim)
    print(f"Published dispute {entry_hash(d)[:16]} against {d['body']['target_entry_hash'][:16]}")


def cmd_show(args):
    node = _node(args)
    names = _names(node)
    pid = records.normalize_paper_id(args.target)
    if pid:
        found = [e for e in node.ledger.entries(kind="paper") if records.normalize_paper_id(e["body"]["paper_id"]) == pid]
    else:
        found = [node.resolve(args.target)]
    if not found:
        sys.exit(f"Nothing in the ledger for {args.target}")
    counted = {(c["confirmed_by"], c["from_paper"], c["to_paper"], c["relation"])
               for c in matching.counted_connections(node.ledger)}
    for e in found:
        h = entry_hash(e)
        print(f"{e['kind']} {h}  by {_who(node, e['author'], names)}  seq {e['seq']}")
        if e["kind"] == "paper":
            _show_record(e["body"])
            p = records.normalize_paper_id(e["body"]["paper_id"])
            for c in node.ledger.entries(kind="connection"):
                b = c["body"]
                if p in (records.normalize_paper_id(b["from_paper"]), records.normalize_paper_id(b["to_paper"])):
                    key = (b["confirmed_by"], records.normalize_paper_id(b["from_paper"]),
                           records.normalize_paper_id(b["to_paper"]), b["relation"])
                    tag = "counts" if key in counted else "does not count"
                    print(f"  connection: {b['from_paper']} {b['relation']} {b['to_paper']} ({tag})")
                    print(f"    {b['explanation']}")
        else:
            print(json.dumps({k: v for k, v in e["body"].items() if not k.endswith("vector")}, indent=2))
        for d in node.disputes_for(h):
            print(f"  DISPUTED by {_who(node, d['author'], names)} on {d['body']['created_at']}:")
            print(f"    {d['body']['claim']}")
        print()


def cmd_reputation(args):
    node = _node(args)
    names = _names(node)
    rep = matching.reputation(node.ledger)
    rows = sorted(rep.items(), key=lambda kv: (-kv[1]["weighted"], kv[0]))
    if args.node_id:
        rows = [(k, v) for k, v in rows if k.startswith(args.node_id)] or [(args.node_id, {"connections": 0,
                                                                                           "weighted": 0})]
    if not rows:
        print("No counted connections in the ledger yet.")
    for node_id, r in rows:
        print(f"{r['weighted']:4d} weighted  {r['connections']:4d} connections  {_who(node, node_id, names)}")


def cmd_verify(args):
    node = _node(args)
    problems = node.ledger.verify_all()
    conflicts = node.ledger.conflicts()
    print(f"{node.ledger.count()} entries from {len(node.ledger.authors())} authors checked.")
    for p in problems:
        print(f"  PROBLEM {p}")
    for c in conflicts:
        print(f"  CONFLICT {_who(node, c['author'])} signed {len(c['hashes'])} different entries at seq {c['seq']}:"
              f" {', '.join(h[:12] for h in c['hashes'])}")
    if not problems and not conflicts:
        print("All signatures and chains check out. No conflicts.")


def cmd_status(args):
    node = _node(args)
    rep = matching.reputation(node.ledger).get(node.node_id, {"connections": 0, "weighted": 0})
    inbox = node.inbox()
    print(f"{node.config['name']}  {node.node_id}")
    print(f"  ledger: {node.ledger.count()} entries, {len(node.ledger.authors())} authors")
    print(f"  my papers: {len(node.my_papers())}   reputation: {rep['weighted']} weighted, "
          f"{rep['connections']} connections")
    print(f"  inbox: {len(inbox)} open   notices: {len(node.notices())}   peers known: "
          f"{len(node.peer_announcements()) - 1}")
    print(f"  relays: {', '.join(node.relays) or 'none'}")


def cmd_notices(args):
    node = _node(args)
    node.fetch_mailbox()
    names = _names(node)
    for n in node.notices():
        c = n["connection"]
        print(f"{n['received_at']}  {_who(node, n['sender'], names)} confirmed: "
              f"{c['from_paper']} {c['relation']} your {c['to_paper']}")
        print(f"    {c['explanation']}")
    if not node.notices():
        print("No connection notices yet.")


def cmd_sync(args):
    node = _node(args)
    print(json.dumps(node.tick(), indent=2))


def cmd_synth(args):
    from .synth import evaluate

    rows = evaluate(args.profiles, args.seed, args.noise)
    print(f"First-level matching over {args.profiles} fake profiles, top {matching.TOP_K} kept, "
          f"side-interest noise {args.noise}.\n")
    print(f"{'area':14s} {'members':>7s} {'matched':>7s} {'precision':>9s} {'recall':>7s}")
    for r in rows:
        print(f"{r['area']:14s} {r['members']:7d} {r['candidates']:7d} {r['precision_at_members']:9.2f} "
              f"{r['recall_at_k']:7.2f}")
    print("\nprecision: of the top N matches (N = people in that area), the share really in that area.")
    print("recall:    of the people in that area, the share that made the top-30 list.")


def cmd_agreement(args):
    from .agreement import FLAG_BELOW, compare_folders

    res = compare_folders(Path(args.folder_a), Path(args.folder_b))
    if not res["pairs"]:
        sys.exit("No papers appear in both folders (files are paired by paper_id).")
    print(f"{'paper':22s} {'contribution':>12s} {'useful_for':>10s} {'topics':>7s} {'evidence':>8s}")
    for r in res["pairs"]:
        print(f"{r['paper_id']:22s} {r['contribution']:12.2f} {r['useful_for']:10.2f} {r['topics']:7.2f} "
              f"{r['evidence']:8.0f}")
    m = res["mean"]
    print(f"{'MEAN':22s} {m['contribution']:12.2f} {m['useful_for']:10.2f} {m['topics']:7.2f} {m['evidence']:8.2f}")
    if res["flagged"]:
        print(f"\nRead these by hand (a score under {FLAG_BELOW}): {', '.join(res['flagged'])}")
    for side in ("only_in_a", "only_in_b"):
        if res[side]:
            print(f"{side}: {', '.join(res[side])}")


def cmd_serve(args):
    node = _node(args)
    node.relay = args.relay
    server = Server(node, host=args.host, port=args.port, relay=args.relay, tls_cert=args.tls_cert,
                    tls_key=args.tls_key).start()
    if args.public_url:
        node.config["public_url"] = args.public_url
        node.save_config()
    print(f"Serving {node.config['name']} on {server.url}{' as a relay' if args.relay else ''}")
    print(f"  public url announced to peers: {node.config.get('public_url') or 'none (reached via relays)'}")
    try:
        while True:
            out = node.tick()
            changed = out["sync"]["added"] or out["mail"] or out["inbox"]["matched"] or out["outbox_sent"]
            if changed:
                print(time.strftime("%H:%M:%S"), json.dumps(out))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        server.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rvn", description="Research Value Network node, prototype v0")
    ap.add_argument("--home", type=Path, default=DEFAULT_HOME, help=f"node folder (default {DEFAULT_HOME})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create your node: key, identity record, first fingerprint")
    p.add_argument("--name", required=True)
    p.add_argument("--orcid", required=True)
    p.add_argument("--affiliation", required=True)
    p.add_argument("--scholar-url")
    p.add_argument("--topics", required=True, help="3 to 10 comma-separated topic labels")
    p.add_argument("--past-work", help="text file: one paragraph per past paper, optionally starting 'YYYY:'")
    p.add_argument("--network", help="network file with relays and embedding model (see network.example.json)")
    p.add_argument("--ai-command", help="CLI that reads a prompt on stdin, e.g. 'claude -p'")
    p.add_argument("--public-url", help="URL where peers can reach this node, if any")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("config", help="show or change a setting, e.g. `rvn config threshold 2`")
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("skill", help="print the extraction prompt to paste into any AI assistant")
    p.set_defaults(fn=cmd_skill)

    p = sub.add_parser("publish", help="review, sign and publish a paper value record, then notify matches")
    p.add_argument("file", help="the JSON the extraction skill produced")
    p.add_argument("--edit", action="store_true", help="open the text fields in $EDITOR first")
    p.add_argument("--yes", action="store_true", help="skip the confirmation question")
    p.add_argument("--extractor", help="override the extractor field, e.g. 'gpt-4o / skill v0.1'")
    p.add_argument("--draft", help="this paper was the draft with this name; release its held connections")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("inbox", help="papers sent to you, with your AI's proposed connections")
    p.add_argument("--all", action="store_true", help="include handled notifications")
    p.set_defaults(fn=cmd_inbox)

    p = sub.add_parser("confirm", help="agree a proposed connection is real and publish it")
    p.add_argument("proposal", type=int)
    p.add_argument("--explanation", help="replace the AI's explanation (under 100 words)")
    p.add_argument("--relation", choices=records.RELATIONS)
    p.set_defaults(fn=cmd_confirm)

    p = sub.add_parser("reject", help="say no to a proposed connection (nothing is published)")
    p.add_argument("proposal", type=int)
    p.set_defaults(fn=cmd_reject)

    p = sub.add_parser("irrelevant", help="mark a whole notification as not for you")
    p.add_argument("notification")
    p.set_defaults(fn=cmd_irrelevant)

    p = sub.add_parser("fingerprint", help="recompute and publish your interest fingerprint now")
    p.set_defaults(fn=cmd_fingerprint)

    p = sub.add_parser("dispute", help="attach a dispute to any ledger entry")
    p.add_argument("target", help="entry hash or the first 10+ characters of it")
    p.add_argument("claim", help="under 200 words")
    p.set_defaults(fn=cmd_dispute)

    p = sub.add_parser("show", help="show a paper (by id) or entry (by hash), with disputes and connections")
    p.add_argument("target")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("reputation", help="counted inbound connections, computed from your ledger")
    p.add_argument("node_id", nargs="?")
    p.set_defaults(fn=cmd_reputation)

    p = sub.add_parser("verify", help="re-check every signature and chain; list conflicts")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("status", help="a one-screen summary of your node")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("notices", help="connections others confirmed to your papers")
    p.set_defaults(fn=cmd_notices)

    p = sub.add_parser("sync", help="one round: announce, find peers, sync, collect mail, match")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("serve", help="run the node (and optionally a relay) until stopped")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8470)
    p.add_argument("--relay", action="store_true", help="also hold mail for offline nodes")
    p.add_argument("--public-url", help="URL peers should use to reach this node")
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    p.add_argument("--interval", type=int, default=60, help="seconds between sync rounds")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("synth", help="test first-level matching on a fake network (build plan step 3)")
    p.add_argument("--profiles", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.5,
                   help="chance (per slot, two slots) that a profile also lists a topic from another area")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("agreement", help="compare two models' extractions of the same papers (build plan step 1)")
    p.add_argument("folder_a")
    p.add_argument("folder_b")
    p.set_defaults(fn=cmd_agreement)

    args = ap.parse_args(argv)
    from .node import LimitError

    try:
        args.fn(args)
    except (ValueError, KeyError, LimitError, FileExistsError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
