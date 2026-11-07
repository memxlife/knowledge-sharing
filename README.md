# Research Value Network: prototype v0

This is a working build of the spec *"Technical Spec: Peer-to-Peer Research Value Network, Prototype v0"*, the companion to the position paper *"From Rejecting Papers to Discovering Value"*.

Each researcher runs a small node on their own machine. They publish what a paper is good for. The network sends it to the people most likely to care. When one of those people agrees the paper connects to their own work, that connection goes into a shared, signed ledger. No central server.

## Try it in two minutes

```bash
cd knowledge-sharing
pip install -e ".[test]"        # needs Python 3.10+; the only runtime dependency is `cryptography`

python scripts/demo.py           # 2 relays + 4 researchers on localhost, the whole protocol, printed step by step
pytest                           # 45 tests, about 15 seconds
rvn synth                        # first-level matching on 50 fake profiles (build plan step 3)
```

## How a researcher uses it

```bash
# 1. Create your node: a key, your identity record, your first interest fingerprint.
rvn init --name "Ada Lovelace" --orcid 0000-0002-1825-0097 --affiliation "Some University" \
         --topics "llm serving,kv cache,batching" \
         --past-work my_past_papers.txt \
         --network network.json \
         --ai-command "claude -p"          # any CLI that reads a prompt on stdin and prints the answer

# 2. Keep it running. It syncs, collects mail from relays, and runs your AI on new arrivals.
rvn serve                                   # add --public-url https://... if peers can reach you directly

# 3. Turn a paper into a value record with your own AI.
rvn skill > prompt.txt                      # paste this plus the PDF into ChatGPT, Claude, or a local model
#    (or drop the skill/ folder into Claude as a skill)

# 4. Review, edit, sign and publish. Matching and notifications happen here.
rvn publish 2309.06180.json --edit

# 5. See what others published that your AI thinks connects to your work.
rvn inbox
rvn confirm 7                               # agree: publishes a connection record, tells the author
rvn reject 8                                # disagree: nothing is published
rvn irrelevant 3                            # not for me: future matching leans away from this

# Other commands
rvn notices                                 # connections others confirmed to your papers
rvn reputation                              # counted inbound connections, from your own ledger copy
rvn show arXiv:2309.06180                   # a paper, its connections, and any disputes on it
rvn dispute 3b44ac0dcb "The evidence field overstates the results."
rvn verify                                  # re-check every signature and chain, list conflicts
rvn agreement out_modelA/ out_modelB/       # build plan step 1: do two models extract the same thing?
```

To run a relay, run a normal node with `rvn serve --relay --host 0.0.0.0 --public-url https://relay.your-uni.edu:8470`, and put that URL in everyone's `network.json`.

## Where each part of the spec lives

| Spec | Code |
|---|---|
| 3. Extraction skill | `skill/prompt.md`, `skill/SKILL.md`, `skill/paper_value_record.schema.json`, `skill/topics.txt`, example in `skill/examples/` |
| 3. Node | `rvn/node.py`, command line in `rvn/cli.py` |
| 3. Ledger | `rvn/ledger.py` (signed entries, hash chain per author, forks kept and shown) |
| 3. Relay nodes | `rvn/node.py` (`relay_store`, `relay_fetch`), served by `rvn/net.py` with `--relay` |
| 3. Topic index | `rvn/matching.py` (`topic_index`) |
| 4. Data formats | `rvn/records.py` (one check per record type) |
| 5. Protocol, steps 1 to 7 | `Node.publish_paper`, `Node.process_inbox`, `Node.confirm`, `Node.reject`, `Node.mark_irrelevant` |
| 6. Reputation | `rvn/matching.py` (`reputation`, `counted_connections`, `over_limit_papers`) |
| 7. Networking | `rvn/net.py` (HTTP or HTTPS, JSON), sync in `Node.sync_from` |
| 9. Build plan 1 | `rvn agreement` (`rvn/agreement.py`) |
| 9. Build plan 2 to 5 | covered by `tests/test_ledger.py`, `tests/test_matching.py`, `tests/test_protocol.py` |

## Choices I made where the spec was silent

Each of these is easy to change. They're listed so nobody is surprised.

1. **A node's id is its public key.** The spec says "public key fingerprint". An Ed25519 public key is only 32 bytes, so the id is the key itself, written as 64 hex characters. That way anyone can check a signature without first finding the identity record.
2. **Every ledger entry has a small envelope.** It holds `kind`, `author`, `seq`, `prev` (the hash of the author's previous entry) and `body` (the record, using the spec's field names exactly). The spec's `signature` field sits on the envelope, so it covers both the record and the chain link.
3. **Sync goes both ways.** The spec describes a pull. A node behind a home router can't be pulled from, so its first entries would never leave the laptop. So after pulling, a node also pushes what the other side lacks. The data format is unchanged.
4. **Extra reputation rules.** On top of the spec's rule (only the author of `from_paper` can confirm):
   - The paper getting the credit (`to_paper`) must be in the ledger, or there is nobody to credit.
   - Connections between your own papers don't count.
   - The same connection confirmed twice counts once.
5. **Weights.** `builds_on` and `inspired` count 2. `cites`, `overlaps` and `contradicts` count 1. The spec says "weighted higher" without numbers.
6. **Time order.** An older paper can't `cite`, `build_on` or be `inspired` by a newer one. Proposals like that are dropped before you see them. `overlaps` and `contradicts` are always allowed.
7. **Direction of a connection.** `from_paper` is always the recipient's work and `to_paper` is the paper that was just sent. So when someone confirms a connection, credit goes to the person who published.
8. **First-level score** is the plain average of the two cosines (recent and long-term).
9. **"Irrelevant" marks** push your recent vector a little away from those papers the next time your fingerprint is rebuilt.
10. **Over-limit papers.** A new node's 4th paper (before anyone has confirmed a connection to it) stays in the ledger, but other nodes don't match or notify it.
11. **Evidence label.** The extraction prompt asks the `evidence` field to start with `Strong:`, `Moderate:` or `Preliminary:`, with a definition for each. That makes two models more likely to agree.

## Open decisions (spec section 10), and what v0 does for now

- **Launch subfield and the ten people.** Not a code question. The starter vocabulary assumes machine learning systems.
- **Embedding model.** v0 ships `hash-bow-v1`: hashed word counts, no download, same answer on every machine. It's crude. It tells a compiler paper from a privacy paper, but it scores a fair rewording low (in one test, 0.20 for a correct rewording of the same contribution). Pick a real model before the pilot. The network file names the model, and nodes skip vectors of the wrong size. Adding one is a class with `name`, `dim` and `embed()` in `rvn/embed.py`.
- **Drafts and "inspired".** v0 takes the cautious answer. Your AI may compare incoming papers against your drafts, if you turn that on. But a connection from a draft stays on your machine until you publish that draft as a paper (`rvn publish FILE --draft NAME`). Nothing about a draft becomes public early.
- **Topic vocabulary.** `skill/topics.txt` has 75 hand-written labels as a placeholder.

## Known gaps

- **ORCID is not verified.** The node checks the format and check digit, nothing more. Anyone can claim anyone's ORCID. For a pilot of ten known people that's fine. Before it grows, it needs ORCID sign-in at the relays, or a link from the ORCID profile back to the node id.
- **No real AI has been run against it yet.** The `command` backend is tested with a fake script that acts like an AI. The demo and tests use a text-similarity stand-in. Try `--ai-command "claude -p"` (or similar) on a few real papers.
- **Build plan step 1 needs your models.** `rvn agreement` does the measuring. Someone still has to run the skill on twenty papers with two models.
- **Relays accept mail from anyone** and keep it 30 days. Messages only carry public ledger data, but there is no rate limit or spam control.
- **The private key is stored unencrypted** in `key.pem` (file mode 600).
- **HTTPS** works with `--tls-cert/--tls-key`, but was only tested over plain HTTP on localhost.
- **The pilot (step 6)** is people, not code.

## Layout

```
rvn/            the node (Python package)
skill/          the extraction skill: prompt, schema, topic list, example
scripts/demo.py the whole network on one machine
tests/          45 tests: records, ledger, matching, AI parsing, the protocol over HTTP
```
