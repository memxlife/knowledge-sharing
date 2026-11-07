---
name: paper-value-extraction
description: Turn one research paper (a PDF or an arXiv ID) into a Paper Value Record JSON for the Research Value Network. Use when the user asks to extract a paper's value record, prepare a paper for the network, or run `rvn publish` on a paper they have not extracted yet.
---

# Paper value extraction (v0.1)

Produce one JSON object that follows `paper_value_record.schema.json` in this folder.

Steps:

1. Get the full text. For an arXiv ID, fetch the abstract page and the PDF. If you only have the abstract, tell the user and stop.
2. Read `prompt.md` in this folder. Its field rules are the contract: follow them literally so different models agree. Use the labels in `topics.txt` (same folder) for `topics`.
3. Write the JSON to a file named `<paper_id>.json` (replace `/` with `_`).
4. Check it: `python -c "import json,sys; from rvn.records import check_extracted; print(check_extracted(json.load(open(sys.argv[1]))) or 'ok')" FILE`
5. Show the user the record and tell them to review it with `rvn publish FILE --edit`. Never publish for them: the human approves every record (spec step 5.1).

Field rules, in short (full version in `prompt.md`):

- `problem` at most 40 words. `contribution` 2 to 4 sentences, at most 90 words. `method_summary` at most 150 words.
- `evidence` starts with `Strong:`, `Moderate:` or `Preliminary:`.
- `builds_on` lists only work the paper says it extends or depends on, at most 5.
- `useful_for` has 2 to 5 items, each "A researcher working on X could use this to Y."
- `limitations` in the authors' framing, or exactly "Not stated by the authors."
- `topics` 3 to 8, from the vocabulary first.
- `extractor` is "<model> / paper-value-extraction v0.1".
