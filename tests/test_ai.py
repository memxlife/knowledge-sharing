import sys

from rvn.ai import CommandAI, build_prompt, clean_proposals

NEW = {"paper_id": "arXiv:2602.00001", "title": "Paged KV cache", "date": "2026-02", "topics": ["kv cache"],
       "problem": "p", "contribution": "c", "useful_for": ["u"], "limitations": "l"}
MINE = [
    {"paper_id": "arXiv:2501.00001", "title": "Old work", "date": "2025-01", "topics": ["kv cache"]},
    {"paper_id": "arXiv:2603.00001", "title": "Later work", "date": "2026-03", "topics": ["kv cache"]},
    {"paper_id": "draft:my-next-paper", "title": "my-next-paper", "text": "draft text"},
]


def test_clean_proposals_enforces_rules():
    raw = [
        {"from_paper": "arXiv:2501.00001", "relation": "overlaps", "explanation": "Same question."},
        {"from_paper": "arXiv:2501.00001", "relation": "builds_on", "explanation": "Older cannot build on newer."},
        {"from_paper": "arXiv:2603.00001", "relation": "builds_on", "explanation": "Later work may."},
        {"from_paper": "draft:my-next-paper", "relation": "inspired", "explanation": "Drafts may."},
        {"from_paper": "arXiv:9999.99999", "relation": "overlaps", "explanation": "Not mine."},
        {"from_paper": "arXiv:2603.00001", "relation": "likes", "explanation": "Bad relation."},
        {"from_paper": "arXiv:2603.00001", "relation": "cites", "explanation": "word " * 150},
    ]
    got = clean_proposals(raw, NEW, MINE)
    pairs = [(p["from_paper"], p["relation"]) for p in got]
    assert pairs == [("arXiv:2501.00001", "overlaps"), ("arXiv:2603.00001", "builds_on"),
                     ("draft:my-next-paper", "inspired"), ("arXiv:2603.00001", "cites")]
    assert len(got[-1]["explanation"].split()) == 99


def test_clean_proposals_accepts_wrapped_object():
    got = clean_proposals({"proposals": [{"from_paper": "arXiv:2501.00001", "relation": "overlaps",
                                          "explanation": "x"}]}, NEW, MINE)
    assert len(got) == 1


def test_prompt_contains_both_sides():
    prompt = build_prompt(NEW, MINE)
    assert "Paged KV cache" in prompt and "arXiv:2501.00001" in prompt and "unpublished draft" in prompt


def test_command_backend_round_trip(tmp_path):
    script = tmp_path / "fake_ai.py"
    script.write_text(
        "import sys, json\n"
        "prompt = sys.stdin.read()\n"
        "assert 'Paged KV cache' in prompt\n"
        "print('Sure! Here you go:')\n"
        "print(json.dumps([{'from_paper': 'arXiv:2603.00001', 'relation': 'builds_on',"
        " 'explanation': 'Uses the paging idea.'}]))\n"
    )
    ai = CommandAI(f"{sys.executable} {script}")
    got = ai.propose(NEW, MINE)
    assert got == [{"from_paper": "arXiv:2603.00001", "relation": "builds_on", "explanation": "Uses the paging idea."}]
