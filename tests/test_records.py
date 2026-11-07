import json

import pytest

from conftest import ROOT, extracted
from rvn import records


def test_orcid_checksum():
    assert records.orcid_ok("0000-0002-1825-0097")
    assert records.orcid_ok("0000-0002-1694-233X")
    assert not records.orcid_ok("0000-0002-1825-0098")  # wrong check digit
    assert not records.orcid_ok("0000000218250097")
    assert not records.orcid_ok(None)


@pytest.mark.parametrize("raw,want", [
    ("2309.06180", "arXiv:2309.06180"),
    ("2309.06180v3", "arXiv:2309.06180"),
    ("arXiv:2309.06180", "arXiv:2309.06180"),
    ("https://arxiv.org/abs/2309.06180v2", "arXiv:2309.06180"),
    ("cs.DC/0101001", "arXiv:cs.DC/0101001"),
    ("hep-th/9901001", "arXiv:hep-th/9901001"),
    ("10.1145/3600006.3613165", "doi:10.1145/3600006.3613165"),
    ("https://doi.org/10.1145/ABC.123", "doi:10.1145/abc.123"),
    ("not a paper", None),
    ("", None),
])
def test_paper_id_normalization(raw, want):
    assert records.normalize_paper_id(raw) == want


def test_example_record_passes_node_checks(example_record):
    assert records.check_extracted(example_record) == []


def test_example_record_passes_json_schema(example_record):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "skill" / "paper_value_record.schema.json").read_text())
    jsonschema.validate(example_record, schema)


def test_extraction_limits():
    rec = extracted("2401.00001", "T", "2024-01", ["a", "b", "c"], "x")
    assert records.check_extracted(rec) == []
    too_long = dict(rec, method_summary="word " * 151)
    assert any("method_summary" in p for p in records.check_extracted(too_long))
    assert any("topics" in p for p in records.check_extracted(dict(rec, topics=["a", "b"])))
    assert any("useful_for" in p for p in records.check_extracted(dict(rec, useful_for=["only one"])))
    assert any("paper_id" in p for p in records.check_extracted(dict(rec, paper_id="nope")))
    assert any("date" in p for p in records.check_extracted(dict(rec, date="last year")))


def test_connection_and_dispute_word_limits():
    base = {"from_paper": "2401.00001", "to_paper": "2401.00002", "relation": "builds_on",
            "explanation": "ok", "confirmed_by": "a" * 64, "created_at": records.now_iso()}
    assert records.check_connection(base) == []
    assert records.check_connection(dict(base, explanation="w " * 100))  # must be under 100
    assert records.check_connection(dict(base, relation="likes"))
    d = {"target_entry_hash": "b" * 64, "claim": "This is wrong.", "author_node_id": "a" * 64,
         "created_at": records.now_iso()}
    assert records.check_dispute(d) == []
    assert records.check_dispute(dict(d, claim="w " * 200))


def test_unknown_fields_are_ignored():
    rec = extracted("2401.00001", "T", "2024-01", ["a", "b", "c"], "x")
    rec["some_future_field"] = {"anything": 1}
    assert records.check_extracted(rec) == []
