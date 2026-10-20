"""Numbers quoted in RESULTS.md must match the committed result files."""
import json, os

import pytest

TAGS = ["qwen05b", "qwen15b", "qwen7b"]


@pytest.mark.parametrize("tag", TAGS)
def test_results_md_quotes_committed_numbers(tag):
    p = f"results/probe_results_{tag}.json"
    if not os.path.exists(p):
        pytest.skip("model results not present")
    txt = open("RESULTS.md", encoding="utf-8").read()
    b = json.load(open(p))["best"]
    assert f"**{b['transfer_mean']}** (layer {b['layer']})" in txt
    assert str(b["maxstat_null_95pct"]) in txt
    m = json.load(open(f"results/monitor_{tag}.json"))
    for v in (m["familyB_relational"]["SATISFIED"], m["familyB_transactional"]["VIOLATED"], m["familyB_halt_precision"]):
        assert str(v) in txt


@pytest.mark.parametrize("tag", TAGS)
def test_results_md_quotes_steering_null(tag):
    p = f"results/steering_{tag}.json"
    if not os.path.exists(p):
        pytest.skip("model results not present")
    n = json.load(open(p)).get("random_null")
    if not n:
        pytest.skip("no steering null in this result file")
    txt = open("RESULTS.md", encoding="utf-8").read()
    assert str(n["frontin_p_two_sided"]) in txt and str(n["probe_frontin_slope"]) in txt
