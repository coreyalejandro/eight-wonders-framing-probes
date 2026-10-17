import json, re
from src.build_dataset import TEMPLATES, build

STOP = set("the a an and or of to for in on at is was what who how whom this it here now will be with from by that which those as about whether does say carries".split())
words = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}

def test_families_share_no_content_words():
    A = set().union(*[words(r) | words(t) for r, t in TEMPLATES["A"]])
    B = set().union(*[words(r) | words(t) for r, t in TEMPLATES["B"]])
    assert A & B == set()

def test_frames_length_matched():
    for fam in TEMPLATES.values():
        for r, t in fam:
            assert len(r.split()) == len(t.split())

def test_dataset_complete_and_deterministic():
    a, b = build("data/prompts_v2.csv"), build("data/prompts_v2.csv")
    assert a == b and len(a) == 1536
    per = {}
    for r in a: per.setdefault(r["scenario_id"], []).append(r)
    assert len(per) == 128 and all(len(v) == 12 for v in per.values())
    for v in per.values():
        assert len({r["scenario"] for r in v}) == 1  # identical scenario across framings

def test_committed_dataset_matches_builder():
    rows = [json.loads(l) for l in open("data/prompts_v3.jsonl", encoding="utf-8")]
    assert rows == build("data/prompts_v2.csv")
