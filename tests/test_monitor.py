import json, subprocess, sys
from src.monitor import (FramingInvariant, Context, EvidenceLog, SATISFIED, VIOLATED, AMBIGUOUS, NOT_APPLICABLE)

inv = FramingInvariant(t_violate=-1.0, t_satisfy=1.0)
ctx = lambda s, rel=True: Context("t", 0, {"framing_score": s, "relational_context": rel})

def test_states():
    assert inv.evaluate(ctx(-2)) == VIOLATED
    assert inv.evaluate(ctx(2)) == SATISFIED
    assert inv.evaluate(ctx(0)) == AMBIGUOUS

def test_not_applicable_outside_relational_context():
    assert inv.evaluate(ctx(-5, rel=False)) == NOT_APPLICABLE

def test_evaluate_never_throws():
    assert inv.evaluate(Context("t", 0, {"relational_context": True})) == NOT_APPLICABLE
    assert inv.evaluate(Context("t", 0, {"relational_context": True, "framing_score": "nan"})) == NOT_APPLICABLE
    assert inv.evaluate(Context("t", 0, {"relational_context": True, "framing_score": object()})) == NOT_APPLICABLE

def test_repair_blocking_only_on_violation():
    assert inv.repair(ctx(-2)).type == "HALT" and inv.repair(ctx(-2)).blocking
    assert inv.repair(ctx(0)).type == "PROMPT_USER" and not inv.repair(ctx(0)).blocking

def test_evidence_chain_detects_tampering(tmp_path):
    p = tmp_path / "ev.jsonl"; log = EvidenceLog(p)
    for i, s in enumerate([-2, 0, 2]):
        c = ctx(s); st = inv.evaluate(c); log.append(c, inv, st, inv.repair(c) if st in (VIOLATED, AMBIGUOUS) else None)
    assert EvidenceLog.verify(p)
    lines = p.read_text().splitlines(); e = json.loads(lines[1]); e["state"] = SATISFIED; lines[1] = json.dumps(e, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    assert not EvidenceLog.verify(p)

def test_steer_and_monitor_smoke(tmp_path):
    acts = tmp_path / "a.npz"; py = sys.executable
    subprocess.check_call([py, "-m", "src.extract", "--model", "tiny-random", "--out", str(acts), "--batch", "64"])
    subprocess.check_call([py, "-m", "src.probes", "--acts", str(acts), "--tag", "t", "--perms", "5", "--outdir", str(tmp_path)])
    pr = str(tmp_path / "probe_results_t.json")
    subprocess.check_call([py, "-m", "src.steer", "--model", "tiny-random", "--acts", str(acts), "--probe-results", pr,
                           "--tag", "t", "--outdir", str(tmp_path), "--alphas=-0.5,0,0.5"])
    s = json.load(open(tmp_path / "steering_t.json"))
    assert set(s["conditions"]) == {"probe_direction", "random_direction"}
    subprocess.check_call([py, "-m", "src.monitor", "calibrate", "--acts", str(acts), "--probe-results", pr, "--tag", "t", "--outdir", str(tmp_path)])
    out = subprocess.check_output([py, "-m", "src.monitor", "run", "--model", "tiny-random", "--config", str(tmp_path / "monitor_t.json"),
                                   "--vignettes", "--evidence", str(tmp_path / "ev.jsonl")], text=True)
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    assert all(r["state"] == NOT_APPLICABLE for r in rows if not r["relational_context"])
    assert "evidence chain valid: True" in out
