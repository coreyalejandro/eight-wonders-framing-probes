import json, subprocess, sys

def test_lexical_transfer_is_chance():
    out = json.loads(subprocess.check_output([sys.executable, "src/lexical_baseline.py"]))
    assert out["v2"]["bow_heldout"][0] > 0.95          # documents the v2 confound
    assert out["v3"]["bow_transfer_A_to_B"] < 0.6      # v3 removes it
    assert out["v3"]["bow_transfer_B_to_A"] < 0.6
    assert out["v3"]["length_transfer_A_to_B"] == 0.5

def test_pipeline_smoke_untrained_model_does_not_transfer(tmp_path):
    acts = tmp_path / "a.npz"
    subprocess.check_call([sys.executable, "-m", "src.extract", "--model", "tiny-random", "--out", str(acts), "--batch", "64"])
    subprocess.check_call([sys.executable, "-m", "src.probes", "--acts", str(acts), "--tag", "pytest_tiny", "--perms", "20"])
    r = json.load(open("results/probe_results_pytest_tiny.json"))
    assert r["best"]["transfer_mean"] < 0.6   # negative control: untrained model cannot pass the test
