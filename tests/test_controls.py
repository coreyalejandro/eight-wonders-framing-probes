import json, subprocess, sys

def test_lexical_transfer_is_chance():
    out = json.loads(subprocess.check_output([sys.executable, "src/lexical_baseline.py"]))
    assert out["v2"]["bow_heldout"][0] > 0.95          # documents the v2 confound
    assert out["v4"]["bow_transfer_A_to_B"] < 0.6      # v4 removes it
    assert out["v4"]["bow_transfer_B_to_A"] < 0.6
    assert out["v4"]["length_transfer_A_to_B"] == 0.5

def test_pipeline_smoke_untrained_model_does_not_transfer(tmp_path):
    acts = tmp_path / "a.npz"
    subprocess.check_call([sys.executable, "-m", "src.extract", "--model", "tiny-random", "--out", str(acts), "--batch", "64"])
    subprocess.check_call([sys.executable, "-m", "src.probes", "--acts", str(acts), "--tag", "pytest_tiny", "--perms", "200", "--outdir", str(tmp_path)])
    r = json.load(open(tmp_path / "probe_results_pytest_tiny.json"))
    assert r["best"]["perm_unit"] == "template pair"
    assert r["best"]["n_template_pairs_tested"] == 16
    assert r["best"]["maxstat_perm_p"] > 0.01   # negative control: untrained model should not pass
