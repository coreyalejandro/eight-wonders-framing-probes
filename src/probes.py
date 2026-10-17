"""Probe analysis on extracted activations.

Tests, per layer and pooling:
  1. within-family accuracy (scenario-grouped 8-fold CV, so a scenario never
     appears in both train and test)
  2. cross-family transfer A->B and B->A (the lexical-confound-proof test)
  3. permutation null for transfer at the best layer (labels shuffled within scenario)
Then, at the best layer:
  4. per-Wonder probe directions: cosine similarity matrix
  5. cross-Wonder transfer: train on Wonder i (family A), test on Wonder j (family B)
Outputs results/probe_results_<tag>.json and figures.
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))

def load(acts, data):
    z = np.load(acts, allow_pickle=False)
    rows = [json.loads(l) for l in open(data, encoding="utf-8")]
    ids = list(z["ids"]); rows = rows[: len(ids)]
    assert ids == [f"{r['scenario_id']}|{r['template']}|{r['frame']}" for r in rows], "prompt order mismatch"
    return z, rows, json.loads(str(z["meta"]))

def within(X, y, g):
    accs = []
    for tr, te in GroupKFold(8).split(X, y, g):
        accs.append(clf().fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(accs))

def transfer(X, y, fam):
    a, b = fam == "A", fam == "B"
    return float(clf().fit(X[a], y[a]).score(X[b], y[b])), float(clf().fit(X[b], y[b]).score(X[a], y[a]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--data", default="data/prompts_v3.jsonl")
    ap.add_argument("--tag", required=True); ap.add_argument("--outdir", default="results"); ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()
    z, rows, meta = load(a.acts, a.data)
    y = np.array([r["label"] for r in rows]); g = np.array([r["scenario_id"] for r in rows])
    fam = np.array([r["family"] for r in rows]); won = np.array([r["wonder"] for r in rows])
    out = {"meta": meta, "n_prompts": len(rows), "layers": {}}
    best = (-1, None, None)
    for pool in ("last", "mean"):
        Xall = z[pool]
        for L in range(Xall.shape[1]):
            X = Xall[:, L]
            ab, ba = transfer(X, y, fam)
            res = {"within_A": within(X[fam == "A"], y[fam == "A"], g[fam == "A"]),
                   "within_B": within(X[fam == "B"], y[fam == "B"], g[fam == "B"]),
                   "transfer_A_to_B": ab, "transfer_B_to_A": ba, "transfer_mean": (ab + ba) / 2}
            out["layers"][f"{pool}_L{L+1}"] = {k: round(v, 4) for k, v in res.items()}
            if res["transfer_mean"] > best[0]: best = (res["transfer_mean"], pool, L)
    _, pool, L = best
    X = z[pool][:, L]
    rng = np.random.default_rng(0)
    null = []
    scen = np.unique(g)
    for _ in range(a.perms):
        flip = dict(zip(scen, rng.integers(0, 2, len(scen))))  # swap labels within each scenario pair-set
        yp = np.array([1 - yi if flip[gi] else yi for yi, gi in zip(y, g)])
        ab, ba = transfer(X, yp, fam); null.append((ab + ba) / 2)
    null = np.array(null)
    obs = out["layers"][f"{pool}_L{L+1}"]["transfer_mean"]
    out["best"] = {"pool": pool, "layer": L + 1, "transfer_mean": obs,
                   "perm_null_mean": round(float(null.mean()), 4), "perm_null_95pct": round(float(np.quantile(null, .95)), 4),
                   "perm_p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4), "n_perms": a.perms}
    W = sorted(set(won)); dirs = {}
    for w in W:
        m = won == w
        p = clf().fit(X[m], y[m]); coef = p[-1].coef_[0] / p[0].scale_
        dirs[w] = coef / np.linalg.norm(coef)
    cos = [[round(float(dirs[i] @ dirs[j]), 3) for j in W] for i in W]
    xw = [[round(float(clf().fit(X[(won == i) & (fam == "A")], y[(won == i) & (fam == "A")])
                         .score(X[(won == j) & (fam == "B")], y[(won == j) & (fam == "B")])), 3) for j in W] for i in W]
    out["wonders"] = {"order": W, "direction_cosine": cos, "cross_wonder_transfer_A_to_B": xw}
    json.dump(out, open(f"{a.outdir}/probe_results_{a.tag}.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for pl, ls in (("last", "-"), ("mean", "--")):
            ks = sorted([k for k in out["layers"] if k.startswith(pl)], key=lambda k: int(k.split("L")[1]))
            xs = [int(k.split("L")[1]) for k in ks]
            ax[0].plot(xs, [out["layers"][k]["transfer_mean"] for k in ks], ls, label=f"cross-family transfer ({pl})")
            ax[0].plot(xs, [(out["layers"][k]["within_A"] + out["layers"][k]["within_B"]) / 2 for k in ks], ls, alpha=.4, label=f"within-family ({pl})")
        ax[0].axhline(out["best"]["perm_null_95pct"], color="gray", lw=1, label="permutation null 95%")
        ax[0].axhline(.5, color="gray", lw=.5, ls=":"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("accuracy"); ax[0].set_ylim(.3, 1.02)
        ax[0].set_title(f"{meta['model']}"); ax[0].legend(fontsize=7)
        im = ax[1].imshow(xw, vmin=.5, vmax=1, cmap="viridis"); ax[1].set_xticks(range(len(W)), [w.split('_')[0] for w in W]); ax[1].set_yticks(range(len(W)), [w.split('_')[0] for w in W])
        ax[1].set_xlabel("test Wonder (family B)"); ax[1].set_ylabel("train Wonder (family A)"); ax[1].set_title(f"cross-Wonder transfer, {pool} L{L+1}")
        fig.colorbar(im, ax=ax[1]); fig.tight_layout(); fig.savefig(f"{a.outdir}/probe_{a.tag}.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps(out["best"], indent=2))

if __name__ == "__main__":
    main()
