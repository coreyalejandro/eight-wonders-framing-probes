"""Probe analysis on extracted activations.

Primary test (pre-specified, 2026-10-17):
  Probe   = difference-of-class-means direction (relational minus transactional),
            fit on one template family, scored on the other (vocabulary-disjoint).
  Metric  = cross-family transfer accuracy, averaged over A->B and B->A.
  Unit    = template pair (all prompts using a template share its wording, so the test
            sample is 16 template pairs, not thousands of independent prompts).
  Null    = max-statistic permutation test: each permutation swaps the frame labels of a
            random subset of template pairs, recomputes transfer for EVERY layer and pooling,
            and keeps the maximum. This corrects for (a) template-level dependence and
            (b) choosing the best layer/pooling on the same data.

Secondary (descriptive): per-template-pair accuracies at the best config; a logistic-
regression probe at the best config (within-family scenario-grouped CV and transfer);
per-Wonder direction cosines and cross-Wonder transfer.
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


class DiffMeans:
    """Fast cross-family transfer for many label permutations at one layer/pooling."""

    def __init__(self, X, y, fam, tpl):
        self.X = X.astype(np.float32); self.y = y; self.fam = fam; self.tpl = tpl
        self.pairs = sorted(set(tpl))
        self.M = {(t, f): self.X[(tpl == t) & (y == f)].mean(0) for t in self.pairs for f in (0, 1)}
        self.pfam = {t: fam[tpl == t][0] for t in self.pairs}
        self.split = {}
        for tr, te in (("A", "B"), ("B", "A")):
            trp = [t for t in self.pairs if self.pfam[t] == tr]
            m = self.fam == te
            self.split[(tr, te)] = (trp, np.stack([self.M[(t, 1)] - self.M[(t, 0)] for t in trp]),
                                    np.stack([self.M[(t, f)] for t in trp for f in (0, 1)]).mean(0),
                                    self.X[m], self.y[m], self.tpl[m])
        del self.X  # keep only the split copies (memory: 7B activations are large)

    def transfer(self, flip=frozenset()):
        accs = []
        for key, (trp, diffs, mid, Xt, yt0, tt) in self.split.items():
            s = np.array([-1.0 if t in flip else 1.0 for t in trp])
            w = (s[:, None] * diffs).mean(0)
            pred = ((Xt - mid) @ w) > 0
            yt = yt0.copy(); fl = np.isin(tt, list(flip)); yt[fl] = 1 - yt[fl]
            accs.append(float((pred == yt.astype(bool)).mean()))
        return float(np.mean(accs)), accs

    def per_template(self):
        out = {}
        for key, (trp, diffs, mid, Xt, yt, tt) in self.split.items():
            w = diffs.mean(0); pred = ((Xt - mid) @ w) > 0
            for t in sorted(set(tt)):
                m = tt == t; out[t] = float((pred[m] == yt[m].astype(bool)).mean())
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True); ap.add_argument("--data", default="data/prompts_v4.jsonl")
    ap.add_argument("--tag", required=True); ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    z, rows, meta = load(a.acts, a.data)
    y = np.array([r["label"] for r in rows]); g = np.array([r["scenario_id"] for r in rows])
    fam = np.array([r["family"] for r in rows]); won = np.array([r["wonder"] for r in rows])
    tpl = np.array([r["template"] for r in rows])
    configs = [(p, L) for p in ("last", "mean") for L in range(z[p].shape[1])]
    dm = {c: DiffMeans(z[c[0]][:, c[1]], y, fam, tpl) for c in configs}

    out = {"meta": meta, "n_prompts": len(rows), "n_template_pairs": len(set(tpl)), "layers": {}}
    obs = {}
    for c in configs:
        t, (ab, ba) = dm[c].transfer()
        obs[c] = t
        out["layers"][f"{c[0]}_L{c[1]+1}"] = {"transfer_A_to_B": round(ab, 4), "transfer_B_to_A": round(ba, 4), "transfer_mean": round(t, 4)}
    best = max(configs, key=lambda c: obs[c])

    rng = np.random.default_rng(0); pairs = np.array(sorted(set(tpl))); null_max = []
    for _ in range(a.perms):
        flip = frozenset(pairs[rng.integers(0, 2, len(pairs)).astype(bool)])
        null_max.append(max(dm[c].transfer(flip)[0] for c in configs))
    null_max = np.array(null_max)

    pt = dm[best].per_template()
    from scipy import stats
    accs = np.array(list(pt.values()))
    X = z[best[0]][:, best[1]]
    lr = {}
    for f in "AB":
        m = fam == f; s = []
        for tr, te in GroupKFold(8).split(X[m], y[m], g[m]):
            s.append(clf().fit(X[m][tr], y[m][tr]).score(X[m][te], y[m][te]))
        lr[f"within_{f}"] = round(float(np.mean(s)), 4)
    lr["transfer_A_to_B"] = round(float(clf().fit(X[fam == "A"], y[fam == "A"]).score(X[fam == "B"], y[fam == "B"])), 4)
    lr["transfer_B_to_A"] = round(float(clf().fit(X[fam == "B"], y[fam == "B"]).score(X[fam == "A"], y[fam == "A"])), 4)

    out["best"] = {
        "pool": best[0], "layer": best[1] + 1, "transfer_mean": round(obs[best], 4),
        "maxstat_perm_p": round(float((1 + (null_max >= obs[best]).sum()) / (1 + len(null_max))), 4),
        "maxstat_null_mean": round(float(null_max.mean()), 4),
        "maxstat_null_95pct": round(float(np.quantile(null_max, .95)), 4),
        "n_perms": a.perms, "n_configs_searched": len(configs), "perm_unit": "template pair",
        "per_template_pair_accuracy": {k: round(v, 4) for k, v in sorted(pt.items())},
        "template_pairs_above_chance": int((accs > 0.5).sum()), "n_template_pairs_tested": int(len(accs)),
        "template_ttest_p_uncorrected": round(float(stats.ttest_1samp(accs, 0.5, alternative="greater").pvalue), 6),
        "logistic_probe_at_best": lr,
    }

    W = sorted(set(won)); dirs = {}
    for w in W:
        m = won == w
        d = X[m & (y == 1)].mean(0) - X[m & (y == 0)].mean(0); dirs[w] = d / np.linalg.norm(d)
    cos = [[round(float(dirs[i] @ dirs[j]), 3) for j in W] for i in W]

    def xw(i, j):
        tr, te = (won == i) & (fam == "A"), (won == j) & (fam == "B")
        d = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); mid = (X[tr & (y == 1)].mean(0) + X[tr & (y == 0)].mean(0)) / 2
        return round(float(((((X[te] - mid) @ d) > 0) == y[te].astype(bool)).mean()), 3)
    out["wonders"] = {"order": W, "direction_cosine": cos, "cross_wonder_transfer_A_to_B": [[xw(i, j) for j in W] for i in W]}

    json.dump(out, open(f"{a.outdir}/probe_results_{a.tag}.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for pl, ls in (("last", "-"), ("mean", "--")):
            ks = sorted([k for k in out["layers"] if k.startswith(pl)], key=lambda k: int(k.split("L")[1]))
            ax[0].plot([int(k.split("L")[1]) for k in ks], [out["layers"][k]["transfer_mean"] for k in ks], ls, label=f"cross-family transfer ({pl})")
        ax[0].axhline(out["best"]["maxstat_null_95pct"], color="gray", lw=1, label="max-stat null 95%")
        ax[0].axhline(.5, color="gray", lw=.5, ls=":"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("accuracy"); ax[0].set_ylim(.3, 1.02)
        ax[0].set_title(meta["model"], fontsize=10); ax[0].legend(fontsize=7)
        M = out["wonders"]["cross_wonder_transfer_A_to_B"]; lab = [w.split("_")[0] for w in W]
        im = ax[1].imshow(M, vmin=.5, vmax=1, cmap="viridis"); ax[1].set_xticks(range(len(W)), lab); ax[1].set_yticks(range(len(W)), lab)
        ax[1].set_xlabel("test Wonder (family B)"); ax[1].set_ylabel("train Wonder (family A)")
        ax[1].set_title(f"cross-Wonder transfer, {best[0]} L{best[1]+1}", fontsize=10); fig.colorbar(im, ax=ax[1])
        fig.tight_layout(); fig.savefig(f"{a.outdir}/probe_{a.tag}.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps({k: v for k, v in out["best"].items() if k != "per_template_pair_accuracy"}, indent=2))


if __name__ == "__main__":
    main()
