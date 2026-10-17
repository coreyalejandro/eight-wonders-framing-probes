"""Causal test: does steering along the framing direction change how the model reads Frontin'?

For each vignette the model answers a forced choice, presented in both option orders
(to cancel letter bias):
  Frontin' vignettes:  adaptive reading vs. theft-risk reading   (on-target)
  Neutral vignettes:   deal-seeking vs. quality-seeking reading  (off-target control)
Metric: logit(option 1) - logit(option 2), averaged over both orders.
  On-target:  positive = adaptive reading preferred.

We add alpha * d to the residual stream at the probe's best layer, where d is the unit
relational-minus-transactional mean difference from the v4 activations and alpha is a
multiple of the mean residual norm at that layer. Controls: a random direction of the
same norm; off-target questions; fluency (mean NLL on neutral text). Alpha = 0 gives the
unsteered base rate of theft-risk readings: a direct measurement of the misreading AHI
predicts.
"""
import argparse, json
import numpy as np
import torch

QUESTIONS = {
    "frontin": ("The shopper is protecting themselves from being treated as a suspect.",
                "The shopper may be planning to steal."),
    "neutral": ("The shopper wants a good deal.", "The shopper wants good quality."),
}
TEMPLATE = "{v}\nQuestion: Which reading fits better?\n(A) {a}\n(B) {b}\nAnswer: ("

def load_vignettes(path="data/frontin_vignettes.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def direction(acts_path, data_path, layer, pool="last"):
    z = np.load(acts_path)
    rows = [json.loads(l) for l in open(data_path, encoding="utf-8")][: len(z["ids"])]
    X = z[pool][:, layer - 1]; y = np.array([r["label"] for r in rows])
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / np.linalg.norm(d), float(np.linalg.norm(X, axis=1).mean())

class Steer:
    def __init__(self, model, layer):
        self.vec = None
        blocks = model.model.layers
        self.h = blocks[layer - 1].register_forward_hook(self.hook)
    def hook(self, mod, inp, out):
        if self.vec is None: return out
        if isinstance(out, tuple): return (out[0] + self.vec.to(out[0].dtype),) + tuple(out[1:])
        return out + self.vec.to(out.dtype)
    def set(self, v): self.vec = v
    def remove(self): self.h.remove()

def letter_ids(tok):
    ids = []
    for c in ("A", "B"):
        t = tok.encode(c, add_special_tokens=False)
        assert len(t) == 1, f"letter {c} is not a single token"
        ids.append(t[0])
    return ids

@torch.no_grad()
def choice_margin(model, tok, vignette, kind, ids):
    o1, o2 = QUESTIONS[kind]
    m = []
    for a, b, sign in ((o1, o2, 1), (o2, o1, -1)):
        enc = tok(TEMPLATE.format(v=vignette, a=a, b=b), return_tensors="pt").to(model.device)
        logits = model(**enc).logits[0, -1].float()
        m.append(sign * float(logits[ids[0]] - logits[ids[1]]))
    return float(np.mean(m))

@torch.no_grad()
def mean_nll(model, tok, texts):
    tot = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        tot.append(float(model(**enc, labels=enc["input_ids"]).loss))
    return float(np.mean(tot))

def main():
    from src.extract import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--acts", required=True)
    ap.add_argument("--probe-results", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data/prompts_v4.jsonl"); ap.add_argument("--outdir", default="results")
    ap.add_argument("--alphas", default="-0.5,-0.25,-0.1,0,0.1,0.25,0.5")
    a = ap.parse_args()
    best = json.load(open(a.probe_results))["best"]; L = best["layer"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok, model, name = load(a.model, device, torch.float16 if device != "cpu" else torch.float32)
    d, rnorm = direction(a.acts, a.data, L, best["pool"])
    rng = np.random.default_rng(0); r = rng.standard_normal(d.shape); r /= np.linalg.norm(r)
    ids = letter_ids(tok); V = load_vignettes()
    neutral_text = [v["text"] for v in V if v["kind"] == "neutral"]
    st = Steer(model, L); out = {"model": name, "layer": L, "residual_norm": rnorm, "alphas": [], "conditions": {}}
    alphas = [float(x) for x in a.alphas.split(",")]; out["alphas"] = alphas
    for cname, vec in (("probe_direction", d), ("random_direction", r)):
        res = {"frontin_margin": [], "neutral_margin": [], "nll_neutral_text": [], "frontin_risk_preferred_frac": []}
        for al in alphas:
            st.set(torch.tensor(vec * al * rnorm, dtype=torch.float32, device=model.device) if al else None)
            fm = [choice_margin(model, tok, v["text"], "frontin", ids) for v in V if v["kind"] == "frontin"]
            nm = [choice_margin(model, tok, v["text"], "neutral", ids) for v in V if v["kind"] == "neutral"]
            res["frontin_margin"].append(round(float(np.mean(fm)), 4))
            res["neutral_margin"].append(round(float(np.mean(nm)), 4))
            res["frontin_risk_preferred_frac"].append(round(float(np.mean(np.array(fm) < 0)), 4))
            res["nll_neutral_text"].append(round(mean_nll(model, tok, neutral_text), 4))
            if al == 0: res["frontin_margins_alpha0"] = [round(x, 4) for x in fm]
            print(cname, al, res["frontin_margin"][-1], flush=True)
        out["conditions"][cname] = res
    st.set(None); st.remove()
    json.dump(out, open(f"{a.outdir}/steering_{a.tag}.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for c, ls in (("probe_direction", "-o"), ("random_direction", "--s")):
            R = out["conditions"][c]
            ax[0].plot(alphas, R["frontin_margin"], ls, label=c); ax[1].plot(alphas, R["neutral_margin"], ls, label=c); ax[2].plot(alphas, R["nll_neutral_text"], ls, label=c)
        for x, t in zip(ax, ("Frontin': adaptive − risk logit", "Neutral (off-target) margin", "Fluency: NLL on neutral text")):
            x.set_title(t, fontsize=10); x.set_xlabel("alpha (× residual norm)"); x.axvline(0, color="gray", lw=.5); x.legend(fontsize=7)
        ax[0].axhline(0, color="gray", lw=.5); fig.suptitle(f"{name}, layer {L}", fontsize=10); fig.tight_layout()
        fig.savefig(f"{a.outdir}/steering_{a.tag}.png", dpi=130)
    except Exception as e: print("plot skipped:", e)

if __name__ == "__main__":
    main()
