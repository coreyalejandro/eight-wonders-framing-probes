"""Lexical baselines: can surface wording alone separate the framings?

Reports held-out accuracy for a bag-of-words classifier and a length-only
classifier, (a) within a template family with scenario-grouped CV and (b) across
families (train A -> test B, train B -> test A). A good control set makes (b)
fall to chance for surface features.
"""
import json, sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def v2_rows(path):
    import csv
    r = list(csv.DictReader(open(path, encoding="utf-8")))
    return [{"text": x["prompt"], "label": int(x["label"] == "sovereign"), "scenario_id": f"S{i//2:03d}"} for i, x in enumerate(r)]

def cv(texts, y, groups, kind):
    F = CountVectorizer().fit_transform(texts) if kind == "bow" else np.array([[len(t.split())] for t in texts])
    s = cross_val_score(LogisticRegression(max_iter=5000), F, y, groups=groups, cv=GroupKFold(8))
    return round(float(s.mean()), 4), round(float(s.std()), 4)

def transfer(tr, te, kind):
    if kind == "bow":
        v = CountVectorizer().fit([r["text"] for r in tr]); Ftr, Fte = v.transform([r["text"] for r in tr]), v.transform([r["text"] for r in te])
    else:
        Ftr = np.array([[len(r["text"].split())] for r in tr]); Fte = np.array([[len(r["text"].split())] for r in te])
    m = LogisticRegression(max_iter=5000).fit(Ftr, [r["label"] for r in tr])
    return round(float(m.score(Fte, [r["label"] for r in te])), 4)

def main(v2="data/prompts_v2.csv", v3="data/prompts_v3.jsonl"):
    out = {"v2": {}, "v3": {}}
    r2 = v2_rows(v2)
    for k in ("bow", "length"):
        out["v2"][f"{k}_heldout"] = cv([r["text"] for r in r2], [r["label"] for r in r2], [r["scenario_id"] for r in r2], k)
    r3 = load(v3)
    for fam in "AB":
        rr = [r for r in r3 if r["family"] == fam]
        for k in ("bow", "length"):
            out["v3"][f"{k}_within_{fam}"] = cv([r["text"] for r in rr], [r["label"] for r in rr], [r["scenario_id"] for r in rr], k)
    A = [r for r in r3 if r["family"] == "A"]; B = [r for r in r3 if r["family"] == "B"]
    for k in ("bow", "length"):
        out["v3"][f"{k}_transfer_A_to_B"] = transfer(A, B, k)
        out["v3"][f"{k}_transfer_B_to_A"] = transfer(B, A, k)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(*sys.argv[1:])
