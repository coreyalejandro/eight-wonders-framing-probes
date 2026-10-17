"""TLC runtime monitor: the framing probe as a ConstitutionalInvariant.

Follows TLC 2.0 src/interfaces/constitutional-invariant.ts:
  evaluate(context) -> SATISFIED | VIOLATED | AMBIGUOUS | NOT_APPLICABLE   (pure, never throws)
  repair(context)   -> RepairAction {type, message, blocking, evidencePath}
  EvidenceEntry     -> append-only JSONL with a SHA-256 hash chain (integrity_hash)

What it watches: the model's internal framing of the current turn. A score below the
violation threshold means the model is reading the situation transactionally — the
collapse AHI predicts for relational behavior such as Frontin'. That triggers TLC's
halt authority and requests the narrative baseline before emission continues.

Calibration is honest by construction: direction and thresholds are fit on template
family A only; reported performance is on family B, whose vocabulary is disjoint.

  python -m src.monitor calibrate --acts results/acts_qwen7b.npz --probe-results results/probe_results_qwen7b.json --tag qwen7b
  python -m src.monitor run --model Qwen/Qwen2.5-7B --config results/monitor_qwen7b.json --text "..."
"""
import argparse, hashlib, json, uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SATISFIED, VIOLATED, AMBIGUOUS, NOT_APPLICABLE = "SATISFIED", "VIOLATED", "AMBIGUOUS", "NOT_APPLICABLE"

@dataclass
class RepairAction:
    type: str
    message: str
    blocking: bool
    evidencePath: str | None = None

@dataclass
class Context:
    sessionId: str
    turn: int
    payload: dict
    history: list = field(default_factory=list)
    narrativeBaseline: str | None = None

class FramingInvariant:
    """EW-FRAME-001: relational situations must not be read through a transactional frame."""
    id = "EW-FRAME-001"
    description = ("Relational framing preserved: the model's internal reading of a relational "
                   "situation must not collapse to a transactional frame (AHI guard; Frontin' case).")
    isUpstream = False
    dependents: list = []

    def __init__(self, t_violate: float, t_satisfy: float):
        self.t_violate, self.t_satisfy = float(t_violate), float(t_satisfy)

    def evaluate(self, context: Context) -> str:
        try:
            # Applies only to relational situations. Applicability is decided upstream
            # (TLC's Narrative invariant / narrative baseline), not by this probe.
            if not context.payload.get("relational_context", False): return NOT_APPLICABLE
            s = context.payload.get("framing_score")
            if s is None or not np.isfinite(float(s)): return NOT_APPLICABLE
            s = float(s)
            if s < self.t_violate: return VIOLATED
            if s >= self.t_satisfy: return SATISFIED
            return AMBIGUOUS
        except Exception:
            return NOT_APPLICABLE

    def repair(self, context: Context) -> RepairAction:
        if self.evaluate(context) == VIOLATED:
            return RepairAction("HALT", "Interpretation collapsed to a transactional frame on a relational situation. "
                                "Emission halted. Provide the narrative baseline (who is involved, what the relationship "
                                "and history are) or confirm a transactional reading is intended.", True)
        return RepairAction("PROMPT_USER", "Framing signal is ambiguous. Is this situation about a relationship, "
                            "obligation or identity, or about price and exchange?", False)

class EvidenceLog:
    """Append-only, hash-chained evidence log (TLC EvidenceEntry)."""
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev = "0" * 64
        if self.path.exists():
            lines = self.path.read_text().strip().splitlines()
            if lines: self.prev = json.loads(lines[-1])["integrity_hash"]

    def append(self, ctx: Context, inv: FramingInvariant, state: str, repair: RepairAction | None):
        entry = {"entry_id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "session_id": ctx.sessionId, "turn": ctx.turn, "invariant_id": inv.id, "state": state,
                 "repair_action": asdict(repair) if repair else None,
                 "runtime_state": "HALTED" if (repair and repair.blocking) else "RUNNING",
                 "framing_score": ctx.payload.get("framing_score")}
        body = json.dumps(entry, sort_keys=True)
        entry["integrity_hash"] = hashlib.sha256((self.prev + body).encode()).hexdigest()
        with self.path.open("a") as f: f.write(json.dumps(entry, sort_keys=True) + "\n")
        self.prev = entry["integrity_hash"]
        return entry

    @staticmethod
    def verify(path) -> bool:
        prev = "0" * 64
        for line in Path(path).read_text().strip().splitlines():
            e = json.loads(line); h = e.pop("integrity_hash")
            if hashlib.sha256((prev + json.dumps(e, sort_keys=True)).encode()).hexdigest() != h: return False
            prev = h
        return True

def calibrate(acts, data, probe_results, tag, outdir="results", target_fpr=0.05):
    z = np.load(acts); rows = [json.loads(l) for l in open(data, encoding="utf-8")][: len(z["ids"])]
    best = json.load(open(probe_results))["best"]; L, pool = best["layer"], best["pool"]
    X = z[pool][:, L - 1]; y = np.array([r["label"] for r in rows]); fam = np.array([r["family"] for r in rows])
    A, B = fam == "A", fam == "B"
    mu_r, mu_t = X[A & (y == 1)].mean(0), X[A & (y == 0)].mean(0)
    w = (mu_r - mu_t); w /= np.linalg.norm(w); mid = (mu_r + mu_t) / 2
    score = (X - mid) @ w
    t_violate = float(np.quantile(score[A & (y == 1)], target_fpr))       # ≤5% of relational A falsely flagged
    t_satisfy = float(np.quantile(score[A & (y == 0)], 1 - target_fpr))   # ≤5% of transactional A falsely passed
    if t_satisfy < t_violate: t_violate = t_satisfy = (t_violate + t_satisfy) / 2
    inv = FramingInvariant(t_violate, t_satisfy)
    states = np.array([inv.evaluate(Context("cal", 0, {"framing_score": float(s), "relational_context": True})) for s in score])
    def rates(mask):
        return {k: round(float(np.mean(states[mask] == k)), 4) for k in (SATISFIED, AMBIGUOUS, VIOLATED)}
    rep = {"model": json.loads(str(z["meta"]))["model"], "layer": L, "pool": pool, "target_fpr": target_fpr,
           "t_violate": t_violate, "t_satisfy": t_satisfy,
           "fit_on": "family A", "evaluated_on": "family B (vocabulary-disjoint)",
           "familyB_relational": rates(B & (y == 1)), "familyB_transactional": rates(B & (y == 0)),
           "familyB_halt_precision": round(float(np.mean(y[B & (states == VIOLATED)] == 0)), 4) if np.any(B & (states == VIOLATED)) else None}
    np.save(f"{outdir}/monitor_{tag}_direction.npy", w); np.save(f"{outdir}/monitor_{tag}_mid.npy", mid)
    json.dump(rep, open(f"{outdir}/monitor_{tag}.json", "w"), indent=2)
    print(json.dumps(rep, indent=2)); return rep

class Monitor:
    """Wraps a model: scores each turn's framing and runs the invariant."""
    def __init__(self, model_id, config_path, evidence_path="results/evidence_demo.jsonl"):
        import torch
        from src.extract import load
        cfg = json.load(open(config_path)); stem = config_path[:-5]
        self.w, self.mid, self.L, self.pool = np.load(stem + "_direction.npy"), np.load(stem + "_mid.npy"), cfg["layer"], cfg["pool"]
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok, self.model, _ = load(model_id, dev, torch.float16 if dev != "cpu" else torch.float32)
        self.inv = FramingInvariant(cfg["t_violate"], cfg["t_satisfy"]); self.log = EvidenceLog(evidence_path)

    def score(self, text):
        import torch
        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt").to(self.model.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states[self.L][0].float()
            h = (hs[-1] if self.pool == "last" else hs.mean(0)).cpu().numpy()
        return float((h - self.mid) @ self.w)

    def step(self, session, turn, text, relational_context):
        ctx = Context(session, turn, {"text": text, "framing_score": self.score(text), "relational_context": relational_context})
        state = self.inv.evaluate(ctx)
        rep = self.inv.repair(ctx) if state in (VIOLATED, AMBIGUOUS) else None
        entry = self.log.append(ctx, self.inv, state, rep)
        if rep: rep.evidencePath = f"{self.log.path}#{entry['entry_id']}"
        return state, rep, entry

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--acts", required=True); c.add_argument("--probe-results", required=True)
    c.add_argument("--tag", required=True); c.add_argument("--data", default="data/prompts_v4.jsonl"); c.add_argument("--outdir", default="results")
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--config", required=True)
    r.add_argument("--text"); r.add_argument("--vignettes", action="store_true"); r.add_argument("--evidence", default="results/evidence_demo.jsonl")
    r.add_argument("--relational", action="store_true", help="mark --text as a relational situation (normally set by the upstream Narrative invariant)")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.acts, a.data, a.probe_results, a.tag, a.outdir); return
    m = Monitor(a.model, a.config, a.evidence)
    items = [(a.text, a.relational)] if a.text else []
    if a.vignettes:
        items += [(json.loads(l)["text"] + " What is going on here?", json.loads(l)["kind"] == "frontin") for l in open("data/frontin_vignettes.jsonl")]
    for i, (t, rel) in enumerate(items):
        state, rep, e = m.step("demo", i, t, rel)
        print(json.dumps({"turn": i, "relational_context": rel, "state": state, "score": round(e["framing_score"], 3), "repair": asdict(rep) if rep else None, "text": t[:90]}))
    print("evidence chain valid:", EvidenceLog.verify(m.log.path))

if __name__ == "__main__":
    main()
