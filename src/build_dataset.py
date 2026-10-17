"""Build the controlled Eight Wonders framing dataset (v4: 8 template pairs per family).

Fixes the v2 confound: in v2 a bag-of-words classifier separates sovereign vs
defensive prompts at 99.6% held-out accuracy, because wording, length and even
scenario text differ between the pair members.

v4 design
- One shared scenario per v2 pair (the scenario sentence(s) before "Analyze" in
  the sovereign version, which keeps the culturally specific details). Both
  framings use the identical scenario.
- The only difference is a one-sentence framing instruction: relational (R) or
  transactional (T).
- Two template families, A and B, whose content words do not overlap. Probes
  train on one family and test on the other; a word-counting classifier cannot
  transfer across disjoint vocabularies, a representation of meaning can.
- Within a family, R and T templates are structurally parallel and length-matched.
Deterministic: no randomness.
"""
import csv, json, re, sys
from pathlib import Path

TEMPLATES = {
    "A": [
        ("Read this as a relationship: who trusted whom, what promise was made, and what is owed now?",
         "Read this as a transaction: who paid whom, what price was set, and what is earned now?"),
        ("Describe the bond at stake here and how the people involved will judge whether it was honored.",
         "Describe the deal at stake here and how the buyers involved will judge whether it was profitable."),
        ("What does this action say to the community about loyalty, respect, and belonging?",
         "What does this action say to the segment about pricing, margins, and conversion?"),
        ("Consider what friendship and gratitude look like in this moment for the neighborhood.",
         "Consider what inventory and turnover look like in this moment for the retailer."),
        ("Whose good faith was tested here, and who was let down?",
         "Whose cash reserves were tested here, and who was charged more?"),
        ("Focus on the care and devotion shown by the regulars.",
         "Focus on the volume and frequency shown by the receipts."),
        ("Speak to the sense of home and family this creates.",
         "Speak to the sense of value and savings this creates."),
        ("Assess whether solidarity among neighbors was strengthened or weakened by this.",
         "Assess whether profit among competitors was strengthened or weakened by this."),
    ],
    "B": [
        ("Interpret it through kinship and memory: which commitment mattered, and how will those affected feel betrayed or affirmed?",
         "Interpret it through demand and supply: which offer mattered, and how will those affected shift spending or stay?"),
        ("Treat this as a covenant: what obligations exist, and what would repair require from the company?",
         "Treat this as a purchase: what costs exist, and what would retention require from the company?"),
        ("Explain the meaning this carries for identity, dignity, and shared history.",
         "Explain the impact this carries for revenue, efficiency, and market share."),
        ("Reflect on the heritage and pride bound up in this choice.",
         "Reflect on the discounts and fees bound up in this choice."),
        ("Who feels seen or erased here, and why would that hurt?",
         "Who gets billed or refunded here, and why would that change?"),
        ("Frame the situation around ancestry, tradition, and collective survival.",
         "Frame the situation around logistics, procurement, and operational throughput."),
        ("Name the grief or hope carried by everyone touched by it.",
         "Name the markup or discount applied to everything sold by it."),
        ("Weigh the reciprocity and tenderness at the heart of this.",
         "Weigh the overhead and liquidity at the heart of this."),
    ],
}

def scenarios(v2_path):
    rows = list(csv.DictReader(open(v2_path, encoding="utf-8")))
    out = []
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        assert a["wonder"] == b["wonder"] and {a["label"], b["label"]} == {"sovereign", "defensive"}
        sov = a if a["label"] == "sovereign" else b
        scen = re.split(r"\s+Analyze\b", sov["prompt"], maxsplit=1)[0].strip()
        out.append({"scenario_id": f"S{i//2:03d}", "wonder": sov["wonder"], "scenario": scen})
    return out

def build(v2_path):
    data = []
    for s in scenarios(v2_path):
        for fam, pairs in TEMPLATES.items():
            for t, (rel, tra) in enumerate(pairs):
                for frame, instr in (("relational", rel), ("transactional", tra)):
                    data.append({**s, "family": fam, "template": f"{fam}{t+1}", "frame": frame,
                                 "label": int(frame == "relational"), "text": f"{s['scenario']} {instr}"})
    return data

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/prompts_v2.csv"
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "data/prompts_v4.jsonl")
    rows = build(src)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} prompts, {len({r['scenario_id'] for r in rows})} scenarios -> {dst}")
