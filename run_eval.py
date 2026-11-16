"""Run the full evaluation and write results/report.md.

  uv run python run_eval.py                     # offline: rules_v1 vs rules_v2, heuristic judge
  EVAL_BASE_URL=... EVAL_MODEL=... uv run python run_eval.py --with-llm
                                                # also scores a real model and an LLM judge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evalkit import data, gate, llm, metrics
from evalkit.judges import HeuristicJudge, swap_consistent
from evalkit.systems import SYSTEMS

ROOT = Path(__file__).resolve().parent
FIELDS = ["category", "priority", "product", "sentiment"]


def run_system(fn, tickets):
    rows = []
    for t in tickets:
        s = metrics.score(metrics.parse(fn(t["text"])), t["gold"])
        rows.append({"id": t["id"], "slice": t["slice"], **s})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    return {"valid": df.valid.mean(), "exact": df.exact.mean(), **{f: df[f].mean() for f in FIELDS},
            "slices": df.groupby("slice").exact.mean().to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    if not (ROOT / "data" / "tickets.jsonl").exists():
        data.build(ROOT / "data")
    tickets = data.load_jsonl(ROOT / "data" / "tickets.jsonl")
    pairs = data.load_jsonl(ROOT / "data" / "pairwise.jsonl")

    systems = dict(SYSTEMS)
    if args.with_llm and llm.configured():
        systems["llm"] = llm.llm_extractor
    results = {name: run_system(fn, tickets) for name, fn in systems.items()}
    summ = {name: summary(df) for name, df in results.items()}

    md = ["# Evaluation report\n",
          f"Dataset: {len(tickets)} synthetic support tickets "
          f"({', '.join(f'{k} {v}' for k, v in pd.Series([t['slice'] for t in tickets]).value_counts().items())}), "
          f"plus {len(pairs)} reply pairs with simulated annotator preferences.\n",
          "## 1. Structured extraction\n",
          "| System | Schema valid | Exact match | Category | Priority | Product | Sentiment | Clean | Noisy | Adversarial |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summ.items():
        md.append(f"| {name} | {s['valid']:.1%} | {s['exact']:.1%} | " + " | ".join(f"{s[f]:.1%}" for f in FIELDS) + " | "
                  + " | ".join(f"{s['slices'].get(sl, float('nan')):.1%}" for sl in ["clean", "noisy", "adversarial"]) + " |")

    base, cand = results["rules_v1"], results["rules_v2"]
    a, b = base.exact.to_numpy(), cand.exact.to_numpy()
    diff = metrics.paired_bootstrap(a.astype(float), b.astype(float))
    p = metrics.mcnemar(a, b)
    md += ["\n## 2. Paired comparison: rules_v2 vs rules_v1\n",
           f"- Exact match difference: {diff[0]*100:+.1f} pts (95% paired bootstrap CI {diff[1]*100:+.1f} to {diff[2]*100:+.1f})",
           f"- McNemar exact test: {int(np.sum(~a & b))} items fixed, {int(np.sum(a & ~b))} items broken, p = {p:.2g}",
           "- Paired tests use the same items for both systems, so they need far fewer examples than comparing two unrelated accuracy numbers."]

    md += ["\n## 3. Release gate\n", "| Rule | Pass | Detail |", "|---|---|---|"]
    checks = gate.decide(summ["rules_v1"], summ["rules_v2"], diff)
    md += [f"| {name} | {'✅' if ok else '❌'} | {detail} |" for name, ok, detail in checks]
    md.append(f"\n**Decision: {'SHIP' if all(ok for _, ok, _ in checks) else 'BLOCK'}**")

    judges = [HeuristicJudge()]
    if args.with_llm and llm.configured():
        judges.append(llm.LLMJudge())
    md += ["\n## 4. Judge quality: agreement with annotators and position bias\n",
           "| Judge | Single-order agreement | Single-order Cohen's κ | Verdict flips when replies are swapped | Swap-consistent agreement (on consistent items) |",
           "|---|---|---|---|---|"]
    human = [r["human_preference"] for r in pairs]
    for j in judges:
        single = [j(r["ticket"], r["reply_a"], r["reply_b"]) for r in pairs]
        cons = [swap_consistent(j, r["ticket"], r["reply_a"], r["reply_b"]) for r in pairs]
        kept = [(c, h) for c, h in zip(cons, human) if c is not None]
        flip = 1 - len(kept) / len(pairs)
        agree_single = np.mean([s == h for s, h in zip(single, human)])
        kappa_single = metrics.cohens_kappa(single, human)
        agree_cons = np.mean([c == h for c, h in kept]) if kept else float("nan")
        md.append(f"| {j.name} | {agree_single:.1%} | {kappa_single:.2f} | {flip:.1%} | {agree_cons:.1%} |")
    md.append("\nAnnotator labels here are simulated with 10% label noise, so agreement over all items tops out near "
              "90%. The swap-consistent column can read higher because it excludes items where the judge cannot "
              "separate the replies (its verdict flips with order). With real annotators, report κ alongside raw "
              "agreement, and drop or re-ask items where the verdict flips with position.")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
