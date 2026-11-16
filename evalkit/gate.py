"""Release gate: a candidate ships only if every rule passes.

Rules (edit to taste):
  - schema validity must not drop
  - exact-match accuracy must improve overall, with the paired 95% CI above zero
  - no slice may regress by more than MAX_SLICE_REGRESSION
  - the adversarial slice must reach MIN_ADVERSARIAL_EXACT
"""
from __future__ import annotations

MAX_SLICE_REGRESSION = 0.01
MIN_ADVERSARIAL_EXACT = 0.80


def decide(baseline: dict, candidate: dict, diff_ci: tuple[float, float, float]) -> list[tuple[str, bool, str]]:
    checks = []
    checks.append(("schema validity does not drop", candidate["valid"] >= baseline["valid"],
                   f"{baseline['valid']:.1%} → {candidate['valid']:.1%}"))
    d, lo, hi = diff_ci
    checks.append(("exact match improves (95% CI above zero)", lo > 0, f"{d*100:+.1f} pts (95% CI {lo*100:+.1f} to {hi*100:+.1f})"))
    for s in baseline["slices"]:
        delta = candidate["slices"][s] - baseline["slices"][s]
        checks.append((f"no regression on '{s}' slice", delta >= -MAX_SLICE_REGRESSION,
                       f"{baseline['slices'][s]:.1%} → {candidate['slices'][s]:.1%}"))
    adv = candidate["slices"].get("adversarial", 1.0)
    checks.append((f"adversarial exact match ≥ {MIN_ADVERSARIAL_EXACT:.0%}", adv >= MIN_ADVERSARIAL_EXACT, f"{adv:.1%}"))
    return checks
