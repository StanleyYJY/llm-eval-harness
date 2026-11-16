# Evaluation report

Dataset: 500 synthetic support tickets (clean 200, noisy 150, adversarial 150), plus 160 reply pairs with simulated annotator preferences.

## 1. Structured extraction

| System | Schema valid | Exact match | Category | Priority | Product | Sentiment | Clean | Noisy | Adversarial |
|---|---|---|---|---|---|---|---|---|---|
| rules_v1 | 100.0% | 64.2% | 90.2% | 88.6% | 89.4% | 92.4% | 97.0% | 58.7% | 26.0% |
| rules_v2 | 100.0% | 86.6% | 99.6% | 96.4% | 91.8% | 98.6% | 97.0% | 70.0% | 89.3% |

## 2. Paired comparison: rules_v2 vs rules_v1

- Exact match difference: +22.4 pts (95% paired bootstrap CI +18.8 to +26.2)
- McNemar exact test: 112 items fixed, 0 items broken, p = 3.9e-34
- Paired tests use the same items for both systems, so they need far fewer examples than comparing two unrelated accuracy numbers.

## 3. Release gate

| Rule | Pass | Detail |
|---|---|---|
| schema validity does not drop | ✅ | 100.0% → 100.0% |
| exact match improves (95% CI above zero) | ✅ | +22.4 pts (95% CI +18.8 to +26.2) |
| no regression on 'adversarial' slice | ✅ | 26.0% → 89.3% |
| no regression on 'clean' slice | ✅ | 97.0% → 97.0% |
| no regression on 'noisy' slice | ✅ | 58.7% → 70.0% |
| adversarial exact match ≥ 80% | ✅ | 89.3% |

**Decision: SHIP**

## 4. Judge quality: agreement with annotators and position bias

| Judge | Single-order agreement | Single-order Cohen's κ | Verdict flips when replies are swapped | Swap-consistent agreement (on consistent items) |
|---|---|---|---|---|
| heuristic_rubric | 85.6% | 0.71 | 14.4% | 92.0% |

Annotator labels here are simulated with 10% label noise, so agreement over all items tops out near 90%. The swap-consistent column can read higher because it excludes items where the judge cannot separate the replies (its verdict flips with order). With real annotators, report κ alongside raw agreement, and drop or re-ask items where the verdict flips with position.
