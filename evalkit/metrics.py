"""Scoring and statistics used by the harness."""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from .data import CATEGORIES, PRIORITIES, PRODUCTS, SENTIMENTS

SCHEMA = {"category": CATEGORIES, "priority": PRIORITIES, "product": PRODUCTS, "sentiment": SENTIMENTS}


def parse(raw: str) -> dict | None:
    """Return the parsed object if it is valid against the schema, else None."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(SCHEMA):
        return None
    if any(obj[k] not in allowed for k, allowed in SCHEMA.items()):
        return None
    return obj


def score(pred: dict | None, gold: dict) -> dict:
    """Per-item scores: schema validity, per-field correctness, exact match."""
    out = {"valid": pred is not None}
    for k in SCHEMA:
        out[k] = bool(pred is not None and pred[k] == gold[k])
    out["exact"] = all(out[k] for k in SCHEMA)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference b - a with a 95% percentile bootstrap interval over items."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired binary outcomes; returns the two-sided p-value."""
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(only_b, only_a + only_b, 0.5).pvalue)


def cohens_kappa(x: list[str], y: list[str]) -> float:
    labels = sorted(set(x) | set(y))
    xa, ya = np.array(x), np.array(y)
    po = float(np.mean(xa == ya))
    pe = sum(float(np.mean(xa == l)) * float(np.mean(ya == l)) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
