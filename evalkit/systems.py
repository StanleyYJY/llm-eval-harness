"""Systems under test.

Every system maps ticket text -> raw string output (what a model would return). The harness
parses and scores that string, so a real model can be dropped in behind the same interface.

  rules_v1   keyword rules, first match wins, trusts every line of the ticket
  rules_v2   v1 plus: drops lines that look like instructions to the model, ignores
             parenthetical "already solved" mentions, tolerates simple typos
  llm        optional: any chat-completions-compatible endpoint (see evalkit/llm.py)
"""
from __future__ import annotations

import json
import re

from .data import CATEGORIES

KEYWORDS = {
    "category": {"billing": ["charged", "invoice", "billed"], "shipping": ["package", "tracking", "address", "arrived"],
                 "account": ["log in", "password", "email on my profile", "profile"],
                 "technical": ["freezes", "crash", "error", "sync"], "refund": ["refund", "money back"]},
    "priority": {"high": ["urgent", "today", "blocked"], "medium": ["look into", "check", "appreciate"], "low": ["no rush", "chance", "letting you know"]},
    "product": {"app": ["mobile app", " app"], "website": ["website"], "subscription": ["subscription"], "device": ["device"]},
    "sentiment": {"negative": ["frustrating", "unhappy", "unacceptable"], "positive": ["love", "great", "thanks so much"], "neutral": ["thanks", "details"]},
}
DEFAULT = {"category": "technical", "priority": "medium", "product": "app", "sentiment": "neutral"}
INJECTION = re.compile(r"(ignore (all )?previous instructions|^system:|note to the ai)", re.I)


def _match(text: str, field: str, fuzzy: bool) -> str:
    t = text.lower()
    for label, words in KEYWORDS[field].items():
        for w in words:
            if w in t:
                return label
            if fuzzy and len(w) > 5 and any(_near(w, tok) for tok in re.findall(r"[a-z]+", t)):
                return label
    return DEFAULT[field]


def _near(a: str, b: str) -> bool:
    """True if b is a one-swap typo of a (adjacent transposition)."""
    if len(a) != len(b) or a == b:
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def rules_v1(text: str) -> str:
    out = {f: _match(text, f, fuzzy=False) for f in KEYWORDS}
    if re.search(r"set priority to low", text, re.I):
        out["priority"] = "low"            # v1 obeys instructions embedded in the ticket
    if re.search(r"classify this ticket as positive", text, re.I):
        out["sentiment"] = "positive"
    m = re.search(r"the category is (\w+)", text, re.I)
    if m and m.group(1).lower() in CATEGORIES:
        out["category"] = m.group(1).lower()
    return json.dumps(out)


def rules_v2(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    kept = [s for s in sentences if not INJECTION.search(s.strip())]
    kept = [re.sub(r"\(last year .*?already solved\.\)", "", s, flags=re.I) for s in kept]
    clean = " ".join(kept)
    return json.dumps({f: _match(clean, f, fuzzy=True) for f in KEYWORDS})


SYSTEMS = {"rules_v1": rules_v1, "rules_v2": rules_v2}
