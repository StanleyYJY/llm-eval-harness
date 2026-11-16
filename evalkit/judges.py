"""Pairwise judges for open-ended replies.

A judge sees a ticket and two replies and returns "A" or "B". The harness always asks twice,
with the replies swapped, and only counts a verdict when both orders agree. That removes the
position bias most LLM judges have.

  HeuristicJudge   offline baseline: rubric scored by simple text checks
  LLMJudge         optional: any chat-completions-compatible endpoint (see evalkit/llm.py)
"""
from __future__ import annotations

import re

RUBRIC = """You are grading two customer-support replies to the same ticket.
Score each reply on: (1) does it address the customer's specific issue, (2) does it give a
concrete next step with a time frame, (3) is it polite. Issue and next step matter most.
Answer with a single letter: A or B."""


class HeuristicJudge:
    name = "heuristic_rubric"

    def score(self, reply: str) -> float:
        s = 0.0
        s += 2.0 if re.search(r"sorry about the \w+ issue", reply, re.I) else 0.0
        s += 1.5 if re.search(r"within \d+ hours|escalated", reply, re.I) else 0.0
        s += 0.3 if reply.lower().startswith("thanks") else 0.0
        return s

    def __call__(self, ticket: str, first: str, second: str) -> str:
        a, b = self.score(first), self.score(second)
        if a == b:
            return "A"          # ties resolve to the first slot, which creates position bias on purpose
        return "A" if a > b else "B"


def swap_consistent(judge, ticket: str, a: str, b: str) -> str | None:
    """Ask in both orders. Return A/B if consistent, None if the verdict flips with position."""
    v1 = judge(ticket, a, b)
    v2 = judge(ticket, b, a)
    v2_mapped = "A" if v2 == "B" else "B"
    return v1 if v1 == v2_mapped else None
