"""A two-token model of the loop-locking experiment.

The paper reduces a loop to the state (n, l): the context already contains n
copies of a unit and the model is about to emit token l of copy n+1. Along the
loop this state recurs with the same next-token distribution, so a loop is
locked exactly when the sampler cannot escape it.

This module keeps that state and samples a 2-token distribution (the loop token
and the best alternative), with top-p applied afterwards. It is a model, not a
measurement; the numbers it produces reproduce the shape of Propositions 5 and
Corollary 4 of arXiv:2610.24715.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .sampler import DRYParams


def logit(p: float) -> float:
    return math.log(max(p, 1e-300))


@dataclass
class Trial:
    mode: str
    escaped: bool
    escape_step: Optional[int]   # loop tokens emitted before the alternative won


def _penalty(mode: str, c: int, g: int, p: DRYParams) -> float:
    """Penalty on the loop token at recurrence count c.

    ``g`` is the length of the longest breaker-free run, i.e. the cap the paper
    derives for DRY's length term: lambda * b ** (g - 1 - a).
    """
    d = 0.0
    if mode in ("dry", "dry-c"):
        exp = g - 1 - p.allowed_length
        if exp >= 0:
            d = p.multiplier * p.base ** exp
    if mode in ("count", "dry-c") and p.count_enabled and c >= p.count_allowed:
        d = max(d, p.multiplier * p.count_base ** (c - p.count_allowed))
    return d


def run_trial(q: float, r: float, pi: float, p: DRYParams, mode: str, g: int,
              max_steps: int = 200, rng: Optional[random.Random] = None) -> Trial:
    rng = rng or random.Random(0)
    c = 0
    for step in range(max_steps):
        d = _penalty(mode, c, g, p)
        z_loop = logit(q) - d
        z_alt = logit(r)
        m = max(z_loop, z_alt)
        e_loop = math.exp(z_loop - m)
        e_alt = math.exp(z_alt - m)
        total = e_loop + e_alt
        p_loop = e_loop / total

        # top-p: the loop token is in the nucleus iff p_loop >= pi (it is the
        # top token before any penalty, and remains the top token here unless
        # the penalty pushes it below the alternative).
        loop_is_top = p_loop >= e_alt / total
        if loop_is_top and p_loop >= pi:
            # nucleus contains only the loop token: the position is locked
            c += 1
            continue
        if not loop_is_top and (e_alt / total) >= pi:
            return Trial(mode, True, step)

        # nucleus has both tokens (or only the alternative); sample
        # renormalized over the two
        nuc = e_loop + e_alt
        if rng.random() < e_loop / nuc:
            c += 1
        else:
            return Trial(mode, True, step)
    return Trial(mode, False, None)


def summarize(trials: List[Trial]) -> Dict[str, object]:
    escaped = [t for t in trials if t.escaped]
    steps = sorted(t.escape_step for t in escaped)
    median = steps[len(steps) // 2] if steps else None
    return {
        "trials": len(trials),
        "escaped": len(escaped),
        "locked_fraction": 1.0 - len(escaped) / len(trials),
        "median_escape_step": median,
        "min_escape_step": steps[0] if steps else None,
        "max_escape_step": steps[-1] if steps else None,
    }


def dry_cap(p: DRYParams, g: int) -> float:
    exp = g - 1 - p.allowed_length
    return p.multiplier * p.base ** exp if exp >= 0 else 0.0


def count_at(p: DRYParams, c: int) -> float:
    if not p.count_enabled or c < p.count_allowed:
        return 0.0
    return p.multiplier * p.count_base ** (c - p.count_allowed)


def unlock_recurrence(delta: float, p: DRYParams) -> Optional[int]:
    """Smallest c with count penalty >= delta, or None if it never gets there."""
    if not p.count_enabled:
        return None
    c = p.count_allowed
    while c < 10_000:
        if count_at(p, c) >= delta:
            return c
        c += 1
    return None
