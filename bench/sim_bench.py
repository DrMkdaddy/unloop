#!/usr/bin/env python3
"""Reproduce the lock/escape behaviour of DRY and DRY-C.

Part 1 is analytic: DRY's length penalty is capped by the longest breaker-free
run g, while the count term grows with the number of recurrences. Part 2 is a
Monte Carlo over the same state machine, so the table is produced by sampling,
not by asserting the formula twice.

    python3 bench/sim_bench.py --out results/sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from dryc import DRYParams  # noqa: E402
from dryc.sim import dry_cap, run_trial, summarize, unlock_recurrence  # noqa: E402

PARAMS = DRYParams(
    multiplier=0.8,
    base=1.75,
    allowed_length=2,
    penalty_last_n=4096,
    count_min_length=24,
    count_allowed=2,
    count_base=3.0,
)

PI = 0.9          # top-p threshold used by the paper's lock criterion
Q = 0.999         # raw probability of the loop token
R = 0.0009        # raw probability of the best alternative
G = 6             # longest breaker-free run, the case the paper calls the blind spot


def delta_for(q: float, r: float, pi: float) -> float:
    return math.log(q / r) - math.log(pi / (1.0 - pi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "sim.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    analytic = []
    for g in range(2, 9):
        cap = dry_cap(PARAMS, g)
        for delta in (1.0, 2.0, 5.0, 10.0, 20.0, 37.6):
            analytic.append({
                "g": g,
                "delta": delta,
                "dry_penalty": cap,
                "dry_locked_forever": cap < delta,
                "count_unlock_recurrence": unlock_recurrence(delta, PARAMS),
            })

    delta = delta_for(Q, R, PI)
    mc = {"q": Q, "r": R, "pi": PI, "delta": delta, "g": G, "max_steps": args.steps,
          "trials": args.trials, "modes": {}}
    for mode in ("top-p", "dry", "count", "dry-c"):
        trials = [run_trial(Q, R, PI, PARAMS, mode, G, args.steps, random.Random(seed))
                  for seed in range(args.trials)]
        mc["modes"][mode] = summarize(trials)

    result = {
        "paper": "arXiv:2610.24715v2",
        "params": PARAMS.__dict__ if hasattr(PARAMS, "__dict__") else {
            "multiplier": PARAMS.multiplier, "base": PARAMS.base,
            "allowed_length": PARAMS.allowed_length, "penalty_last_n": PARAMS.penalty_last_n,
            "count_min_length": PARAMS.count_min_length,
            "count_allowed": PARAMS.count_allowed, "count_base": PARAMS.count_base,
        },
        "analytic": analytic,
        "monte_carlo": mc,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"delta = {delta:.3f}  (q={Q}, r={R}, top-p={PI}), g={G}")
    print(f"{'mode':<8} {'locked':>8} {'escaped':>8} {'median step':>12}")
    for mode, s in mc["modes"].items():
        print(f"{mode:<8} {s['locked_fraction']:>8.1%} {s['escaped']:>8} "
              f"{str(s['median_escape_step']):>12}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
