#!/usr/bin/env python3
"""Run the teacher-forced C++ driver over a sweep of repetition counts.

    python3 bench/teacher_forced_run.py --binary bin-teacher-forced \
        --model model.gguf --out results/teacher_forced.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UNIT = "k: v\n"
REPS = [2, 4, 6, 8, 10, 12, 16, 20]


def run(binary: str, model: str, unit: str, reps: int, top_k: int = 20,
        count_allowed: int = 2) -> dict:
    out = subprocess.run(
        [binary, model, unit, str(reps), str(top_k), str(count_allowed)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "teacher_forced.json"))
    ap.add_argument("--unit", default=UNIT)
    ap.add_argument("--reps", default=",".join(str(r) for r in REPS))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--count-allowed", type=int, default=2)
    args = ap.parse_args()

    reps_list = [int(r) for r in args.reps.split(",")]
    runs = [run(args.binary, args.model, args.unit, r, args.top_k, args.count_allowed)
            for r in reps_list]

    # the loop token is the top-p favourite at the largest repetition count
    last = runs[-1]
    top_p = next(c for c in last["configs"] if c["name"] == "top-p")
    loop = top_p["top"][0]
    loop_id = loop["id"]

    rows = []
    for r, run_data in zip(reps_list, runs):
        row = {"reps": r, "tokens": run_data["tokens"], "loop_token": loop["token"]}
        for cfg in run_data["configs"]:
            probs = {t["id"]: t for t in cfg["top"]}
            hit = probs.get(loop_id)
            row[cfg["name"]] = {
                "loop_prob": hit["prob"] if hit else 0.0,
                "loop_rank": (cfg["top"].index(hit) + 1) if hit else None,
                "top_token": cfg["top"][0]["token"],
                "top_prob": cfg["top"][0]["prob"],
            }
        rows.append(row)

    result = {
        "model": Path(args.model).name,
        "unit": args.unit,
        "loop_token_id": loop_id,
        "loop_token": loop["token"],
        "count_allowed": args.count_allowed,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"loop token {loop['token']!r} (id {loop_id})")
    print(f"{'reps':>5} " + " ".join(f"{n:>20}" for n in ("top-p", "dry", "dry-c", "count-only")))
    for row in rows:
        cells = []
        for n in ("top-p", "dry", "dry-c", "count-only"):
            d = row[n]
            rank = f"#{d['loop_rank']}" if d["loop_rank"] else "out"
            cells.append(f"{d['loop_prob']:.6f} {rank:>4}")
        print(f"{row['reps']:>5} " + " ".join(f"{c:>20}" for c in cells))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
