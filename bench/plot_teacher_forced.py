#!/usr/bin/env python3
"""Plot the teacher-forced sweep from results/teacher_forced.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

FLOOR = 1e-5
STYLE = {
    "top-p": ("#8a8a8a", "o", "-"),
    "dry": ("#d1495b", "s", "-"),
    "dry-c": ("#1b6ca8", "^", "-"),
    "count-only": ("#2a9d8f", "D", "--"),
}


def main() -> int:
    data = json.loads((ROOT / "results" / "teacher_forced.json").read_text())
    reps = [r["reps"] for r in data["rows"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in ("top-p", "dry", "dry-c"):
        color, marker, ls = STYLE[name]
        ys = [max(r[name]["loop_prob"], FLOOR) for r in data["rows"]]
        censored = [r[name]["loop_prob"] == 0.0 for r in data["rows"]]
        lw = 4.0 if name == "top-p" else 1.6
        alpha = 0.45 if name == "top-p" else 1.0
        ax.plot(reps, ys, ls, color=color, marker=marker, ms=5, lw=lw, alpha=alpha,
                label={"top-p": "top-p only", "dry": "DRY", "dry-c": "DRY-C"}[name])
        if any(censored):
            ax.scatter([x for x, c in zip(reps, censored) if c],
                       [y for y, c in zip(ys, censored) if c],
                       facecolors="none", edgecolors=color, marker=marker, s=46, zorder=4)

    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 1.4)
    ax.set_xticks(reps)
    ax.set_xlabel("repetitions of the unit `k: v\\n` in the prompt")
    ax.set_ylabel("P(next token = `k`)")
    ax.set_title("DRY's penalty is capped by the trailing newline; the count term is not")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    ax.annotate("outside top 20", xy=(12, FLOOR), xytext=(12.6, FLOOR * 3),
                fontsize=8, color=STYLE["dry-c"][0],
                arrowprops=dict(arrowstyle="-", color=STYLE["dry-c"][0], lw=0.8))

    fig.tight_layout()
    out = ROOT / "figures" / "teacher_forced.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
