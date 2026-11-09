#!/usr/bin/env python3
"""Measure what DRY and DRY-C do to exact, legitimate repetition.

The maintainers' objection to loop samplers is that repetition penalties break
tool calls: a path or identifier that must be repeated exactly gets mutated.
This runs the same prompts under several sampler configurations and counts
exact copies, so the trade-off is measured instead of argued.

    python3 bench/exact_repetition.py --llama-server ... --model ... \
        --out results/exact_repetition.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dryc import find_text_loop  # noqa: E402

PATH = "/usr/local/lib/python3.11/site-packages/numpy/core/include"
ASSIGN = 'CACHE_DIR = "/var/cache/unloop/models"'
CSV_ROW = "abc,123,/opt/data/input.bin"

CONFIGS = {
    "top-p": [],
    "dry-0.05": [
        "--dry-multiplier", "0.05", "--dry-base", "1.75",
        "--dry-allowed-length", "2", "--dry-penalty-last-n", "4096",
    ],
    "dry-0.8": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "2", "--dry-penalty-last-n", "4096",
    ],
    "dry-c": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "2", "--dry-penalty-last-n", "4096",
        "--dry-count",
    ],
    "count-only": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "100000", "--dry-penalty-last-n", "4096",
        "--dry-count",
    ],
    "count-flag": [
        "--dry-count-only",
    ],
    "count-c3": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "100000", "--dry-penalty-last-n", "4096",
        "--dry-count", "--dry-count-allowed", "3",
    ],
    "count-c4": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "100000", "--dry-penalty-last-n", "4096",
        "--dry-count", "--dry-count-allowed", "4",
    ],
    "count-c5": [
        "--dry-multiplier", "0.8", "--dry-base", "1.75",
        "--dry-allowed-length", "100000", "--dry-penalty-last-n", "4096",
        "--dry-count", "--dry-count-allowed", "5",
    ],
}

TASKS = [
    {
        "id": "path_lines",
        "kind": "lines",
        "expected": PATH,
        "copies": 12,
        "prompt": f"Output this exact path 12 times, one per line, and nothing else:\n{PATH}",
    },
    {
        "id": "json_calls",
        "kind": "json_paths",
        "expected": PATH,
        "copies": 10,
        "prompt": (
            "Return a JSON array of 10 objects. Every object must be exactly "
            f'{{"tool":"read","path":"{PATH}"}}. Output only the JSON array.'
        ),
    },
    {
        "id": "code_assign",
        "kind": "lines",
        "expected": ASSIGN,
        "copies": 15,
        "prompt": (
            "Write a Python file containing exactly 15 identical lines:\n"
            f"{ASSIGN}\nOutput only the file contents."
        ),
    },
    {
        "id": "csv_rows",
        "kind": "lines",
        "expected": CSV_ROW,
        "copies": 20,
        "prompt": (
            "Write 20 CSV rows. Every row must be exactly:\n"
            f"{CSV_ROW}\nOutput only the rows."
        ),
    },
    {
        "id": "edit_task",
        "kind": "edit",
        "expected": "def compute_sum(items):",
        "forbidden": "def compute_total(items):",
        "copies": 1,
        "prompt": (
            "Here is a function signature:\n\ndef compute_total(items):\n"
            "Rename it to def compute_sum(items):\nOutput only the new signature line."
        ),
    },
    {
        "id": "tool_once",
        "kind": "json_one",
        "expected": PATH,
        "copies": 1,
        "prompt": (
            f'Call the read tool on {PATH}. Output only JSON of the form '
            f'{{"tool":"read","path":"<exact path>"}}.'
        ),
    },
]


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_healthy(base: str, proc: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            http_json(base + "/health", timeout=2.0)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise TimeoutError("server did not become healthy")


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^```.*$", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    line = line.strip()
    if len(line) > 2 and line[0] == line[-1] == "`":
        line = line[1:-1].strip()
    return line


def score(task: dict, text: str) -> dict:
    kind = task["kind"]
    expected = task["expected"]
    row = {
        "exact": 0,
        "expected": task["copies"],
        "valid": True,
        "mutated_lines": 0,
        "loop": find_text_loop(text, min_unit=40, max_unit=1600, min_repeats=3) is not None,
    }

    if kind == "lines":
        lines = [clean_line(l) for l in text.splitlines() if clean_line(l)]
        row["lines"] = len(lines)
        row["exact"] = sum(1 for l in lines if l == expected)
        row["mutated_lines"] = sum(
            1 for l in lines if l != expected and (expected[:12] in l or l[:12] == expected[:12])
        )
        # a line broken by the tokenizer still counts as a mutation, not a copy

    elif kind == "json_paths":
        match = re.search(r"\[.*\]", text, re.S)
        data = None
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                row["valid"] = False
        if isinstance(data, list):
            row["objects"] = len(data)
            row["exact"] = sum(
                1 for item in data
                if isinstance(item, dict) and item.get("path") == expected
            )
        else:
            # A mutation can make the array invalid JSON. Count the exact path
            # strings anyway, so a broken array does not erase partial copies.
            row["valid"] = False
            paths = re.findall(r'"path"\s*:\s*"([^"]*)"', text)
            row["exact"] = sum(1 for p in paths if p == expected)
            row["objects"] = len(paths)

    elif kind == "json_one":
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            row["valid"] = False
        else:
            try:
                item = json.loads(match.group(0))
            except json.JSONDecodeError:
                row["valid"] = False
                item = {}
            if isinstance(item, dict):
                row["exact"] = int(item.get("path") == expected)

    elif kind == "edit":
        row["exact"] = int(expected in text and task["forbidden"] not in text)
        row["valid"] = row["exact"] == 1

    return row


def run_config(name: str, args: list[str], server: str, model: str, seeds: list[int],
               max_tokens: int, log_dir: Path) -> list[dict]:
    port = 18400 + (abs(hash(name)) % 300)
    base = f"http://127.0.0.1:{port}"
    cmd = [server, "-m", model, "-ngl", "99", "-c", "8192", "--port", str(port),
           "--no-webui", "-np", "1", "--no-warmup"] + args
    log = (log_dir / f"exact-{name}.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        wait_healthy(base, proc)
        for task in TASKS:
            for seed in seeds:
                payload = {
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "seed": seed,
                    "stream": False,
                }
                try:
                    out = http_json(base + "/v1/chat/completions", payload)
                except Exception as e:  # noqa: BLE001
                    rows.append({"task": task["id"], "config": name, "seed": seed, "error": repr(e)})
                    continue
                text = out["choices"][0]["message"]["content"]
                timings = out.get("timings", {})
                row = {
                    "task": task["id"],
                    "config": name,
                    "seed": seed,
                    "text": text,
                    "decode_tok_s": timings.get("predicted_per_second"),
                }
                row.update(score(task, text))
                rows.append(row)
                print(f"  {task['id']:<14} {name:<11} seed={seed} "
                      f"exact={row['exact']}/{row['expected']} valid={row['valid']}", flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in CONFIGS:
        runs = [r for r in rows if r["config"] == name and "error" not in r]
        by_task = {}
        for task in TASKS:
            tr = [r for r in runs if r["task"] == task["id"]]
            exact = sum(r["exact"] for r in tr)
            expected = sum(r["expected"] for r in tr)
            by_task[task["id"]] = {
                "exact": exact,
                "expected": expected,
                "rate": exact / expected if expected else None,
                "valid": sum(1 for r in tr if r.get("valid")) / len(tr) if tr else None,
                "loops": sum(1 for r in tr if r.get("loop")),
            }
        total_exact = sum(v["exact"] for v in by_task.values())
        total_expected = sum(v["expected"] for v in by_task.values())
        speeds = [r["decode_tok_s"] for r in runs if r.get("decode_tok_s")]
        out[name] = {
            "exact": total_exact,
            "expected": total_expected,
            "exact_rate": total_exact / total_expected if total_expected else None,
            "mean_decode_tok_s": sum(speeds) / len(speeds) if speeds else None,
            "tasks": by_task,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "exact_repetition.json"))
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    log_dir = Path(args.out).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.configs.split(","):
        print(f"config {name}", flush=True)
        rows.extend(run_config(name, CONFIGS[name], args.llama_server, args.model, seeds,
                               args.max_tokens, log_dir))

    result = {
        "model": os.path.basename(args.model),
        "llama_server": args.llama_server,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": args.max_tokens,
        "seeds": seeds,
        "configs": CONFIGS,
        "tasks": [{k: v for k, v in t.items() if k != "prompt"} for t in TASKS],
        "summary": summarize(rows),
        "runs": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"\n{'config':<11} {'exact':>10} {'rate':>7} {'valid':>7}")
    for name, s in result["summary"].items():
        n = sum(1 for r in rows if r["config"] == name)
        if n == 0:
            continue
        valid = sum(1 for r in rows if r["config"] == name and r.get("valid")) / n
        print(f"{name:<11} {s['exact']:>4}/{s['expected']:<5} {s['exact_rate']:>7.1%} {valid:>7.1%}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
