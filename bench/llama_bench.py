#!/usr/bin/env python3
"""Measure DRY vs DRY-C on a real model through llama-server.

Each configuration is a separate server process, because the sampler is
configured on the command line. Requests are raw completions, so the prompt is
not wrapped in a chat template.

    python3 bench/llama_bench.py --llama-server /path/to/llama-server \
        --model /path/to/model.gguf --out results/llama_dryc.json

The server must be built from the patch in patches/. A stock server rejects
--dry-count-*.
"""

from __future__ import annotations

import argparse
import json
import os
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

CONFIGS = {
    "top-p": [],
    "dry": [
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "2",
        "--dry-penalty-last-n", "4096",
    ],
    "dry-c": [
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "2",
        "--dry-penalty-last-n", "4096",
        "--dry-count",
    ],
    "count-only": [
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "100000",
        "--dry-penalty-last-n", "4096",
        "--dry-count",
    ],
}


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


def run_config(name: str, args: list[str], server: str, model: str, prompts: list[dict],
               n_predict: int, temperature: float, seed: int, log_dir: Path) -> list[dict]:
    port = 18080 + (abs(hash(name)) % 500)
    base = f"http://127.0.0.1:{port}"
    cmd = [
        server, "-m", model, "-ngl", "99", "-c", "8192", "--port", str(port),
        "--no-webui", "-np", "1", "--no-warmup",
    ] + args
    log = (log_dir / f"server-{name}.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        wait_healthy(base, proc)
        for p in prompts:
            t0 = time.time()
            payload = {
                "messages": [{"role": "user", "content": p["prompt"]}],
                "max_tokens": n_predict,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "seed": seed,
                "stream": False,
            }
            try:
                out = http_json(base + "/v1/chat/completions", payload)
            except Exception as e:  # noqa: BLE001
                rows.append({"id": p["id"], "config": name, "error": repr(e)})
                continue
            text = out["choices"][0]["message"]["content"]
            loop = find_text_loop(text, min_unit=40, max_unit=1600, min_repeats=3)
            timings = out.get("timings", {})
            rows.append({
                "id": p["id"],
                "config": name,
                "category": p["category"],
                "requested_repetition": p["requested_repetition"],
                "wall_s": round(time.time() - t0, 3),
                "prompt_tokens": timings.get("prompt_n"),
                "generated_tokens": timings.get("predicted_n"),
                "decode_tok_s": timings.get("predicted_per_second"),
                "loop": None if loop is None else {"unit_chars": loop[0], "repetitions": loop[1]},
                "text": text,
            })
            status = "LOOP" if loop else "ok"
            print(f"  {p['id']:<16} {name:<11} {status}", flush=True)
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
        rr = [r for r in rows if r["config"] == name and "error" not in r]
        real = [r for r in rr if not r["requested_repetition"]]
        requested = [r for r in rr if r["requested_repetition"]]
        speeds = [r["decode_tok_s"] for r in rr if r.get("decode_tok_s")]
        out[name] = {
            "runs": len(rr),
            "loops": sum(1 for r in real if r["loop"]),
            "runs_without_requested_repetition": len(real),
            "loop_rate": (sum(1 for r in real if r["loop"]) / len(real)) if real else None,
            "requested_repetition_loops": sum(1 for r in requested if r["loop"]),
            "requested_repetition_runs": len(requested),
            "mean_decode_tok_s": (sum(speeds) / len(speeds)) if speeds else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "bench" / "prompts.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "llama_dryc.json"))
    ap.add_argument("--n-predict", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    log_dir = Path(args.out).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.configs.split(","):
        print(f"config {name}", flush=True)
        rows.extend(run_config(name, CONFIGS[name], args.llama_server, args.model, prompts,
                               args.n_predict, args.temperature, args.seed, log_dir))

    result = {
        "model": os.path.basename(args.model),
        "llama_server": args.llama_server,
        "n_predict": args.n_predict,
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 40,
        "seed": args.seed,
        "configs": CONFIGS,
        "summary": summarize(rows),
        "runs": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
