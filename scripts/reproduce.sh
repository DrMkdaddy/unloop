#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m unittest discover -s tests -v
python3 bench/sim_bench.py --out results/sim.json

if [[ -n "${LLAMA_SERVER:-}" && -n "${MODEL:-}" ]]; then
    python3 bench/llama_bench.py \
        --llama-server "$LLAMA_SERVER" \
        --model "$MODEL" \
        --out results/llama_dryc.json
else
    echo "set LLAMA_SERVER and MODEL to also run the real-model benchmark"
fi
