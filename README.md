# unloop

[![test](https://github.com/DrMkdaddy/unloop/actions/workflows/test.yml/badge.svg)](https://github.com/DrMkdaddy/unloop/actions/workflows/test.yml)

**A count term for repetition loops that DRY cannot break.** One llama.cpp
patch, one OpenCode plugin, and the measurements that justify them.

DRY penalizes a token by `λ · b^(match_length − allowed)`. The match length
stops at a sequence breaker. A loop whose unit ends in `\n` or `:` therefore
gets no penalty at all, and a position that is already locked stays locked.
Raising `dry_base` does not help: it raises a number that is still capped.

DRY-C ([arXiv:2610.24715](https://arxiv.org/abs/2610.24715)) adds a second
penalty built from how many times a token followed an earlier copy of the
current suffix. Sequence breakers do not cap a count. The two penalties
combine with `max()`, so DRY never gets weaker than it is today.

## Why it matters

Teacher-forced on four instruction models, 1.5B to 8B. The prompt is `k: v\n`
repeated R times, ending on a unit boundary; the next token should be `k`. The
trailing newline is a DRY breaker, so the length term is silent.

At 20 repetitions, `P(next token = k)`:

| model | DRY | unloop (DRY-C) |
|---|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M | 0.990 | outside top 20 |
| Qwen2.5-3B-Instruct Q6_K | 0.997 | outside top 20 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 0.999 | outside top 20 |
| Qwen3-8B Q4_K_M | 0.999 | outside top 20 |

All four cross between 8 and 10 repetitions and are out of the top 20 by 12.
The 4B detail:

| repetitions | DRY | unloop (DRY-C) |
|---:|---:|---:|
| 2 | 0.796 | 0.796 |
| 8 | 0.996 | 0.991 |
| 10 | 0.993 | **0.092** |
| 12 | 0.998 | outside top 20 |
| 20 | 0.999 | outside top 20 |

DRY does not move across the sweep. Same weights, same logits, one extra term.

![Teacher-forced sweep](figures/teacher_forced.png)

The paper's lock model gives the same shape. With a breaker-free run of 6, the
token margin is 4.815 and DRY's cap is 4.29, so DRY never unlocks; the count
term reaches the margin at the fourth recurrence. In 300 simulated trials:
DRY locked 100%, unloop locked 0%, median escape step 4.

## Use it

### If you run llama.cpp locally

Apply the patch, build, and turn the term on:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 7d7f0287530bc8981690f361d8c1627dbae1c99a
git apply /path/to/unloop/patches/llama.cpp-dry-c.patch

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server test-sampling -j

./build/bin/llama-server -m model.gguf \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 \
  --dry-penalty-last-n 4096 --dry-count
```

`--dry-count` uses the paper's values: window 4096, suffix 24, 2 free
recurrences, base 3. The count term is off by default, so existing DRY
configs are unchanged. `--dry-allowed-length 100000 --dry-count` is the
paper's count-only variant.

### If the model is not yours to patch

Install the OpenCode plugin. It finds loops in text OpenCode already received:

```jsonc
{
  "plugins": [
    { "package": "/absolute/path/to/unloop/plugin", "options": { "guard": true } },
  ],
}
```

It adds an `unloop_scan` tool (`tools.scan` in Code Mode), a `/unloop` command,
and an opt-in idle guard.
The plugin is post-hoc detection; the server patch is prevention.

## What is in the repo

| Path | What it is |
|---|---|
| `patches/llama.cpp-dry-c.patch` | +237/−8 across 7 files, against `7d7f028` |
| `dryc/sampler.py` | dependency-free reference for the length and count terms |
| `dryc/sim.py` | the paper's `(n, l)` lock state machine |
| `plugin/` | OpenCode plugin: `unloop_scan`, `/unloop`, idle guard |
| `bench/teacher_forced.cpp` | one-pass real-model measurement |
| `bench/llama_bench.py` | 16-prompt spontaneous suite through `llama-server` |
| `results/` | raw JSON, generated texts, and the plotting data |

## Verify

```bash
# Python reference: 12 tests
python3 -m unittest discover -s tests -v

# C++ sampler: existing cases plus 4 new ones
./build/bin/test-sampling

# Plugin: 11 tests and a typecheck
cd plugin && npm ci && npx tsc --noEmit && node --test test/*.test.ts

# Plugin end to end (one model call)
cd plugin && ./scripts/e2e.sh

# Rebuild the simulation and the teacher-forced sweep
./scripts/reproduce.sh
```

CI runs the Python tests, the simulation, the plugin typecheck, and the plugin
tests. Both jobs are green on `main`.

The plugin was also tested end to end: in a real OpenCode session, a model
read a file containing one line repeated five times, called the tool, and
reported the unit length (47) and repetition count (5). There is no unit test
standing in for that path.

## Limits

- The teacher-forced numbers are four models on one GPU. They demonstrate the
  mechanism and its consistency, not the paper's 1,800-generation loop-count
  study.
- The 16-prompt spontaneous suite produced **zero** loops on the 4B model at
  temperature 1.0, so it does not separate the configs. The teacher-forced
  test is the evidence; the suite is a sanity check.
- The patch is proposed upstream, not merged. The plugin is in this repo, not
  on npm.
- The plugin matches exact characters. A unit whose whitespace changes is a
  different unit, and the idle guard does not stream.
- DRY-C makes requested verbatim repetition harder. On the repetition control,
  top-p repeated a sentence 25 times, count-only stopped at 9, and DRY/DRY-C
  mutated it into near-copies. That cost is real.

## Citation

```bibtex
@article{chen2026loops,
  title   = {When Do Repetition Loops Lock? Truncation Sampling, Bounded
             Penalties, and a Blind Spot in DRY},
  author  = {Chen, Yichuan and Wang, Xin},
  journal = {arXiv preprint arXiv:2610.24715},
  year    = {2026}
}
```
