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

## Exact repetition is the cost

The reason maintainers resist loop samplers is that repetition penalties break
tool calls. Measured on the same server: six tasks that require exact strings,
two seeds, temperature 0.2, exact copies counted.

| config | exact copies | single tool call |
|---|---:|---:|
| top-p | 118/118 | 2/2 |
| DRY, λ=0.05 | 72/118 | 0/2 |
| DRY, λ=0.8 | 32/118 | 0/2 |
| DRY-C (both terms) | 28/118 | 0/2 |
| count-only, c0=2 | 76/118 | 2/2 |
| count-only, c0=3 | 92/118 | 2/2 |
| count-only, c0=4 | 96/118 | 2/2 |
| **count-only, c0=5** | **104/118** | **2/2** |

The length term is the tool-call problem. `llama-server` seeds the sampler with
the prompt tokens (`tools/server/server-context.cpp`, the
`common_sampler_accept(..., false)` call), so a path that appears in the prompt
is already in DRY's history. Re-emitting it once is penalized; even λ=0.05
mutates the path. The count term is silent until the third recurrence, and with
`c0=5` a single exact tool call survives.

Raising `c0` buys exactness and delays loop escape:

| c0 | exact copies | loop token out at |
|---:|---:|---:|
| 2 | 64.4% | 12 repetitions |
| 3 | 78.0% | 12 repetitions |
| 4 | 81.4% | 16 repetitions |
| 5 | 88.1% | 16 repetitions |

`c0=5` is the tool-safe default: one exact call is preserved, 88% of exact
copies survive, and a breaker-terminated loop is still gone by 16 repetitions.
Lower `--dry-count-allowed` if you would rather break loops sooner and accept
more tool-call damage.

No setting is free. Count-only still breaks long requested repetition (18/40
CSV rows, 12/20 JSON paths, and the array becomes invalid once it mutates), and
the additive DRY-C is the worst of both worlds for exactness: 26/30 on the
quoted assignment line, against 30/30 for plain DRY. The defensible upstream
variant is count-only with a tuned `c0`, not an additive term.

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

For tool-heavy use, the recommended mode is count-only with five free
recurrences:

```bash
./build/bin/llama-server -m model.gguf --dry-count-only
```

`--dry-count-only` disables the length term, sets λ=0.8, and uses window 4096,
suffix 24, c0=5, base 3. `--dry-count` keeps the additive mode with the paper's
c0=2; use it when breaking loops as early as possible matters more than exact
repetition. The count term is off by default, so existing DRY configs are
unchanged.

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
| `patches/llama.cpp-dry-c.patch` | +251/−8 across 7 files, against `7d7f028` |
| `dryc/sampler.py` | dependency-free reference for the length and count terms |
| `dryc/sim.py` | the paper's `(n, l)` lock state machine |
| `plugin/` | OpenCode plugin: `unloop_scan`, `/unloop`, idle guard |
| `bench/teacher_forced.cpp` | one-pass real-model measurement |
| `bench/llama_bench.py` | 16-prompt spontaneous suite through `llama-server` |
| `bench/exact_repetition.py` | tool-call / exact-string cost of each sampler config |
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

# Exact-string cost (needs llama-server built from the patch)
python3 bench/exact_repetition.py \
  --llama-server /path/to/llama-server --model model.gguf

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
- DRY-C with both terms makes requested verbatim repetition harder; count-only
  is the variant that preserves a single exact tool call. The exactness table
  above is the trade-off, not a footnote.

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
