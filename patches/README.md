# llama.cpp patch

`llama.cpp-dry-c.patch` adds the count term to the DRY sampler. It is against
`ggml-org/llama.cpp` at commit `7d7f0287530bc8981690f361d8c1627dbae1c99a`
(2026-11-08).

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 7d7f0287530bc8981690f361d8c1627dbae1c99a
git apply /path/to/unloop/patches/llama.cpp-dry-c.patch

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-sampling llama-server -j
./build/bin/test-sampling
```

What it changes:

| File | Change |
|---|---|
| `src/llama-sampler.cpp` | count-term computation, `max()` composition, count-aware init/clone |
| `src/llama-sampler.h` | testing entry point for the new cases |
| `include/llama.h` | `llama_sampler_init_dry_count` |
| `common/common.h` | `dry_count_min_length`, `dry_count_allowed`, `dry_count_base` |
| `common/arg.cpp` | `--dry-count`, `--dry-count-only`, `--dry-count-min-length`, `--dry-count-allowed`, `--dry-count-base` |
| `common/sampling.cpp` | wire the parameters into the sampler chain |
| `tests/test-sampling.cpp` | four deterministic cases, including a breaker token |

The count term is off by default (`dry_count_min_length = 0`), so
`llama_sampler_init_dry` and every existing DRY configuration behave exactly as
before.

- `--dry-count` is the additive mode with the paper's values: window 4096,
  suffix 24, 2 free recurrences, base 3.0.
- `--dry-count-only` is the tool-call-safe mode: the length term is disabled,
  lambda defaults to 0.8, and the count term uses 5 free recurrences. A single
  exact tool call survives; a breaker-terminated loop is still gone by 16
  repetitions in the teacher-forced test.

`--dry-count-allowed N` overrides `c0` in either mode.
