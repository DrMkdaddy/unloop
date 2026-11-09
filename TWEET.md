# Tweet drafts

Numbers are from `results/teacher_forced.json` and `results/sim.json`. Keep the
claim to what the repo measured: the count term is implemented, tested, and
isolated on one model; the paper's full study is larger.

## Short

> DRY's repetition penalty stops growing at a sequence breaker. A line ending
> in `\n` can loop forever while DRY reports no penalty at all.
>
> I implemented the count term from arXiv:2610.24715 in llama.cpp.
>
> Qwen3-4B, same logits, 20 repetitions:
> DRY: P(loop token) = 0.998
> DRY-C: out of the top 20
>
> Patch + tests: https://github.com/DrMkdaddy/dry-c-lab

## Technical thread

1/ DRY penalizes `λ·b^(match_length − allowed)`. But the match length is capped
by sequence breakers. If the loop unit ends in `\n` or `:`, the penalty resets
every iteration — and a locked position stays locked.

2/ arXiv:2610.24715 adds a second signal: count how many times a token followed
an earlier occurrence of the current suffix. `d = λ·b_c^(count − 2)`. Breakers
do not cap a count.

3/ I put it in llama.cpp's DRY sampler. Off by default; `--dry-count` enables
the paper's values (W=4096, a'=24, c0=2, b_c=3). The two terms combine with
`max()`, not addition.

4/ The measurement is teacher-forced, so sampling luck is not involved: repeat
`k: v\n` R times, one forward pass, look at the probability of `k`.

R=20: DRY leaves it at 0.998. DRY-C pushes it out of the top 20. The crossover
is between 8 and 10 repetitions, where the count crosses the model's margin.

5/ In the paper's `(n, l)` state machine, DRY's cap at `g=6` is 4.29 and the
token's margin is 4.815, so DRY can never escape. The count term reaches 4.815
at the fourth recurrence. 300/300 trials.

6/ Patch is +237/−8 over 7 files against llama.cpp `7d7f028`, with four
deterministic sampler tests. The breaker case is the point: the length term
skips the token, the count term does not.

Repo: https://github.com/DrMkdaddy/dry-c-lab

## One number

> DRY: 0.998 after 20 repetitions.
> DRY-C: outside the top 20 at the same state.
>
> The difference is one uncapped count in the sampler.
> https://github.com/DrMkdaddy/dry-c-lab
