# Committed results

| File | Produced by | What it contains |
|---|---|---|
| `sim.json` | `bench/sim_bench.py` | analytic cap/unlock grid + the 300-trial lock table |
| `teacher_forced.json` | `bench/teacher_forced_run.py` | Qwen3-4B, `P(loop token)` for R = 2..20, three configs |
| `teacher_forced_qwen3_8b.json` | same | Qwen3-8B sweep |
| `teacher_forced_qwen25_3b.json` | same | Qwen2.5-3B sweep |
| `teacher_forced_deepseek_1.5b.json` | same | DeepSeek-R1-Distill-Qwen-1.5B sweep |
| `llama_dryc.json` | `bench/llama_bench.py` | 16-prompt spontaneous suite, four configs, raw texts |
| `exact_repetition.json` | `bench/exact_repetition.py` | exact-string and tool-call cost, five configs, two seeds |

`logs/` is git-ignored; it holds the server logs from the spontaneous run.
