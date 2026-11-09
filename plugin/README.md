# opencode-unloop

An [OpenCode](https://opencode.ai) plugin that finds verbatim repetition loops
in model output. It is the TypeScript port of the detector in the
[unloop](../) project: a unit of `minUnit`..`maxUnit` characters repeated at
least `minRepeats` times, back to back.

It gives you three things:

| Surface | What it does |
|---|---|
| `unloop_scan` tool | Scan any text; returns the unit, repetition count, offsets, and a trimmed copy |
| `/unloop` command | Scan the last assistant message, or the text after the command, and post a report |
| idle guard (opt-in) | When a turn ends in a loop that covers ≥25% of the message, leave a synthetic notice once |

The plugin is **post-hoc**: it finds a loop in text OpenCode already received.
The [llama.cpp patch](../patches/llama.cpp-dry-c.patch) is **preventive**: it
removes the loop at sampling time for local models. Use the plugin when the
provider is not yours to patch.

## Install

Local checkout:

```jsonc title="opencode.jsonc"
{
  "plugins": [
    {
      "package": "/absolute/path/to/unloop/plugin",
      "options": { "guard": true }
    },
  ],
}
```

Published package (after `npm publish` in this directory):

```jsonc
{
  "plugins": ["opencode-unloop"],
}
```

OpenCode reloads plugins when the config is saved; `/reload` forces a full
reload. No restart is needed.

## Options

| Option | Default | Meaning |
|---|---:|---|
| `minUnit` | `40` | Smallest repeated unit, in characters |
| `maxUnit` | `1600` | Largest repeated unit, in characters |
| `minRepeats` | `3` | Consecutive copies required |
| `maxChars` | `50000` | Only scan the first N characters |
| `guard` | `false` | Enable the `experimental.idle` notice |

Tool calls can override `minUnit`, `maxUnit`, `minRepeats`, and `maxChars` per
call.

## Use

Ask the agent to check its own answer:

```text
Call unloop_scan on your previous answer before sending it.
```

Or run the command:

```text
/unloop
```

The report looks like:

```text
unloop: loop found — 47 chars repeated 25× (1175 chars, 100% of 1175).
start=0 end=1175
unit: "The quick brown fox jumps over the lazy dog.  \n"
trimmed: 47 chars (1128 removed)
```

## Development

```bash
npm install
npx tsc --noEmit
node --test test/
```

The tests cover the detector, the tool and command wiring, and the idle
guard's coverage threshold and per-message dedupe. No OpenCode server or model
call is needed.

## Limits

- Exact characters only. A unit whose whitespace changes is not the same unit.
- The command scans the last assistant message; it does not stream.
- The idle guard uses the experimental `experimental.idle` hook and only
  reports; it does not ask the model to continue.
- It is a detector, not a sampler. To stop loops at the source, patch the
  server.
