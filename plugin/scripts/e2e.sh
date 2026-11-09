#!/usr/bin/env bash
# End-to-end check: load the plugin in a real OpenCode project, have a model
# read a file with a known 5x loop, and assert that the tool reports 47/5.
#
# Needs OpenCode with a tool-calling model configured. It makes one model call.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT

python3 - "$HERE" "$DIR" <<'PY'
import pathlib, sys

here, out = sys.argv[1], pathlib.Path(sys.argv[2])
line = "The quick brown fox jumps over the lazy dog.  \n"
(out / "loop.txt").write_text(line * 5)
(out / "opencode.jsonc").write_text(
    '{\n'
    '  "$schema": "https://opencode.ai/config.json",\n'
    '  "plugins": [{"package": "%s", "options": {}}]\n'
    '}\n' % here
)
PY

cd "$DIR"
opencode run --auto --format json \
  "Read loop.txt, call the unloop_scan tool on its exact contents, then report the unit length and repetition count." \
  > run.jsonl

python3 - <<'PY'
import json

events = [json.loads(line) for line in open("run.jsonl") if line.strip()]

def is_result(value):
    return isinstance(value, dict) and value.get("unitLength") == 47 and value.get("repetitions") == 5

ok = False
for event in events:
    if event.get("type") != "tool_use":
        continue
    output = (event.get("part", {}).get("state") or {}).get("output")
    if not isinstance(output, str):
        continue

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None

    if is_result(parsed):
        ok = True
        break
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
        try:
            if is_result(json.loads(parsed["result"])):
                ok = True
                break
        except json.JSONDecodeError:
            pass

    flat = output.replace('\\"', '"').replace("\\\\", "\\")
    if '"unitLength": 47' in flat and '"repetitions": 5' in flat:
        ok = True
        break

assert ok, "tool did not return unitLength 47 / repetitions 5"
print("ok: unloop_scan returned unitLength=47 repetitions=5")
PY
